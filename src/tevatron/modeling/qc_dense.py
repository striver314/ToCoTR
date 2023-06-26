from dataclasses import dataclass
import torch
import torch.nn as nn
from torch import Tensor
import torch.distributed as dist

from transformers import AutoConfig, AutoModel, PreTrainedModel, T5ForConditionalGeneration, T5Model
from transformers.modeling_outputs import ModelOutput

from typing import Optional, Dict

from tevatron.arguments_typo import ModelArguments, DataArguments, \
    DenseTrainingArguments as TrainingArguments
import logging

logger = logging.getLogger(__name__)


@dataclass
class QCDenseOutput(ModelOutput):
    q_reps: Tensor = None
    loss: Tensor = None
    scores: Tensor = None


class QCDenseModel(nn.Module):
    def __init__(
            self,
            lm: PreTrainedModel,
            model_args: ModelArguments = None,
            data_args: DataArguments = None,
            train_args: TrainingArguments = None,
    ):
        super().__init__()

        self.lm = lm
        if model_args.model_type != "t5":
            self.lm_head = nn.Linear(768, 32128, bias=False)
            self.cross_entropy = nn.CrossEntropyLoss(reduction='mean')

        self.model_args = model_args
        self.train_args = train_args
        self.data_args = data_args

        if train_args.negatives_x_device:
            if not dist.is_initialized():
                raise ValueError('Distributed training has not been initialized for representation all gather.')
            self.process_rank = dist.get_rank()
            self.world_size = dist.get_world_size()

    def forward(
            self,
            typo_text: Dict[str, Tensor] = None,
            text: Dict[str, Tensor] = None,
    ):
        if self.training:
            if self.model_args.model_type != "t5":
                typo_q_hidden, typo_q_reps = self.encode_query(typo_text)
                lm_logits = self.lm_head(typo_q_hidden)
                if typo_q_reps is None:
                    return QCDenseOutput(
                        q_reps=typo_q_reps
                    )
                loss = self.cross_entropy(lm_logits.view(-1, lm_logits.size(-1)), text.input_ids.view(-1))
            else:
                text.input_ids[text.input_ids[:, :] == 0] = -100
                loss = self.lm(input_ids = typo_text.input_ids, labels = text.input_ids).loss
            logger.warning("batch qc loss: %f", loss)
            
            if self.train_args.negatives_x_device:
                loss = loss * self.world_size  # counter average weight reduction
            return QCDenseOutput(
                loss=loss,
            )
        else:
            return QCDenseOutput(
                loss=None
            )

    def encode_query(self, qry, typoq=None):
        if qry is None:
            return None, None
        elif self.model_args.model_type == "t5":
            if not typoq:
                raise ValueError('T5 for QC training need typo_text as input.')
            else:
                decoder_input_ids = self.lm._shift_right(qry.input_ids)
                qry_out = self.lm(input_ids=typoq.input_ids, decoder_input_ids=decoder_input_ids, return_dict=True)
                logits = qry_out.logits
                return None, logits
        else:
            qry_out = self.lm(**qry, return_dict=True)
            q_hidden = qry_out.last_hidden_state
        
            q_reps = q_hidden[:, 0]
            return q_hidden, q_reps

    @classmethod
    def build(
            cls,
            model_args: ModelArguments,
            data_args: DataArguments,
            train_args: TrainingArguments,
            **hf_kwargs,
    ):
        if model_args.model_type == "t5":
            # lm_q = T5Model.from_pretrained(model_args.model_name_or_path, **hf_kwargs)
            lm_q = T5ForConditionalGeneration.from_pretrained(model_args.model_name_or_path, **hf_kwargs)
        else:
            lm_q = AutoModel.from_pretrained(model_args.model_name_or_path, **hf_kwargs)

        model = cls(
            lm=lm_q,
            model_args=model_args,
            data_args=data_args,
            train_args=train_args
        )
        return model

    def save(self, output_dir: str):
        self.lm.save_pretrained(output_dir)

    def dist_gather_tensor(self, t: Optional[torch.Tensor]):
        if t is None:
            return None
        t = t.contiguous()

        all_tensors = [torch.empty_like(t) for _ in range(self.world_size)]
        dist.all_gather(all_tensors, t)

        all_tensors[self.process_rank] = t
        all_tensors = torch.cat(all_tensors, dim=0)

        return all_tensors
