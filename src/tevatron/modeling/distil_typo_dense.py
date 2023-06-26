import torch
import torch.nn as nn
from torch import Tensor
import torch.distributed as dist   #TODO: dist train
from transformers import PreTrainedModel
from typing import Dict

from .dense import DenseModel, DenseOutput
from tevatron.arguments import ModelArguments, DataArguments, \
    DenseTrainingArguments as TrainingArguments
import logging

logger = logging.getLogger(__name__)


class DistilTypoDenseModel(DenseModel):
    def __init__(
            self,
            lm_q: PreTrainedModel,
            lm_p: PreTrainedModel,
            pooler: nn.Module = None,
            model_args: ModelArguments = None,
            data_args: DataArguments = None,
            train_args: TrainingArguments = None,
    ):
        super().__init__(lm_q, lm_p, pooler, model_args, data_args, train_args)
        self.KL = nn.KLDivLoss(reduction='batchmean', log_target=True)
        self.MSE = nn.MSELoss()
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(
            self,
            query: Dict[str, Tensor] = None,
            typo_query: Dict[str, Tensor] = None,
            passage: Dict[str, Tensor] = None,
    ):
        q_hidden, q_reps = self.encode_query(query)      # query: ptbs * 32 * 50
        if self.model_args.model_type == "t5":
            typo_q_hidden, typo_q_reps = self.encode_query(query, typo_query)
        else:
            typo_q_hidden, typo_q_reps = self.encode_query(typo_query)
        p_hidden, p_reps = self.encode_passage(passage)  # passage: 16 * 128 * 50

        if q_reps is None or p_reps is None:
            return DenseOutput(
                q_reps=q_reps,
                p_reps=p_reps
            )

        if self.training:
            if self.train_args.negatives_x_device:
                q_reps = self.dist_gather_tensor(q_reps)
                typo_q_reps = self.dist_gather_tensor(typo_q_reps)
                p_reps = self.dist_gather_tensor(p_reps)

            effective_bsz = self.train_args.per_device_train_batch_size * self.world_size \
                if self.train_args.negatives_x_device \
                else self.train_args.per_device_train_batch_size

            scores = torch.matmul(q_reps, p_reps.transpose(0, 1))
            scores = scores.view(effective_bsz, -1)

            typo_scores = torch.matmul(typo_q_reps, p_reps.transpose(0, 1))
            typo_scores = typo_scores.view(effective_bsz, -1)

            target = torch.arange(
                scores.size(0),
                device=scores.device,
                dtype=torch.long
            )
            target = target * self.data_args.train_n_passages
            ce_loss = self.cross_entropy(scores, target)
            kl_loss = self.KL(self.log_softmax(typo_scores),
                              self.log_softmax(scores.detach()/self.train_args.temperature))
            loss = ce_loss + kl_loss

            if self.train_args.negatives_x_device:
                loss = loss * self.world_size  # counter average weight reduction
            return DenseOutput(
                loss=loss,
                scores=scores,
                q_reps=q_reps,
                p_reps=p_reps
            )

        else:
            loss = None
            if query and passage:
                scores = (q_reps * p_reps).sum(1)
            else:
                scores = None

            return DenseOutput(
                loss=loss,
                scores=scores,
                q_reps=q_reps,
                p_reps=p_reps
            )
