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


class RodrTypoDenseModel(DenseModel):
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

        self.mse = torch.nn.MSELoss(reduction='mean')
        self.kl = torch.nn.KLDivLoss(reduction='batchmean')

        # loss print
        self.total_loss = 0.
        self.total_qvp_nll_loss = 0.
        self.total_qp_lra_loss = 0.
        self.total_pq_lra_loss = 0.

    def forward(
            self,
            query: Dict[str, Tensor] = None,
            typo_query: Dict[str, Tensor] = None,
            passage: Dict[str, Tensor] = None,
            original_reps: Tensor = None,
            current_step: int = None
    ):
        ## The default ordering of encoding forward: original query -> passage -> query variation
        ## It may affect the DR training slightly.
        if self.data_args.rodr_training:   ## local ranking alignment
            assert typo_query is not None
            assert original_reps is not None
            q_hidden, q_reps = None, None
            p_hidden, p_reps = self.encode_passage(passage)
            if self.model_args.model_type == "t5":
                typo_q_hidden, typo_q_reps = self.encode_query(query, typo_query)
            else:
                typo_q_hidden, typo_q_reps = self.encode_query(typo_query)
        else:
            raise NotImplementedError('Please choose the correct training mode.')

        if (q_reps is None and typo_q_reps is None) or p_reps is None:
            return DenseOutput(
                q_reps=q_reps,
                typo_q_reps=typo_q_reps,
                p_reps=p_reps
            )

        if self.training:
            if self.train_args.negatives_x_device:
                if q_reps is not None:
                    q_reps = self.dist_gather_tensor(q_reps)
                p_reps = self.dist_gather_tensor(p_reps)
                if typo_q_reps is not None:
                    typo_q_reps = self.dist_gather_tensor(typo_q_reps)
                if original_reps is not None:
                    original_reps.q_reps = self.dist_gather_tensor(original_reps.q_reps)
                    original_reps.p_reps = self.dist_gather_tensor(original_reps.p_reps)

            effective_bsz = self.train_args.per_device_train_batch_size * self.world_size \
                if self.train_args.negatives_x_device \
                else self.train_args.per_device_train_batch_size

            if self.data_args.rodr_training:
                ### qv_nll
                qvp_scores = torch.matmul(typo_q_reps, p_reps.transpose(0, 1))
                target = torch.arange(qvp_scores.size(0), device=qvp_scores.device, dtype=torch.long)
                target = target * self.data_args.train_n_passages
                qvp_nll_loss = self.cross_entropy(qvp_scores, target)

                ### qp_lra and pq_lra
                ori_qp_scores = torch.matmul(original_reps.q_reps, original_reps.p_reps.transpose(0, 1))
                ori_pq_scores = torch.matmul(original_reps.p_reps, original_reps.q_reps.transpose(0, 1))
                pqv_scores = torch.matmul(p_reps, typo_q_reps.transpose(0, 1))
                qp_lra_loss = self.kl(qvp_scores.log_softmax(dim=-1), ori_qp_scores.softmax(dim=-1))
                pq_lra_loss = self.kl(pqv_scores.log_softmax(dim=-1), ori_pq_scores.softmax(dim=-1))

                loss = self.train_args.w1 * qvp_nll_loss + self.train_args.w2 * qp_lra_loss + self.train_args.w3 * pq_lra_loss

                if current_step % self.train_args.logging_steps == 0:
                    self.total_loss = 0.
                    self.total_qvp_nll_loss = 0.
                    self.total_qp_lra_loss = 0.
                    self.total_pq_lra_loss = 0.

                self.total_loss += loss.item()
                self.total_qvp_nll_loss += qvp_nll_loss.item()
                self.total_qp_lra_loss += qp_lra_loss.item()
                self.total_pq_lra_loss += pq_lra_loss.item()

                if (current_step + 1) % self.train_args.logging_steps == 0:
                    logger.info('****print loss****')
                    logger.info('%s: nll loss: %s', str(current_step + 1), str(self.total_qvp_nll_loss / self.train_args.logging_steps))
                    logger.info('%s: qp lra loss: %s', str(current_step + 1), str(self.total_qp_lra_loss / self.train_args.logging_steps))
                    logger.info('%s: pq lra loss: %s', str(current_step + 1), str(self.total_pq_lra_loss / self.train_args.logging_steps))
                    logger.info('%s: loss: %s', str(current_step + 1), str(self.total_loss / self.train_args.logging_steps))

            else:
                raise NotImplementedError

            if self.train_args.negatives_x_device:
                loss = loss * self.world_size  # counter average weight reduction

            return DenseOutput(
                loss=loss,
                q_reps=q_reps,
                typo_q_reps=typo_q_reps,
                p_reps=p_reps
            )

        else:
            return DenseOutput(
                q_reps=q_reps,
                typo_q_reps=typo_q_reps,
                p_reps=p_reps
            )
