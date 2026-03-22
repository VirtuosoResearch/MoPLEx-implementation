"""
SPL Trainer: Soft Preference Learning

SPL decouples the KL-regularization term into separate cross-entropy and entropy terms.
The key difference from DPO is that SPL uses different coefficients:
- α for the policy log ratio: α * (log π_θ(y_w|x) - log π_θ(y_l|x))
- β for the reference log ratio: -β * (log π_ref(y_w|x) - log π_ref(y_l|x))

Standard DPO uses β for both terms.
"""

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import PreTrainedModel, PreTrainedTokenizerBase
from typing import Optional, Union, Dict, List, Tuple, Any
import numpy as np
from dataclasses import dataclass

from trainers.dpo_trainer import DPOTrainer
from trainers.dpo_config import DPOConfig
from trainers.utils import logprobs_from_logits, flatten_dict
import typing

PreTrainedModelWrapper = typing.Union[nn.Module, nn.DataParallel]


@dataclass
class SPLConfig(DPOConfig):
    """Configuration for SPL training."""
    
    alpha: float = 1.0
    """Coefficient for policy log ratio (diversity control)"""
    
    beta: float = 0.1
    """Coefficient for reference log ratio (reference model prior strength)"""


class SPLTrainer(DPOTrainer):
    """
    SPL Trainer: Soft Preference Learning
    
    SPL objective (DPO-style):
    max_π E_{y,y'~D} [log σ(α log(π(y|x)/π(y'|x)) - β log(π_ref(y|x)/π_ref(y'|x)))]
    
    Key difference from DPO:
    - DPO: β * (log π_θ - log π_ref)
    - SPL: α * log π_θ - β * log π_ref
    """
    
    def _step(
        self,
        queries: torch.LongTensor,
        responses_w: torch.LongTensor,
        responses_l: torch.LongTensor,
        preference_mask: Optional[torch.BoolTensor] = None,  # (bs,)
    ):
        input_ids_w = torch.cat((queries, responses_w), dim=1)  # [B, Lq+Lw]
        input_ids_l = torch.cat((queries, responses_l), dim=1)  # [B, Lq+Ll]
        pad_id = self.tokenizer.pad_token_id

        def process_input_ids(input_ids):
            # attention_mask: pad=0, others=1
            attention_mask = (input_ids != pad_id).long()  # [B, L]
            input_data = {"input_ids": input_ids, "attention_mask": attention_mask}

            logits, _, _ = self.model(**input_data)
            with torch.no_grad():
                old_logits, _, _ = self.ref_model(**input_data)
                old_logprobs = logprobs_from_logits(old_logits[:, :-1, :], input_ids[:, 1:])  # [B, L-1]
            logprobs = logprobs_from_logits(logits[:, :-1, :], input_ids[:, 1:])  # [B, L-1]
            attn_mask_shifted = attention_mask[:, 1:]  # [B, L-1]

            return logprobs, old_logprobs, attn_mask_shifted

        logprobs_w, old_logprobs_w, attn_shift_w = process_input_ids(input_ids_w)
        logprobs_l, old_logprobs_l, attn_shift_l = process_input_ids(input_ids_l)

        q_len = queries.size(1)
        mask_w = attn_shift_w.clone()  # [B, L-1]
        mask_l = attn_shift_l.clone()
        if q_len > 1:
            mask_w[:, :q_len - 1] = 0
            mask_l[:, :q_len - 1] = 0

        if preference_mask is not None:
            sample_mask = preference_mask.to(logprobs_w.device).float()  # [B]
        else:
            sample_mask = torch.ones(logprobs_w.size(0), device=logprobs_w.device)

        seq_logprob_w = (logprobs_w * mask_w).sum(dim=1)          # [B]
        seq_logprob_l = (logprobs_l * mask_l).sum(dim=1)          # [B]
        seq_old_logprob_w = (old_logprobs_w * mask_w).sum(dim=1)  # [B]
        seq_old_logprob_l = (old_logprobs_l * mask_l).sum(dim=1)  # [B]

        pi_logratios_seq = seq_logprob_w - seq_logprob_l               # [B]
        ref_logratios_seq = seq_old_logprob_w - seq_old_logprob_l      # [B]

        # SPL: α * log π_θ - β * log π_ref (different from DPO: β * (log π_θ - log π_ref))
        alpha = getattr(self.config, 'alpha', 1.0)
        beta = getattr(self.config, 'beta', 0.1)
        spl_logit_seq = alpha * pi_logratios_seq - beta * ref_logratios_seq  # [B]

        if self.config.ipo_loss:
            # For IPO, use average of alpha and beta as temperature (or can use separate logic)
            avg_temperature = (alpha + beta) / 2.0
            spl_loss_vec = (spl_logit_seq - 1.0 / (2 * avg_temperature)) ** 2  # [B]
        else:
            spl_loss_vec = -F.logsigmoid(spl_logit_seq)  # [B]

        denom = sample_mask.sum()
        if denom.item() == 0:
            spl_loss = spl_loss_vec.mean()
        else:
            spl_loss = (spl_loss_vec * sample_mask).sum() / denom

        delta_w = seq_logprob_w - seq_old_logprob_w  # [B]
        delta_l = seq_logprob_l - seq_old_logprob_l  # [B]

        # For stats, use alpha for chosen and rejected rewards (can be adjusted)
        rewards_chosen = alpha * delta_w.detach()
        rewards_rejected = alpha * delta_l.detach()
        reward_margin = rewards_chosen - rewards_rejected

        stats = dict(
            loss=dict(
                spl_loss=spl_loss.detach(),
            ),
            policy=dict(
                rewards_chosen=rewards_chosen.mean().detach(),
                rewards_rejected=rewards_rejected.mean().detach(),
                reward_margin=reward_margin.mean().detach(),
                logprobs_w=seq_logprob_w.mean().detach(),
                logprobs_l=seq_logprob_l.mean().detach(),
                spl_logit_mean=spl_logit_seq.mean().detach(),
                classifier_accuracy=(reward_margin > 0).float().mean().detach(),
            ),
            spl=dict(
                alpha=alpha,
                beta=beta,
                pi_logratio_mean=pi_logratios_seq.mean().detach(),
                ref_logratio_mean=ref_logratios_seq.mean().detach(),
            ),
        )
        return spl_loss, flatten_dict(stats)

