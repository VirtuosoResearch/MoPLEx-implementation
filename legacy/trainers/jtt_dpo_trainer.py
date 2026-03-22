"""
JTT-DPO Trainer: Just Train Twice for DPO

Adapts the Just Train Twice (JTT) method to DPO.
This trainer extends DPOTrainer to support float preference_mask for upweighting.
"""

import torch
from torch import nn
import torch.nn.functional as F
from trainers.dpo_trainer import DPOTrainer
from trainers.dpo_config import DPOConfig
from trainers.utils import logprobs_from_logits
from typing import Optional
from torch.utils.data import Dataset


class JTTDPOTrainer(DPOTrainer):
    """
    JTT-DPO Trainer
    
    Extends DPOTrainer to support float preference_mask for upweighting error examples.
    The preference_mask can be float values (not just 0/1) to allow different weights.
    """
    
    def step(
        self,
        queries: torch.LongTensor,
        responses_w: torch.LongTensor,
        responses_l: torch.LongTensor,
        preference_mask: Optional[torch.Tensor] = None,  # Can be float or bool
    ):
        """
        Override step to accept float preference_mask for upweighting.
        """
        assert queries.ndim == 2 and responses_w.ndim == 2 and responses_l.ndim == 2
        self.model.train()
        bs = self.config.batch_size
        sub_bs = self.config.mini_batch_size
        assert bs % sub_bs == 0
        
        all_stats = []
        for i in range(0, bs, sub_bs):
            queries_ = queries[i : i + sub_bs]
            responses_w_ = responses_w[i : i + sub_bs]
            responses_l_ = responses_l[i : i + sub_bs]
            preference_mask_ = preference_mask[i : i + sub_bs] if preference_mask is not None else None

            loss, stats = self._step(
                queries=queries_,
                responses_w=responses_w_,
                responses_l=responses_l_,
                preference_mask=preference_mask_,
            )
            
            self.optimizer.zero_grad()
            self.accelerator.backward(loss)
            self.optimizer.step()
            self.current_step += 1
            
            all_stats.append(stats)
        
        # Aggregate stats
        aggregated_stats = {}
        for key in all_stats[0].keys():
            values = [s[key] for s in all_stats if key in s]
            if values:
                if isinstance(values[0], torch.Tensor):
                    aggregated_stats[key] = torch.stack(values).mean()
                else:
                    aggregated_stats[key] = sum(values) / len(values)
        
        return aggregated_stats
    
    def _step(
        self,
        queries: torch.LongTensor,
        responses_w: torch.LongTensor,
        responses_l: torch.LongTensor,
        preference_mask: Optional[torch.Tensor] = None,  # Can be float or bool
    ):
        """
        Override _step to support float preference_mask for upweighting.
        """
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

        # Support both float and bool preference_mask
        if preference_mask is not None:
            if preference_mask.dtype == torch.bool:
                sample_mask = preference_mask.to(logprobs_w.device).float()  # [B]
            else:
                sample_mask = preference_mask.to(logprobs_w.device).float()  # [B] (already float)
        else:
            sample_mask = torch.ones(logprobs_w.size(0), device=logprobs_w.device)

        seq_logprob_w = (logprobs_w * mask_w).sum(dim=1)          # [B]
        seq_logprob_l = (logprobs_l * mask_l).sum(dim=1)          # [B]
        seq_old_logprob_w = (old_logprobs_w * mask_w).sum(dim=1)  # [B]
        seq_old_logprob_l = (old_logprobs_l * mask_l).sum(dim=1)  # [B]

        pi_logratios_seq = seq_logprob_w - seq_logprob_l               # [B]
        ref_logratios_seq = seq_old_logprob_w - seq_old_logprob_l      # [B]

        dpo_logit_seq = self.config.temperature * (pi_logratios_seq - ref_logratios_seq)  # [B]

        if self.config.ipo_loss:
            dpo_loss_vec = (dpo_logit_seq - 1.0 / (2 * self.config.temperature)) ** 2  # [B]
        else:
            dpo_loss_vec = -F.logsigmoid(dpo_logit_seq)  # [B]

        denom = sample_mask.sum()
        if denom.item() == 0:
            dpo_loss = dpo_loss_vec.mean()
        else:
            # Weighted average using sample_mask
            dpo_loss = (dpo_loss_vec * sample_mask).sum() / denom

        delta_w = seq_logprob_w - seq_old_logprob_w  # [B]
        delta_l = seq_logprob_l - seq_old_logprob_l  # [B]

        rewards_chosen = self.config.temperature * delta_w.detach()
        rewards_rejected = self.config.temperature * delta_l.detach()
        reward_margin = rewards_chosen - rewards_rejected

        from trainers.utils import flatten_dict
        stats = dict(
            loss=dict(
                dpo_loss=dpo_loss.detach(),
            ),
            policy=dict(
                rewards_chosen=rewards_chosen.mean().detach(),
                rewards_rejected=rewards_rejected.mean().detach(),
                reward_margin=reward_margin.mean().detach(),
                logprobs_w=seq_logprob_w.mean().detach(),
                logprobs_l=seq_logprob_l.mean().detach(),
                dpo_logit_mean=dpo_logit_seq.mean().detach(),
                classifier_accuracy=(reward_margin > 0).float().mean().detach(),
            ),
        )
        return dpo_loss, flatten_dict(stats)

