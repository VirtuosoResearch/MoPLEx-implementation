import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import PreTrainedModel, PreTrainedTokenizerBase
from transformers import logging
from typing import Optional, Union, Dict, List, Tuple, NamedTuple, Callable, Iterable, Any, Mapping
import numpy as np
import random
import typing
from trainers.dpo_config import DPOConfig
from trainers.utils import (
    RunningMoments,
    AdaptiveKLController,
    FixedKLController,
    logprobs_from_logits,
    entropy_from_logits,
    masked_mean,
    masked_mean_sum,
    flatten_dict,
    set_seed,
    is_torch_greater_2_0,
    create_reference_model,
    empty_cache,
    empty_cache_decorator
)
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, is_deepspeed_available
import warnings
from transformers import DataCollatorForLanguageModeling
from torch.optim import Adam
import sys
import inspect
from packaging import version
import datasets
from copy import deepcopy
import tqdm

PreTrainedModelWrapper = typing.Union[nn.Module, nn.DataParallel]

class ApproxDPOTrainer():
    def __init__(
        self,
        config: DPOConfig = None,
        model: PreTrainedModelWrapper = None,
        ref_model: Optional[PreTrainedModelWrapper] = None,
        tokenizer: PreTrainedTokenizerBase = None,
        dataset: Optional[Union[torch.utils.data.Dataset, Dataset]] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        data_collator: Optional[typing.Callable] = None,
        num_shared_layers: Optional[int] = None,
        lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        additional_config_kwargs: Optional[dict] = None,
    ):
        self.config = config
        set_seed(self.config.seed)

        # Accelerator setup
        self.accelerator = Accelerator(
            log_with=config.log_with,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            project_config=ProjectConfiguration(**config.project_kwargs),
            **config.accelerator_kwargs,
        )

        self.model = model
        if ref_model is None:
            self.ref_model = create_reference_model(self.model, num_shared_layers=num_shared_layers)
        else:
            self.ref_model = ref_model
        self.model_params = filter(lambda p: p.requires_grad, self.model.parameters())
        self.is_encoder_decoder = hasattr(self.model, "is_encoder_decoder")
        if self.is_encoder_decoder:
            raise ValueError("ApproxDPOTrainer does not support encoder-decoder models.")

        self.is_peft_model = getattr(self.model, "is_peft_model", False)
        config.is_encoder_decoder = self.is_encoder_decoder
        config.is_peft_model = self.is_peft_model

        self.accelerator.init_trackers(
            config.tracker_project_name,
            config=config.to_dict(),
            init_kwargs=config.tracker_kwargs,
        )
        self.tokenizer = tokenizer

        self.dataset = dataset
        self._signature_columns = None
        if self.dataset is not None:
            self.dataloader = self.prepare_dataloader(self.dataset, data_collator)
        else:
            self.dataloader = None

        self.data_collator = DataCollatorForLanguageModeling(self.tokenizer, mlm=False)
        if optimizer is None:
            self.optimizer = Adam(self.model_params, lr=self.config.learning_rate)
        else:
            self.optimizer = optimizer

        self.lr_scheduler = lr_scheduler
        if self.config.adap_kl_ctrl:
            self.kl_ctl = AdaptiveKLController(self.config.init_kl_coef, self.config.target, self.config.horizon)
        else:
            self.kl_ctl = FixedKLController(self.config.init_kl_coef)

        (
            self.model,
            self.optimizer,
            self.data_collator,
            self.dataloader,
            self.lr_scheduler,
        ) = self.accelerator.prepare(
            self.model,
            self.optimizer,
            self.data_collator,
            self.dataloader,
            self.lr_scheduler,
        )
        self.ref_model = self.accelerator.prepare(self.ref_model)
        self.is_distributed = self.accelerator.distributed_type == "MULTI_GPU"
        self.current_step = 0
        self.current_device = self.accelerator.device
        self.running = RunningMoments(self.accelerator)

    def _set_signature_columns_if_needed(self):
        if self._signature_columns is None:
            signature = inspect.signature(self.model.forward)
            self._signature_columns = list(signature.parameters.keys())
            self._signature_columns += ["label", "query", "response"]

    def _remove_unused_columns(self, dataset: "Dataset"):
        if not self.config.remove_unused_columns:
            return dataset
        self._set_signature_columns_if_needed()
        signature_columns = self._signature_columns

        ignored_columns = list(set(dataset.column_names) - set(signature_columns))

        columns = [k for k in signature_columns if k in dataset.column_names]

        if version.parse(datasets.__version__) < version.parse("1.4.0"):
            dataset.set_format(
                type=dataset.format["type"],
                columns=columns,
                format_kwargs=dataset.format["format_kwargs"],
            )
            return dataset
        else:
            return dataset.remove_columns(ignored_columns)

    def prepare_dataloader(self, dataset: Union[torch.utils.data.Dataset, Dataset], data_collator=None):
        if isinstance(dataset, Dataset):
            dataset = self._remove_unused_columns(dataset)
        dataloader = DataLoader(
            dataset,
            batch_size=self.config.dataloader_batch_size or self.config.batch_size,
            collate_fn=data_collator,
            shuffle=True,
            drop_last=True,
        )
        return dataloader

    def _step(
        self,
        queries: torch.LongTensor,
        responses_w: torch.LongTensor,
        responses_l: torch.LongTensor,
        return_stats: bool = False,
        preference_mask: Optional[torch.BoolTensor] = None,
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
            entropy = entropy_from_logits(logits)

            return logprobs, old_logprobs, attn_mask_shifted, entropy

        logprobs_w, old_logprobs_w, attn_shift_w, entropy_w = process_input_ids(input_ids_w)
        logprobs_l, old_logprobs_l, attn_shift_l, entropy_l = process_input_ids(input_ids_l)

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

        # Compute sequence-level logprobs
        seq_logprob_w = (logprobs_w * mask_w).sum(dim=1)          # [B]
        seq_logprob_l = (logprobs_l * mask_l).sum(dim=1)          # [B]
        seq_old_logprob_w = (old_logprobs_w * mask_w).sum(dim=1)  # [B]
        seq_old_logprob_l = (old_logprobs_l * mask_l).sum(dim=1)  # [B]

        # Compute sequence-level logratios
        pi_logratios_seq = seq_logprob_w - seq_logprob_l               # [B]
        ref_logratios_seq = seq_old_logprob_w - seq_old_logprob_l      # [B]

        # Compute r_hat (score difference at theta*)
        r_hat_diff = self.config.temperature * (pi_logratios_seq.detach() - ref_logratios_seq.detach())  # [B]

        # Compute gradients of sequence-level log_probs w.r.t model parameters
        grads_w = torch.autograd.grad(
            outputs=seq_logprob_w.sum(),  # Sum over batch for gradient computation
            inputs=self.model.parameters(),
            create_graph=True, retain_graph=True, allow_unused=True
        )
        grads_l = torch.autograd.grad(
            outputs=seq_logprob_l.sum(),  # Sum over batch for gradient computation
            inputs=self.model.parameters(),
            create_graph=True, retain_graph=True, allow_unused=True
        )

        # Compute theta - theta_star and grad_diff dot product in batches
        first_order_term = torch.tensor(0.0, device=self.current_device)
        for (p, p_star, gw, gl) in zip(self.model.parameters(), self.ref_model.parameters(), grads_w, grads_l):
            if p.requires_grad and gw is not None and gl is not None:
                theta_diff = (p - p_star).detach()
                grad_diff = (gw - gl)
                # Reshape to vectors
                theta_diff_flat = theta_diff.view(-1)
                grad_diff_flat = grad_diff.view(-1)
                # Incrementally accumulate dot product
                first_order_term += torch.dot(grad_diff_flat, theta_diff_flat)

        # Final approximated logits for DPO loss (sequence-level)
        # Use beta if available in config, otherwise use 1.0
        beta = getattr(self.config, 'beta', 1.0)
        approx_logits = r_hat_diff + beta * first_order_term  # [B]

        if self.config.ipo_loss:
            dpo_loss_vec = (approx_logits - 1.0 / (2 * self.config.temperature)) ** 2  # [B]
        else:
            dpo_loss_vec = -F.logsigmoid(approx_logits)  # [B]

        # Apply preference_mask for weighted average
        denom = sample_mask.sum()
        if denom.item() == 0:
            dpo_loss = dpo_loss_vec.mean()
        else:
            dpo_loss = (dpo_loss_vec * sample_mask).sum() / denom

        if return_stats:
            delta_w = seq_logprob_w - seq_old_logprob_w  # [B]
            delta_l = seq_logprob_l - seq_old_logprob_l  # [B]
            rewards_chosen = self.config.temperature * delta_w.detach()
            rewards_rejected = self.config.temperature * delta_l.detach()
            reward_margin = rewards_chosen - rewards_rejected

            stats = dict(
                loss=dict(dpo_loss=dpo_loss.detach()),
                policy=dict(
                    entropy=torch.cat((entropy_w, entropy_l), dim=0).detach(),
                    rewards_chosen=rewards_chosen.mean().detach(),
                    rewards_rejected=rewards_rejected.mean().detach(),
                    reward_margin=reward_margin.mean().detach(),
                    logprobs_w=seq_logprob_w.mean().detach(),
                    logprobs_l=seq_logprob_l.mean().detach(),
                    pi_logratios=pi_logratios_seq.mean().detach(),
                    ref_logratios=ref_logratios_seq.mean().detach(),
                    dpo_logit_mean=approx_logits.mean().detach(),
                    classifier_accuracy=(reward_margin > 0).float().mean().detach(),
                )
            )
            return dpo_loss, flatten_dict(stats)
        else:
            return dpo_loss

    def step(
        self,
        queries: torch.LongTensor,
        responses_w: torch.LongTensor,
        responses_l: torch.LongTensor,
        preference_mask: Optional[torch.BoolTensor] = None,
    ):
        self.model.train()
        bs = self.config.batch_size
        sub_bs = self.config.mini_batch_size
        assert bs % sub_bs == 0
        for i in tqdm.tqdm(range(0, bs, sub_bs), desc="Training with Minibatches", leave=False):
            queries_ = queries[i:i + sub_bs]
            responses_w_ = responses_w[i:i + sub_bs]
            responses_l_ = responses_l[i:i + sub_bs]
            preference_mask_ = preference_mask[i:i + sub_bs] if preference_mask is not None else None

            loss, stats = self._step(
                queries=queries_,
                responses_w=responses_w_,
                responses_l=responses_l_,
                return_stats=True,
                preference_mask=preference_mask_,
            )

            self.optimizer.zero_grad()
            self.accelerator.backward(loss)
            self.optimizer.step()
            self.current_step += 1
        return stats
