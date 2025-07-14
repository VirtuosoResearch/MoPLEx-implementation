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
        input_ids_w = torch.cat((queries, responses_w), dim=1)
        input_ids_l = torch.cat((queries, responses_l), dim=1)
        mask_w = torch.cat((torch.zeros_like(queries), torch.ones_like(responses_w)), dim=1)[:, :-1]
        mask_l = torch.cat((torch.zeros_like(queries), torch.ones_like(responses_l)), dim=1)[:, :-1]
        mask = mask_w
        if preference_mask is not None:
            preference_mask = preference_mask.unsqueeze(1).repeat(1, mask.shape[1])
            mask = mask * preference_mask.to(mask.dtype).to(mask.device)

        def process_input_ids(input_ids):
            input_data = {"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids)}
            logits, _, _ = self.model(**input_data)
            with torch.no_grad():
                old_logits, _, _ = self.ref_model(**input_data)
                old_logprobs = logprobs_from_logits(old_logits[:, :-1, :], input_ids[:, 1:])
            logprobs = logprobs_from_logits(logits[:, :-1, :], input_ids[:, 1:])
            entropy = entropy_from_logits(logits)
            return logprobs, old_logprobs, entropy, logits

        logprobs_w, old_logprobs_w, entropy_w, logits_w = process_input_ids(input_ids_w)
        logprobs_l, old_logprobs_l, entropy_l, logits_l = process_input_ids(input_ids_l)
        
        pi_logratios = logprobs_w - logprobs_l
        ref_logratios = old_logprobs_w - old_logprobs_l
        dpo_logit = self.config.temperature * (pi_logratios - ref_logratios)

        if self.config.use_approx_dpo:
            # Approximate DPO loss using Taylor expansion
            sigmoid_dpo_logit = torch.sigmoid(dpo_logit.detach())
            grad_term = (1 - sigmoid_dpo_logit)
            taylor_loss = grad_term * dpo_logit
            dpo_loss = masked_mean(taylor_loss, mask)
        else:
            dpo_loss = -F.logsigmoid(dpo_logit)
            dpo_loss = masked_mean(dpo_loss, mask)

        if return_stats:
            stats = dict(
                loss=dict(dpo_loss=dpo_loss.detach()),
                policy=dict(
                    entropy=torch.cat((entropy_w, entropy_l), dim=0).detach(),
                    pi_logratios=torch.mean(pi_logratios).detach(),
                    ref_logratios=torch.mean(ref_logratios).detach(),
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
