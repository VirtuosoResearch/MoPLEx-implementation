import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import PreTrainedModel, PreTrainedTokenizerBase
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
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

class DPOTrainer():
    
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
        
        # Step 1: Initialize Accelerator
        self.accelerator = Accelerator(
            log_with=config.log_with,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            project_config=ProjectConfiguration(**config.project_kwargs),
            deepspeed_plugin=None,            # disable DeepSpeed
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
            raise ValueError("Reinforce does not support encoder-decoder models.")

        self.is_peft_model = getattr(self.model, "is_peft_model", False)
        config.is_encoder_decoder = self.is_encoder_decoder
        config.is_peft_model = self.is_peft_model

        is_using_tensorboard = config.log_with is not None and config.log_with == "tensorboard"
        current_config = dict(trl_reinforce_trainer_config=config.to_dict()) if not is_using_tensorboard else config.to_dict()
        current_config.update(flatten_dict(additional_config_kwargs or {}))
        
        self.accelerator.init_trackers(
            config.tracker_project_name,
            config=current_config,
            init_kwargs=config.tracker_kwargs,
        )
        self.is_using_text_environment = getattr(config, "use_text_environment", False)
        self.tokenizer = tokenizer
        
        self._signature_columns = None

        # Step 3: Initialize optimizer and data collator
        self.data_collator = DataCollatorForLanguageModeling(self.tokenizer, mlm=False)
        if optimizer is None:
            self.optimizer = Adam(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=self.config.learning_rate,
            )
        else:
            self.optimizer = optimizer

        self.lr_scheduler = lr_scheduler
        if self.lr_scheduler is not None:
            lr_scheduler_class = (
                torch.optim.lr_scheduler._LRScheduler
                if not is_torch_greater_2_0()
                else torch.optim.lr_scheduler.LRScheduler
            )

            if not isinstance(self.lr_scheduler, lr_scheduler_class):
                raise ValueError(
                    "lr_scheduler must be a torch.optim.lr_scheduler._LRScheduler or torch.optim.lr_scheduler.LRScheduler (for torch >= 2.0)"
                )

        if self.config.adap_kl_ctrl:
            self.kl_ctl = AdaptiveKLController(self.config.init_kl_coef, self.config.target, self.config.horizon)
        else:
            self.kl_ctl = FixedKLController(self.config.init_kl_coef)

        # Safety checkers for DS integration
        is_deepspeed_used = self.accelerator.distributed_type == "DEEPSPEED" and hasattr(
            self.accelerator.state, "deepspeed_plugin"
        )

        (
            self.model,
            self.optimizer,
            self.data_collator,
            self.lr_scheduler,
        ) = self.accelerator.prepare(
            self.model,
            self.optimizer,
            self.data_collator,
            self.lr_scheduler,
        )
        if is_deepspeed_used:
            if not self.is_peft_model and not (
                getattr(self.ref_model.pretrained_model, "is_loaded_in_8bit", False)
                or getattr(self.ref_model.pretrained_model, "is_loaded_in_4bit", False)
            ):
                self.ref_model = self._prepare_deepspeed(self.ref_model)
        else:
            self.ref_model = self.accelerator.prepare(self.ref_model)
            
        self.is_distributed = self.accelerator.distributed_type == "MULTI_GPU"

        self.current_step = 0

        if config.push_to_hub_if_best_kwargs:
            if "repo_id" not in config.push_to_hub_if_best_kwargs:
                raise ValueError("You have to specify repo_id in order to push the model to the hub!")
            self.push_to_hub_kwargs = config.push_to_hub_if_best_kwargs
            self.compare_step = 0
            self.highest_reward = torch.tensor(-float("inf"))

        self.current_device = self.accelerator.device
        self.running = RunningMoments(self.accelerator)
        self.save_pstar_at_epoch = getattr(self.config, "save_pstar_at_epoch", -1)
        self.pstar_save_path = getattr(self.config, "pstar_save_path", "./pstar.pt")
        self.has_saved_pstar = False

        self.save_p_at_epoch = getattr(self.config, "save_p_at_epoch", -1)
        self.p_save_path = getattr(self.config, "p_save_path", "./p.pt")
        self.has_saved_p = False

    
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

        dpo_logit_seq = self.config.temperature * (pi_logratios_seq - ref_logratios_seq)  # [B]

        if self.config.ipo_loss:
            dpo_loss_vec = (dpo_logit_seq - 1.0 / (2 * self.config.temperature)) ** 2  # [B]
        else:
            dpo_loss_vec = -F.logsigmoid(dpo_logit_seq)  # [B]

        denom = sample_mask.sum()
        if denom.item() == 0:
            dpo_loss = dpo_loss_vec.mean()
        else:
            dpo_loss = (dpo_loss_vec * sample_mask).sum() / denom

        delta_w = seq_logprob_w - seq_old_logprob_w  # [B]
        delta_l = seq_logprob_l - seq_old_logprob_l  # [B]

        rewards_chosen = self.config.temperature * delta_w.detach()
        rewards_rejected = self.config.temperature * delta_l.detach()
        reward_margin = rewards_chosen - rewards_rejected

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
    
    def step(
        self,
        queries: torch.LongTensor,
        responses_w: torch.LongTensor,
        responses_l: torch.LongTensor,
        preference_mask: Optional[torch.BoolTensor] = None, # (bs,)
    ):
        assert queries.ndim == 2 and responses_w.ndim == 2 and responses_l.ndim == 2
        self.model.train()
        bs = self.config.batch_size
        sub_bs = self.config.mini_batch_size
        assert bs % sub_bs == 0
        
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
        return stats
    
    def end_of_epoch_step(self, epoch: int):
        """Performs tasks at the end of an epoch, like saving models."""
        print(f"Executing end-of-epoch tasks for epoch: {epoch}")

        # save p* (params + grads) for ApproxDPO
        if (
            self.save_pstar_at_epoch >= 0 and
            epoch == self.save_pstar_at_epoch and
            not self.has_saved_pstar and
            self.accelerator.is_main_process
        ):
            self.has_saved_pstar = True
            print(f"Saving p* (params + grads) at epoch {epoch} to {self.pstar_save_path} ...")
            self.save_pstar()

        # save p (params only)
        if (
            hasattr(self, 'save_p_at_epoch') and
            self.save_p_at_epoch >= 0 and
            epoch == self.save_p_at_epoch and
            not getattr(self, 'has_saved_p', False) and
            self.accelerator.is_main_process
        ):
            self.has_saved_p = True
            print(f"Saving p (params only) at epoch {epoch} to {self.p_save_path} ...")
            self.save_p()

    def save_pstar(self):
        import os
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(self.pstar_save_path) if os.path.dirname(self.pstar_save_path) else ".", exist_ok=True)
        self.model.eval()
        device = next(self.model.parameters()).device
        with torch.no_grad():
            params_vector = torch.cat([p.detach().to(device).flatten() for p in self.model.parameters()])
        torch.save({"params": params_vector}, self.pstar_save_path)
        print(f"Saved p* to {self.pstar_save_path}")
        self.model.train()

    def save_p(self):
        import os
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(self.p_save_path) if os.path.dirname(self.p_save_path) else ".", exist_ok=True)
        self.model.eval()
        device = next(self.model.parameters()).device
        with torch.no_grad():
            params_vector = torch.cat([p.detach().to(device).flatten() for p in self.model.parameters()])
        torch.save({"params": params_vector}, self.p_save_path)
        print(f"Saved p to {self.p_save_path}")
        self.model.train()


    def log_stats(
            self,
            stats: dict,
            batch: dict,
            rewards: List[torch.FloatTensor],
            columns_to_log: List[str] = ["query", "response"],
        ):
            """
            A function that logs all the training stats. Call it at the end of each epoch.

            Args:
                stats (dict[str, Any]):
                    A dictionary of training stats.
                batch (dict[str, Any]):
                    A dictionary of batch data, this contains the queries and responses.
                rewards (`List[torch.FloatTensor]`):
                    A tensor of rewards.
            """
            # Log only if we are in the main process
            if self.accelerator.is_main_process:
                logs = {}

                # Log stats
                if not isinstance(rewards, torch.Tensor):
                    rewards = torch.tensor(rewards).to(self.current_device)

                if self.config.log_with == "wandb":
                    import wandb

                    if any([column_to_log not in batch.keys() for column_to_log in columns_to_log]):
                        raise ValueError(f"Columns to log {columns_to_log} are not present in the batch {batch.keys()}.")

                    batch_list = [batch[column_to_log] for column_to_log in columns_to_log]

                    table_rows = [list(r) for r in zip(*batch_list, rewards.cpu().tolist())]
                    logs.update({"game_log": wandb.Table(columns=[*columns_to_log, "reward"], rows=table_rows)})

                logs.update(stats)

                # manually cast in fp32 for bf16 torch tensors
                for k, v in logs.items():
                    if isinstance(v, torch.Tensor) and v.dtype == torch.bfloat16:
                        logs[k] = v.float()

                self.accelerator.log(
                    logs,
                    step=self.current_step if self.config.log_with == "tensorboard" else None,
                )
    
    @torch.no_grad()
    def evaluate(self, eval_dataloader):
        self.model.eval()
        self.ref_model.eval()

        total_loss = 0.0
        total_acc = 0.0
        total_examples = 0

        for batch in eval_dataloader:
            preference_mask = batch.get("preference_mask", None)

            if isinstance(batch["query"], list) or isinstance(batch["query"], tuple):
                queries = self.tokenizer(
                    batch["query"],
                    padding=True,
                    truncation=True,
                    max_length=128,
                    return_tensors="pt",
                ).input_ids.to(self.current_device)

                all_pref = batch["response_w"] + batch["response_l"]
                tokenized = self.tokenizer(
                    all_pref,
                    padding=True,
                    truncation=True,
                    max_length=256,
                    return_tensors="pt",
                ).input_ids.to(self.current_device)

                n_w = len(batch["response_w"])
                responses_w = tokenized[:n_w]
                responses_l = tokenized[n_w:]

                if preference_mask is not None:
                    preference_mask = torch.as_tensor(preference_mask, device=self.current_device)
            else:
                queries = batch["query"].to(self.current_device)
                responses_w = batch["response_w"].to(self.current_device)
                responses_l = batch["response_l"].to(self.current_device)
                if preference_mask is not None:
                    preference_mask = preference_mask.to(self.current_device)

            loss, stats = self._step(
                queries=queries,
                responses_w=responses_w,
                responses_l=responses_l,
                preference_mask=preference_mask,
            )

            batch_size = queries.size(0)
            total_examples += batch_size

            total_loss += loss.item() * batch_size
            batch_acc = stats["policy/classifier_accuracy"].item()
            total_acc += batch_acc * batch_size

        if total_examples == 0:
            mean_loss = 0.0
            mean_acc = 0.0
        else:
            mean_loss = total_loss / total_examples
            mean_acc = total_acc / total_examples

        metrics = {
            "eval/dpo_loss": mean_loss,
            "eval/classification_accuracy": mean_acc,
        }
        return metrics