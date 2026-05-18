# Copyright 2020-2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
# Full training
python scripts/dpo.py \
    --dataset_name trl-lib/ultrafeedback_binarized \
    --model_name_or_path Qwen/Qwen2-0.5B-Instruct \
    --learning_rate 5.0e-7 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --gradient_checkpointing \
    --logging_steps 25 \
    --eval_strategy steps \
    --eval_steps 50 \
    --output_dir Qwen2-0.5B-DPO \
    --no_remove_unused_columns

# LoRA:
python scripts/dpo.py \
    --dataset_name trl-lib/ultrafeedback_binarized \
    --model_name_or_path Qwen/Qwen2-0.5B-Instruct \
    --learning_rate 5.0e-6 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --gradient_checkpointing \
    --logging_steps 25 \
    --eval_strategy steps \
    --eval_steps 50 \
    --output_dir Qwen2-0.5B-DPO \
    --no_remove_unused_columns \
    --use_peft \
    --lora_r 32 \
    --lora_alpha 16
"""

import logging
import os
import sys

import datasets
import torch
import transformers
from transformers import TrainerCallback, set_seed
from transformers.trainer_utils import get_last_checkpoint

from alignment import (
    DPOConfig,
    ListwiseDPOTrainer,
    MixtureBTTrainer,
    MixtureDPOTrainer,
    MixtureEMDPOTrainer,
    MixturePLConfig,
    ScriptArguments,
    get_dataset,
    get_ranking_dataset,
    get_model,
    get_tokenizer,
)
from alignment.ranking_eval import _dimension_mapping, evaluate_ranking_split, evaluate_ranking_splits
from trl import DPOTrainer, ModelConfig, TrlParser, get_peft_config


logger = logging.getLogger(__name__)


class RankingEvaluationCallback(TrainerCallback):
    """Log full ranking metrics on the configured evaluation split during Trainer.evaluate()."""

    def __init__(self, ranking_dataset, tokenizer, script_args, training_args):
        self.ranking_dataset = ranking_dataset
        self.tokenizer = tokenizer
        self.script_args = script_args
        self.training_args = training_args
        self.eval_split = script_args.dataset_test_split
        self.dimension_to_id = _dimension_mapping(script_args.preference_dimensions, ranking_dataset)
        self.trainer = None

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        del args, state, control, kwargs
        if self.trainer is None or self.eval_split not in self.ranking_dataset:
            return

        split_metrics, _ = evaluate_ranking_split(
            trainer=self.trainer,
            split_dataset=self.ranking_dataset[self.eval_split],
            tokenizer=self.tokenizer,
            training_args=self.training_args,
            dimension_to_id=self.dimension_to_id,
            cluster_alignment=None,
        )
        logged_metrics = {f"ranking_{self.eval_split}/{key}": value for key, value in split_metrics.items()}
        best_model_aliases = {f"eval_{key}": value for key, value in logged_metrics.items()}
        if metrics is not None:
            metrics.update(logged_metrics)
            metrics.update(best_model_aliases)
        self.trainer.log(logged_metrics)


def main(script_args, training_args, model_args):
    # Set seed for reproducibility
    set_seed(training_args.seed)

    ###############
    # Setup logging
    ###############
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    logger.info(f"Model parameters {model_args}")
    logger.info(f"Script parameters {script_args}")
    logger.info(f"Training parameters {training_args}")

    # Check for last checkpoint
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
    if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
        logger.info(f"Checkpoint detected, resuming training at {last_checkpoint=}.")

    ###################
    # Model & Tokenizer
    ###################
    model = get_model(model_args, training_args)
    ref_model = get_model(model_args, training_args)
    tokenizer = get_tokenizer(model_args, training_args)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if script_args.ignore_bias_buffers:
        # torch distributed hack
        model._ddp_params_and_buffers_to_ignore = [
            name for name, buffer in model.named_buffers() if buffer.dtype == torch.bool
        ]

    #########
    # Dataset
    #########
    dataset = get_dataset(script_args)
    ranking_dataset = get_ranking_dataset(script_args) if script_args.run_ranking_eval else None
    # print dataset sizes
    for split in dataset:
        logger.info(f"Loaded {len(dataset[split])} examples from the '{split}' split.")
    if ranking_dataset is not None:
        for split in ranking_dataset:
            logger.info(f"Loaded {len(ranking_dataset[split])} ranking examples from the '{split}' split.")
    
    # print a few examples from the dataset for sanity check
    for split in dataset:
        logger.info(f"Sample examples from the '{split}' split:")
        for i in range(min(12, len(dataset[split]))):
            logger.info(dataset[split][i])
    
    for split in dataset:
        if "messages" in dataset[split].column_names:
            dataset[split] = dataset[split].remove_columns("messages")

    ##########
    # Training
    ##########
    peft_config = get_peft_config(model_args)
    # When using PEFT adapters with DPO, don't pass a separate ref_model.
    # DPOTrainer will handle creating the reference model internally.
    use_listwise = script_args.dataset_format == "listwise" or training_args.listwise
    use_mixture = training_args.use_mixture if hasattr(training_args, "use_mixture") else False

    # Select trainer class
    if use_mixture:
        if getattr(training_args, "mixture_objective", "pl") == "bt":
            trainer_cls = MixtureBTTrainer
        else:
            trainer_cls = MixtureEMDPOTrainer if training_args.mixture_training_mode == "em_only" else MixtureDPOTrainer
        logger.info(
            "Using Mixture DPO trainer with objective=%s, %d clusters, mode=%s, reward_backend=%s, mixture_nll_weight=%.4f",
            getattr(training_args, "mixture_objective", "pl"),
            training_args.num_clusters,
            training_args.mixture_training_mode,
            training_args.mixture_reward_backend,
            training_args.mixture_nll_weight,
        )
    elif use_listwise:
        trainer_cls = ListwiseDPOTrainer
        logger.info("Using listwise DPO trainer for dimensions '%s'", script_args.preference_dimensions)
    else:
        trainer_cls = DPOTrainer
        logger.info("Using standard pairwise DPO trainer")

    trainer_kwargs = dict(
        model=model,
        ref_model=None if peft_config is not None else ref_model,
        args=training_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    if use_mixture:
        trainer_kwargs["mixture_config"] = training_args
    if use_listwise:
        trainer_kwargs["listwise_beta"] = training_args.listwise_beta

    trainer = trainer_cls(**trainer_kwargs)
    if (
        ranking_dataset is not None
        and script_args.ranking_eval_during_training
        and training_args.eval_strategy != "no"
        and script_args.dataset_test_split in ranking_dataset
    ):
        ranking_callback = RankingEvaluationCallback(
            ranking_dataset=ranking_dataset,
            tokenizer=tokenizer,
            script_args=script_args,
            training_args=training_args,
        )
        ranking_callback.trainer = trainer
        trainer.add_callback(ranking_callback)

    logger.info("*** Train ***")
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    metrics = train_result.metrics
    metrics["train_samples"] = len(dataset[script_args.dataset_train_split])
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    if training_args.eval_strategy != "no":
        metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    if ranking_dataset is not None:
        logger.info("*** Ranking evaluation on available ranking splits ***")
        ranking_results = evaluate_ranking_splits(
            trainer=trainer,
            ranking_dataset=ranking_dataset,
            tokenizer=tokenizer,
            script_args=script_args,
            training_args=training_args,
        )
        for split_name, split_metrics in ranking_results.items():
            trainer.log({f"ranking_{split_name}/{key}": value for key, value in split_metrics.items()})
            trainer.log_metrics(f"ranking_{split_name}", split_metrics)
            trainer.save_metrics(f"ranking_{split_name}", split_metrics)

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)


if __name__ == "__main__":
    parser = TrlParser((ScriptArguments, MixturePLConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)
