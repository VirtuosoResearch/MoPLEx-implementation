from __future__ import annotations

import logging
import os
from pathlib import Path
import sys
import time
import types

import datasets
import torch
import transformers
from transformers import set_seed
from transformers.trainer_utils import get_last_checkpoint
from trl import ModelConfig, TrlParser

from alignment import ScriptArguments, get_ranking_dataset, get_tokenizer
from alignment.gpo import (
    GPOConfig,
    GPOPairwiseDataCollator,
    GPORankingEvaluationCallback,
    GPORewardTrainer,
    evaluate_gpo_ranking_splits,
    listwise_to_gpo_pairwise_dataset,
)


logger = logging.getLogger(__name__)

_BYTES_PER_MIB = 1024**2


def _ensure_general_preference_on_path() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    gp_root = repo_root.parent / "general-preference-model"
    if gp_root.exists():
        sys.path.insert(0, str(gp_root))


def _load_gp_reward_model(model_args: ModelConfig, training_args: GPOConfig):
    _ensure_general_preference_on_path()
    if "transformers.deepspeed" not in sys.modules:
        try:
            from transformers.integrations.deepspeed import HfDeepSpeedConfig

            deepspeed_compat = types.ModuleType("transformers.deepspeed")
            deepspeed_compat.HfDeepSpeedConfig = HfDeepSpeedConfig
            sys.modules["transformers.deepspeed"] = deepspeed_compat
        except ImportError:
            pass
    try:
        from general_preference.models import get_reward_model
    except ImportError as exc:
        raise ImportError(
            "Could not import official `general_preference` package. "
            "Expected it at ../general-preference-model or on PYTHONPATH."
        ) from exc

    use_peft = bool(getattr(model_args, "use_peft", False))
    lora_rank = int(getattr(model_args, "lora_r", 0) or 0) if use_peft else 0
    target_modules = getattr(model_args, "lora_target_modules", None)
    attn_implementation = getattr(model_args, "attn_implementation", None)

    model = get_reward_model(
        model_args.model_name_or_path,
        bf16=bool(getattr(training_args, "bf16", False)),
        load_in_4bit=bool(getattr(model_args, "load_in_4bit", False)),
        lora_rank=lora_rank,
        lora_alpha=int(getattr(model_args, "lora_alpha", 16) or 16),
        target_modules=target_modules,
        lora_dropout=float(getattr(model_args, "lora_dropout", 0.0) or 0.0),
        use_flash_attention_2=attn_implementation == "flash_attention_2",
        init_value_head=True,
        is_general_preference=True,
        value_head_dim=training_args.gpo_value_head_dim,
        add_prompt_head=False,
    )
    if getattr(training_args, "gradient_checkpointing", False):
        model.config.use_cache = False
    return model


def _cuda_synchronize() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _reset_cuda_peak_memory() -> None:
    if torch.cuda.is_available():
        _cuda_synchronize()
        torch.cuda.reset_peak_memory_stats()


def _cuda_memory_metrics(prefix: str) -> dict[str, float]:
    if not torch.cuda.is_available():
        return {}
    _cuda_synchronize()
    device = torch.cuda.current_device()
    return {
        f"{prefix}_gpu_memory_allocated_mib": torch.cuda.memory_allocated(device) / _BYTES_PER_MIB,
        f"{prefix}_gpu_memory_reserved_mib": torch.cuda.memory_reserved(device) / _BYTES_PER_MIB,
        f"{prefix}_gpu_peak_memory_allocated_mib": torch.cuda.max_memory_allocated(device) / _BYTES_PER_MIB,
        f"{prefix}_gpu_peak_memory_reserved_mib": torch.cuda.max_memory_reserved(device) / _BYTES_PER_MIB,
    }


def _log_benchmark_metrics(stage: str, metrics: dict[str, float]) -> None:
    metric_items = []
    for suffix in (
        "wall_time_seconds",
        "gpu_memory_allocated_mib",
        "gpu_memory_reserved_mib",
        "gpu_peak_memory_allocated_mib",
        "gpu_peak_memory_reserved_mib",
    ):
        key = f"{stage}_{suffix}"
        if key in metrics:
            metric_items.append(f"{key}={metrics[key]:.3f}")
    if metric_items:
        logger.info("Benchmark %s: %s", stage, ", ".join(metric_items))


def main(script_args: ScriptArguments, training_args: GPOConfig, model_args: ModelConfig) -> None:
    set_seed(training_args.seed)

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

    logger.info("Model parameters %s", model_args)
    logger.info("Script parameters %s", script_args)
    logger.info("Training parameters %s", training_args)

    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
    if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
        logger.info("Checkpoint detected, resuming training at %s.", last_checkpoint)

    tokenizer = get_tokenizer(model_args, training_args)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "right"

    model = _load_gp_reward_model(model_args, training_args)

    ranking_dataset = get_ranking_dataset(script_args)
    if ranking_dataset is None:
        raise ValueError("GPO training requires listwise/ranking dataset splits.")
    for split in ranking_dataset:
        logger.info("Loaded %d listwise ranking examples from the '%s' split.", len(ranking_dataset[split]), split)

    train_split = script_args.dataset_train_split
    eval_split = script_args.dataset_test_split
    train_dataset = listwise_to_gpo_pairwise_dataset(
        ranking_dataset[train_split],
        strategy=training_args.gpo_pairwise_strategy,
    )
    eval_dataset = None
    if training_args.eval_strategy != "no" and eval_split in ranking_dataset:
        eval_dataset = listwise_to_gpo_pairwise_dataset(
            ranking_dataset[eval_split],
            strategy=training_args.gpo_pairwise_strategy,
        )
    logger.info("Built %d GP pairwise training examples from '%s'.", len(train_dataset), train_split)
    if eval_dataset is not None:
        logger.info("Built %d GP pairwise evaluation examples from '%s'.", len(eval_dataset), eval_split)

    collator = GPOPairwiseDataCollator(
        tokenizer=tokenizer,
        max_length=getattr(training_args, "max_length", 1024),
        max_prompt_length=getattr(training_args, "max_prompt_length", 512),
    )
    trainer = GPORewardTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        processing_class=tokenizer,
        gpo_tau=training_args.gpo_tau,
        gpo_value_head_dim=training_args.gpo_value_head_dim,
    )

    if (
        script_args.ranking_eval_during_training
        and training_args.eval_strategy != "no"
        and eval_split in ranking_dataset
    ):
        ranking_callback = GPORankingEvaluationCallback(
            ranking_dataset=ranking_dataset,
            tokenizer=tokenizer,
            script_args=script_args,
            training_args=training_args,
        )
        ranking_callback.trainer = trainer
        trainer.add_callback(ranking_callback)

    logger.info("*** Train GP ranker ***")
    checkpoint = training_args.resume_from_checkpoint or last_checkpoint
    _reset_cuda_peak_memory()
    train_wall_start = time.perf_counter()
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    _cuda_synchronize()
    metrics = train_result.metrics
    metrics["train_wall_time_seconds"] = time.perf_counter() - train_wall_start
    metrics.update(_cuda_memory_metrics("train"))
    _log_benchmark_metrics("train", metrics)
    metrics["train_samples"] = len(train_dataset)
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    if training_args.eval_strategy != "no" and eval_dataset is not None:
        logger.info("*** Pairwise eval loss ***")
        _reset_cuda_peak_memory()
        eval_wall_start = time.perf_counter()
        metrics = trainer.evaluate()
        _cuda_synchronize()
        metrics["eval_wall_time_seconds"] = time.perf_counter() - eval_wall_start
        metrics.update(_cuda_memory_metrics("eval"))
        _log_benchmark_metrics("eval", metrics)
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    if script_args.run_ranking_eval:
        logger.info("*** Ranking evaluation on available ranking splits ***")
        _reset_cuda_peak_memory()
        ranking_wall_start = time.perf_counter()
        ranking_results = evaluate_gpo_ranking_splits(
            trainer=trainer,
            ranking_dataset=ranking_dataset,
            tokenizer=tokenizer,
            script_args=script_args,
            training_args=training_args,
        )
        _cuda_synchronize()
        ranking_benchmark_metrics = {
            "ranking_wall_time_seconds": time.perf_counter() - ranking_wall_start,
            **_cuda_memory_metrics("ranking"),
        }
        _log_benchmark_metrics("ranking", ranking_benchmark_metrics)
        trainer.log(ranking_benchmark_metrics)
        trainer.log_metrics("ranking_benchmark", ranking_benchmark_metrics)
        trainer.save_metrics("ranking_benchmark", ranking_benchmark_metrics)
        for split_name, split_metrics in ranking_results.items():
            trainer.log({f"ranking_{split_name}/{key}": value for key, value in split_metrics.items()})
            trainer.log_metrics(f"ranking_{split_name}", split_metrics)
            trainer.save_metrics(f"ranking_{split_name}", split_metrics)

    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)


if __name__ == "__main__":
    parser = TrlParser((ScriptArguments, GPOConfig, ModelConfig))
    parsed_script_args, parsed_training_args, parsed_model_args = parser.parse_args_and_config()
    main(parsed_script_args, parsed_training_args, parsed_model_args)
