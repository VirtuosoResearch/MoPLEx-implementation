#!/usr/bin/env python
# coding=utf-8
"""
Annotator-specific evaluation for preference optimization models.
Evaluates the model on each annotator separately and records all results.
"""

import logging
import json
import os
from collections import defaultdict
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass

import numpy as np
import torch
from datasets import Dataset
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from simpo_trainer import SimPOTrainer
from dpo_trainer import DPOTrainer

logger = logging.getLogger(__name__)


@dataclass
class AnnotatorEvaluationResult:
    """Container for annotator-specific evaluation results."""
    annotator_id: Union[str, int]
    num_samples: int
    accuracy: float
    chosen_reward_mean: float
    rejected_reward_mean: float
    reward_margin_mean: float
    loss_mean: float
    metrics: Dict[str, float]


class AnnotatorEvaluator:
    """
    Evaluator that performs evaluation on each annotator separately.
    """
    
    def __init__(
        self,
        trainer: Union[SimPOTrainer, DPOTrainer],
        tokenizer: PreTrainedTokenizerBase,
        annotator_column: str = "annotator",
        criterion_column: str = "criterion",
        save_results: bool = True,
        results_dir: Optional[str] = None,
    ):
        """
        Initialize the annotator evaluator.
        
        Args:
            trainer: The trainer instance (SimPOTrainer or DPOTrainer)
            tokenizer: The tokenizer
            annotator_column: Name of the column containing annotator IDs
            criterion_column: Name of the column containing criteria
            save_results: Whether to save results to file
            results_dir: Directory to save results (defaults to trainer output_dir)
        """
        self.trainer = trainer
        self.tokenizer = tokenizer
        self.annotator_column = annotator_column
        self.criterion_column = criterion_column
        self.save_results = save_results
        self.results_dir = results_dir or trainer.args.output_dir
        
        # Ensure results directory exists
        if self.save_results:
            os.makedirs(self.results_dir, exist_ok=True)
    
    def evaluate_by_annotator(
        self, 
        test_dataset: Dataset,
        include_overall: bool = True,
        min_samples_per_annotator: int = 1,
    ) -> Dict[str, Any]:
        """
        Evaluate the model on each annotator separately.
        
        Args:
            test_dataset: The test dataset
            include_overall: Whether to include overall evaluation results
            min_samples_per_annotator: Minimum number of samples required per annotator
            
        Returns:
            Dictionary containing results for each annotator and overall results
        """
        logger.info(f"Starting annotator-specific evaluation on {len(test_dataset)} samples")
        
        # Check if annotator column exists
        if self.annotator_column not in test_dataset.column_names:
            logger.warning(f"Annotator column '{self.annotator_column}' not found in dataset. "
                         f"Available columns: {test_dataset.column_names}")
            if include_overall:
                logger.info("Falling back to overall evaluation only")
                overall_metrics = self.trainer.evaluate(test_dataset)
                return {
                    "overall": {
                        "num_samples": len(test_dataset),
                        "metrics": overall_metrics
                    }
                }
            else:
                logger.error("Cannot perform annotator-specific evaluation without annotator column")
                return {}
        
        # Get unique annotators
        annotators = list(set(test_dataset[self.annotator_column]))
        logger.info(f"Found {len(annotators)} unique annotators: {annotators}")
        
        # Filter annotators by minimum sample count
        annotator_counts = defaultdict(int)
        for annotator in test_dataset[self.annotator_column]:
            annotator_counts[annotator] += 1
        
        valid_annotators = [
            ann for ann in annotators 
            if annotator_counts[ann] >= min_samples_per_annotator
        ]
        
        logger.info(f"Evaluating on {len(valid_annotators)} annotators with >= {min_samples_per_annotator} samples")
        
        # Store results
        results = {
            "annotator_results": {},
            "summary": {
                "total_annotators": len(annotators),
                "evaluated_annotators": len(valid_annotators),
                "total_samples": len(test_dataset),
                "min_samples_per_annotator": min_samples_per_annotator
            }
        }
        
        # Evaluate on each annotator
        for annotator_id in valid_annotators:
            logger.info(f"Evaluating annotator {annotator_id}")
            
            # Filter dataset for this annotator
            annotator_dataset = test_dataset.filter(
                lambda x: x[self.annotator_column] == annotator_id
            )
            
            if len(annotator_dataset) == 0:
                logger.warning(f"No samples found for annotator {annotator_id}")
                continue
            
            # Evaluate on this annotator's data
            try:
                annotator_metrics = self.trainer.evaluate(annotator_dataset)
                
                # Create result object
                annotator_result = AnnotatorEvaluationResult(
                    annotator_id=annotator_id,
                    num_samples=len(annotator_dataset),
                    accuracy=annotator_metrics.get("eval_rewards/accuracies", 0.0),
                    chosen_reward_mean=annotator_metrics.get("eval_rewards/chosen", 0.0),
                    rejected_reward_mean=annotator_metrics.get("eval_rewards/rejected", 0.0),
                    reward_margin_mean=annotator_metrics.get("eval_rewards/margins", 0.0),
                    loss_mean=annotator_metrics.get("eval_loss", 0.0),
                    metrics=annotator_metrics
                )
                
                results["annotator_results"][str(annotator_id)] = {
                    "annotator_id": annotator_id,
                    "num_samples": len(annotator_dataset),
                    "accuracy": float(annotator_result.accuracy),
                    "chosen_reward_mean": float(annotator_result.chosen_reward_mean),
                    "rejected_reward_mean": float(annotator_result.rejected_reward_mean),
                    "reward_margin_mean": float(annotator_result.reward_margin_mean),
                    "loss_mean": float(annotator_result.loss_mean),
                    "metrics": {k: float(v) for k, v in annotator_metrics.items()}
                }
                
                logger.info(f"Annotator {annotator_id}: accuracy={annotator_result.accuracy:.4f}, "
                          f"samples={len(annotator_dataset)}")
                
            except Exception as e:
                logger.error(f"Error evaluating annotator {annotator_id}: {e}")
                continue
        
        # Add overall evaluation if requested
        if include_overall:
            logger.info("Performing overall evaluation")
            overall_metrics = self.trainer.evaluate(test_dataset)
            results["overall"] = {
                "num_samples": len(test_dataset),
                "metrics": {k: float(v) for k, v in overall_metrics.items()}
            }
        
        # Compute summary statistics
        self._compute_summary_statistics(results)
        
        # Save results if requested
        if self.save_results:
            self._save_results(results)
        
        logger.info("Annotator-specific evaluation completed")
        return results
    
    def _compute_summary_statistics(self, results: Dict[str, Any]) -> None:
        """Compute summary statistics across all annotators."""
        annotator_results = results["annotator_results"]
        
        if not annotator_results:
            logger.warning("No annotator results to compute summary statistics")
            return
        
        # Collect metrics across all annotators
        accuracies = [r["accuracy"] for r in annotator_results.values()]
        reward_margins = [r["reward_margin_mean"] for r in annotator_results.values()]
        losses = [r["loss_mean"] for r in annotator_results.values()]
        sample_counts = [r["num_samples"] for r in annotator_results.values()]
        
        # Compute statistics
        summary_stats = {
            "accuracy": {
                "mean": float(np.mean(accuracies)),
                "std": float(np.std(accuracies)),
                "min": float(np.min(accuracies)),
                "max": float(np.max(accuracies)),
                "median": float(np.median(accuracies))
            },
            "reward_margin": {
                "mean": float(np.mean(reward_margins)),
                "std": float(np.std(reward_margins)),
                "min": float(np.min(reward_margins)),
                "max": float(np.max(reward_margins)),
                "median": float(np.median(reward_margins))
            },
            "loss": {
                "mean": float(np.mean(losses)),
                "std": float(np.std(losses)),
                "min": float(np.min(losses)),
                "max": float(np.max(losses)),
                "median": float(np.median(losses))
            },
            "sample_distribution": {
                "mean": float(np.mean(sample_counts)),
                "std": float(np.std(sample_counts)),
                "min": int(np.min(sample_counts)),
                "max": int(np.max(sample_counts)),
                "total": int(np.sum(sample_counts))
            }
        }
        
        results["summary"]["statistics"] = summary_stats
        
        # Log summary
        logger.info(f"Summary statistics:")
        logger.info(f"  Accuracy: {summary_stats['accuracy']['mean']:.4f} ± {summary_stats['accuracy']['std']:.4f}")
        logger.info(f"  Reward margin: {summary_stats['reward_margin']['mean']:.4f} ± {summary_stats['reward_margin']['std']:.4f}")
        logger.info(f"  Loss: {summary_stats['loss']['mean']:.4f} ± {summary_stats['loss']['std']:.4f}")
    
    def _save_results(self, results: Dict[str, Any]) -> None:
        """Save results to JSON file."""
        results_file = os.path.join(self.results_dir, "annotator_evaluation_results.json")
        
        try:
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            logger.info(f"Results saved to {results_file}")
        except Exception as e:
            logger.error(f"Failed to save results: {e}")
    
    def evaluate_by_criterion_and_annotator(
        self,
        test_dataset: Dataset,
        min_samples_per_annotator: int = 1,
    ) -> Dict[str, Any]:
        """
        Evaluate by both criterion and annotator.
        
        Args:
            test_dataset: The test dataset
            min_samples_per_annotator: Minimum number of samples required per annotator
            
        Returns:
            Dictionary containing results grouped by criterion and annotator
        """
        logger.info("Starting criterion and annotator-specific evaluation")
        
        # Check required columns
        required_columns = [self.annotator_column, self.criterion_column]
        missing_columns = [col for col in required_columns if col not in test_dataset.column_names]
        
        if missing_columns:
            logger.error(f"Missing required columns: {missing_columns}")
            return {}
        
        # Get unique combinations
        unique_combinations = set()
        for i in range(len(test_dataset)):
            annotator = test_dataset[i][self.annotator_column]
            criterion = test_dataset[i][self.criterion_column]
            unique_combinations.add((annotator, criterion))
        
        logger.info(f"Found {len(unique_combinations)} unique (annotator, criterion) combinations")
        
        results = {
            "criterion_annotator_results": {},
            "summary": {
                "total_combinations": len(unique_combinations),
                "total_samples": len(test_dataset)
            }
        }
        
        # Evaluate each combination
        for annotator_id, criterion in unique_combinations:
            logger.info(f"Evaluating annotator {annotator_id}, criterion {criterion}")
            
            # Filter dataset for this combination
            combination_dataset = test_dataset.filter(
                lambda x: (x[self.annotator_column] == annotator_id and 
                          x[self.criterion_column] == criterion)
            )
            
            if len(combination_dataset) < min_samples_per_annotator:
                logger.warning(f"Not enough samples for annotator {annotator_id}, criterion {criterion}: {len(combination_dataset)}")
                continue
            
            # Evaluate
            try:
                combination_metrics = self.trainer.evaluate(combination_dataset)
                
                key = f"{annotator_id}_{criterion}"
                results["criterion_annotator_results"][key] = {
                    "annotator_id": annotator_id,
                    "criterion": criterion,
                    "num_samples": len(combination_dataset),
                    "metrics": {k: float(v) for k, v in combination_metrics.items()}
                }
                
                logger.info(f"Combination {key}: samples={len(combination_dataset)}")
                
            except Exception as e:
                logger.error(f"Error evaluating combination {annotator_id}, {criterion}: {e}")
                continue
        
        # Save results
        if self.save_results:
            results_file = os.path.join(self.results_dir, "criterion_annotator_evaluation_results.json")
            try:
                with open(results_file, 'w', encoding='utf-8') as f:
                    json.dump(results, f, indent=2, ensure_ascii=False)
                logger.info(f"Results saved to {results_file}")
            except Exception as e:
                logger.error(f"Failed to save results: {e}")
        
        logger.info("Criterion and annotator-specific evaluation completed")
        return results


def evaluate_model_by_annotator(
    trainer: Union[SimPOTrainer, DPOTrainer],
    test_dataset: Dataset,
    tokenizer: PreTrainedTokenizerBase,
    annotator_column: str = "annotator",
    criterion_column: str = "criterion",
    include_overall: bool = True,
    min_samples_per_annotator: int = 1,
    save_results: bool = True,
    results_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Convenience function to evaluate model by annotator.
    
    Args:
        trainer: The trainer instance
        test_dataset: The test dataset
        tokenizer: The tokenizer
        annotator_column: Name of the column containing annotator IDs
        criterion_column: Name of the column containing criteria
        include_overall: Whether to include overall evaluation results
        min_samples_per_annotator: Minimum number of samples required per annotator
        save_results: Whether to save results to file
        results_dir: Directory to save results
        
    Returns:
        Dictionary containing evaluation results
    """
    evaluator = AnnotatorEvaluator(
        trainer=trainer,
        tokenizer=tokenizer,
        annotator_column=annotator_column,
        criterion_column=criterion_column,
        save_results=save_results,
        results_dir=results_dir,
    )
    
    return evaluator.evaluate_by_annotator(
        test_dataset=test_dataset,
        include_overall=include_overall,
        min_samples_per_annotator=min_samples_per_annotator,
    )
