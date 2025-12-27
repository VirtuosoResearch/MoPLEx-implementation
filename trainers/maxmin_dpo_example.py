"""
Example usage of MaxMin-DPO Trainer with EM Algorithm

This script demonstrates how to use the MaxMin-DPO trainer to align models
with diverse human preferences using the EM algorithm for reward learning.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from trainers.maxmin_dpo_trainer import (
    MaxMinDPOTrainer,
    MaxMinDPOConfig,
    EMRewardLearner,
    RewardModel,
)
from trainers.model_value_head import AutoModelForCausalLMWithValueHead
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def prepare_preference_data(dataset, tokenizer):
    """
    Prepare preference data from dataset.
    
    Expected format: each item should have:
    - 'prompt': the input prompt
    - 'chosen': the preferred response
    - 'rejected': the non-preferred response
    - 'user_id' or 'annotator': identifier for the user/annotator
    """
    preference_data = []
    
    for item in dataset:
        # Extract fields (adjust based on your dataset format)
        prompt = item.get('prompt', item.get('instruction', ''))
        chosen = item.get('chosen', item.get('response_w', ''))
        rejected = item.get('rejected', item.get('response_l', ''))
        user_id = item.get('user_id', item.get('annotator', 0))
        
        # Tokenize and format
        preference_data.append({
            'prompt': prompt,
            'chosen': chosen,
            'rejected': rejected,
            'user_id': user_id,
        })
    
    return preference_data


def main():
    # Configuration
    model_name = "gpt2"  # Change to your model
    reward_model_name = "distilbert-base-uncased"  # Base model for reward models
    num_clusters = 2  # Number of user clusters
    num_epochs = 3
    batch_size = 32
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    base_model = AutoModelForCausalLM.from_pretrained(model_name)
    model = AutoModelForCausalLMWithValueHead(base_model)
    model.to(device)
    
    # Load preference dataset (adjust path/name based on your dataset)
    # Example: dataset = load_dataset("your_dataset", split="train")
    # For now, we'll create dummy data
    logger.info("Loading preference dataset...")
    
    # TODO: Replace with your actual dataset loading
    # dataset = load_dataset("your_preference_dataset", split="train")
    # preference_data = prepare_preference_data(dataset, tokenizer)
    
    # Example dummy data structure
    preference_data = [
        {
            'prompt': "What is the capital of France?",
            'chosen': "The capital of France is Paris.",
            'rejected': "I don't know.",
            'user_id': 0,
        },
        {
            'prompt': "Explain quantum computing.",
            'chosen': "Quantum computing uses quantum mechanical phenomena...",
            'rejected': "It's a type of computer.",
            'user_id': 1,
        },
        # Add more examples...
    ]
    
    logger.info(f"Loaded {len(preference_data)} preference examples")
    
    # Step 1: Learn reward models using EM algorithm (Algorithm 2)
    logger.info("\n=== Step 1: Learning Reward Models with EM Algorithm ===")
    em_learner = EMRewardLearner(
        num_clusters=num_clusters,
        reward_model_name=reward_model_name,
        tokenizer=tokenizer,
        device=device,
        convergence_threshold=1e-4,
        max_iterations=20,
        reward_learning_rate=1e-5,
        reward_num_epochs=3,
    )
    
    reward_models = em_learner.fit(preference_data)
    user_assignments = em_learner.user_assignments
    
    logger.info(f"EM algorithm completed. User assignments: {user_assignments}")
    
    # Step 2: Train model using MaxMin-DPO
    logger.info("\n=== Step 2: Training with MaxMin-DPO ===")
    
    config = MaxMinDPOConfig(
        model_name=model_name,
        learning_rate=1e-5,
        batch_size=batch_size,
        mini_batch_size=4,
        temperature=0.1,
        num_clusters=num_clusters,
        maxmin_weight=0.1,
        steps=1000,  # Adjust based on your dataset size
        seed=42,
    )
    
    trainer = MaxMinDPOTrainer(
        config=config,
        model=model,
        tokenizer=tokenizer,
        reward_models=reward_models,
        user_assignments=user_assignments,
        num_clusters=num_clusters,
    )
    
    # Prepare training data
    # In practice, you would tokenize your dataset properly
    # For this example, we'll show the structure
    
    logger.info("Starting MaxMin-DPO training...")
    logger.info("Note: This is a simplified example. In practice, you would:")
    logger.info("1. Properly tokenize your dataset")
    logger.info("2. Create data loaders with user_ids")
    logger.info("3. Run training loop with trainer.step()")
    
    # Example training loop structure:
    # for epoch in range(num_epochs):
    #     for batch in dataloader:
    #         queries = batch['query']
    #         responses_w = batch['response_w']
    #         responses_l = batch['response_l']
    #         user_ids = batch['user_id']
    #
    #         stats = trainer.step(
    #             queries=queries,
    #             responses_w=responses_w,
    #             responses_l=responses_l,
    #             user_ids=user_ids,
    #         )
    #         logger.info(f"Stats: {stats}")
    
    logger.info("\nMaxMin-DPO implementation complete!")
    logger.info("Key components:")
    logger.info("1. EMRewardLearner: Implements Algorithm 2 for learning reward models")
    logger.info("2. MaxMinDPOTrainer: Extends DPO trainer with MaxMin objective")
    logger.info("3. Cluster assignments learned via EM algorithm guide the MaxMin optimization")


if __name__ == "__main__":
    main()

