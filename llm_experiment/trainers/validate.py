import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from llm_experiment.trainers.dpo_trainer import DPOTrainer
from llm_experiment.trainers.approx_dpo_trainer import ApproxDPOTrainer
from llm_experiment.trainers.dpo_config import DPOConfig
from datasets import load_dataset
import matplotlib.pyplot as plt
import random
import os

def preprocess_batch(tokenizer, batch, max_seq_len, device):
    queries = tokenizer(batch["query"], truncation=True, max_length=max_seq_len, padding="max_length", return_tensors="pt")["input_ids"]
    responses_w = tokenizer(batch["response_w"], truncation=True, max_length=max_seq_len, padding="max_length", return_tensors="pt")["input_ids"]
    responses_l = tokenizer(batch["response_l"], truncation=True, max_length=max_seq_len, padding="max_length", return_tensors="pt")["input_ids"]
    mask = torch.ones_like(queries)
    return queries.to(device), responses_w.to(device), responses_l.to(device), mask.to(device)

def compute_loss(trainer, queries, responses_w, responses_l, mask):
    trainer.model.eval()
    with torch.no_grad():
        loss, _ = trainer._step(
            queries, responses_w, responses_l, return_stats=True, preference_mask=None
        )
    return loss.item()

def main():
    # Configuration
    MODEL_NAME = FLAGS.pretrained_dir
    DATASET_PATH = FLAGS.preference_dataset_path
    DATASET_SPLIT = FLAGS.preference_dataset_split or "train"
    DOWNSAMPLE_RATIO = float(FLAGS.downsample_ratio)
    BATCH_SIZE = FLAGS.batch_size
    SEQ_LEN = 512
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    LEARNING_RATE = FLAGS.learning_rate

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(DEVICE)
    ref_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(DEVICE)

    # Config for trainer
    config = DPOConfig(
        model_name=MODEL_NAME,
        learning_rate=LEARNING_RATE,
        batch_size=BATCH_SIZE,
        mini_batch_size=BATCH_SIZE,
        temperature=FLAGS.temperature,
        beta=FLAGS.beta
    )

    # Initialize trainers
    approx_trainer = ApproxDPOTrainer(config, model, ref_model, tokenizer)
    exact_trainer = DPOTrainer(config, model, ref_model, tokenizer)

    # Load dataset
    dataset = load_dataset(DATASET_PATH, split=DATASET_SPLIT)
    if DOWNSAMPLE_RATIO < 1.0:
        dataset = dataset.shuffle(seed=FLAGS.seed).select(range(int(len(dataset) * DOWNSAMPLE_RATIO)))
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Validation loop
    exact_losses, approx_losses, diffs = [], [], []

    print("=== Starting ApproxDPO Validation ===")
    for batch_idx, batch in enumerate(dataloader, 1):
        queries, responses_w, responses_l, mask = preprocess_batch(tokenizer, batch, SEQ_LEN, DEVICE)

        exact_loss = compute_loss(exact_trainer, queries, responses_w, responses_l, mask)
        approx_loss = compute_loss(approx_trainer, queries, responses_w, responses_l, mask)
        abs_diff = abs(exact_loss - approx_loss)

        exact_losses.append(exact_loss)
        approx_losses.append(approx_loss)
        diffs.append(abs_diff)

        print(f"[Batch {batch_idx}] Exact: {exact_loss:.6f} | Approx: {approx_loss:.6f} | Diff: {abs_diff:.6e}")

    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(exact_losses, label="Exact DPO Loss")
    plt.plot(approx_losses, label="Approx DPO Loss", linestyle="--")
    plt.title("Exact vs Approximate DPO Loss")
    plt.xlabel("Batch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(8, 5))
    plt.hist(diffs, bins=30, color="orange", edgecolor="black")
    plt.title("Absolute Difference between Exact and Approximate DPO Loss")
    plt.xlabel("Absolute Difference")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    import absl.app as app
    from absl import flags
    FLAGS = flags.FLAGS
    app.run(main)
