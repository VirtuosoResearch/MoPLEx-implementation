# Mixture of Plackett-Luce Models in DPO Training - Implementation Summary

## Overview

Successfully integrated Mixture of Plackett-Luce (MoPL) models into the DPO training pipeline. The implementation enables EM-based clustering of rankings by preference dimension while jointly optimizing the DPO objective.

## What Was Implemented

### 1. Core Mixture PL Components (`alignment/mixture_pl_components.py`)

Extracted and refactored the Mixture PL utilities from the synthetic experiments:

#### Functions:
- **`pl_log_prob(rewards, rankings)`** — Compute Plackett-Luce log probability
  - Input: [B, M] rewards, [B, M] ranking indices  
  - Output: [B] log probabilities
  
- **`mixture_pl_nll(router_logits, rewards, rankings)`** — Mixture model NLL  
  - Input: [B, K] router logits, [B, K, M] component rewards, [B, M] rankings
  - Output: scalar NLL, [B, K] component log-probabilities
  
- **`em_responsibilities(router_logits, comp_logp, temperature)`** — E-step soft assignments
  - Input: [B, K] router logits, [B, K] component log-probs, temperature scalar
  - Output: [B, K] posterior probabilities (soft cluster assignments)
  
- **`em_expected_complete_nll(router_logits, comp_logp, gamma)`** — M-step objective  
  - Input: [B, K] router logits, [B, K] comp log-probs, [B, K] responsibilities
  - Output: scalar loss (expected complete-data negative log-likelihood)

#### Classes:
- **`MixtureRouterHead(hidden_size, num_clusters, use_contextual, router_hidden_size)`** — Router network
  - **Contextual mode**: MLP on LM hidden states (per-instance mixture weights)
  - **Global mode**: Learnable parameters (shared mixture weights)
  
- **`MixtureEvalMetrics`** — Metrics dataclass
  - `nll`, `router_acc`, `posterior_acc_raw`, `posterior_acc_aligned`, `aligned_mean_corr`, `assignment`
  
- **`compute_mixture_eval_metrics(...)`** — Batch evaluation with optional correlation alignment

### 2. Configuration (`alignment/configs.py`)

Added `MixturePLConfig` extending `trl.DPOConfig`:

```python
@dataclass
class MixturePLConfig(trl.DPOConfig):
    use_mixture: bool = False  # Enable mixture modeling
    num_clusters: Optional[int] = None  # Auto-detect from preference_dimensions
    em_temperature: float = 1.0  # Soft/hard assignment control
    m_step_updates: int = 10  # Gradient steps per E-step  
    mixture_nll_weight: float = 0.1  # DPO_loss + weight * mixture_nll
    use_contextual_router: bool = False  # Contextual vs global router
    router_hidden_size: int = 256  # MLP hidden dimension
    use_auxiliary_dimension_loss: bool = False  # Future: supervised alignment
    log_cluster_metrics: bool = True  # Track cluster accuracy
```

### 3. MixtureDPOTrainer (`alignment/listwise_dpo.py`)

Extended `ListwiseDPOTrainer` with mixture modeling:

#### Key Methods:
- **`__init__(..., mixture_config)`** — Initialize router + component reward heads
  
- **`get_batch_mixture_output(model, batch)`** — Compute router logits and per-component rewards
  - Extracts LM hidden states, pools them
  - Passes through router for cluster logits  
  - Per-cluster reward heads score each item
  
- **`_create_rankings_from_utilities(utilities, candidate_mask)`** — Synthetic ranking creation
  - Converts DPO utilities to full permutation rankings for mixture loss
  
- **`get_batch_loss_metrics_with_mixture(model, batch, train_eval)`** — Hybrid loss computation
  - Computes DPO loss via listwise PL
  - Computes mixture loss via mixture_pl_nll
  - Hybrid loss = DPO_loss + mixture_nll_weight × mixture_nll
  - Tracks cluster accuracy vs ground-truth preference_dimension
  
- **`compute_loss(model, inputs, ...)`** — Override to include mixture loss  
  
- **`prediction_step(model, inputs, ...)`** — Override to compute eval metrics with mixture

#### Training Workflow:
1. Forward pass: Get model logits, router logits, component rewards
2. Compute DPO utility scores (β × (policy_logp - ref_logp))
3. E-step: Compute soft cluster assignments from router logits and component log-probs
4. M-step: Gradient updates on hybrid loss (DPO + mixture)
5. Eval: Compute cluster accuracy by comparing router predictions vs ground-truth dimensions

### 4. Entry Point Updates (`scripts/dpo.py`)

- Import `MixtureDPOTrainer` and `MixturePLConfig`
- Updated trainer selection logic:
  - If `use_mixture=True` → `MixtureDPOTrainer`
  - Elif `listwise=True` → `ListwiseDPOTrainer`  
  - Else → `DPOTrainer`
- Updated `TrlParser` to use `MixturePLConfig` (inherits from `DPOConfig`)

### 5. Module Exports (`alignment/__init__.py`)

Exported new classes:
- `MixturePLConfig`
- `MixtureDPOTrainer`

## Usage Example

### Command Line

```bash
python scripts/dpo.py \
    --use_mixture \
    --num_clusters 2 \
    --mixture_nll_weight 0.1 \
    --em_temperature 1.0 \
    --use_contextual_router \
    --router_hidden_size 256 \
    --dataset_name trl-lib/ultrafeedback_binarized \
    --model_name_or_path Qwen/Qwen2-0.5B-Instruct \
    --dataset_format listwise \
    --preference_dimensions '["helpfulness", "harmlessness"]' \
    --output_dir ./outputs/mixture_dpo_qwen
```

### Programmatic

```python
from alignment import MixturePLConfig, MixtureDPOTrainer, get_dataset

# Configuration
mixture_config = MixturePLConfig(
    use_mixture=True,
    num_clusters=2,
    mixture_nll_weight=0.1,
    em_temperature=1.0,
    use_contextual_router=True,
)

# Trainer
trainer = MixtureDPOTrainer(
    model=model,
    ref_model=ref_model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    processing_class=tokenizer,
    mixture_config=mixture_config,
)

# Training
trainer.train()
```

## Testing

### Unit Tests (19 passing tests)

Location: `experiments-lm-alignment/tests/test_mixture_pl_components.py`

Test Coverage:
- ✅ `TestPlLogProb` — PL log probability shape, gradients, values
- ✅ `TestMixturePLNLL` — Mixture NLL shape and gradients
- ✅ `TestEMResponsibilities` — Responsibility shape, summation, temperature effect
- ✅ `TestEMExpectedCompleteNLL` — M-step loss computation and descent
- ✅ `TestMixtureRouterHead` — Router shapes (contextual & global), gradients, consistency
- ✅ `TestPearsonCorr` — Correlation computation (perfect, negative, zero variance)
- ✅ `TestIntegration` — Full EM loop, training loop simulation

Run tests:
```bash
cd experiments-lm-alignment
PYTHONPATH=src python -m unittest tests.test_mixture_pl_components -v
```

## Key Design Decisions

1. **Reward Heads**: Per-cluster reward heads are initialized lazily on first batch
   - Pro: Flexible to hidden size; no pre-allocation needed
   - Con: Slightly more complex initialization

2. **Synthetic Rankings**: DPO utilities are converted to rankings for mixture loss
   - Pro: Enables mixture loss without explicit ranking data
   - Con: Not ground-truth rankings; could miss true structure

3. **Contextual vs Global Router**: Both modes supported
   - Contextual (default): More expressive, per-instance cluster assignment
   - Global: Simpler, shared mixture weights

4. **EM Loop Frequency**: Single E-step per batch, configurable M-step updates
   - Pro: Efficient during training
   - Con: Less frequent responsibility updates; could adjust if needed

5. **Cluster Metrics**: Use ground-truth preference_dimension for eval-time accuracy
   - Pro: Meaningful evaluation; enables cluster alignment via Hungarian matching
   - Con: Requires dimension labels in data

## Known Limitations & Future Work

1. **Auxiliary Dimension Loss** — Defined but not yet implemented
   - Could add supervised loss term to align router with ground-truth dimensions during training
   - Flag: `use_auxiliary_dimension_loss`

2. **Pairwise DPO Support** — Currently listwise-only
   - Converting pairwise (chosen/rejected) to full rankings is non-trivial
   - Would need preference transitivity modeling

3. **Component Initialization** — Reward heads initialized in compute_loss
   - Could pre-allocate in `__init__` for cleaner code

4. **Computational Cost** — Not yet profiled on large datasets
   - EM loop overhead could be significant
   - Recommendations: reduce m_step_updates, batch caching of component log-probs

5. **Correlation Matrix** — Currently optional; not integrated into main loss
   - Could optionally add correlation alignment loss for unsupervised cluster discovery

## Performance Metrics Logged

During training, MixtureDPOTrainer logs:

- **DPO metrics** (from listwise trainer):
  - `eval_listwise/top1_acc` — Top-1 ranking accuracy
  - `eval_listwise/pairwise_acc` — Pairwise ranking accuracy
  - `eval_listwise/utility_*` — Utility statistics

- **Mixture metrics** (new):
  - `mixture/nll` — Mixture PL negative log-likelihood
  - `mixture/cluster_acc` — Cluster assignment accuracy vs ground-truth dimension

All metrics are gathered and averaged across all processes in distributed training.

## Backward Compatibility

✅ **Fully backward compatible**:
- Default `use_mixture=False` preserves original ListwiseDPOTrainer behavior
- Existing pairwise/listwise DPO scripts work unchanged
- No modifications to data loading or preprocessing

## File Changes Summary

| File | Changes |
|------|---------|
| `src/alignment/mixture_pl_components.py` | ✨ NEW — Core mixture PL utilities |
| `src/alignment/configs.py` | ✏️ ADDED `MixturePLConfig` class |
| `src/alignment/listwise_dpo.py` | ✏️ ADDED `MixtureDPOTrainer` class, updated imports |
| `src/alignment/__init__.py` | ✏️ ADDED exports for new classes |
| `scripts/dpo.py` | ✏️ UPDATED trainer selection, imports, parser |
| `tests/test_mixture_pl_components.py` | ✨ NEW — 19 unit tests |

## Verification Checklist

- ✅ All files compile without syntax errors
- ✅ 19 unit tests pass (pl_log_prob, mixture_nll, em_steps, router, integration tests)
- ✅ New classes properly exported from alignment module
- ✅ dpo.py successfully imports and instantiates MixtureDPOTrainer
- ✅ Backward compatibility preserved (use_mixture=False by default)
- ✅ Configuration validates correctly
- ✅ No breaking changes to existing ListwiseDPOTrainer or DPOTrainer

## Next Steps (Phase 5 Continuation)

1. **Integration Test** — Create toy listwise dataset and run full training loop
2. **Real Dataset Test** — Run on UltraFeedback subset with --dataset_format listwise
3. **Metric Validation** — Verify cluster accuracy improves over epochs
4. **Profiling** — Measure computational overhead on realistic batch sizes
5. **Documentation** — Add example configs and training recipes
6. **Optional Enhancements**:
   - Implement auxiliary dimension loss
   - Add pairwise DPO support
   - Optimize component head initialization
   - Add correlation-based unsupervised clustering loss
