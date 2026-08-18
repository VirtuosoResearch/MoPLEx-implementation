# Learning Mixtures of Plackett-Luce Models for Preference Optimization

This repository contains the code for reproducing the main experiments of **MoPLEx**, a method for learning mixtures of Plackett-Luce ranking models from heterogeneous preference data.

The main experiments report clustering and ranking accuracy on:

- UltraFeedback disagreement data
- PERSONA-12 heterogeneous-user data

## Usage

Here is the procedure to reproduce the main results.

1. Environment reconstruction

We use Python 3.10 for all the experiments.
```bash
pip install -r requirements.txt
cd src
pip install -e .
```

2. Generate UltraFeedback disagreement data

```bash
bash scripts/bash_scripts/generate_disagreement_ultrafeedback_dataset.sh
```

MoPLEx uses augmented ranking slates with generated tail responses. Generate the augmented training data with:

```bash
bash scripts/bash_scripts/augment_ultrafeedback_disagreement/augment_train_downsampled_ultrafeedback_disagreement_0.25.sh
```

3. Run MoPLEx on UltraFeedback

This command runs the main mixture-of-PL model with block EM, per-cluster LoRA adapters, response augmentation, and linear reward approximation.

```bash
bash scripts/bash_scripts/evaluate_ultrafeedback_disagreement/run_mixture_pl_linear_approx_qlora_ultrafeedback_disagreement_original_augmented.sh
```

For the non-approximated original-slate variant:

```bash
bash scripts/bash_scripts/evaluate_ultrafeedback_disagreement/run_mixture_pl_qlora_ultrafeedback_disagreement_original.sh
```

4. Run UltraFeedback baselines

```bash
# DPO:
bash scripts/bash_scripts/evaluate_ultrafeedback_disagreement/run_dpo_qlora.sh

# LiPO / single listwise PL model:
bash scripts/bash_scripts/evaluate_ultrafeedback_disagreement/ablations/run_single_pl_qlora_ultrafeedback_disagreement_all_criteria.sh

# MiCRo, MaxMin-RLHF, and EM-DPO clustering baselines:

bash scripts/bash_scripts/evaluate_ultrafeedback_disagreement/run_micro_cluster.sh
bash scripts/bash_scripts/evaluate_ultrafeedback_disagreement/run_maxmin_cluster.sh
bash scripts/bash_scripts/evaluate_ultrafeedback_disagreement/run_emdpo_cluster.sh
```

5. Generate PERSONA-12 data

```bash
bash scripts/bash_scripts/evaluate_persona/generate_persona_subset.sh
```

6. Run MoPLEx on PERSONA-12

```bash
bash scripts/bash_scripts/evaluate_persona/run_persona_12_mixture_pl_linear_approx_response_sweep.sh
```

7. Run PERSONA-12 baselines

```bash
# DPO and LiPO:
bash scripts/bash_scripts/evaluate_persona/run_persona_12_all_baselines.sh

# MiCRo, MaxMin-RLHF, and EM-DPO clustering baselines:
bash scripts/bash_scripts/evaluate_persona/run_persona_12_cluster_baselines.sh

# Optional per-persona DPO diagnostic:
bash scripts/bash_scripts/evaluate_persona/run_persona_12_dpo_by_dimension.sh
```

## Project structure

```text
MMPO-code/
├── README.md
├── requirements.txt
└── src/
    ├── src/alignment/                  # mixture PL, ListDPO, MoPLEx trainers
    ├── scripts/dpo.py                  # main TRL training entry point
    ├── scripts/bash_scripts/           # experiment launch scripts
    │   ├── evaluate_ultrafeedback_disagreement/
    │   ├── evaluate_persona/
    │   └── augment_ultrafeedback_disagreement/
    ├── recipes/                        # QLoRA and training configs
    ├── data/                           # generated datasets
    └── outputs/                        # checkpoints and logs
```

## Main metrics

The scripts log metrics to Weights & Biases and write checkpoints under `src/outputs/`.

For the main table, report the mean and standard deviation over three seeds:

- Clustering accuracy: aligned mixture posterior / cluster assignment accuracy
- Ranking accuracy: validation top-1 or pairwise ranking accuracy

## Reference

If you find this repository useful or happen to use it in a research paper, please cite our work with the following Bib information.

```bibtex
@article{moplex2026,
  title={Learning Mixtures of Plackett-Luce Models for Preference Optimization},
  author={Li, Dongyue and Zhang, Ziniu and Wang, Lu and Zhang, Hongyang R.},
  year={2026}
}
```
