# Scalable direct preference optimization and inference

Document: https://docs.google.com/document/d/1K3690xE6Axo1hSqcxgbScRxwCSHCZf4RgrM27pF2Xrc/edit?usp=sharing

- Environment
  ```bash
  pip install -e .
  ```
  
- Finetuning with standard DPO
  ```bash
  bash scripts/downsample_dpo_single.sh
  ```

- First-order approximation
  ```bash
  python motivation.py
  ```
