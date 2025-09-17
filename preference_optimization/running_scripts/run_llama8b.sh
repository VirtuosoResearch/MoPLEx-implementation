export WANDB_PROJECT="scalable-preference-optimization"
export WANDB_ENTITY="VirtuosoResearch"
export WANDB_NAME="simpo"
CUDA_VISIBLE_DEVICES=0 ACCELERATE_LOG_LEVEL=info accelerate launch --num_processes 1 --config_file accelerate_configs/deepspeed_zero3.yaml scripts/run_simpo.py training_configs/llama-3-8b-base-simpo.yaml
