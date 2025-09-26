export WANDB_PROJECT="scalable-preference-optimization"
export WANDB_ENTITY="VirtuosoResearch"
export WANDB_NAME="dpo"

criterion="overall_score"
CUDA_VISIBLE_DEVICES=0 ACCELERATE_LOG_LEVEL=info accelerate launch run_simpo.py training_configs/base-simpo.yaml \
    beta=2.0 training_method="dpo" \
    load_multi_preference_criterions="$criterion" \
    output_dir="outputs/base-dpo-qwen-single-$criterion" \
    run_name="base-dpo-single-$criterion"
# --num_processes 1 --config_file accelerate_configs/deepspeed_zero3.yaml