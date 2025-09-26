export WANDB_PROJECT="scalable-preference-optimization"
export WANDB_ENTITY="VirtuosoResearch"
export WANDB_NAME="dpo"

criterions=("overall_score" "helpfulness" "honesty" "instruction_following" "truthfulness")

for criterion in "${criterions[@]}"; do
CUDA_VISIBLE_DEVICES=0 ACCELERATE_LOG_LEVEL=info accelerate launch run_simpo.py training_configs/base-dpo.yaml \
    beta=2.0 training_method="dpo" \
    load_multi_preference_criterions="$criterion" \
    output_dir="outputs/base-dpo-qwen-single-$criterion-v2" \
    run_name="base-dpo-single-$criterion-v2"
done

CUDA_VISIBLE_DEVICES=0 ACCELERATE_LOG_LEVEL=info accelerate launch run_simpo.py training_configs/base-dpo.yaml \
    beta=2.0 training_method="dpo" \
    load_multi_preference_criterions="overall_score,helpfulness,honesty,instruction_following,truthfulness" \
    output_dir="outputs/base-dpo-qwen-multi" \
    run_name="base-dpo-multi"