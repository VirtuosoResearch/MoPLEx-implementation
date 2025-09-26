export WANDB_PROJECT="scalable-preference-optimization"
export WANDB_ENTITY="VirtuosoResearch"
export WANDB_NAME="simpo"

criterions=("overall_score" "helpfulness" "honesty" "instruction_following" "truthfulness")

for criterion in "${criterions[@]}"; do
    CUDA_VISIBLE_DEVICES=0 ACCELERATE_LOG_LEVEL=info accelerate launch run_simpo.py training_configs/base-simpo.yaml \
        load_multi_preference_criterions="$criterion" \
        output_dir="outputs/base-simpo-qwen-single-$criterion" \
        run_name="base-simpo-single-$criterion"
done
# CUDA_VISIBLE_DEVICES=0 ACCELERATE_LOG_LEVEL=info accelerate launch run_simpo.py training_configs/base-simpo.yaml \
#     load_multi_preference_criterions="overall_score" \
#     output_dir="outputs/base-simpo-qwen-single-overall" \
#     run_name="base-simpo-single-overall"
