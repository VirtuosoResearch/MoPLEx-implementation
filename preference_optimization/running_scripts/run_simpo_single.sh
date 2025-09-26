export WANDB_PROJECT="scalable-preference-optimization"
export WANDB_ENTITY="VirtuosoResearch"
export WANDB_NAME="simpo"

criterions=("overall_score" "helpfulness" "honesty" "instruction_following" "truthfulness")

for criterion in "${criterions[@]}"; do
    CUDA_VISIBLE_DEVICES=0 ACCELERATE_LOG_LEVEL=info accelerate launch run_simpo.py training_configs/base-simpo.yaml \
        beta=2.0 training_method="simpo" \
        load_multi_preference_criterions="$criterion" \
        output_dir="outputs/base-simpo-qwen-single-$criterion-controlled" \
        run_name="base-simpo-single-$criterion-controlled" \
        load_specific_pairs=True num_train_epochs=10 test_size=1000
done

# export WANDB_PROJECT="scalable-preference-optimization"
# export WANDB_ENTITY="VirtuosoResearch"
# export WANDB_NAME="dpo"

# criterions=("overall_score" "helpfulness" "honesty" "instruction_following" "truthfulness")

# for criterion in "${criterions[@]}"; do
# CUDA_VISIBLE_DEVICES=0 ACCELERATE_LOG_LEVEL=info accelerate launch run_simpo.py training_configs/base-simpo.yaml \
#     beta=2.0 training_method="dpo" \
#     load_multi_preference_criterions="$criterion" \
#     output_dir="outputs/base-dpo-qwen-single-$criterion" \
#     run_name="base-dpo-single-$criterion"
# done

# CUDA_VISIBLE_DEVICES=0 ACCELERATE_LOG_LEVEL=info accelerate launch run_simpo.py training_configs/base-simpo.yaml \
#     beta=2.0 training_method="dpo" \
#     load_multi_preference_criterions="overall_score,helpfulness,honesty,instruction_following,truthfulness" \
#     output_dir="outputs/base-dpo-qwen-multi" \
#     run_name="base-dpo-multi"