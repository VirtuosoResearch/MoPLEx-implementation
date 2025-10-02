export WANDB_PROJECT="scalable-preference-optimization"
export WANDB_ENTITY="VirtuosoResearch"
export WANDB_NAME="dpo"

criterions=("honesty" "instruction_following" "truthfulness") # "overall_score" "helpfulness"

for criterion in "${criterions[@]}"; do
    CUDA_VISIBLE_DEVICES=0 ACCELERATE_LOG_LEVEL=info accelerate launch run_simpo.py training_configs/base-dpo.yaml \
        beta=0.1 training_method="dpo" \
        load_multi_preference_criterions="$criterion" \
        output_dir="outputs/base-dpo-qwen-single-$criterion-controlled" \
        run_name="base-dpo-single-$criterion-controlled" \
        load_specific_pairs=True num_train_epochs=5 test_size=2000
done

export WANDB_NAME="dpo"
CUDA_VISIBLE_DEVICES=1 ACCELERATE_LOG_LEVEL=info accelerate launch run_simpo.py training_configs/base-dpo.yaml \
    beta=0.1 training_method="dpo" \
    load_multi_preference_criterions="overall_score,helpfulness,honesty,instruction_following,truthfulness" \
    output_dir="outputs/base-dpo-qwen-multi-controlled" \
    run_name="base-dpo-multi-controlled" \
    load_specific_pairs=True num_train_epochs=5 test_size=2000
