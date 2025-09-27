export WANDB_PROJECT="scalable-preference-optimization"
export WANDB_ENTITY="VirtuosoResearch"
export WANDB_NAME="reward_modeling"
criterions=("overall_score" "helpfulness" "honesty" "instruction_following" "truthfulness")

for criterion in "${criterions[@]}"; do
CUDA_VISIBLE_DEVICES=2 ACCELERATE_LOG_LEVEL=info accelerate launch run_reward_modeling.py training_configs/base-reward-model.yaml \
    load_multi_preference=True load_multi_preference_criterions="$criterion" \
    output_dir="outputs/base-reward-model-qwen-$criterion" \
    run_name="base-reward-model-$criterion" 
    # load_specific_pairs=True num_train_epochs=5 test_size=2000
done