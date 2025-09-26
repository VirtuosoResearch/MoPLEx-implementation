export WANDB_PROJECT="scalable-preference-optimization"
export WANDB_ENTITY="VirtuosoResearch"
export WANDB_NAME="simpo"
CUDA_VISIBLE_DEVICES=1 ACCELERATE_LOG_LEVEL=info accelerate launch run_simpo.py training_configs/base-simpo.yaml \
    beta=2.0 training_method="simpo" \
    load_multi_preference_criterions="overall_score,helpfulness,honesty,instruction_following,truthfulness" \
    output_dir="outputs/base-simpo-qwen-multi-controlled" \
    run_name="base-simpo-multi-controlled" \
    load_specific_pairs=True num_train_epochs=10 test_size=1000
