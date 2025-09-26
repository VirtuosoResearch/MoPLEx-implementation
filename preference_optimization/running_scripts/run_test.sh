
# Full training
# python trl/scripts/dpo.py \
#     --dataset_name trl-lib/ultrafeedback_binarized \
#     --model_name_or_path Qwen/Qwen2-0.5B-Instruct \
#     --learning_rate 5.0e-7 \
#     --num_train_epochs 1 \
#     --per_device_train_batch_size 2 \
#     --max_steps 1000 \
#     --gradient_accumulation_steps 8 \
#     --gradient_checkpointing \
#     --eval_strategy steps \
#     --eval_steps 50 \
#     --output_dir Qwen2-0.5B-DPO \
#     --no_remove_unused_columns

# LoRA:
# python trl/scripts/dpo.py \
#     --dataset_name trl-lib/ultrafeedback_binarized \
#     --model_name_or_path Qwen/Qwen2-0.5B-Instruct \
#     --learning_rate 5.0e-6 \
#     --num_train_epochs 1 \
#     --per_device_train_batch_size 2 \
#     --max_steps 1000 \
#     --gradient_accumulation_steps 8 \
#     --gradient_checkpointing \
#     --eval_strategy steps \
#     --eval_steps 50 \
#     --output_dir Qwen2-0.5B-DPO \
#     --no_remove_unused_columns \
#     --use_peft \
#     --lora_r 32 \
#     --lora_alpha 16

export WANDB_PROJECT="scalable-preference-optimization"
export WANDB_ENTITY="VirtuosoResearch"
export WANDB_NAME="simpo"
CUDA_VISIBLE_DEVICES=1 ACCELERATE_LOG_LEVEL=info accelerate launch run_simpo.py training_configs/base-simpo.yaml \
        beta=2.0 training_method="simpo" \
        load_multi_preference_criterions="overall_score" \
        output_dir="outputs/base-simpo-qwen-single-overall_score-controlled-v2" \
        run_name="base-simpo-single-overall_score-controlled-v2" \
        load_specific_pairs=False num_train_epochs=10