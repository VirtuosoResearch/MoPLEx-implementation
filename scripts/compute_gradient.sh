python trainers/precompute_gradients_for_approx_dpo.py \
    --model_name meta-llama/Llama-3.2-1B \
    --preference_dataset_path Asap7772/relabeled_alpacafarm_pythiasft_20K_preference_data_minlength \
    --batch_size 4 \
    --use_lora \
    --projection_dim 200 \
    --downsample_ratio 0.05 \
    # --max_samples 100