for subset_id in {0..20}; do
    CUDA_VISIBLE_DEVICES=0 ACCELERATE_LOG_LEVEL=info accelerate launch run_simpo.py training_configs/base-simpo.yaml \
        beta=2.0 training_method="simpo" \
        output_dir="outputs/subset-simpo-qwen-multi-controlled-$subset_id" \
        run_name="subset-simpo-multi-controlled-$subset_id" \
        load_specific_pairs=True num_train_epochs=1 test_size=2000 \
        load_multi_preference_dataset=openai/collective-alignment-1 \
        subset_id=$subset_id
done