for subset_id in {0..50}; do
    CUDA_VISIBLE_DEVICES=0 ACCELERATE_LOG_LEVEL=info accelerate launch run_simpo.py training_configs/base-simpo.yaml \
        beta=2.0 training_method="simpo" \
        output_dir="outputs/subset-simpo-qwen-imdb-$subset_id" \
        run_name="subset-simpo-imdb-$subset_id" \
        load_specific_pairs=True num_train_epochs=1 test_size=2000 \
        load_multi_preference_dataset=imdb_preference_with_source \
        subset_id=$subset_id
done

CUDA_VISIBLE_DEVICES=0 ACCELERATE_LOG_LEVEL=info screen accelerate launch run_simpo.py training_configs/base-simpo.yaml \
        beta=2.0 training_method="simpo" \
        output_dir="outputs/subset-simpo-qwen-imdb-0" \
        run_name="subset-simpo-imdb-0" \
        load_specific_pairs=True num_train_epochs=5 test_size=2000 \
        load_multi_preference_dataset=imdb_preference_with_source \
        subset_id=0