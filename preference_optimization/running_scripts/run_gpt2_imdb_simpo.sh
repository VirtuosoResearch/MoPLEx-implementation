for subset_id in {0..50}; do
    CUDA_VISIBLE_DEVICES=1 ACCELERATE_LOG_LEVEL=info accelerate launch run_simpo.py training_configs/gpt2-imdb-simpo.yaml \
        training_method="simpo" \
        load_multi_preference=True \
        load_multi_preference_dataset=imdb_preference_with_source \
        model_name_or_path=gpt2 \
        output_dir="outputs/gpt2-imdb-simpo-${subset_id}" \
        run_name="gpt2-imdb-simpo-${subset_id}" \
        subset_id=${subset_id} \
        seed=42 "$@"
done

CUDA_VISIBLE_DEVICES=1 ACCELERATE_LOG_LEVEL=info accelerate launch run_simpo.py training_configs/gpt2-imdb-simpo.yaml \
        training_method="simpo" \
        load_multi_preference=True \
        load_multi_preference_dataset=imdb_preference_with_source \
        model_name_or_path=gpt2 \
        output_dir="outputs/gpt2-imdb-simpo-1" \
        run_name="gpt2-imdb-simpo-1" \
        subset_id=1 \
        seed=42 "$@"
