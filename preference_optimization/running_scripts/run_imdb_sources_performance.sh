CUDA_VISIBLE_DEVICES=0 ACCELERATE_LOG_LEVEL=info accelerate launch run_simpo.py training_configs/base-simpo.yaml \
    beta=2.0 training_method="simpo" \
    output_dir="outputs/subset-simpo-qwen-imdb-positive" \
    run_name="subset-simpo-multi-controlled-positive" \
    load_specific_pairs=True num_train_epochs=5 test_size=2000 \
    load_multi_preference_dataset=imdb_preference_with_source \
    preference_sources="positive"

CUDA_VISIBLE_DEVICES=0 ACCELERATE_LOG_LEVEL=info accelerate launch run_simpo.py training_configs/base-simpo.yaml \
    beta=2.0 training_method="simpo" \
    output_dir="outputs/subset-simpo-qwen-imdb-negative" \
    run_name="subset-simpo-multi-controlled-negative" \
    load_specific_pairs=True num_train_epochs=1 test_size=2000 \
    load_multi_preference_dataset=imdb_preference_with_source \
    preference_sources="negative"

CUDA_VISIBLE_DEVICES=0 ACCELERATE_LOG_LEVEL=info accelerate launch run_simpo.py training_configs/base-simpo.yaml \
    beta=2.0 training_method="simpo" \
    output_dir="outputs/subset-simpo-qwen-imdb-long" \
    run_name="subset-simpo-multi-controlled-long" \
    load_specific_pairs=True num_train_epochs=1 test_size=2000 \
    load_multi_preference_dataset=imdb_preference_with_source \
    preference_sources="long"

CUDA_VISIBLE_DEVICES=0 ACCELERATE_LOG_LEVEL=info accelerate launch run_simpo.py training_configs/base-simpo.yaml \
    beta=2.0 training_method="simpo" \
    output_dir="outputs/subset-simpo-qwen-imdb-short" \
    run_name="subset-simpo-multi-controlled-short" \
    load_specific_pairs=True num_train_epochs=1 test_size=2000 \
    load_multi_preference_dataset=imdb_preference_with_source \
    preference_sources="short"