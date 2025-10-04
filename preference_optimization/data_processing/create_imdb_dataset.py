from datasets import load_dataset, Dataset
import pandas as pd
import numpy as np
import random
import json
import os

def filter_by_scores(data, score_threshold=0.5):
    """
    Filter data based on scores field, keeping only data where max-min > threshold
    """
    filtered_data = []
    
    for item in data:
        scores = item['scores']
        if isinstance(scores, list) and len(scores) > 0:
            # Calculate max - min
            score_diff = max(scores) - min(scores)
            if score_diff > score_threshold:
                filtered_data.append(item)
    
    return filtered_data

def add_metadata_fields(data, source_name="ZHZisZZ/imdb_preference", annotator_name="default"):
    """
    Add source and annotator fields to all data
    """
    enhanced_data = []
    
    for item in data:
        enhanced_item = item.copy()
        enhanced_item['source'] = source_name
        enhanced_item['annotator'] = annotator_name
        enhanced_data.append(enhanced_item)
    
    return enhanced_data

def load_imdb_preference_train(score_threshold=0.5, source_name="ZHZisZZ/imdb_preference", annotator_name="default"):
    """
    Load ZHZisZZ/imdb_preference dataset train split from Hugging Face,
    filter based on scores, and add metadata fields
    """
    print("Loading ZHZisZZ/imdb_preference dataset from Hugging Face...")
    
    # Load dataset
    dataset = load_dataset("ZHZisZZ/imdb_preference")
    train_data = dataset['train']
    
    print(f"Original train split data count: {len(train_data)}")
    
    # Convert to list for filtering
    data_list = list(train_data)
    
    # Filter data based on scores
    print(f"Filtering data based on scores (threshold={score_threshold})...")
    filtered_data = filter_by_scores(data_list, score_threshold)
    print(f"Filtered data count: {len(filtered_data)}")
    
    # Add source and annotator fields
    print("Adding source and annotator fields...")
    enhanced_data = add_metadata_fields(filtered_data, source_name, annotator_name)
    
    # Display data sample
    if len(enhanced_data) > 0:
        print("\nProcessed data sample:")
        sample = enhanced_data[0]
        print(f"Data columns: {list(sample.keys())}")
        print(f"Sample data:")
        for key, value in sample.items():
            print(f"  {key}: {value}")
    
    return enhanced_data

def split_data_into_four_sets(data):
    """
    Split data into 4 sets with ratio 3:2:3:2 based on different criteria:
    1. First set: Keep original 'chosen', set 'source' to 'positive'
    2. Second set: Flip 'chosen' (0->1, 1->0), set 'source' to 'negative'
    3. Third set: Choose longer response index as 'chosen', set 'source' to 'long'
    4. Fourth set: Choose shorter response index as 'chosen', set 'source' to 'short'
    """
    # Shuffle data to ensure random distribution
    random.shuffle(data)
    
    total_count = len(data)
    
    # Calculate split sizes based on 3:2:3:2 ratio
    # Total parts = 3 + 2 + 3 + 2 = 10
    part1_size = int(total_count * 3 / 10)  # positive
    part2_size = int(total_count * 2 / 10)  # negative
    part3_size = int(total_count * 3 / 10)  # long
    part4_size = total_count - part1_size - part2_size - part3_size  # short (remaining)
    
    print(f"Splitting {total_count} items into 4 sets:")
    print(f"  Set 1 (positive): {part1_size} items")
    print(f"  Set 2 (negative): {part2_size} items")
    print(f"  Set 3 (long): {part3_size} items")
    print(f"  Set 4 (short): {part4_size} items")
    
    # Split data
    part1_data = data[:part1_size]
    part2_data = data[part1_size:part1_size + part2_size]
    part3_data = data[part1_size + part2_size:part1_size + part2_size + part3_size]
    part4_data = data[part1_size + part2_size + part3_size:]
    
    # Process each part
    processed_sets = []
    
    # Set 1: Keep original chosen, set source to 'positive'
    set1 = []
    for item in part1_data:
        new_item = item.copy()
        new_item['source'] = 'positive'
        set1.append(new_item)
    processed_sets.append(('positive', set1))
    
    # Set 2: Flip chosen (0->1, 1->0), set source to 'negative'
    set2 = []
    for item in part2_data:
        new_item = item.copy()
        # Flip chosen: 0->1, 1->0
        new_item['chosen'] = 1 - item['chosen']
        new_item['source'] = 'negative'
        set2.append(new_item)
    processed_sets.append(('negative', set2))
    
    # Set 3: Choose longer response index, set source to 'long'
    set3 = []
    for item in part3_data:
        new_item = item.copy()
        responses = item['responses']
        if len(responses) >= 2:
            # Choose index of longer response (0 or 1)
            if len(responses[0]) >= len(responses[1]):
                new_item['chosen'] = 0
            else:
                new_item['chosen'] = 1
        else:
            # Fallback to original chosen if not enough responses
            new_item['chosen'] = item['chosen']
        new_item['source'] = 'long'
        set3.append(new_item)
    processed_sets.append(('long', set3))
    
    # Set 4: Choose shorter response index, set source to 'short'
    set4 = []
    for item in part4_data:
        new_item = item.copy()
        responses = item['responses']
        if len(responses) >= 2:
            # Choose index of shorter response (0 or 1)
            if len(responses[0]) <= len(responses[1]):
                new_item['chosen'] = 0
            else:
                new_item['chosen'] = 1
        else:
            # Fallback to original chosen if not enough responses
            new_item['chosen'] = item['chosen']
        new_item['source'] = 'short'
        set4.append(new_item)
    processed_sets.append(('short', set4))
    
    return processed_sets

def load_imdb_preference_train_with_splits(score_threshold=0.5, source_name="ZHZisZZ/imdb_preference", annotator_name="default"):
    """
    Load and process data with 4-way splitting
    """
    print("Loading ZHZisZZ/imdb_preference dataset from Hugging Face...")
    
    # Load dataset
    dataset = load_dataset("ZHZisZZ/imdb_preference")
    train_data = dataset['train']
    
    print(f"Original train split data count: {len(train_data)}")
    
    # Convert to list for filtering
    data_list = list(train_data)
    
    # Filter data based on scores
    print(f"Filtering data based on scores (threshold={score_threshold})...")
    filtered_data = filter_by_scores(data_list, score_threshold)
    print(f"Filtered data count: {len(filtered_data)}")
    
    # Add initial metadata fields
    print("Adding initial metadata fields...")
    enhanced_data = add_metadata_fields(filtered_data, source_name, annotator_name)
    
    # Split into 4 sets
    print("Splitting data into 4 sets...")
    split_sets = split_data_into_four_sets(enhanced_data)
    
    # Display summary
    total_processed = sum(len(split[1]) for split in split_sets)
    print(f"\nTotal processed data: {total_processed}")
    
    for set_name, set_data in split_sets:
        print(f"\n{set_name.capitalize()} set:")
        print(f"  Count: {len(set_data)}")
        if len(set_data) > 0:
            sample = set_data[0]
            print(f"  Sample chosen: {sample['chosen']}")
            print(f"  Sample source: {sample['source']}")
            print(f"  Sample annotator: {sample['annotator']}")
    
    return split_sets

def assign_annotator_ids_and_split(all_data, samples_per_annotator=500, train_ratio=0.9):
    """
    Assign annotator IDs to data and perform train/test split within each annotator
    
    Args:
        all_data: Combined data from all 4 sets
        samples_per_annotator: Number of samples per annotator (default: 500)
        train_ratio: Ratio for train split (default: 0.9 for 9:1 split)
    
    Returns:
        Dictionary with 'train' and 'test' splits
    """
    print(f"Assigning annotator IDs and performing train/test split...")
    print(f"Samples per annotator: {samples_per_annotator}")
    print(f"Train ratio: {train_ratio}")
    
    # Group data by source
    source_groups = {}
    for item in all_data:
        source = item['source']
        if source not in source_groups:
            source_groups[source] = []
        source_groups[source].append(item)
    
    print(f"\nSource groups found: {list(source_groups.keys())}")
    for source, data in source_groups.items():
        print(f"  {source}: {len(data)} items")
    
    # Assign annotator IDs
    annotated_data = []
    global_annotator_id = 0
    
    for source, source_data in source_groups.items():
        print(f"\nProcessing source '{source}' with {len(source_data)} items...")
        
        # Shuffle data within each source for random assignment
        random.shuffle(source_data)
        
        # Group into annotators (500 samples each)
        num_annotators = (len(source_data) + samples_per_annotator - 1) // samples_per_annotator
        print(f"  Creating {num_annotators} annotators")
        
        for annotator_idx in range(num_annotators):
            start_idx = annotator_idx * samples_per_annotator
            end_idx = min(start_idx + samples_per_annotator, len(source_data))
            annotator_data = source_data[start_idx:end_idx]
            
            # Assign annotator ID to each item
            for item in annotator_data:
                new_item = item.copy()
                new_item['annotator'] = global_annotator_id
                annotated_data.append(new_item)
            
            print(f"    Annotator {global_annotator_id}: {len(annotator_data)} items")
            global_annotator_id += 1
    
    print(f"\nTotal annotators created: {global_annotator_id}")
    print(f"Total annotated data: {len(annotated_data)}")
    
    # Perform train/test split within each annotator
    train_data = []
    test_data = []
    
    for annotator_id in range(global_annotator_id):
        # Get all data for this annotator
        annotator_items = [item for item in annotated_data if item['annotator'] == annotator_id]
        
        # Shuffle within annotator
        random.shuffle(annotator_items)
        
        # Calculate split point
        train_size = int(len(annotator_items) * train_ratio)
        
        # Split data
        annotator_train = annotator_items[:train_size]
        annotator_test = annotator_items[train_size:]
        
        train_data.extend(annotator_train)
        test_data.extend(annotator_test)
        
        print(f"Annotator {annotator_id}: {len(annotator_train)} train, {len(annotator_test)} test")
    
    print(f"\nFinal split results:")
    print(f"  Train data: {len(train_data)} items")
    print(f"  Test data: {len(test_data)} items")
    
    # Display source distribution in train/test
    train_sources = {}
    test_sources = {}
    
    for item in train_data:
        source = item['source']
        train_sources[source] = train_sources.get(source, 0) + 1
    
    for item in test_data:
        source = item['source']
        test_sources[source] = test_sources.get(source, 0) + 1
    
    print(f"\nTrain set source distribution:")
    for source, count in train_sources.items():
        print(f"  {source}: {count}")
    
    print(f"\nTest set source distribution:")
    for source, count in test_sources.items():
        print(f"  {source}: {count}")
    
    return {
        'train': train_data,
        'test': test_data,
        'annotated_data': annotated_data
    }

def load_and_process_imdb_data(score_threshold=0.5, samples_per_annotator=500, train_ratio=0.9):
    """
    Complete pipeline: load, filter, split into 4 sets, assign annotator IDs, and train/test split
    """
    print("=" * 60)
    print("Starting complete IMDB data processing pipeline")
    print("=" * 60)
    
    # Step 1: Load and split into 4 sets
    split_sets = load_imdb_preference_train_with_splits(score_threshold)
    
    # Step 2: Combine all sets
    all_data = []
    for set_name, set_data in split_sets:
        all_data.extend(set_data)
    
    print(f"\nCombined data from all sets: {len(all_data)} items")
    
    # Step 3: Assign annotator IDs and perform train/test split
    result = assign_annotator_ids_and_split(all_data, samples_per_annotator, train_ratio)
    
    return result

def save_data_to_json(train_data, test_data, annotated_data, output_dir="output"):
    """
    Save train, test, and annotated data to JSON files
    
    Args:
        train_data: List of training data items
        test_data: List of test data items
        annotated_data: List of all annotated data items
        output_dir: Directory to save JSON files
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\nSaving data to JSON files in directory: {output_dir}")
    
    # Save train data
    train_file = os.path.join(output_dir, "train_data.json")
    with open(train_file, 'w', encoding='utf-8') as f:
        json.dump(train_data, f, indent=2, ensure_ascii=False)
    print(f"  Saved train data: {train_file} ({len(train_data)} items)")
    
    # Save test data
    test_file = os.path.join(output_dir, "test_data.json")
    with open(test_file, 'w', encoding='utf-8') as f:
        json.dump(test_data, f, indent=2, ensure_ascii=False)
    print(f"  Saved test data: {test_file} ({len(test_data)} items)")
    
    # Save all annotated data
    annotated_file = os.path.join(output_dir, "annotated_data.json")
    with open(annotated_file, 'w', encoding='utf-8') as f:
        json.dump(annotated_data, f, indent=2, ensure_ascii=False)
    print(f"  Saved annotated data: {annotated_file} ({len(annotated_data)} items)")
    
    # Save metadata summary
    metadata = {
        "total_items": len(annotated_data),
        "train_items": len(train_data),
        "test_items": len(test_data),
        "train_ratio": len(train_data) / len(annotated_data) if len(annotated_data) > 0 else 0,
        "test_ratio": len(test_data) / len(annotated_data) if len(annotated_data) > 0 else 0,
        "sources": list(set(item['source'] for item in annotated_data)),
        "annotators": list(set(item['annotator'] for item in annotated_data)),
        "total_annotators": len(set(item['annotator'] for item in annotated_data))
    }
    
    metadata_file = os.path.join(output_dir, "metadata.json")
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f"  Saved metadata: {metadata_file}")
    
    # Save source distribution
    source_dist = {}
    for item in annotated_data:
        source = item['source']
        if source not in source_dist:
            source_dist[source] = 0
        source_dist[source] += 1
    
    source_dist_file = os.path.join(output_dir, "source_distribution.json")
    with open(source_dist_file, 'w', encoding='utf-8') as f:
        json.dump(source_dist, f, indent=2, ensure_ascii=False)
    print(f"  Saved source distribution: {source_dist_file}")
    
    # Save annotator distribution
    annotator_dist = {}
    for item in annotated_data:
        annotator = item['annotator']
        if annotator not in annotator_dist:
            annotator_dist[annotator] = 0
        annotator_dist[annotator] += 1
    
    annotator_dist_file = os.path.join(output_dir, "annotator_distribution.json")
    with open(annotator_dist_file, 'w', encoding='utf-8') as f:
        json.dump(annotator_dist, f, indent=2, ensure_ascii=False)
    print(f"  Saved annotator distribution: {annotator_dist_file}")
    
    print(f"\nAll data saved successfully!")
    return {
        "train_file": train_file,
        "test_file": test_file,
        "annotated_file": annotated_file,
        "metadata_file": metadata_file,
        "source_dist_file": source_dist_file,
        "annotator_dist_file": annotator_dist_file
    }

def load_and_process_imdb_data_with_save(score_threshold=0.5, samples_per_annotator=500, train_ratio=0.9, output_dir="output"):
    """
    Complete pipeline with JSON saving functionality
    """
    print("=" * 60)
    print("Starting complete IMDB data processing pipeline with JSON saving")
    print("=" * 60)
    
    # Run the processing pipeline
    result = load_and_process_imdb_data(score_threshold, samples_per_annotator, train_ratio)
    
    train_data = result['train']
    test_data = result['test']
    annotated_data = result['annotated_data']
    
    # Save to JSON files
    file_paths = save_data_to_json(train_data, test_data, annotated_data, output_dir)
    
    return {
        'train_data': train_data,
        'test_data': test_data,
        'annotated_data': annotated_data,
        'file_paths': file_paths
    }

if __name__ == "__main__":
    # Run the complete pipeline with JSON saving
    result = load_and_process_imdb_data_with_save(
        score_threshold=0.5,
        samples_per_annotator=500,
        train_ratio=0.9,
        output_dir="imdb_processed_data"
    )
    
    train_data = result['train_data']
    test_data = result['test_data']
    annotated_data = result['annotated_data']
    file_paths = result['file_paths']
    
    print(f"\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(f"Total annotated data: {len(annotated_data)}")
    print(f"Train data: {len(train_data)}")
    print(f"Test data: {len(test_data)}")
    
    print(f"\nFiles saved:")
    for file_type, file_path in file_paths.items():
        print(f"  {file_type}: {file_path}")
    
    # Display sample data
    if train_data:
        print(f"\nSample train data:")
        sample = train_data[0]
        for key, value in sample.items():
            print(f"  {key}: {value}")
    
    if test_data:
        print(f"\nSample test data:")
        sample = test_data[0]
        for key, value in sample.items():
            print(f"  {key}: {value}")
    
    # Convert to DataFrames for analysis
    train_df = pd.DataFrame(train_data)
    test_df = pd.DataFrame(test_data)
    
    print(f"\nTrain DataFrame shape: {train_df.shape}")
    print(f"Test DataFrame shape: {test_df.shape}")
    
    print(f"\nTrain set annotator distribution:")
    print(train_df['annotator'].value_counts().sort_index())
    
    print(f"\nTest set annotator distribution:")
    print(test_df['annotator'].value_counts().sort_index())
    
    print(f"\nTrain set source distribution:")
    print(train_df['source'].value_counts())
    
    print(f"\nTest set source distribution:")
    print(test_df['source'].value_counts())
    
    print(f"\n" + "=" * 60)
    print("PROCESSING COMPLETED SUCCESSFULLY!")
    print("All data has been saved to JSON files.")
    print("=" * 60)

