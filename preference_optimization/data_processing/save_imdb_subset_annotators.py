#!/usr/bin/env python3
"""
Create and save annotator subsets for IMDB preference dataset.
Randomly samples 50% of annotators, creates 50 different subsets.
"""

import json
import random
import os
from typing import List, Dict

def get_all_annotators() -> List[int]:
    """Get all available annotator IDs from the processed data"""
    metadata_path = "./imdb_processed_data/metadata.json"
    
    if not os.path.exists(metadata_path):
        # Fallback: assume we have annotators 0-35
        return list(range(36))
    
    with open(metadata_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    
    return metadata.get('annotators', list(range(36)))

def create_annotator_subsets(total_annotators: List[int], num_subsets: int = 50, sample_ratio: float = 0.5) -> List[Dict]:
    """
    Create multiple annotator subsets by random sampling
    
    Args:
        total_annotators: List of all available annotator IDs
        num_subsets: Number of subsets to create
        sample_ratio: Ratio of annotators to sample (0.5 = 50%)
    
    Returns:
        List of subset dictionaries
    """
    subsets = []
    sample_size = int(len(total_annotators) * sample_ratio)
    
    print(f"Creating {num_subsets} subsets with {sample_size} annotators each (from {len(total_annotators)} total)")
    
    for subset_id in range(num_subsets):
        # Randomly sample annotators
        sampled_annotators = random.sample(total_annotators, sample_size)
        sampled_annotators.sort()  # Sort for consistency
        
        subset_info = {
            "subset_id": subset_id,
            "subset_size": sample_size,
            "annotator_ids": sampled_annotators,
            "description": f"Random subset {subset_id} with {sample_size} annotators"
        }
        
        subsets.append(subset_info)
        
        print(f"  Subset {subset_id}: {len(sampled_annotators)} annotators")
    
    return subsets

def save_subsets_to_json(subsets: List[Dict], output_path: str = "./imdb_annotator_subsets.json"):
    """Save subsets to JSON file"""
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    
    # Create metadata for the file
    output_data = {
        "metadata": {
            "total_subsets": len(subsets),
            "sample_ratio": 0.5,
            "description": "Random annotator subsets for IMDB preference dataset",
            "created_by": "save_imdb_subset_annotators.py"
        },
        "subsets": subsets
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"\nSaved {len(subsets)} subsets to: {output_path}")
    return output_path

def load_subsets_from_json(json_path: str = "./imdb_annotator_subsets.json") -> Dict:
    """Load subsets from JSON file"""
    
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Subsets file not found: {json_path}")
    
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    return data

def get_subset_annotators(subset_id: int, json_path: str = "./imdb_annotator_subsets.json") -> List[int]:
    """
    Get annotator IDs for a specific subset
    
    Args:
        subset_id: ID of the subset to retrieve
        json_path: Path to the subsets JSON file
    
    Returns:
        List of annotator IDs for the specified subset
    """
    data = load_subsets_from_json(json_path)
    subsets = data['subsets']
    
    for subset in subsets:
        if subset['subset_id'] == subset_id:
            return subset['annotator_ids']
    
    raise ValueError(f"Subset ID {subset_id} not found. Available subset IDs: {[s['subset_id'] for s in subsets]}")

def main():
    """Main function to create and save annotator subsets"""
    
    print("Creating IMDB annotator subsets...")
    
    # Set random seed for reproducibility
    random.seed(42)
    
    # Get all available annotators
    all_annotators = get_all_annotators()
    print(f"Found {len(all_annotators)} total annotators: {all_annotators}")
    
    # Create subsets
    subsets = create_annotator_subsets(all_annotators, num_subsets=50, sample_ratio=0.5)
    
    # Save to JSON
    output_path = save_subsets_to_json(subsets)
    
    # Display some statistics
    print(f"\n=== Subset Statistics ===")
    print(f"Total subsets created: {len(subsets)}")
    print(f"Annotators per subset: {subsets[0]['subset_size']}")
    print(f"Total annotators: {len(all_annotators)}")
    print(f"Sample ratio: 50%")
    
    # Show first few subsets as examples
    print(f"\n=== Example Subsets ===")
    for i in range(min(5, len(subsets))):
        subset = subsets[i]
        print(f"Subset {subset['subset_id']}: {subset['annotator_ids']}")
    
    # Test loading functionality
    print(f"\n=== Testing Load Functionality ===")
    try:
        test_subset_0 = get_subset_annotators(0, output_path)
        print(f"Successfully loaded subset 0: {test_subset_0}")
        
        test_subset_25 = get_subset_annotators(25, output_path)
        print(f"Successfully loaded subset 25: {test_subset_25}")
        
    except Exception as e:
        print(f"Error testing load functionality: {e}")
    
    print(f"\n=== Creation Complete ===")
    print(f"Use subset_id parameter in run_simpo.py to select specific annotator combinations.")

if __name__ == "__main__":
    main()
