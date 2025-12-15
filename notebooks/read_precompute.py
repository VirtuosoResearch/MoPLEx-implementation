# %%
import torch
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Construct path relative to this script's location
script_dir = os.path.dirname(os.path.abspath(__file__))
precompute_path = os.path.join(script_dir, '..', 'pre_compute', 'precomputed_gradients_b.pt')
precompute_path = os.path.normpath(precompute_path)

# Load precomputed data with error handling
precompute_data = torch.load(precompute_path, map_location='cpu', weights_only=False)
# Print available keys
print("Available keys in precomputed data:", list(precompute_data.keys()))

# Check and print projection matrix info if available
if 'projection_matrix' in precompute_data:
    print(f"Projection matrix shape: {precompute_data['projection_matrix'].shape}")

if 'projection_dim' in precompute_data:
    print(f"Projection dimension: {precompute_data['projection_dim']}")

# Print other available information
if 'gradients' in precompute_data:
    print(f"Number of gradients: {len(precompute_data['gradients'])}")
    if len(precompute_data['gradients']) > 0:
        print(f"Gradient dimension: {len(precompute_data['gradients'][0])}")
    print(precompute_data['gradients'][0])

if 'b_values' in precompute_data:
    print(f"Number of b values: {len(precompute_data['b_values'])}")
    print(f"b_values shape: {precompute_data['b_values'].shape}")
    print(precompute_data['b_values'])
# %%