import numpy as np
import json

file = 'imdb_preference_dataset_with_source.json'

with open(file, 'r') as f:
    data = json.load(f)

results = []
for i in range(100):
    subset_indices = np.random.choice(100, 50, replace=False)
    subset_indices = np.sort(subset_indices)
    print(subset_indices)
    subset_data = [data[i] for i in subset_indices]
    results.append({'subset_indices': subset_indices.tolist(), 'subset_data': subset_data})
with open('imdb_preference_dataset_with_source_subset_indices.json', 'w') as f:
    json.dump(results, f)



