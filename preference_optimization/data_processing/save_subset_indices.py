import numpy as np
import json

annotator_ids = np.load('./collective-alignment/annotator_ids.npy', allow_pickle=True).tolist()

# Sample 100 subsets with 500 annotator ids each, save to json
results = []
for i in range(100):
    subset_indices = np.random.choice(len(annotator_ids), 500, replace=False)
    subset_indices = np.sort(subset_indices)
    print(subset_indices)
    subset_annotator_ids = [annotator_ids[i] for i in subset_indices]
    results.append({'subset_indices': subset_indices.tolist(), 'subset_annotator_ids': subset_annotator_ids})
with open('./collective-alignment/subset_indices.json', 'w') as f:
    json.dump(results, f)



