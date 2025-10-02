# %%
from datasets import load_dataset

# raw_datasets = load_dataset("openai/collective-alignment-1", "merged_comparison_annotators")
# example = raw_datasets["train"][0]
raw_datasets_comparisons = load_dataset("openai/collective-alignment-1", "comparisons")
raw_datasets_annotators = load_dataset("openai/collective-alignment-1", "annotators")

# %%
country_groups = ['Argentina',
 'Australia',
 'Canada',
 'Chile',
 'Germany',
 'Greece',
 'India',
 'Italy',
 'Japan',
 'Kenya',
 'Korea, Republic of',
 'Malaysia',
 'Mexico',
 'Netherlands',
 'Portugal',
 'South Africa',
 'Switzerland',
 'United Kingdom',
 'United States']
age_groups = ['<44', '>45']
continents_groups = ['Africa', 'Asia', 'Europe', 'North_America', 'South_America']
gender_groups = ['Female', 'Male'] # remove 'Non-binary or Prefer not to say' since the samples are too few

country_to_continents = {
    'Argentina': 'South_America',
    'Australia': 'Asia', # Geopolitically considered part of Asia-Pacific as there is only one Australian annotator
    'Canada': 'North_America',
    'Chile': 'South_America',
    'Germany': 'Europe',
    'Greece': 'Europe',
    'India': 'Asia',
    'Italy': 'Europe',
    'Japan': 'Asia',
    'Kenya': 'Africa',
    'Korea, Republic of': 'Asia',
    'Malaysia': 'Asia',
    'Mexico': 'South_America',
    'Netherlands': 'Europe',
    'Portugal': 'Europe',
    'South Africa': 'Africa',
    'Switzerland': 'Europe',
    'United Kingdom': 'Europe',
    'United States': 'North_America'}
age_to_age_group = {'18-24': '<44',
    '25-34': '<44',
    '35-44': '<44',
    '45-54': '>45',
    '55-64': '>45',
    '65+': '>45'}

# %%
from collections import defaultdict
from itertools import product

# divide annotators into groups: age x continent x gender
groups = list(product(age_groups, continents_groups, gender_groups)) 
groups = ["{}_{}_{}".format(age, continent, gender) for age, continent, gender in groups]
group_to_annotators = defaultdict(list)
annotator_ids = list()
ground_truth_cluster_labels = []
for i, annotator in enumerate(raw_datasets_annotators['train']):
    demographics = annotator['demographics']
    # preferences = annotator['assessments']
    age = demographics['age']
    country = demographics['country_of_residence']
    gender = demographics['gender'] 
    if gender in 'Non-binary or Prefer not to say':
        ground_truth_cluster_labels.append(0) # skip non-binary or prefer not to say due to small sample size
        annotator_ids.append(annotator['annotator_id'])
        continue
    group_to_annotators["{}_{}_{}".format(age_to_age_group[age], country_to_continents[country], gender)].append(annotator['annotator_id'])
    ground_truth_cluster_labels.append(groups.index("{}_{}_{}".format(age_to_age_group[age], country_to_continents[country], gender)))
    annotator_ids.append(annotator['annotator_id'])
# %%
import numpy as np
np.save('./data_processing/collective-alignment/annotator_ids.npy', np.array(annotator_ids))

# %%
annotator_ids = np.load('./data_processing/collective-alignment/annotator_ids.npy', allow_pickle=True).tolist()

# %%
for key, val in group_to_annotators.items():
    print(key, len(val))
# %%
# construct annotator to prompt label matrix
def parse_ranking(ranking_str):
    """
    Parse a ranking string like 'B>A=C=D' into groups.
    """
    groups = ranking_str.split(">")        # split by '>'
    parsed = [group.split("=") for group in groups]
    # flatten the list
    return [item.strip() for sublist in parsed for item in sublist]

def kendall_tau_distance(rank1, rank2):
    """
    Compute Kendall Tau distance between two rankings.

    rank1, rank2: lists of items in ranked order
    """
    # Map items to their positions in each ranking
    pos1 = {item: i for i, item in enumerate(rank1)}
    pos2 = {item: i for i, item in enumerate(rank2)}

    # Get all pairs of distinct items
    items = rank1
    n = len(items)
    disagreements = 0

    for i in range(n):
        for j in range(i + 1, n):
            a, b = items[i], items[j]

            # Check order in both rankings
            order1 = pos1[a] - pos1[b]
            order2 = pos2[a] - pos2[b]

            # Disagreement if one ranking says a<b and other says b<a
            if order1 * order2 < 0:
                disagreements += 1

    return disagreements

def get_common_conversations(assessments1, assessments2):
    convs1 = set([item['conversation_id'] for item in assessments1])
    convs2 = set([item['conversation_id'] for item in assessments2])
    common_convs = convs1.intersection(convs2)
    return common_convs

import numpy as np
label_similarity_matrix = np.zeros((len(raw_datasets_annotators['train']), len(raw_datasets_annotators['train'])))
for i, annotator in enumerate(raw_datasets_annotators['train']):
    for j, other_annotator in enumerate(raw_datasets_annotators['train']):
        if i >= j: continue
        common_convs = get_common_conversations(annotator['assessments'], other_annotator['assessments'])
        if len(common_convs) == 0:
            label_similarity_matrix[i, j] = -100
            label_similarity_matrix[j, i] = -100
            print("No common conversations between annotator {} and {}".format(i, j))
            continue
        
        avg_similarity = 0
        for conv_id in common_convs:
            id1 = [item['conversation_id'] for item in annotator['assessments']].index(conv_id)
            try:
                ranking1 = parse_ranking(annotator['assessments'][id1]['ranking_blocks']['personal'][0]['ranking'])
            except:
                ranking1 = parse_ranking(annotator['assessments'][id1]['ranking_blocks']['world'][0]['ranking'])
            id2 = [item['conversation_id'] for item in other_annotator['assessments']].index(conv_id)
            try:
                ranking2 = parse_ranking(other_annotator['assessments'][id1]['ranking_blocks']['personal'][0]['ranking'])
            except:
                ranking2 = parse_ranking(other_annotator['assessments'][id2]['ranking_blocks']['world'][0]['ranking'])
            distance = kendall_tau_distance(ranking1, ranking2)
            avg_similarity += 1 - distance/6
        avg_similarity /= len(common_convs)
        label_similarity_matrix[i, j] = avg_similarity
        label_similarity_matrix[j, i] = avg_similarity

# %%
np.save('./data_processing/collective-alignment/annotator_label_similarity_matrix.npy', label_similarity_matrix)

# %%
import numpy as np
# import from parent directory
from utils.clustering import run_spectral_clustering, run_sdp_clustering

label_similarity_matrix = np.load('./data_processing/collective-alignment/annotator_label_similarity_matrix.npy')

clustering_assignment = run_spectral_clustering(label_similarity_matrix, k=20)
# %%
y_true = np.array(ground_truth_cluster_labels)
y_pred = np.array(clustering_assignment)

from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score
import numpy as np
from scipy.optimize import linear_sum_assignment

ARI = adjusted_rand_score(y_true, y_pred)
AMI = adjusted_mutual_info_score(y_true, y_pred, average_method="arithmetic")

# Clustering Accuracy with Hungarian
def clustering_accuracy(y_true, y_pred):
    classes = np.unique(y_true)
    clusters = np.unique(y_pred)
    L = len(classes); K = len(clusters)
    # Build confusion
    C = np.zeros((K, L), dtype=int)
    for k_idx, k in enumerate(clusters):
        for l_idx, l in enumerate(classes):
            C[k_idx, l_idx] = np.sum((y_pred==k) & (y_true==l))
    # Hungarian on negative counts (maximize matches)
    row_ind, col_ind = linear_sum_assignment(-C)
    return C[row_ind, col_ind].sum() / len(y_true)

ACC = clustering_accuracy(y_true, y_pred)

# %%
print("ARI: {:.4f}, AMI: {:.4f}, ACC: {:.4f}".format(ARI, AMI, ACC))
# ARI: 0.0027, AMI: 0.0125, ACC: 0.121

# %%
from datasets import load_dataset, DatasetDict, concatenate_datasets
import hashlib
import random
import time

ds = load_dataset("openai/collective-alignment-1", "comparisons")
total_rows = ds.num_rows

def explode_assessments(batch):
    out = {
        "prompt_id": [],
        "prompt": [],
        "responses": [],          # keep the full candidates for context
        "assessment": [],         # the single assessment for this new row
    }
    for i in range(len(batch["prompt_id"])):
        meta = batch["metadata"][i]
        assessments = meta.get("assessments", [])
        for a in assessments:
            out["prompt_id"].append(batch["prompt_id"][i])
            out["prompt"].append(batch["prompt"][i])
            out["responses"].append(batch["responses"][i])
            out["assessment"].append(a)
    return out

if isinstance(ds, DatasetDict):
    ds =  DatasetDict({
        split: ds[split].map(
            explode_assessments,
            batched=True,
            remove_columns=ds[split].column_names,  # drop originals; only keep what we return
        )
        for split in ds
    })
else:
    ds = ds.map(
        explode_assessments,
        batched=True,
        remove_columns=ds.column_names,
    )

# %%

load_specific_pairs = True
load_specific_pairs_idxes = [0, 1]

def parse_ranking(ranking_str):
    """
    Parse a ranking string like 'B>A=C=D' into groups.
    """
    groups = ranking_str.split(">")        # split by '>'
    parsed = [group.split("=") for group in groups]
    # flatten the list
    return [item.strip() for sublist in parsed for item in sublist]


def get_pairwise_completions(responses, ranking, seed=42):
    random.seed(seed)
    start = time.time()

    ranking = parse_ranking(ranking)
    scores_and_completions = [(len(ranking) - ranking.index(resp['response_index']), resp['messages'][0], resp['response_index']) for resp in responses]
    scores_and_completions.sort(key=lambda x: x[2]) # sort by score descending
    
    if load_specific_pairs:
        pairs = [scores_and_completions[idx] for idx in load_specific_pairs_idxes]
        chosen = max(pairs, key=lambda x: x[0])
        if pairs[0][0] == pairs[1][0]:
            rejected = pairs[1]
        else:
            rejected = min(pairs, key=lambda x: x[0])
    else:
        chosen = max(scores_and_completions, key=lambda x: x[0])
        rejected = random.choice(scores_and_completions)
        while rejected == chosen:
            end = time.time()
            if end - start > 3:
                print("Timeout")
                print(chosen, rejected)
                break
            rejected = random.choice(scores_and_completions)
    return chosen, rejected

def format_prompt(x):
    prompt = x["prompt"]['messages'][-1]
    ranking = x["assessment"]['ranking_blocks']['personal']
    if not ranking:
        ranking = x["assessment"]['ranking_blocks']['world']
    ranking = ranking[0]['ranking']
    chosen, rejected = get_pairwise_completions(x["responses"], ranking)
    chosen_messages = []
    rejected_messages = []
    chosen_messages = [
        prompt,
        chosen[1],
    ]
    rejected_messages = [
        prompt,
        rejected[1],
    ]
    return {
        "prompt": prompt,
        "prompt_id": x["prompt_id"],
        "chosen": chosen_messages,
        "rejected": rejected_messages,
        "messages": chosen_messages, # Use best-ranked example for SFT
        "score_chosen": chosen[0] if chosen is not None else -100.0,
        "score_rejected": rejected[0] if rejected is not None else -100.0,
        "criterion": x['assessment']['annotator_id'],
    }

ds_list = []
ds = ds['train']
tmp_ds = ds.map(format_prompt, num_proc=8, remove_columns=ds.column_names)
tmp_ds = tmp_ds.filter(lambda x: x["score_chosen"] != -100 or x["score_rejected"] != -100, num_proc=8)
ds_list.append(tmp_ds)

# %%
from data_processing.load_collective_alignment import load_collective_alignment

dataset = load_collective_alignment(annotators = None,
                             load_specific_pairs = True, 
                             load_specific_pairs_idxes = [[0,1], [0,2], [0,3], [1,2], [1,3], [2,3]],
                             load_indexes_path = None)
# %%
