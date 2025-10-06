"""Build annotator affinity matrices with optional surrogate models and cluster them."""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import torch

from data_processing.load_imdb_preference_with_source import load_imdb_preference_with_source

from .annotator_surrogates import (
    fit_krr_surrogates,
    fit_lstsq_surrogates,
    predict_krr_surrogates,
    predict_lstsq_surrogates,
)
from .cluster import spectral_clustering_groups
from .score_matrix_io import load_score_matrix_npz, get_matrix_layout


def build_feature_matrix(
    subset_ids: Sequence[int],
    trained_lists: Sequence[Sequence[int]],
    annotator_ids: Sequence[int],
    include_bias: bool = True,
) -> torch.Tensor:
    annotator_order = sorted(int(a) for a in annotator_ids)
    index = {ann: pos for pos, ann in enumerate(annotator_order)}
    features = torch.zeros((len(subset_ids), len(annotator_order)), dtype=torch.float32)
    for row_idx, members in enumerate(trained_lists):
        for annotator in members:
            pos = index.get(int(annotator))
            if pos is not None:
                features[row_idx, pos] = 1.0
    if include_bias:
        bias = torch.ones((features.shape[0], 1), dtype=features.dtype)
        features = torch.cat([features, bias], dim=1)
    return features


def split_train_eval(
    features: torch.Tensor,
    scores: torch.Tensor,
    subset_ids: Sequence[int],
    trained_lists: Sequence[Sequence[int]],
    train_fraction: float,
) -> Tuple[torch.Tensor, torch.Tensor, List[int], List[List[int]], torch.Tensor, torch.Tensor, List[int], List[List[int]]]:
    if scores.shape[0] < 2:
        raise ValueError("At least two subsets required for train/eval split")
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be within (0, 1)")

    total = scores.shape[0]
    train_rows = max(1, min(int(math.floor(total * train_fraction)), total - 1))

    train_slice = slice(0, train_rows)
    eval_slice = slice(train_rows, total)

    return (
        features[train_slice],
        scores[train_slice],
        list(subset_ids[:train_rows]),
        [list(x) for x in trained_lists[:train_rows]],
        features[eval_slice],
        scores[eval_slice],
        list(subset_ids[train_rows:]),
        [list(x) for x in trained_lists[train_rows:]],
    )


def sample_eval_rows(
    features: torch.Tensor,
    scores: torch.Tensor,
    subset_ids: Sequence[int],
    trained_lists: Sequence[Sequence[int]],
    sample_count: int,
    rng: random.Random,
) -> List[Tuple[torch.Tensor, torch.Tensor, int, List[int]]]:
    num_rows = scores.shape[0]
    if num_rows == 0:
        return []
    if sample_count >= num_rows:
        indices = list(range(num_rows))
    else:
        indices = rng.sample(range(num_rows), sample_count)
    samples: List[Tuple[torch.Tensor, torch.Tensor, int, List[int]]] = []
    for idx in indices:
        samples.append((features[idx : idx + 1], scores[idx], int(subset_ids[idx]), list(trained_lists[idx])))
    return samples


def build_affinity_matrix(
    annotator_ids: Sequence[int],
    sample_records: Mapping[int, List[Dict[str, object]]],
) -> Tuple[np.ndarray, np.ndarray]:
    n = len(annotator_ids)
    sums = np.zeros((n, n), dtype=np.float64)
    counts = np.zeros((n, n), dtype=np.float64)
    index = {int(a): i for i, a in enumerate(annotator_ids)}

    for row_id, records in sample_records.items():
        row_idx = index[int(row_id)]
        for record in records:
            value = float(record["prediction"])
            for member in record["subset_annotators"]:
                col = index.get(int(member))
                if col is None:
                    continue
                sums[row_idx, col] += value
                counts[row_idx, col] += 1

    with np.errstate(divide="ignore", invalid="ignore"):
        affinity = sums / counts
    return affinity, counts


def serialize_samples(samples: Mapping[int, List[Dict[str, object]]]) -> Dict[str, List[Dict[str, object]]]:
    return {
        str(annotator): [
            {
                "subset_id": record["subset_id"],
                "subset_annotators": [int(x) for x in record["subset_annotators"]],
                "prediction": float(record["prediction"]),
            }
            for record in records
        ]
        for annotator, records in samples.items()
    }


def collect_source_counts() -> Dict[int, Counter[str]]:
    dataset = load_imdb_preference_with_source(test_size=0.1, seed=42)
    counts: Dict[int, Counter[str]] = defaultdict(Counter)
    for split in ("train", "test"):
        for example in dataset[split]:
            annotator = example.get("annotator")
            source = example.get("preference_source") or example.get("source")
            if annotator is None or source is None:
                continue
            counts[int(annotator)][str(source)] += 1
    return counts


def evaluate_clusters(
    clusters: Iterable[Sequence[int]],
    annotator_ids: Sequence[int],
    source_counts: Mapping[int, Counter[str]],
) -> Dict[str, Dict[str, object]]:
    evaluation: Dict[str, Dict[str, object]] = {}
    for idx, group in enumerate(clusters):
        annotators = [int(annotator_ids[i]) for i in group]
        total_counter: Counter[str] = Counter()
        for ann in annotators:
            total_counter.update(source_counts.get(ann, Counter()))
        total = sum(total_counter.values())
        if total == 0:
            distribution: Dict[str, float] = {}
            majority = None
            entropy = None
        else:
            distribution = {src: count / total for src, count in total_counter.items()}
            majority_src, majority_count = max(total_counter.items(), key=lambda item: item[1])
            majority = {"source": majority_src, "proportion": majority_count / total}
            entropy = -sum(p * math.log(p + 1e-12) for p in distribution.values())
        evaluation[str(idx)] = {
            "annotators": annotators,
            "total_samples": total,
            "source_distribution": distribution,
            "majority": majority,
            "entropy": entropy,
        }
    return evaluation


def run_clustering(
    affinity_matrices: Mapping[str, np.ndarray],
    annotator_ids: Sequence[int],
    source_counts: Mapping[int, Counter[str]],
    output_dir: Path,
    k: int = 4,
) -> Dict[str, Dict[str, object]]:
    summary: Dict[str, Dict[str, object]] = {}
    for name, matrix in affinity_matrices.items():
        symmetric = (matrix + matrix.T) / 2.0
        symmetric = np.nan_to_num(symmetric, nan=np.nanmean(symmetric))
        groups = spectral_clustering_groups(symmetric, k=k, use_exp=False)
        clusters_indices = {str(i): group.tolist() for i, group in enumerate(groups)}
        evaluation = evaluate_clusters(groups, annotator_ids, source_counts)
        with (output_dir / f"clusters_{name}.json").open("w", encoding="utf-8") as handle:
            json.dump(
                {
                    "clusters": {cluster_id: [int(annotator_ids[i]) for i in indices]
                                 for cluster_id, indices in clusters_indices.items()},
                    "groups_indices": clusters_indices,
                    "evaluation": evaluation,
                },
                handle,
                indent=2,
            )
        summary[name] = {"evaluation": evaluation}
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--score-matrix",
        required=True,
        type=str,
        help="Unified score matrix (.npz) produced by prepare_score_matrix.py",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/annotator_affinity",
        help="Directory to store surrogate artifacts and clustering outputs",
    )
    parser.add_argument(
        "--train-fraction",
        type=float,
        default=0.8,
        help="Fraction of subset rows used to train surrogate models",
    )
    parser.add_argument(
        "--sample-count",
        type=int,
        default=100,
        help="Number of evaluation subsets to sample for affinity estimation",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling",
    )
    parser.add_argument(
        "--skip-surrogate",
        action="store_true",
        help="Skip surrogate fitting and only use original score matrix",
    )
    parser.add_argument(
        "--krr-alpha",
        type=float,
        default=1e-3,
        help="Kernel ridge regression alpha",
    )
    parser.add_argument(
        "--krr-kernel",
        type=str,
        default="rbf",
        choices=["rbf", "linear", "poly"],
        help="Kernel for kernel ridge regression",
    )
    parser.add_argument(
        "--krr-gamma",
        type=float,
        default=None,
        help="Gamma parameter for kernel ridge regression",
    )
    parser.add_argument(
        "--krr-degree",
        type=int,
        default=3,
        help="Polynomial degree when kernel='poly'",
    )
    parser.add_argument(
        "--krr-coef0",
        type=float,
        default=1.0,
        help="Polynomial coef0 when kernel='poly'",
    )
    parser.add_argument(
        "--clusters",
        type=int,
        default=4,
        help="Number of clusters for spectral clustering",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    payload = load_score_matrix_npz(Path(args.score_matrix))
    subset_ids = payload["subset_ids"]
    trained_lists = payload["trained_annotators"]
    annotator_ids = payload["annotator_ids"]
    scores_np = np.asarray(payload["scores"], dtype=np.float32)

    layout = get_matrix_layout(payload)
    if layout == "annotator_rows":
        # rows are annotators, columns are subsets
        scores_np = scores_np.T  # rows become subsets
        # subset_ids/trained_lists already aligned with columns, keep order
        subset_ids = [int(s) for s in subset_ids]
        trained_lists = [list(t) for t in trained_lists]
    else:
        order = np.argsort(subset_ids)
        subset_ids = [int(subset_ids[i]) for i in order]
        trained_lists = [trained_lists[i] for i in order]
        scores_np = scores_np[order]

    features = build_feature_matrix(subset_ids, trained_lists, annotator_ids)
    scores = torch.tensor(scores_np, dtype=torch.float32)

    (
        train_features,
        train_scores,
        _,
        _,
        eval_features,
        eval_scores,
        eval_subset_ids,
        eval_trained_lists,
    ) = split_train_eval(features, scores, subset_ids, trained_lists, args.train_fraction)

    rng = random.Random(args.seed)
    samples = sample_eval_rows(
        eval_features,
        eval_scores,
        eval_subset_ids,
        eval_trained_lists,
        args.sample_count,
        rng,
    )

    annotator_tuple = tuple(int(a) for a in annotator_ids)
    perform_surrogate = not args.skip_surrogate

    if perform_surrogate:
        lstsq_coeffs, lstsq_order = fit_lstsq_surrogates(train_features, train_scores, annotator_tuple)
        krr_packed, krr_order = fit_krr_surrogates(
            train_features,
            train_scores,
            annotator_ids=annotator_tuple,
            alpha=args.krr_alpha,
            kernel=args.krr_kernel,
            gamma=args.krr_gamma,
            degree=args.krr_degree,
            coef0=args.krr_coef0,
        )
        if lstsq_order != krr_order:
            raise ValueError("Annotator ordering mismatch between surrogates")
        surrogate_order = lstsq_order
    else:
        surrogate_order = annotator_tuple

    if len(surrogate_order) != len(annotator_tuple):
        raise ValueError("Unexpected annotator ordering mismatch")

    sample_records_actual: Dict[int, List[Dict[str, object]]] = defaultdict(list)
    sample_records_lstsq: Dict[int, List[Dict[str, object]]] = defaultdict(list)
    sample_records_krr: Dict[int, List[Dict[str, object]]] = defaultdict(list)

    for feature_row, score_row, subset_id, members in samples:
        record_common = {
            "subset_id": subset_id,
            "subset_annotators": [int(x) for x in members],
        }
        actual_values = score_row
        preds_lstsq = predict_lstsq_surrogates(lstsq_coeffs, feature_row).squeeze(0) if perform_surrogate else None
        preds_krr = predict_krr_surrogates(krr_packed, feature_row).squeeze(0) if perform_surrogate else None
        for idx, annotator in enumerate(surrogate_order):
            actual = float(actual_values[idx].item())
            sample_records_actual[int(annotator)].append({**record_common, "prediction": actual})
            if perform_surrogate:
                sample_records_lstsq[int(annotator)].append({**record_common, "prediction": float(preds_lstsq[idx].item())})
                sample_records_krr[int(annotator)].append({**record_common, "prediction": float(preds_krr[idx].item())})

    affinity_matrices: Dict[str, np.ndarray] = {}
    counts_matrices: Dict[str, np.ndarray] = {}

    actual_affinity, actual_counts = build_affinity_matrix(surrogate_order, sample_records_actual)
    affinity_matrices["actual"] = actual_affinity
    counts_matrices["actual"] = actual_counts

    if perform_surrogate:
        lstsq_affinity, lstsq_counts = build_affinity_matrix(surrogate_order, sample_records_lstsq)
        krr_affinity, krr_counts = build_affinity_matrix(surrogate_order, sample_records_krr)
        affinity_matrices["lstsq"] = lstsq_affinity
        affinity_matrices["krr"] = krr_affinity
        counts_matrices["lstsq"] = lstsq_counts
        counts_matrices["krr"] = krr_counts

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for name, matrix in affinity_matrices.items():
        np.save(output_dir / f"{name}_matrix.npy", matrix)
    for name, counts in counts_matrices.items():
        np.save(output_dir / f"{name}_counts.npy", counts)

    with (output_dir / "samples_actual.json").open("w", encoding="utf-8") as handle:
        json.dump(serialize_samples(sample_records_actual), handle, indent=2)
    if perform_surrogate:
        with (output_dir / "samples_lstsq.json").open("w", encoding="utf-8") as handle:
            json.dump(serialize_samples(sample_records_lstsq), handle, indent=2)
        with (output_dir / "samples_krr.json").open("w", encoding="utf-8") as handle:
            json.dump(serialize_samples(sample_records_krr), handle, indent=2)

    source_counts = collect_source_counts()
    clustering_summary = run_clustering(affinity_matrices, surrogate_order, source_counts, output_dir, k=args.clusters)

    with (output_dir / "cluster_evaluation_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(clustering_summary, handle, indent=2)

    print(f"Affinity matrices and clustering summaries saved to {output_dir}")


if __name__ == "__main__":
    main()
