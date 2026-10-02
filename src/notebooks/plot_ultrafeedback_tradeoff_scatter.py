#!/usr/bin/env python
"""Bubble scatter plots for pairwise UltraFeedback score trade-offs."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datasets import DatasetDict, load_from_disk


DEFAULT_DIMENSIONS = (
    "instruction_following",
    "honesty",
    "truthfulness",
    "helpfulness",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", default="data/ultrafeedback_disagreement")
    parser.add_argument("--output_dir", default="notebooks/figures/ultrafeedback_tradeoffs")
    parser.add_argument("--dimensions", nargs="+", default=list(DEFAULT_DIMENSIONS))
    parser.add_argument("--splits", nargs="+", default=["train", "validation", "test"])
    parser.add_argument("--max_marker_size", type=float, default=2600.0)
    parser.add_argument("--min_marker_size", type=float, default=80.0)
    parser.add_argument("--formats", nargs="+", default=["png", "pdf"])
    return parser.parse_args()


def project_path(path: str) -> Path:
    path_obj = Path(path)
    if path_obj.is_absolute():
        return path_obj
    return Path(__file__).resolve().parents[1] / path_obj


def pretty_name(dimension: str) -> str:
    return dimension.replace("_", " ").title()


def safe_name(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in text).strip("_")


def load_dataset_dict(dataset_dir: Path) -> DatasetDict:
    dataset = load_from_disk(str(dataset_dir))
    if isinstance(dataset, DatasetDict):
        return dataset
    return DatasetDict({"train": dataset})


def collect_scores_by_source(
    dataset: DatasetDict,
    splits: list[str],
    dimensions: set[str],
) -> dict[tuple[str, str], dict[str, dict[str, float]]]:
    grouped = defaultdict(dict)
    for split in splits:
        if split not in dataset:
            continue
        for row_idx, row in enumerate(dataset[split]):
            dimension = row.get("preference_dimension")
            if dimension not in dimensions:
                continue
            responses = row.get("responses")
            scores = row.get("scores")
            if not isinstance(responses, list) or not isinstance(scores, list):
                continue
            source_index = row.get("source_index", row_idx)
            grouped[(split, str(source_index))][dimension] = {
                str(response): float(score)
                for response, score in zip(responses, scores)
            }
    return grouped


def score_pair_counts(
    grouped_scores: dict[tuple[str, str], dict[str, dict[str, float]]],
    dim_x: str,
    dim_y: str,
) -> Counter[tuple[float, float]]:
    counts: Counter[tuple[float, float]] = Counter()
    for score_maps in grouped_scores.values():
        if dim_x not in score_maps or dim_y not in score_maps:
            continue
        x_scores = score_maps[dim_x]
        y_scores = score_maps[dim_y]
        for response in x_scores.keys() & y_scores.keys():
            counts[(x_scores[response], y_scores[response])] += 1
    return counts


def marker_sizes(counts: list[int], min_size: float, max_size: float) -> list[float]:
    max_count = max(counts) if counts else 1
    return [min_size + (max_size - min_size) * (count / max_count) for count in counts]


def plot_pair(
    counts: Counter[tuple[float, float]],
    dim_x: str,
    dim_y: str,
    output_dir: Path,
    formats: list[str],
    min_marker_size: float,
    max_marker_size: float,
) -> None:
    if not counts:
        print(f"Skipping {dim_x} vs {dim_y}: no matched scores")
        return

    items = sorted(counts.items())
    xs = [score_pair[0] for score_pair, _ in items]
    ys = [score_pair[1] for score_pair, _ in items]
    ns = [count for _, count in items]
    sizes = marker_sizes(ns, min_marker_size, max_marker_size)

    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    scatter = ax.scatter(
        xs,
        ys,
        s=sizes,
        c=ns,
        cmap="viridis",
        alpha=0.78,
        edgecolors="#111827",
        linewidths=0.6,
    )

    for x, y, count in zip(xs, ys, ns):
        ax.text(x, y, str(count), ha="center", va="center", fontsize=8, color="white", weight="bold")

    ax.set_xlim(0.5, 5.5)
    ax.set_ylim(0.5, 5.5)
    ax.set_xticks([1, 2, 3, 4, 5])
    ax.set_yticks([1, 2, 3, 4, 5])
    ax.grid(True, color="#e5e7eb", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_xlabel(pretty_name(dim_x))
    ax.set_ylabel(pretty_name(dim_y))
    ax.set_title(f"{pretty_name(dim_x)} vs {pretty_name(dim_y)}")
    cbar = fig.colorbar(scatter, ax=ax, pad=0.02)
    cbar.set_label("Number of candidate responses")
    fig.tight_layout()

    stem = f"score_counts_{safe_name(dim_x)}_vs_{safe_name(dim_y)}"
    for fmt in formats:
        fig.savefig(output_dir / f"{stem}.{fmt}", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    dataset_dir = project_path(args.dataset_dir)
    output_dir = project_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset_dict(dataset_dir)
    dimensions = tuple(args.dimensions)
    grouped_scores = collect_scores_by_source(dataset, args.splits, set(dimensions))

    print(f"Loaded {sum(len(split) for split in dataset.values()):,} listwise rows")
    print(f"Grouped into {len(grouped_scores):,} source prompts")
    print(f"Writing figures to {output_dir}")

    for dim_x, dim_y in combinations(dimensions, 2):
        counts = score_pair_counts(grouped_scores, dim_x, dim_y)
        print(f"{dim_x} vs {dim_y}: {sum(counts.values()):,} responses, {len(counts):,} score pairs")
        plot_pair(
            counts,
            dim_x,
            dim_y,
            output_dir,
            args.formats,
            args.min_marker_size,
            args.max_marker_size,
        )


if __name__ == "__main__":
    main()
