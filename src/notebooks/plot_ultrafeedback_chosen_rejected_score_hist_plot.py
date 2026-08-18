#!/usr/bin/env python
"""Bar plots of chosen/rejected responses measured by one score dimension."""

# %%
from __future__ import annotations
import argparse
from collections import defaultdict
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from datasets import DatasetDict, load_from_disk


class args:
    dataset_dir = "data/ultrafeedback_disagreement"
    output_dir = "notebooks/figures/ultrafeedback_chosen_rejected_score_bars"
    dimension_pair = ["truthfulness", "helpfulness"]
    score_axis = "both"
    splits = ["train", "validation", "test"]
    formats = ["png", "pdf"]

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


def collect_rows_by_source(
    dataset: DatasetDict,
    splits: list[str],
    dimensions: set[str],
) -> dict[tuple[str, str], dict[str, dict[str, object]]]:
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
            if not isinstance(responses, list) or not isinstance(scores, list) or not responses:
                continue
            source_index = row.get("source_index", row_idx)
            grouped[(split, str(source_index))][dimension] = {
                "responses": [str(response) for response in responses],
                "score_by_response": {
                    str(response): float(score)
                    for response, score in zip(responses, scores)
                },
            }
    return grouped


def chosen_rejected_scores(
    grouped_rows: dict[tuple[str, str], dict[str, dict[str, object]]],
    selector_dim: str,
    score_axis: str,
) -> dict[str, list[float]]:
    values = {"chosen": [], "rejected": []}

    for rows_by_dim in grouped_rows.values():
        if selector_dim not in rows_by_dim or score_axis not in rows_by_dim:
            continue
        selector_responses = rows_by_dim[selector_dim]["responses"]
        axis_scores = rows_by_dim[score_axis]["score_by_response"]
        if not isinstance(selector_responses, list) or not selector_responses or not isinstance(axis_scores, dict):
            continue

        chosen_response = selector_responses[0]
        rejected_response = selector_responses[-1]
        if chosen_response in axis_scores:
            values["chosen"].append(float(axis_scores[chosen_response]))
        if rejected_response in axis_scores:
            values["rejected"].append(float(axis_scores[rejected_response]))

    return values

# %%
import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rc

rc("font", **{"family": "sans-serif", "sans-serif": ["Helvetica"]})
mpl.rcParams["savefig.dpi"] = 1200
mpl.rcParams["text.usetex"] = True

def gaussian_kde_1d(samples: np.ndarray, grid: np.ndarray, bandwidth: float) -> np.ndarray:
    if samples.size == 0:
        return np.zeros_like(grid)
    diffs = (grid[:, None] - samples[None, :]) / bandwidth
    kernel = np.exp(-0.5 * diffs ** 2) / (np.sqrt(2 * np.pi) * bandwidth)
    return kernel.mean(axis=1)

def plot_chosen_rejected_bars(
    values: dict[str, list[float]],
    selector_dim: str,
    score_axis: str,
    output_dir: Path,
    formats: list[str],
) -> None:
    if not values["chosen"] and not values["rejected"]:
        print(f"Skipping selector={selector_dim}, x={score_axis}: no matched responses")
        return

    fig, ax = plt.subplots(figsize=(6, 4.5))
    x_grid = np.linspace(1, 5.1, 800)
    bandwidth = 0.5
    chosen_kde = gaussian_kde_1d(np.asarray(values["chosen"]), x_grid, bandwidth)
    rejected_kde = gaussian_kde_1d(np.asarray(values["rejected"]), x_grid, bandwidth)
    ax.fill_between(x_grid, chosen_kde, color="royalblue", alpha=0.5)
    ax.plot(
        x_grid,
        chosen_kde,
        color="royalblue",
        linewidth=2.5,
        label=r"$\mathrm{Top~ranked}$",
    )
    ax.fill_between(x_grid, rejected_kde, color="darkorange", alpha=0.5)
    ax.plot(
        x_grid,
        rejected_kde,
        color="darkorange",
        linewidth=2.5,
        label=r"$\mathrm{Least~ranked}$",
    )

    ax.set_xlim(0.5, 5.5)
    ax.set_xticks([1, 2, 3, 4, 5])
    ax.set_yticks(np.arange(0, 1.1, 0.2))
    
    ax.set_xlabel(r"$\mathrm{Helpfulness~score}$", fontsize=32)
    ax.set_ylabel(r"$\mathrm{Density}$", fontsize=32)
    # set axis font size
    ax.tick_params(axis="both", which="major", labelsize=32)
    ax.title.set_fontsize(32)
    ax.set_title(r"$\mathrm{Rankings~by~helpfulness}$", fontsize=32)
    ax.grid(True, axis="y", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=True, fontsize=28, loc="upper left")
    fig.tight_layout()

    stem = f"chosen_rejected_by_{safe_name(selector_dim)}_x_{safe_name(score_axis)}"
    fig.savefig(output_dir / f"{stem}.pdf", dpi=220, bbox_inches="tight")
    plt.show(fig)


dataset_dir = project_path(args.dataset_dir)
output_dir = project_path(args.output_dir)
output_dir.mkdir(parents=True, exist_ok=True)

dim_a, dim_b = tuple(args.dimension_pair)
score_axes = (dim_a, dim_b) if args.score_axis == "both" else (args.score_axis,)
needed_dimensions = {dim_a, dim_b, *score_axes}

dataset = load_dataset_dict(dataset_dir)
grouped_rows = collect_rows_by_source(dataset, args.splits, needed_dimensions)

print(f"Loaded {sum(len(split) for split in dataset.values()):,} listwise rows")
print(f"Grouped into {len(grouped_rows):,} source prompts")
print(f"Writing figures to {output_dir}")

score_axis = "helpfulness"
selector_dim = "helpfulness"
values = chosen_rejected_scores(grouped_rows, selector_dim, score_axis)
plot_chosen_rejected_bars(values, selector_dim, score_axis, output_dir, args.formats)

# %%
