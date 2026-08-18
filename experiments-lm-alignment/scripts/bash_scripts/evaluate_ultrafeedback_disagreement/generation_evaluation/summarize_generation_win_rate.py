#!/usr/bin/env python
"""Resolve checkpoints and summarize generation win-rate outputs."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any


def _checkpoint_step(path: Path) -> int:
    match = re.fullmatch(r"checkpoint-(\d+)", path.name)
    return int(match.group(1)) if match else -1


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object.")
    return payload


def resolve_best_checkpoint(path: Path) -> Path:
    """Return an adapter directory, preferring trainer_state.best_model_checkpoint."""
    path = path.expanduser()
    if (path / "adapter_config.json").exists():
        return path
    if not path.exists():
        raise FileNotFoundError(path)

    state_files = []
    if (path / "trainer_state.json").exists():
        state_files.append(path / "trainer_state.json")
    state_files.extend(sorted(path.glob("checkpoint-*/trainer_state.json"), key=lambda p: _checkpoint_step(p.parent)))

    for state_file in reversed(state_files):
        state = _read_json(state_file)
        best = state.get("best_model_checkpoint")
        if not best:
            continue
        best_path = Path(str(best)).expanduser()
        if not best_path.is_absolute():
            best_path = Path.cwd() / best_path
        if (best_path / "adapter_config.json").exists():
            return best_path

    checkpoint_dirs = sorted(
        [candidate for candidate in path.glob("checkpoint-*") if (candidate / "adapter_config.json").exists()],
        key=_checkpoint_step,
    )
    if checkpoint_dirs:
        return checkpoint_dirs[-1]

    raise FileNotFoundError(f"Could not resolve an adapter checkpoint under {path}")


def _tag_metadata(model_tag: str) -> dict[str, Any]:
    match = re.fullmatch(r"(mixture_mp[24])_cluster_(\d+)", model_tag)
    if match:
        return {"group": match.group(1), "cluster": int(match.group(2))}
    if model_tag == "dpo":
        return {"group": "dpo", "cluster": None}
    if model_tag == "listdpo":
        return {"group": "listdpo", "cluster": None}
    return {"group": model_tag, "cluster": None}


def _metric_row(
    *,
    model_tag: str,
    dimension: str,
    metrics: dict[str, Any],
    metrics_path: Path,
    row_type: str = "run",
) -> dict[str, Any]:
    meta = _tag_metadata(model_tag)
    return {
        "row_type": row_type,
        "model_tag": model_tag,
        "group": meta["group"],
        "cluster": "" if meta["cluster"] is None else meta["cluster"],
        "dimension": dimension,
        "num_examples": metrics.get("num_examples", 0.0),
        "win_rate": metrics.get("win_rate", 0.0),
        "loss_rate": metrics.get("loss_rate", 0.0),
        "tie_rate": metrics.get("tie_rate", 0.0),
        "mean_score_margin": metrics.get("mean_score_margin", 0.0),
        "metrics_path": str(metrics_path),
    }


def summarize_outputs(output_root: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    runs: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []

    for metrics_path in sorted(output_root.glob("*/metrics.json")):
        model_tag = metrics_path.parent.name
        metrics = _read_json(metrics_path)
        by_dimension = metrics.get("by_dimension", {})
        if not isinstance(by_dimension, dict):
            by_dimension = {}

        runs[model_tag] = {
            "metrics_path": str(metrics_path),
            "overall": {
                key: metrics.get(key)
                for key in ("num_examples", "win_rate", "loss_rate", "tie_rate", "mean_score_margin")
            },
            "by_dimension": by_dimension,
        }
        rows.append(_metric_row(model_tag=model_tag, dimension="overall", metrics=metrics, metrics_path=metrics_path))
        for dimension, dim_metrics in sorted(by_dimension.items()):
            if isinstance(dim_metrics, dict):
                rows.append(
                    _metric_row(
                        model_tag=model_tag,
                        dimension=dimension,
                        metrics=dim_metrics,
                        metrics_path=metrics_path,
                    )
                )

    best_mixture_by_dimension: dict[str, dict[str, Any]] = {}
    for row in rows:
        if row["row_type"] != "run" or row["dimension"] == "overall":
            continue
        if not str(row["group"]).startswith("mixture_"):
            continue
        key = f"{row['group']}::{row['dimension']}"
        current = best_mixture_by_dimension.get(key)
        if current is None or float(row["win_rate"]) > float(current["win_rate"]):
            best_mixture_by_dimension[key] = row

    best_rows = []
    for key, row in sorted(best_mixture_by_dimension.items()):
        group, dimension = key.split("::", 1)
        best_row = dict(row)
        best_row["row_type"] = "best_mixture_dimension"
        best_row["model_tag"] = f"{group}_best"
        best_row["dimension"] = dimension
        best_rows.append(best_row)

    summary = {
        "output_root": str(output_root),
        "runs": runs,
        "best_mixture_by_dimension": best_rows,
    }
    return summary, rows + best_rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "row_type",
        "model_tag",
        "group",
        "cluster",
        "dimension",
        "num_examples",
        "win_rate",
        "loss_rate",
        "tie_rate",
        "mean_score_margin",
        "metrics_path",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    resolve_parser = subparsers.add_parser("resolve-checkpoint")
    resolve_parser.add_argument("path")

    summarize_parser = subparsers.add_parser("summarize")
    summarize_parser.add_argument("--output_root", required=True)
    summarize_parser.add_argument("--output_json", required=True)
    summarize_parser.add_argument("--output_csv", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "resolve-checkpoint":
        print(resolve_best_checkpoint(Path(args.path)))
        return

    output_root = Path(args.output_root)
    summary, rows = summarize_outputs(output_root)
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    write_csv(Path(args.output_csv), rows)
    print(f"Wrote {output_json} and {args.output_csv}")


if __name__ == "__main__":
    main()
