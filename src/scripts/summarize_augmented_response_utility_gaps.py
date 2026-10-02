#!/usr/bin/env python
"""Summarize utility gaps among generated responses from augmented ranking evals."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        action="append",
        required=True,
        metavar="TEMPERATURE=CSV",
        help="Temperature label and response_scores.csv path. Repeat for each temperature.",
    )
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--tolerance", type=float, default=1e-12)
    return parser.parse_args()


def parse_inputs(values: list[str]) -> list[tuple[str, Path]]:
    parsed = []
    for value in values:
        if "=" not in value:
            raise ValueError(f"Invalid --input {value!r}; expected TEMPERATURE=CSV.")
        temperature, path = value.split("=", 1)
        temperature = temperature.strip()
        path = Path(path.strip()).expanduser()
        if not temperature:
            raise ValueError(f"Invalid --input {value!r}; temperature is empty.")
        if not path.is_file():
            raise FileNotFoundError(f"Missing response score CSV for temperature {temperature}: {path}")
        parsed.append((temperature, path))
    return parsed


def read_augmented_rows(inputs: list[tuple[str, Path]]) -> dict[tuple[str, str, str], list[dict[str, Any]]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for temperature, path in inputs:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                if row.get("response_type") != "augmented":
                    continue
                criterion = str(row.get("criterion", ""))
                source_index = str(row.get("source_index") or row.get("dataset_row_index") or "")
                if not criterion or not source_index:
                    raise ValueError(f"Missing criterion/source_index in {path}: {row}")
                try:
                    utility = float(row["pl_utility"])
                except (KeyError, ValueError) as exc:
                    raise ValueError(f"Invalid pl_utility in {path}: {row}") from exc
                grouped[(temperature, criterion, source_index)].append(
                    {
                        "candidate_label": row.get("candidate_label", ""),
                        "dataset_row_index": row.get("dataset_row_index", ""),
                        "utility": utility,
                    }
                )
    return grouped


def pairwise_gaps(values: list[float]) -> list[float]:
    gaps = []
    for left_idx, left in enumerate(values):
        for right in values[left_idx + 1 :]:
            gaps.append(abs(left - right))
    return gaps


def population_variance(values: list[float]) -> float:
    if not values:
        return float("nan")
    mean = sum(values) / len(values)
    return sum((value - mean) ** 2 for value in values) / len(values)


def coefficient_of_variation(values: list[float], *, mean_tolerance: float) -> float:
    if not values:
        return float("nan")
    mean = sum(values) / len(values)
    if abs(mean) <= mean_tolerance:
        return float("nan")
    return math.sqrt(population_variance(values)) / abs(mean)


def parse_utility_list(value: Any) -> list[float]:
    if value is None:
        return []
    text = str(value)
    if not text:
        return []
    return [float(item) for item in text.split("|")]


def build_gap_rows(
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]],
    *,
    tolerance: float,
) -> list[dict[str, Any]]:
    records = []
    for (temperature, criterion, source_index), rows in sorted(grouped.items()):
        rows = sorted(rows, key=lambda row: str(row["candidate_label"]))
        utilities = [float(row["utility"]) for row in rows]
        gaps = pairwise_gaps(utilities)
        min_gap = min(gaps) if gaps else float("nan")
        max_utility = max(utilities) if utilities else float("nan")
        min_utility = min(utilities) if utilities else float("nan")
        utility_mean = sum(utilities) / len(utilities) if utilities else float("nan")
        utility_variance = population_variance(utilities)
        utility_std = math.sqrt(utility_variance) if math.isfinite(utility_variance) else float("nan")
        records.append(
            {
                "temperature": temperature,
                "criterion": criterion,
                "source_index": source_index,
                "dataset_row_index": rows[0]["dataset_row_index"] if rows else "",
                "num_generated_responses": len(rows),
                "candidate_labels": "|".join(str(row["candidate_label"]) for row in rows),
                "generated_utilities": "|".join(f"{value:.17g}" for value in utilities),
                "all_generated_utilities_same": (max_utility - min_utility) <= tolerance,
                "any_generated_utility_tie": any(gap <= tolerance for gap in gaps),
                "min_pairwise_utility_gap": min_gap,
                "generated_utility_mean": utility_mean,
                "generated_utility_std": utility_std,
                "generated_utility_variance": utility_variance,
                "generated_utility_coefficient_of_variation": coefficient_of_variation(
                    utilities,
                    mean_tolerance=tolerance,
                ),
            }
        )
    return records


def _finite(values: list[float]) -> list[float]:
    return [value for value in values if math.isfinite(value)]


def summarize_group(records: list[dict[str, Any]], *, tolerance: float) -> dict[str, Any]:
    count = len(records)
    min_gaps = _finite([float(row["min_pairwise_utility_gap"]) for row in records])
    means = _finite([float(row["generated_utility_mean"]) for row in records])
    stds = _finite([float(row["generated_utility_std"]) for row in records])
    variances = _finite([float(row["generated_utility_variance"]) for row in records])
    cvs = _finite([float(row["generated_utility_coefficient_of_variation"]) for row in records])
    pooled_utilities = [
        utility
        for row in records
        for utility in parse_utility_list(row.get("generated_utilities"))
    ]
    pooled_mean = sum(pooled_utilities) / len(pooled_utilities) if pooled_utilities else float("nan")
    pooled_variance = population_variance(pooled_utilities)
    pooled_std = math.sqrt(pooled_variance) if math.isfinite(pooled_variance) else float("nan")
    pooled_cv = coefficient_of_variation(pooled_utilities, mean_tolerance=tolerance)
    same_count = sum(str(row["all_generated_utilities_same"]) == "True" for row in records)
    tie_count = sum(str(row["any_generated_utility_tie"]) == "True" for row in records)
    return {
        "num_prompt_criterion_rows": count,
        "num_pooled_generated_utilities": len(pooled_utilities),
        "num_all_generated_utilities_same": same_count,
        "all_generated_utilities_same_rate": same_count / count if count else 0.0,
        "num_any_generated_utility_tie": tie_count,
        "any_generated_utility_tie_rate": tie_count / count if count else 0.0,
        "min_pairwise_utility_gap_min": min(min_gaps) if min_gaps else None,
        "min_pairwise_utility_gap_mean": sum(min_gaps) / len(min_gaps) if min_gaps else None,
        "min_pairwise_utility_gap_max": max(min_gaps) if min_gaps else None,
        "generated_utility_mean_mean": sum(means) / len(means) if means else None,
        "generated_utility_std_mean": sum(stds) / len(stds) if stds else None,
        "generated_utility_variance_mean": sum(variances) / len(variances) if variances else None,
        "generated_utility_variance_max": max(variances) if variances else None,
        "num_generated_utility_coefficient_of_variation_defined": len(cvs),
        "num_generated_utility_coefficient_of_variation_undefined": count - len(cvs),
        "generated_utility_coefficient_of_variation_mean": sum(cvs) / len(cvs) if cvs else None,
        "generated_utility_coefficient_of_variation_min": min(cvs) if cvs else None,
        "generated_utility_coefficient_of_variation_max": max(cvs) if cvs else None,
        "pooled_generated_utility_mean": pooled_mean if math.isfinite(pooled_mean) else None,
        "pooled_generated_utility_std": pooled_std if math.isfinite(pooled_std) else None,
        "pooled_generated_utility_variance": pooled_variance if math.isfinite(pooled_variance) else None,
        "pooled_generated_utility_coefficient_of_variation": (
            pooled_cv if math.isfinite(pooled_cv) else None
        ),
    }


def build_summary_rows(records: list[dict[str, Any]], *, tolerance: float) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[(str(record["temperature"]), str(record["criterion"]))].append(record)

    summary_rows = []
    for (temperature, criterion), rows in sorted(grouped.items()):
        summary_rows.append(
            {
                "temperature": temperature,
                "criterion": criterion,
                **summarize_group(rows, tolerance=tolerance),
            }
        )
    return summary_rows


def write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    if not records:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)


def main() -> None:
    args = parse_args()
    if args.tolerance < 0:
        raise ValueError("--tolerance must be non-negative.")
    inputs = parse_inputs(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    gap_rows = build_gap_rows(read_augmented_rows(inputs), tolerance=args.tolerance)
    summary_rows = build_summary_rows(gap_rows, tolerance=args.tolerance)
    write_csv(output_dir / "generated_utility_gap_rows.csv", gap_rows)
    write_csv(output_dir / "generated_utility_gap_summary.csv", summary_rows)

    by_temperature: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in gap_rows:
        by_temperature[str(record["temperature"])].append(record)
    payload = {
        "metadata": {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "tolerance": args.tolerance,
            "inputs": [{"temperature": temperature, "path": str(path)} for temperature, path in inputs],
        },
        "overall": summarize_group(gap_rows, tolerance=args.tolerance),
        "per_temperature": {
            temperature: summarize_group(rows, tolerance=args.tolerance)
            for temperature, rows in sorted(by_temperature.items())
        },
        "per_temperature_criterion": summary_rows,
    }
    with (output_dir / "generated_utility_gap_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


if __name__ == "__main__":
    main()
