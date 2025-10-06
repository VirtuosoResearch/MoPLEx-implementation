"""Utilities for loading and saving annotator score matrices in a unified format."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence

import numpy as np


ScoreMatrix = MutableMapping[str, object]


def get_matrix_layout(payload: Mapping[str, object]) -> str:
    metadata = payload.get("metadata")
    if isinstance(metadata, Mapping):
        layout = metadata.get("layout")
        if isinstance(layout, str):
            return layout
    return "subset_rows"


def _validate_matrix_payload(payload: Mapping[str, object]) -> None:
    required = {"scores", "annotator_ids", "subset_ids", "trained_annotators"}
    missing = required.difference(payload.keys())
    if missing:
        raise ValueError(f"Score matrix payload missing keys: {sorted(missing)}")

    scores = np.asarray(payload["scores"])
    annotators = list(payload["annotator_ids"])
    subsets = list(payload["subset_ids"])
    trained_lists = list(payload["trained_annotators"])

    if scores.ndim != 2:
        raise ValueError("scores must be a 2D matrix")

    layout = get_matrix_layout(payload)
    if layout == "subset_rows":
        if scores.shape[0] != len(subsets):
            raise ValueError("Number of rows in scores must match subset_ids length")
        if scores.shape[1] != len(annotators):
            raise ValueError("Number of columns in scores must match annotator_ids length")
        if len(trained_lists) != len(subsets):
            raise ValueError("trained_annotators length must match subset_ids length")
    elif layout == "annotator_rows":
        if scores.shape[0] != len(annotators):
            raise ValueError("Number of rows in scores must match annotator_ids length for annotator_rows layout")
        if scores.shape[1] != len(subsets):
            raise ValueError("Number of columns in scores must match subset_ids length for annotator_rows layout")
        if len(trained_lists) != len(subsets):
            raise ValueError("trained_annotators length must match subset_ids length for annotator_rows layout")
    else:
        raise ValueError(f"Unknown score matrix layout: {layout}")


def save_score_matrix_npz(payload: Mapping[str, object], path: Path) -> None:
    """Persist a score matrix payload to npz format."""
    _validate_matrix_payload(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        scores=np.asarray(payload["scores"], dtype=np.float32),
        annotator_ids=np.asarray(payload["annotator_ids"], dtype=np.int64),
        subset_ids=np.asarray(payload["subset_ids"], dtype=np.int64),
        trained_annotators=np.asarray(payload["trained_annotators"], dtype=object),
        metadata=json.dumps(payload.get("metadata", {})),
    )


def load_score_matrix_npz(path: Path) -> ScoreMatrix:
    """Load a score matrix payload stored via :func:`save_score_matrix_npz`."""
    with np.load(path, allow_pickle=True) as data:
        payload: ScoreMatrix = {
            "scores": data["scores"],
            "annotator_ids": data["annotator_ids"].tolist(),
            "subset_ids": data["subset_ids"].tolist(),
            "trained_annotators": data["trained_annotators"].tolist(),
        }
        if "metadata" in data:
            payload["metadata"] = json.loads(str(data["metadata"]))
    _validate_matrix_payload(payload)
    return payload


def score_matrix_from_logistic_results(path: Path) -> ScoreMatrix:
    """Create a score matrix payload from logistic_regression JSON output."""
    with path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, dict):
        raise ValueError("Expected logistic results to be a JSON object")

    rows: List[List[float]] = []
    subset_ids: List[int] = []
    trained_lists: List[List[int]] = []
    evaluated_annotators: Optional[List[int]] = None

    for subset_idx, (key, payload) in enumerate(sorted(raw.items())):
        per = payload.get("per_annotator_accuracy")
        eval_ids = payload.get("evaluated_annotators")
        trained = payload.get("trained_annotators")
        if per is None or eval_ids is None:
            raise ValueError(f"Entry '{key}' missing required fields")
        annotators = [int(a) for a in eval_ids]
        per_map = {int(k): float(v) for k, v in per.items()}
        if evaluated_annotators is None:
            evaluated_annotators = annotators
        elif annotators != evaluated_annotators:
            raise ValueError("Inconsistent evaluated_annotators across entries")
        row = [per_map[int(a)] for a in evaluated_annotators]
        rows.append(row)
        subset_ids.append(int(payload.get("subset_id", subset_idx)))
        trained_lists.append([int(x) for x in (trained or eval_ids)])

    if evaluated_annotators is None:
        raise ValueError("No entries found in logistic results")

    scores = np.asarray(rows, dtype=np.float32)
    payload_matrix: ScoreMatrix = {
        "scores": scores,
        "annotator_ids": evaluated_annotators,
        "subset_ids": subset_ids,
        "trained_annotators": trained_lists,
        "metadata": {
            "source": "logistic_regression",
            "path": str(path),
            "layout": "subset_rows",
        },
    }
    _validate_matrix_payload(payload_matrix)
    return payload_matrix


def score_matrix_from_evaluation(path: Path) -> ScoreMatrix:
    """Create a score matrix payload from annotator_evaluation JSON output."""
    with path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, dict):
        raise ValueError("Evaluation results must be a JSON object")

    annotator_results = raw.get("annotator_results")
    if annotator_results is None:
        raise ValueError("annotator_results missing in evaluation JSON")

    subset_entries = raw.get("subset_metrics") or []
    annotator_ids = sorted(int(a) for a in annotator_results.keys())

    if subset_entries:
        rows: List[List[float]] = []
        subset_ids: List[int] = []
        trained_lists: List[List[int]] = []

        for entry in subset_entries:
            subset_id = int(entry.get("subset_id", len(subset_ids)))
            scores = entry.get("per_annotator_scores")
            trained = entry.get("subset_annotators", annotator_ids)
            if scores is None:
                raise ValueError(f"subset_metrics entry {subset_id} missing scores")
            row = [float(scores[str(a)]) for a in annotator_ids]
            rows.append(row)
            subset_ids.append(subset_id)
            trained_lists.append([int(x) for x in trained])

        scores_array = np.asarray(rows, dtype=np.float32)
    else:
        # Some evaluation exports only per-annotator aggregated scores
        scores_array = np.asarray(
            [[float(annotator_results[str(a)].get("accuracy", 0.0)) for a in annotator_ids]],
            dtype=np.float32,
        )
        subset_ids = [0]
        trained_lists = [annotator_ids[:]]

    payload_matrix = {
        "scores": scores_array,
        "annotator_ids": annotator_ids,
        "subset_ids": subset_ids,
        "trained_annotators": trained_lists,
        "metadata": {
            "source": "annotator_evaluation",
            "path": str(path),
            "layout": "subset_rows",
        },
    }

    _validate_matrix_payload(payload_matrix)
    return payload_matrix
