"""CLI to convert raw annotator results into the unified score-matrix format."""

from __future__ import annotations

import argparse
from pathlib import Path
import glob
import numpy as np

from .score_matrix_io import (
    save_score_matrix_npz,
    score_matrix_from_evaluation,
    score_matrix_from_logistic_results,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__ or "prepare_score_matrix")
    parser.add_argument(
        "input",
        nargs="?",
        type=str,
        help="Path to raw annotator results JSON (single-file mode)",
    )
    parser.add_argument(
        "output",
        nargs="?",
        type=str,
        help="Destination .npz path (single-file mode)",
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["logistic", "evaluation"],
        required=True,
        help="Input JSON format",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default=None,
        help="Directory prefix to batch-convert (will expand to prefix*)."
    )
    parser.add_argument(
        "--glob",
        type=str,
        default=None,
        help="Explicit glob pattern overriding --prefix."
    )
    parser.add_argument(
        "--input-name",
        type=str,
        default="annotator_evaluation_results.json",
        help="File name to look for inside each run directory when using batch mode.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="scores",
        help="Directory used for batch outputs or as default combined output location.",
    )
    parser.add_argument(
        "--output-suffix",
        type=str,
        default="_score.npz",
        help="Suffix appended to each run name when saving per-run outputs (requires --per-run).",
    )
    parser.add_argument(
        "--combined-output",
        type=str,
        default=None,
        help="Path for aggregated score matrix in batch mode (defaults to <output-dir>/combined_score_matrix.npz).",
    )
    parser.add_argument(
        "--per-run",
        action="store_true",
        help="Also save individual score matrices for each matched directory.",
    )
    return parser.parse_args()


def _load_payload(input_path: Path, fmt: str):
    if fmt == "logistic":
        return score_matrix_from_logistic_results(input_path)
    return score_matrix_from_evaluation(input_path)


def main() -> None:
    args = parse_args()

    if args.prefix or args.glob:
        if args.input or args.output:
            raise ValueError("Do not supply positional input/output when using --prefix or --glob")
        pattern = args.glob or f"{args.prefix}*"
        matched = sorted(glob.glob(pattern))
        if not matched:
            raise ValueError(f"No directories matched pattern '{pattern}'")

        per_run_dir = Path(args.output_dir)
        if args.per_run:
            per_run_dir.mkdir(parents=True, exist_ok=True)

        combined_output = Path(args.combined_output) if args.combined_output else Path(args.output_dir) / "combined_score_matrix.npz"
        combined_output.parent.mkdir(parents=True, exist_ok=True)

        row_dicts: List[Dict[int, float]] = []
        aggregated_subset_ids: List[int] = []
        aggregated_trained: List[List[int]] = []
        subset_metadata: List[Dict[str, object]] = []
        annotator_union: List[int] = []
        subset_counter = 0

        for run_path_str in matched:
            run_path = Path(run_path_str)
            if not run_path.is_dir():
                continue
            input_path = run_path / args.input_name
            if not input_path.exists():
                print(f"[skip] {input_path} not found")
                continue

            payload = _load_payload(input_path, args.format)
            annotators = [int(a) for a in payload["annotator_ids"]]

            existing = set(annotator_union)
            added = False
            for ann in annotators:
                if ann not in existing:
                    annotator_union.append(ann)
                    existing.add(ann)
                    added = True
            if added:
                annotator_union.sort()

            scores_array = np.asarray(payload["scores"], dtype=np.float32)
            aggregated_trained.extend([list(map(int, lst)) for lst in payload["trained_annotators"]])

            for local_idx, orig_subset in enumerate(payload["subset_ids"]):
                row_dicts.append({
                    annotators[col_idx]: float(scores_array[local_idx, col_idx])
                    for col_idx in range(len(annotators))
                })
                aggregated_subset_ids.append(subset_counter)
                subset_metadata.append(
                    {
                        "run_dir": run_path.name,
                        "original_subset_id": int(orig_subset),
                        "row_index": subset_counter,
                    }
                )
                subset_counter += 1

            if args.per_run:
                output_path = per_run_dir / f"{run_path.name}{args.output_suffix}"
                save_score_matrix_npz(payload, output_path)
                print(f"Saved score matrix to {output_path}")

        if not row_dicts:
            raise ValueError("No valid runs found for aggregation")

        annotator_union.sort()
        annotator_index = {ann: idx for idx, ann in enumerate(annotator_union)}
        combined_scores = np.full((len(annotator_union), len(row_dicts)), np.nan, dtype=np.float32)
        for col, value_dict in enumerate(row_dicts):
            for ann, score in value_dict.items():
                combined_scores[annotator_index[ann], col] = float(score)

        combined_payload = {
            "scores": combined_scores,
            "annotator_ids": annotator_union,
            "subset_ids": aggregated_subset_ids,
            "trained_annotators": aggregated_trained,
            "metadata": {
                "aggregated_runs": matched,
                "subset_mapping": subset_metadata,
                "format": args.format,
                "layout": "annotator_rows",
            },
        }
        save_score_matrix_npz(combined_payload, combined_output)
        print(f"Saved combined score matrix to {combined_output}")
    else:
        if not args.input or not args.output:
            raise ValueError("input and output paths are required in single-file mode")
        payload = _load_payload(Path(args.input), args.format)
        save_score_matrix_npz(payload, Path(args.output))
        print(f"Saved score matrix to {args.output}")


if __name__ == "__main__":
    main()
