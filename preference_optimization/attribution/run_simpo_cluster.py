"""Train SIMPO runs per annotator cluster and compute weighted accuracy."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

import numpy as np


def load_clusters(path: Path) -> Dict[str, List[int]]:
    with path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    clusters = raw.get("clusters") or raw.get("cluster_ids") or raw
    if not isinstance(clusters, Mapping):
        raise ValueError(f"Unrecognized cluster format in {path}")

    cluster_map: Dict[str, List[int]] = {}
    for key, value in clusters.items():
        annotators: List[int]
        if isinstance(value, Mapping):
            annotators = [int(v) for v in value.get("annotators", [])]
        else:
            annotators = [int(v) for v in value]
        cluster_map[str(key)] = annotators
    return cluster_map


def extract_accuracy(eval_path: Path) -> float:
    with eval_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    summary = data.get("summary", {})
    stats = summary.get("statistics", {})
    accuracy_stats = stats.get("accuracy")
    if isinstance(accuracy_stats, Mapping) and "mean" in accuracy_stats:
        return float(accuracy_stats["mean"])

    overall = data.get("overall")
    if isinstance(overall, Mapping):
        metrics = overall.get("metrics", {})
        for key in ("eval_accuracy", "accuracy"):
            if key in metrics:
                return float(metrics[key])

    annotator_results = data.get("annotator_results", {})
    if isinstance(annotator_results, Mapping):
        values = []
        for result in annotator_results.values():
            if isinstance(result, Mapping):
                acc = result.get("accuracy")
                if acc is None and "metrics" in result:
                    metrics = result["metrics"]
                    acc = metrics.get("accuracy") or metrics.get("eval_accuracy")
                if acc is not None:
                    values.append(float(acc))
        if values:
            return float(sum(values) / len(values))

    raise ValueError(f"Could not extract accuracy from {eval_path}")


def build_command(
    accelerate_cmd: str,
    run_simpo_path: Path,
    config_path: Path,
    overrides: Sequence[str],
) -> List[str]:
    cmd = shlex.split(accelerate_cmd)
    cmd.append(str(run_simpo_path))
    cmd.append(str(config_path))
    cmd.extend(overrides)
    return cmd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=str, help="Base YAML configuration for run_simpo.py")
    parser.add_argument("clusters", type=str, help="Cluster assignment JSON file")
    parser.add_argument(
        "--output-root",
        type=str,
        default="outputs/cluster_runs",
        help="Directory where per-cluster outputs are stored",
    )
    parser.add_argument(
        "--run-simpo",
        type=str,
        default=None,
        help="Path to run_simpo.py (defaults to repository version)",
    )
    parser.add_argument(
        "--accelerate-cmd",
        type=str,
        default="accelerate launch",
        help="Accelerate command prefix",
    )
    parser.add_argument(
        "--overrides",
        nargs="*",
        default=None,
        help="Additional key=value overrides passed to run_simpo",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="cluster_",
        help="Prefix used for cluster-specific output directories",
    )
    parser.add_argument(
        "--run-name-prefix",
        type=str,
        default="cluster-run-",
        help="Prefix for the run_name override",
    )
    parser.add_argument(
        "--weight-mode",
        type=str,
        choices=["annotators"],
        default="annotators",
        help="Weighting scheme for overall accuracy",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip training if annotator_evaluation_results.json already exists",
    )
    parser.add_argument(
        "--workdir",
        type=str,
        default=".",
        help="Working directory used for subprocess calls",
    )
    parser.add_argument(
        "--summary-output",
        type=str,
        default=None,
        help="Optional path to save JSON summary with per-cluster metrics",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config_path = Path(args.config)
    cluster_path = Path(args.clusters)
    run_simpo_path = Path(args.run_simpo) if args.run_simpo else Path(__file__).resolve().parents[1] / "run_simpo.py"
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    clusters = load_clusters(cluster_path)
    overrides = args.overrides or []
    weighted_sum = 0.0
    total_weight = 0.0
    cluster_metrics: Dict[str, Dict[str, float]] = {}

    for cluster_id, annotators in clusters.items():
        annotators = [int(a) for a in annotators]
        if not annotators:
            continue

        annotator_str = ",".join(str(a) for a in annotators)
        output_dir = output_root / f"{args.output_prefix}{cluster_id}"
        run_name = f"{args.run_name_prefix}{cluster_id}"
        eval_path = output_dir / "annotator_evaluation_results.json"

        if not (args.skip_existing and eval_path.exists()):
            output_dir.mkdir(parents=True, exist_ok=True)
            cluster_overrides = list(overrides)
            cluster_overrides.extend(
                [
                    f'output_dir="{output_dir}"',
                    f'run_name="{run_name}"',
                    f'annotator_ids="{annotator_str}"',
                ]
            )
            cmd = build_command(args.accelerate_cmd, run_simpo_path, config_path, cluster_overrides)
            print("Running:", " ".join(cmd))
            subprocess.run(cmd, cwd=args.workdir, check=True)

        if not eval_path.exists():
            raise FileNotFoundError(f"Expected evaluation results at {eval_path}")

        accuracy = float(extract_accuracy(eval_path))
        weight = float(len(annotators)) if args.weight_mode == "annotators" else 1.0
        cluster_metrics[cluster_id] = {
            "annotators": annotators,
            "num_annotators": len(annotators),
            "accuracy": accuracy,
            "weight": weight,
            "output_dir": str(output_dir),
        }
        if not np.isnan(accuracy):
            weighted_sum += accuracy * weight
            total_weight += weight

    overall = weighted_sum / total_weight if total_weight > 0 else float("nan")
    cluster_metrics["overall"] = {
        "weighted_accuracy": overall,
        "total_weight": total_weight,
    }

    if args.summary_output:
        summary_path = Path(args.summary_output)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with summary_path.open("w", encoding="utf-8") as handle:
            json.dump(cluster_metrics, handle, indent=2)
        print(f"Saved summary to {summary_path}")
    else:
        print(json.dumps(cluster_metrics, indent=2))


if __name__ == "__main__":
    main()
