import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
from datasets import Dataset


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "evaluate_generation_win_rate.py"
spec = importlib.util.spec_from_file_location("evaluate_generation_win_rate", SCRIPT_PATH)
eval_script = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = eval_script
spec.loader.exec_module(eval_script)


def test_balanced_sampling_round_robins_by_preference_dimension():
    dataset = Dataset.from_list(
        [
            {"prompt": "a0", "preference_dimension": "a"},
            {"prompt": "a1", "preference_dimension": "a"},
            {"prompt": "a2", "preference_dimension": "a"},
            {"prompt": "b0", "preference_dimension": "b"},
            {"prompt": "b1", "preference_dimension": "b"},
            {"prompt": "c0", "preference_dimension": "c"},
        ]
    )

    selected = eval_script.select_eval_examples(
        dataset,
        max_examples=5,
        sample_strategy="balanced_by_dimension",
        seed=123,
    )

    assert [row["prompt"] for row in selected] == ["a0", "b0", "c0", "a1", "b1"]
    assert [row["preference_dimension"] for row in selected].count("a") == 2
    assert [row["preference_dimension"] for row in selected].count("b") == 2
    assert [row["preference_dimension"] for row in selected].count("c") == 1


def test_reward_adapter_map_parses_json_file_and_repeated_args(tmp_path):
    map_path = tmp_path / "reward_map.json"
    map_path.write_text(json.dumps({"persona_0000": "rm0"}), encoding="utf-8")

    parsed = eval_script.parse_reward_adapter_map(
        reward_adapter_map_json=str(map_path),
        reward_adapters=["persona_0001=rm1"],
    )

    assert parsed == {"persona_0000": "rm0", "persona_0001": "rm1"}


def test_reward_adapter_map_rejects_missing_dimensions():
    with pytest.raises(ValueError, match="Missing reward adapter"):
        eval_script.validate_reward_adapter_map(
            {"persona_0000", "persona_0001"},
            {"persona_0000": "rm0"},
        )


def test_reward_adapter_map_rejects_duplicate_dimensions(tmp_path):
    map_path = tmp_path / "reward_map.json"
    map_path.write_text(json.dumps({"persona_0000": "rm0"}), encoding="utf-8")

    with pytest.raises(ValueError, match="Duplicate reward adapter"):
        eval_script.parse_reward_adapter_map(
            reward_adapter_map_json=str(map_path),
            reward_adapters=["persona_0000=rm0b"],
        )


def test_select_top_ranked_response_uses_highest_score():
    selected = eval_script.select_top_ranked_response(
        {
            "responses": ["low", "best", "middle"],
            "scores": [1.0, 4.0, 3.0],
        }
    )

    assert selected == {"response": "best", "index": 1, "score": 4.0}


def test_select_top_ranked_response_uses_earliest_max_score_on_tie():
    selected = eval_script.select_top_ranked_response(
        {
            "responses": ["first best", "also best", "low"],
            "scores": [4.0, 4.0, 1.0],
        }
    )

    assert selected == {"response": "first best", "index": 0, "score": 4.0}


@pytest.mark.parametrize(
    "row,match",
    [
        ({}, "list-valued"),
        ({"responses": [], "scores": []}, "non-empty"),
        ({"responses": ["a"], "scores": [1.0, 2.0]}, "same length"),
        ({"responses": ["a"], "scores": ["bad"]}, "Invalid score"),
    ],
)
def test_select_top_ranked_response_rejects_malformed_rows(row, match):
    with pytest.raises(ValueError, match=match):
        eval_script.select_top_ranked_response(row)


def test_aggregate_metrics_sets_winners_and_per_dimension_stats():
    records = [
        {
            "preference_dimension": "persona_0000",
            "reference_reward_score": 0.0,
            "policy_reward_score": 1.0,
        },
        {
            "preference_dimension": "persona_0000",
            "reference_reward_score": 2.0,
            "policy_reward_score": 1.0,
        },
        {
            "preference_dimension": "persona_0001",
            "reference_reward_score": 3.0,
            "policy_reward_score": 3.0,
        },
    ]

    metrics = eval_script.aggregate_metrics(records, tie_epsilon=0.0)

    assert metrics["num_examples"] == 3.0
    assert metrics["win_rate"] == pytest.approx(1 / 3)
    assert metrics["loss_rate"] == pytest.approx(1 / 3)
    assert metrics["tie_rate"] == pytest.approx(1 / 3)
    assert metrics["mean_score_margin"] == pytest.approx(0.0)
    assert [row["winner"] for row in records] == ["policy", "reference", "tie"]
    assert metrics["by_dimension"]["persona_0000"]["num_examples"] == 2.0
    assert metrics["by_dimension"]["persona_0000"]["win_rate"] == pytest.approx(0.5)
    assert metrics["by_dimension"]["persona_0001"]["tie_rate"] == pytest.approx(1.0)


def test_aggregate_metrics_keeps_base_winner_label_for_base_generation():
    records = [
        {
            "preference_dimension": "persona_0000",
            "reference_response_source": "base_generation",
            "reference_reward_score": 1.0,
            "policy_reward_score": 0.0,
        }
    ]

    eval_script.aggregate_metrics(records, tie_epsilon=0.0)

    assert records[0]["winner"] == "base"


def test_score_records_writes_reference_scores_and_aliases(monkeypatch):
    scorer = eval_script.DPOUtilityRewardScorer.__new__(eval_script.DPOUtilityRewardScorer)
    scorer.args = SimpleNamespace(batch_size=2, disable_tqdm=True)

    def fake_score_dimension(dimension, pairs):
        assert dimension == "persona_0000"
        assert pairs == [("p0", "ref0"), ("p0", "pol0"), ("p1", "ref1"), ("p1", "pol1")]
        return [0.1, 0.5, 0.8, 0.4]

    monkeypatch.setattr(scorer, "score_dimension", fake_score_dimension)
    records = [
        {
            "prompt": "p0",
            "preference_dimension": "persona_0000",
            "reference_response_source": "base_generation",
            "reference_response": "ref0",
            "policy_response": "pol0",
        },
        {
            "prompt": "p1",
            "preference_dimension": "persona_0000",
            "reference_response_source": "dataset_top_ranked",
            "reference_response": "ref1",
            "policy_response": "pol1",
        },
    ]

    scorer.score_records(records)

    assert records[0]["reference_reward_score"] == pytest.approx(0.1)
    assert records[0]["base_reward_score"] == pytest.approx(0.1)
    assert records[0]["policy_reward_score"] == pytest.approx(0.5)
    assert records[1]["reference_reward_score"] == pytest.approx(0.8)
    assert records[1]["top_ranked_reward_score"] == pytest.approx(0.8)
    assert records[1]["policy_reward_score"] == pytest.approx(0.4)


def test_policy_mixture_path_defaults_to_persona_cluster_layout():
    path = eval_script.resolve_policy_adapter_path(
        "outputs/persona/run",
        "dimension_mapped_mixture",
        "persona_0003",
    )

    assert path.endswith("outputs/persona/run/mixture_cluster_3")


def test_parse_args_smoke_does_not_load_models():
    args = eval_script.parse_args(
        [
            "--dataset_name",
            "data/persona",
            "--base_model_name_or_path",
            "Qwen/Qwen3-0.6B",
            "--policy_adapter_path",
            "outputs/policy",
            "--reward_adapter",
            "persona_0000=outputs/rm0",
            "--output_dir",
            "outputs/eval",
        ]
    )

    assert args.split == "validation"
    assert args.sample_strategy == "balanced_by_dimension"
    assert args.reference_response_source == "base_generation"
    assert args.reward_adapter == ["persona_0000=outputs/rm0"]


def test_parse_args_accepts_dataset_top_ranked_reference_source():
    args = eval_script.parse_args(
        [
            "--dataset_name",
            "data/persona",
            "--base_model_name_or_path",
            "Qwen/Qwen3-0.6B",
            "--policy_adapter_path",
            "outputs/policy",
            "--reference_response_source",
            "dataset_top_ranked",
            "--reward_adapter",
            "persona_0000=outputs/rm0",
            "--output_dir",
            "outputs/eval",
        ]
    )

    assert args.reference_response_source == "dataset_top_ranked"
