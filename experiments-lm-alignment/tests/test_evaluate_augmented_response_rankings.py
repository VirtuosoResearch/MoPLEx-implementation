import csv
import importlib.util
import json
from pathlib import Path
import sys

import pytest
from datasets import Dataset


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "evaluate_augmented_response_rankings.py"
spec = importlib.util.spec_from_file_location("evaluate_augmented_response_rankings", SCRIPT_PATH)
eval_script = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = eval_script
spec.loader.exec_module(eval_script)


def make_row(**updates):
    row = {
        "prompt": "prompt",
        "responses": ["O1", "O2", "O3", "O4", "A1", "A2", "A3", "A4"],
        "scores": [4.0, 3.0, 2.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        "preference_dimension": "helpfulness",
        "source_index": 7,
        "ranked_prefix_length": 4,
        "num_generated_responses": 4,
    }
    row.update(updates)
    return row


def scored_candidates(utilities):
    candidates = eval_script.label_candidates(make_row())
    for candidate, utility in zip(candidates, utilities):
        candidate["pl_utility"] = utility
    return candidates


def test_label_candidates_uses_augmented_metadata_and_dataset_positions():
    candidates = eval_script.label_candidates(make_row())

    assert [item["candidate_label"] for item in candidates] == [
        "original_1",
        "original_2",
        "original_3",
        "original_4",
        "augmented_1",
        "augmented_2",
        "augmented_3",
        "augmented_4",
    ]
    assert [item["dataset_position"] for item in candidates] == list(range(8))
    assert [item["response_type"] for item in candidates] == ["original"] * 4 + ["augmented"] * 4


@pytest.mark.parametrize(
    "updates, message",
    [
        ({"ranked_prefix_length": None}, "must be integers"),
        ({"num_generated_responses": 3}, "Unexpected original/augmented"),
        ({"responses": ["too short"]}, "Response count"),
    ],
)
def test_label_candidates_rejects_invalid_metadata(updates, message):
    with pytest.raises((TypeError, ValueError), match=message):
        eval_script.label_candidates(make_row(**updates))


def test_label_candidates_requires_augmentation_metadata():
    row = make_row()
    del row["num_generated_responses"]
    with pytest.raises(ValueError, match="ranked_prefix_length.*num_generated_responses"):
        eval_script.label_candidates(row)


def test_rank_candidates_descends_and_uses_dataset_order_for_ties():
    candidates = scored_candidates([8.0, 7.0, 6.0, 5.0, 7.0, 4.0, 3.0, 2.0])
    ranked, flags = eval_script.rank_candidates(candidates)

    assert [item["candidate_label"] for item in ranked] == [
        "original_1",
        "original_2",
        "augmented_1",
        "original_3",
        "original_4",
        "augmented_2",
        "augmented_3",
        "augmented_4",
    ]
    assert flags["ranking"] == (
        "original_1 > original_2 > augmented_1 > original_3 > original_4 > "
        "augmented_2 > augmented_3 > augmented_4"
    )
    assert flags["top_augmented_rank"] == 3
    assert flags["num_original_augmented_ties"] == 1
    assert not flags["all_augmented_weakly_bottom"]


def test_rank_candidates_distinguishes_strict_weak_and_boundary_tie():
    _, strict = eval_script.rank_candidates(scored_candidates([8, 7, 6, 5, 4, 3, 2, 1]))
    _, tied = eval_script.rank_candidates(scored_candidates([8, 7, 6, 5, 5, 3, 2, 1]))

    assert strict["num_augmented_lower_pairs"] == 16
    assert strict["num_augmented_below_all_originals"] == 4
    assert strict["all_augmented_strictly_bottom"]
    assert strict["all_augmented_weakly_bottom"]
    assert not strict["boundary_tie"]
    assert strict["top_augmented_rank"] == 5

    assert not tied["all_augmented_strictly_bottom"]
    assert tied["all_augmented_weakly_bottom"]
    assert tied["boundary_tie"]
    assert tied["top_augmented_rank"] == 5


def test_summarize_rankings_computes_pairwise_bottom_and_rank_statistics():
    records = []
    for utilities in ([8, 7, 6, 5, 4, 3, 2, 1], [8, 7, 6, 5, 9, 4, 3, 2]):
        _, flags = eval_script.rank_candidates(scored_candidates(utilities))
        records.append(flags)

    summary = eval_script.summarize_rankings(records)

    assert summary["num_prompt_criterion_rows"] == 2
    assert summary["num_responses"] == 16
    assert summary["num_original_augmented_pairs"] == 32
    assert summary["num_augmented_lower_pairs"] == 28
    assert summary["augmented_below_original_pairwise_rate"] == pytest.approx(28 / 32)
    assert summary["augmented_below_all_originals_rate"] == pytest.approx(7 / 8)
    assert summary["all_augmented_strictly_bottom_rate"] == pytest.approx(0.5)
    assert summary["mean_top_augmented_rank"] == pytest.approx(3.0)
    assert summary["top_augmented_rank_histogram"] == {"1": 1, "2": 0, "3": 0, "4": 0, "5": 1}


def test_validate_checkpoints_rejects_incomplete_and_mismatched_models(tmp_path):
    incomplete = tmp_path / "incomplete"
    incomplete.mkdir()
    with pytest.raises(FileNotFoundError, match="Incomplete checkpoint"):
        eval_script.validate_checkpoints({"helpfulness": str(incomplete)})

    paths = []
    for index, base_model in enumerate(("base/one", "base/two")):
        path = tmp_path / f"adapter-{index}"
        path.mkdir()
        (path / "adapter_config.json").write_text(
            json.dumps({"base_model_name_or_path": base_model}), encoding="utf-8"
        )
        (path / "adapter_model.safetensors").write_bytes(b"weights")
        paths.append(path)
    with pytest.raises(ValueError, match="share one base model"):
        eval_script.validate_checkpoints({"helpfulness": str(paths[0]), "honesty": str(paths[1])})


def test_prepare_rows_rejects_absent_and_unmapped_criteria():
    dataset = Dataset.from_list([make_row()])
    with pytest.raises(ValueError, match="no selected rows"):
        eval_script.prepare_rows(
            dataset,
            {"honesty": "/checkpoint"},
            max_rows=None,
            expected_num_original=4,
            expected_num_augmented=4,
        )

    mixed = Dataset.from_list([make_row(), make_row(preference_dimension="honesty", source_index=8)])
    with pytest.raises(ValueError, match="criteria without checkpoints"):
        eval_script.prepare_rows(
            mixed,
            {"helpfulness": "/checkpoint"},
            max_rows=None,
            expected_num_original=4,
            expected_num_augmented=4,
        )


class FakeScorer:
    def __init__(self):
        self.calls = []

    def score_pairs(self, pairs, criterion):
        self.calls.append((criterion, list(pairs)))
        if criterion is None:
            return [-10.0 - index for index in range(len(pairs))]
        return [-9.0 - 0.5 * index for index in range(len(pairs))]


def test_score_outputs_preserve_each_raw_score_and_have_one_long_row_per_candidate(tmp_path):
    rows = eval_script.prepare_rows(
        Dataset.from_list([make_row()]),
        {"helpfulness": "/checkpoint"},
        max_rows=None,
        expected_num_original=4,
        expected_num_augmented=4,
    )
    scorer = FakeScorer()

    ranking_records, response_records = eval_script.score_and_rank_rows(rows, scorer, beta=0.2)

    assert len(ranking_records) == 1
    assert len(response_records) == 8
    assert len({row["candidate_label"] for row in response_records}) == 8
    assert [call[0] for call in scorer.calls] == [None, "helpfulness"]
    first = response_records[0]
    assert first["adapter_log_probability"] == -9.0
    assert first["base_log_probability"] == -10.0
    assert first["log_probability_difference"] == 1.0
    assert first["pl_utility"] == 0.2

    output = tmp_path / "response_scores.csv"
    eval_script.write_csv(output, response_records)
    with output.open(encoding="utf-8", newline="") as handle:
        saved = list(csv.DictReader(handle))
    assert len(saved) == 8
    assert float(saved[0]["adapter_log_probability"]) == first["adapter_log_probability"]
    assert float(saved[0]["base_log_probability"]) == first["base_log_probability"]
    assert float(saved[0]["log_probability_difference"]) == first["log_probability_difference"]
    assert float(saved[0]["pl_utility"]) == first["pl_utility"]


def test_build_statistics_reports_micro_macro_and_per_criterion():
    records = []
    for criterion in ("helpfulness", "honesty"):
        _, flags = eval_script.rank_candidates(scored_candidates([8, 7, 6, 5, 4, 3, 2, 1]))
        records.append({"criterion": criterion, **flags})

    statistics = eval_script.build_statistics(records, metadata={"beta": 0.2})

    assert statistics["metadata"] == {"beta": 0.2}
    assert set(statistics["by_criterion"]) == {"helpfulness", "honesty"}
    assert statistics["overall_micro"]["num_prompt_criterion_rows"] == 2
    assert statistics["overall_macro_by_criterion"]["all_augmented_strictly_bottom_rate"] == 1.0
