# coding=utf-8

from datasets import Dataset

from alignment.cyclic_data import build_cyclic_rows, has_strict_rotated_cycle, split_rows_to_dataset_dict


def _make_cyclic_completion(text: str, instr: float, honesty: float, truth: float, helpf: float):
    return {
        "response": text,
        "annotations": {
            "instruction_following": {"Rating": instr},
            "honesty": {"Rating": honesty},
            "truthfulness": {"Rating": truth},
            "helpfulness": {"Rating": helpf},
        },
    }


def test_has_strict_rotated_cycle_accepts_expected_pattern():
    # A, B, C, D aligned at indices 0..3.
    scores = {
        "instruction_following": [4.0, 3.0, 2.0, 1.0],
        "honesty": [1.0, 4.0, 3.0, 2.0],
        "truthfulness": [2.0, 1.0, 4.0, 3.0],
        "helpfulness": [3.0, 2.0, 1.0, 4.0],
    }
    assert has_strict_rotated_cycle(scores)


def test_has_strict_rotated_cycle_rejects_ties():
    scores = {
        "instruction_following": [4.0, 3.0, 2.0, 1.0],
        "honesty": [1.0, 4.0, 3.0, 2.0],
        "truthfulness": [2.0, 1.0, 4.0, 3.0],
        "helpfulness": [3.0, 2.0, 2.0, 4.0],
    }
    assert not has_strict_rotated_cycle(scores)


def test_build_cyclic_rows_keeps_only_matching_example():
    dataset = Dataset.from_list(
        [
            {
                "instruction": "p-good",
                "completions": [
                    _make_cyclic_completion("A", 4, 1, 2, 3),
                    _make_cyclic_completion("B", 3, 4, 1, 2),
                    _make_cyclic_completion("C", 2, 3, 4, 1),
                    _make_cyclic_completion("D", 1, 2, 3, 4),
                ],
            },
            {
                "instruction": "p-bad",
                "completions": [
                    _make_cyclic_completion("A", 4, 4, 4, 4),
                    _make_cyclic_completion("B", 3, 3, 3, 3),
                    _make_cyclic_completion("C", 2, 2, 2, 2),
                    _make_cyclic_completion("D", 1, 1, 1, 1),
                ],
            },
        ]
    )

    rows, stats = build_cyclic_rows(dataset, seed=123)

    assert stats.total_rows == 2
    assert stats.eligible_rows == 2
    assert stats.cycle_rows == 1
    assert len(rows) == 4
    assert all(row["prompt"] == "p-good" for row in rows)
    assert all(len(row["responses"]) == 4 for row in rows)
    assert all(len(row["scores"]) == 4 for row in rows)
    assert all(row["scores"] == sorted(row["scores"], reverse=True) for row in rows)
    assert {row["preference_dimension"] for row in rows} == {
        "instruction_following",
        "honesty",
        "truthfulness",
        "helpfulness",
    }


def test_has_strict_rotated_cycle_accepts_any_permutation_base_order():
    # Base order is [2, 0, 3, 1] instead of fixed [0, 1, 2, 3].
    scores = {
        "instruction_following": [3.0, 1.0, 4.0, 2.0],
        "honesty": [4.0, 2.0, 1.0, 3.0],
        "truthfulness": [1.0, 3.0, 2.0, 4.0],
        "helpfulness": [2.0, 4.0, 3.0, 1.0],
    }
    assert has_strict_rotated_cycle(scores)


def test_has_strict_rotated_cycle_subset_dimensions_m2():
    scores = {
        "instruction_following": [4.0, 3.0, 2.0, 1.0],
        # rotate by 1 relative to instruction_following
        "helpfulness": [1.0, 4.0, 3.0, 2.0],
    }
    assert has_strict_rotated_cycle(scores, dimensions=("instruction_following", "helpfulness"))


def test_has_strict_rotated_cycle_subset_dimensions_m2_allows_shift_2():
    scores = {
        "instruction_following": [4.0, 3.0, 2.0, 1.0],
        # rotate by 2 relative to instruction_following
        "helpfulness": [2.0, 1.0, 4.0, 3.0],
    }
    assert has_strict_rotated_cycle(scores, dimensions=("instruction_following", "helpfulness"))


def test_has_strict_rotated_cycle_rejects_duplicate_nonzero_shifts():
    scores = {
        "instruction_following": [4.0, 3.0, 2.0, 1.0],
        # both rotate by 1 from base -> should be rejected because shifts must differ
        "honesty": [1.0, 4.0, 3.0, 2.0],
        "helpfulness": [1.0, 4.0, 3.0, 2.0],
    }
    assert not has_strict_rotated_cycle(scores, dimensions=("instruction_following", "honesty", "helpfulness"))


def test_build_cyclic_rows_subset_dimensions_emits_only_requested_rows():
    dataset = Dataset.from_list(
        [
            {
                "instruction": "p-subset",
                "completions": [
                    _make_cyclic_completion("A", 4, 1, 2, 1),
                    _make_cyclic_completion("B", 3, 4, 1, 4),
                    _make_cyclic_completion("C", 2, 3, 4, 3),
                    _make_cyclic_completion("D", 1, 2, 3, 2),
                ],
            }
        ]
    )

    rows, stats = build_cyclic_rows(dataset, dimensions=("instruction_following", "helpfulness"), seed=0)

    assert stats.cycle_rows == 1
    assert len(rows) == 2
    assert {row["preference_dimension"] for row in rows} == {"instruction_following", "helpfulness"}


def test_build_cyclic_rows_finds_combination_when_more_than_four_candidates():
    dataset = Dataset.from_list(
        [
            {
                "instruction": "p1",
                "completions": [
                    _make_cyclic_completion("X", 0, 0, 0, 0),
                    _make_cyclic_completion("Y", 0, 0, 0, 0),
                    _make_cyclic_completion("A", 4, 1, 2, 3),
                    _make_cyclic_completion("B", 3, 4, 1, 2),
                    _make_cyclic_completion("C", 2, 3, 4, 1),
                    _make_cyclic_completion("D", 1, 2, 3, 4),
                ],
            }
        ]
    )

    rows, stats = build_cyclic_rows(dataset, seed=0)

    assert stats.total_rows == 1
    assert stats.eligible_rows == 1
    assert stats.cycle_rows == 1
    assert len(rows) == 4
    assert {row["preference_dimension"] for row in rows} == {
        "instruction_following",
        "honesty",
        "truthfulness",
        "helpfulness",
    }


def test_split_rows_is_deterministic_for_same_seed():
    rows = [
        {
            "prompt": f"p{i}",
            "responses": ["A", "B", "C", "D"],
            "scores": [4.0, 3.0, 2.0, 1.0],
            "preference_dimension": "instruction_following",
            "source_index": i,
        }
        for i in range(20)
    ]

    d1 = split_rows_to_dataset_dict(rows, seed=7)
    d2 = split_rows_to_dataset_dict(rows, seed=7)

    assert len(d1["train"]) == 16
    assert len(d1["validation"]) == 2
    assert len(d1["test"]) == 2
    assert d1["train"]["prompt"] == d2["train"]["prompt"]
    assert d1["validation"]["prompt"] == d2["validation"]["prompt"]
    assert d1["test"]["prompt"] == d2["test"]["prompt"]