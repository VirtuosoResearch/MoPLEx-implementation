import os
import sys

from datasets import Dataset
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src"))

from alignment.gpo import (
    aggregate_gp_pairwise_utilities,
    gp_pairwise_scores,
    listwise_to_gpo_pairwise_dataset,
)


def test_listwise_to_gpo_pairwise_respects_ranked_prefix_length():
    dataset = Dataset.from_list(
        [
            {
                "prompt": "p",
                "responses": ["A", "B", "C", "D"],
                "scores": [4.0, 3.0, 2.0, 1.0],
                "preference_dimension": "helpfulness",
                "ranked_prefix_length": 2,
                "source_index": 7,
            }
        ]
    )

    pairwise = listwise_to_gpo_pairwise_dataset(dataset)

    assert len(pairwise) == 5
    assert {(row["chosen"], row["rejected"]) for row in pairwise} == {
        ("A", "B"),
        ("A", "C"),
        ("A", "D"),
        ("B", "C"),
        ("B", "D"),
    }
    assert set(pairwise["preference_dimension"]) == {"helpfulness"}
    assert set(pairwise["source_index"]) == {7}


def test_listwise_to_gpo_pairwise_extreme_uses_top_and_bottom_without_prefix():
    dataset = Dataset.from_list(
        [
            {
                "prompt": "p",
                "responses": ["A", "B", "C"],
                "scores": [3.0, 2.0, 1.0],
                "preference_dimension": "truthfulness",
            }
        ]
    )

    pairwise = listwise_to_gpo_pairwise_dataset(dataset, strategy="extreme")

    assert len(pairwise) == 1
    assert pairwise[0]["chosen"] == "A"
    assert pairwise[0]["rejected"] == "C"


def test_gp_pairwise_scores_are_antisymmetric_for_high_dimensional_head():
    rewards_a = torch.tensor([[1.0, 0.0, 0.5, 1.0]])
    rewards_b = torch.tensor([[0.0, 1.0, 1.0, -0.5]])

    score_ab = gp_pairwise_scores(rewards_a, rewards_b, value_head_dim=4)
    score_ba = gp_pairwise_scores(rewards_b, rewards_a, value_head_dim=4)

    assert score_ab.item() == pytest.approx(-score_ba.item())


def test_aggregate_gp_pairwise_utilities_recovers_pairwise_winner():
    pairwise_scores = torch.tensor(
        [
            [
                [0.0, 2.0, 1.0],
                [-2.0, 0.0, -1.0],
                [-1.0, 1.0, 0.0],
            ]
        ]
    )
    candidate_mask = torch.tensor([[True, True, True]])

    utilities = aggregate_gp_pairwise_utilities(pairwise_scores, candidate_mask)

    assert utilities.argmax(dim=1).tolist() == [0]
    assert utilities[0, 0] > utilities[0, 2] > utilities[0, 1]
