"""Unit tests for the length-matched linear reward approximation."""

import unittest

import torch

from alignment.listwise_dpo import ListwiseDPOTrainer


def _build_inputs():
    """B=1, M=3 candidates, S=6 input positions (token frame S-1=5), H=2.

    Response spans (shifted frame): candidate 0 -> start 0, len 3 (the anchor);
    candidate 1 -> start 0, len 2; candidate 2 -> start 0, len 4.
    """
    labels = torch.tensor(
        [
            [-100, 10, 11, 12, -100, -100],
            [-100, 20, 21, -100, -100, -100],
            [-100, 30, 31, 32, 33, -100],
        ]
    )
    starts, lengths = ListwiseDPOTrainer._response_spans(labels)
    response_starts = starts.view(1, 3)
    response_lengths = lengths.view(1, 3)

    candidate_mask = torch.ones(1, 3, dtype=torch.bool)
    anchor_positions = [torch.tensor([0])]

    # Anchor per-token scores: 1, 2, 3 on its three response tokens.
    anchor_token_scores = torch.tensor([[1.0, 2.0, 3.0, 0.0, 0.0]], requires_grad=True)
    # Gradient [1, 0] at every input position: dot terms count delta[..., 0].
    anchor_grads = torch.zeros(1, 6, 2)
    anchor_grads[..., 0] = 1.0

    # Embeddings: candidate 0 all zeros, candidate 1 has delta 1 per position,
    # candidate 2 has delta 2 per position (in the first hidden dim).
    candidate_embeddings = torch.zeros(1, 3, 6, 2)
    candidate_embeddings[0, 1, :, 0] = 1.0
    candidate_embeddings[0, 2, :, 0] = 2.0

    return (
        candidate_embeddings,
        candidate_mask,
        anchor_positions,
        anchor_token_scores,
        anchor_grads,
        response_starts,
        response_lengths,
    )


class TestResponseSpans(unittest.TestCase):
    def test_spans(self):
        labels = torch.tensor(
            [
                [-100, 10, 11, 12, -100, -100],
                [-100, -100, -100, 40, 41, -100],
                [-100, -100, -100, -100, -100, -100],
            ]
        )
        starts, lengths = ListwiseDPOTrainer._response_spans(labels)
        self.assertEqual(starts.tolist(), [0, 2, 0])
        self.assertEqual(lengths.tolist(), [3, 2, 0])


class TestLengthMatchedEstimator(unittest.TestCase):
    def test_matched_prefix_values_without_ref(self):
        args = _build_inputs()
        estimates = ListwiseDPOTrainer._linear_approx_length_matched(*args)
        # anchor: exact full score 1+2+3 = 6
        # (0->1): L=2, prefix 1+2=3, dot over positions {0,1} = 2*1 = 2 -> 5
        # (0->2): L=3, prefix 6, dot over positions {0,1,2} = 3*2 = 6 -> 12
        self.assertTrue(torch.allclose(estimates, torch.tensor([[6.0, 5.0, 12.0]])))

    def test_matched_prefix_values_with_ref(self):
        args = _build_inputs()
        ref_token_logps = torch.tensor(
            [
                [
                    [0.2, 0.2, 0.2, 0.0, 0.0],
                    [0.5, 0.5, 0.0, 0.0, 0.0],
                    [1.0, 1.0, 1.0, 1.0, 0.0],
                ]
            ]
        )
        estimates = ListwiseDPOTrainer._linear_approx_length_matched(
            *args, ref_token_logps=ref_token_logps
        )
        # anchor: 6 - ref_full 0.6 = 5.4
        # (0->1): 5 - ref prefix at L=2 (0.5+0.5) = 4
        # (0->2): 12 - ref prefix at L=3 (3.0) = 9
        self.assertTrue(torch.allclose(estimates, torch.tensor([[5.4, 4.0, 9.0]])))

    def test_case_a_uses_partial_anchor_value(self):
        # For the shorter candidate the estimate must build on the anchor's
        # prefix sum (3.0), not its full score (6.0).
        args = _build_inputs()
        estimates = ListwiseDPOTrainer._linear_approx_length_matched(*args)
        self.assertAlmostEqual(float(estimates[0, 1]), 5.0)
        self.assertNotAlmostEqual(float(estimates[0, 1]), 8.0)

    def test_gradient_flows_only_through_anchor(self):
        args = _build_inputs()
        anchor_token_scores = args[3]
        estimates = ListwiseDPOTrainer._linear_approx_length_matched(*args)
        estimates.sum().backward()
        # Non-anchor entries are stop-gradient; the only graph path is the
        # anchor's exact full score, so every token contributes exactly once.
        self.assertTrue(torch.allclose(anchor_token_scores.grad, torch.ones(1, 5)))

    def test_equal_lengths_match_existing_estimator(self):
        labels = torch.tensor(
            [
                [-100, 10, 11, 12, -100, -100],
                [-100, 20, 21, 22, -100, -100],
                [-100, 30, 31, 32, -100, -100],
            ]
        )
        starts, lengths = ListwiseDPOTrainer._response_spans(labels)
        candidate_mask = torch.ones(1, 3, dtype=torch.bool)
        anchor_positions = [torch.tensor([0, 2])]

        torch.manual_seed(0)
        anchor_token_scores = torch.rand(2, 5)
        anchor_token_scores[:, 3:] = 0.0  # only response tokens carry score
        anchor_grads = torch.rand(2, 6, 2)
        candidate_embeddings = torch.rand(1, 3, 6, 2)
        # A real full-score gradient is zero from position start + L = 3 on:
        # the last response token's own embedding (position 3) influences no
        # labeled token. Mirror that so the full-grid dot equals the masked dot.
        candidate_embeddings[:, :, 3:, :] = 0.0
        anchor_grads[:, 3:, :] = 0.0

        matched = ListwiseDPOTrainer._linear_approx_length_matched(
            candidate_embeddings,
            candidate_mask,
            anchor_positions,
            anchor_token_scores,
            anchor_grads,
            starts.view(1, 3),
            lengths.view(1, 3),
        )
        legacy = ListwiseDPOTrainer._linear_approx_from_anchor_tensors(
            candidate_embeddings,
            candidate_mask,
            anchor_positions,
            anchor_token_scores.sum(dim=-1),
            anchor_grads,
        )
        self.assertTrue(torch.allclose(matched, legacy, atol=1e-6))

    def test_padded_candidate_masked_out(self):
        args = list(_build_inputs())
        candidate_mask = torch.tensor([[True, True, False]])
        args[1] = candidate_mask
        estimates = ListwiseDPOTrainer._linear_approx_length_matched(*args)
        self.assertEqual(float(estimates[0, 2]), 0.0)


if __name__ == "__main__":
    unittest.main()
