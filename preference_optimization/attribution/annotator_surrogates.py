"""Per-annotator surrogate predictors built on top of ``datamodels`` utilities.

This module provides helper functions to train and evaluate surrogate predictors for
individual annotators. The predictors share the same feature representation
(``train_w``) but produce annotator-specific score estimates.
"""

from __future__ import annotations

from typing import Callable, Dict, Iterable, Mapping, MutableMapping, Optional, Sequence, Tuple
import torch
from scipy.stats import spearmanr

from .datamodels import calculate_normalized_error, make_krr_predictor

Predictor = Callable[[torch.Tensor], torch.Tensor]


def _ensure_2d_tensor(tensor: torch.Tensor, name: str) -> torch.Tensor:
    if tensor.dim() != 2:
        raise ValueError(f"{name} must be a 2D tensor, got shape {tuple(tensor.shape)}")
    return tensor


def _ensure_scores_tensor(scores: torch.Tensor) -> torch.Tensor:
    if scores.dim() == 1:
        return scores.unsqueeze(1)
    if scores.dim() != 2:
        raise ValueError(f"annotator_scores must have shape (n, m); got {tuple(scores.shape)}")
    return scores


def build_surrogate_predictors(
    train_w: torch.Tensor,
    annotator_scores: torch.Tensor,
    annotator_ids: Optional[Sequence[int]] = None,
    *,
    alpha: float = 1e-3,
    kernel: str = "rbf",
    gamma: Optional[float] = None,
    degree: int = 3,
    coef0: float = 1.0,
    jitter: float = 1e-8,
) -> Dict[int, Predictor]:
    """Train kernel ridge surrogate predictors for each annotator.

    Args:
        train_w: Feature matrix of shape (n_samples, d).
        annotator_scores: Target scores of shape (n_samples, n_annotators).
        annotator_ids: Optional explicit annotator ordering. Defaults to
            ``range(n_annotators)``.
        alpha: Ridge regularization strength.
        kernel: Kernel choice passed to :func:`make_krr_predictor`.
        gamma: Kernel coefficient for RBF/poly kernels.
        degree: Polynomial degree when ``kernel='poly'``.
        coef0: Independent term in polynomial kernel.
        jitter: Diagonal jitter added during linear solve.

    Returns:
        Mapping from annotator id to prediction callable.
    """
    train_w = _ensure_2d_tensor(train_w, "train_w")
    annotator_scores = _ensure_scores_tensor(annotator_scores)

    n_targets = annotator_scores.shape[1]
    if annotator_ids is None:
        annotator_ids = list(range(n_targets))
    if len(annotator_ids) != n_targets:
        raise ValueError(
            "Length of annotator_ids must match annotator_scores' second dimension."
        )

    predictors: Dict[int, Predictor] = {}
    for idx, annotator_id in enumerate(annotator_ids):
        predictor = make_krr_predictor(
            train_w,
            annotator_scores[:, idx],
            alpha=alpha,
            kernel=kernel,
            gamma=gamma,
            degree=degree,
            coef0=coef0,
            jitter=jitter,
        )
        predictors[int(annotator_id)] = predictor
    return predictors


def predict_with_surrogates(
    predictors: Mapping[int, Predictor],
    features: torch.Tensor,
    annotator_ids: Optional[Sequence[int]] = None,
) -> torch.Tensor:
    """Stack surrogate predictions for all requested annotators."""
    features = _ensure_2d_tensor(features, "features")
    if annotator_ids is None:
        annotator_ids = list(predictors.keys())
    outputs = []
    for annotator_id in annotator_ids:
        predictor = predictors[annotator_id]
        outputs.append(predictor(features).unsqueeze(1))
    return torch.cat(outputs, dim=1)


def evaluate_surrogates(
    predictors: Mapping[int, Predictor],
    eval_w: torch.Tensor,
    eval_scores: torch.Tensor,
    annotator_ids: Optional[Sequence[int]] = None,
    *,
    compute_spearman: bool = True,
) -> Dict[int, MutableMapping[str, float]]:
    """Evaluate surrogate predictors on held-out data.

    Returns
    -------
    Dict mapping annotator id to metric dictionary with the fields returned by
    :func:`calculate_normalized_error`. Optionally adds ``spearman``.
    """
    eval_w = _ensure_2d_tensor(eval_w, "eval_w")
    eval_scores = _ensure_scores_tensor(eval_scores)

    if annotator_ids is None:
        annotator_ids = list(predictors.keys())
    if eval_scores.shape[1] != len(annotator_ids):
        raise ValueError(
            "eval_scores second dimension must align with annotator_ids length."
        )

    metrics: Dict[int, MutableMapping[str, float]] = {}
    for col, annotator_id in enumerate(annotator_ids):
        predictor = predictors[annotator_id]
        pred_scores = predictor(eval_w).detach()
        true_scores = eval_scores[:, col]

        # Align tensor devices for metric computation
        pred_scores = pred_scores.to(true_scores.device)

        metric = calculate_normalized_error(pred_scores, true_scores)
        if compute_spearman:
            with torch.no_grad():
                pred_np = pred_scores.detach().cpu().numpy()
                true_np = true_scores.detach().cpu().numpy()
            rho, _ = spearmanr(pred_np, true_np)
            metric["spearman"] = float(rho) if rho is not None else float("nan")
        metrics[int(annotator_id)] = metric
    return metrics


def prepare_score_matrix(
    annotator_score_dict: Mapping[int, Sequence[float]],
    annotator_ids: Optional[Sequence[int]] = None,
) -> torch.Tensor:
    """Convert a mapping of annotator scores to a consistent tensor layout."""
    if annotator_ids is None:
        annotator_ids = sorted(annotator_score_dict.keys())
    columns = []
    for annotator_id in annotator_ids:
        if annotator_id not in annotator_score_dict:
            raise KeyError(f"Missing scores for annotator {annotator_id}")
        columns.append(torch.as_tensor(annotator_score_dict[annotator_id], dtype=torch.float32))
    stacked = torch.stack(columns, dim=1)
    return stacked


def split_train_eval(
    tensor: torch.Tensor,
    split: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Split a tensor along the first dimension at ``split`` index."""
    if split <= 0 or split >= tensor.size(0):
        raise ValueError("split must be in (0, len)")
    return tensor[:split], tensor[split:]


def fit_lstsq_surrogates(
    train_w: torch.Tensor,
    annotator_scores: torch.Tensor,
    annotator_ids: Optional[Sequence[int]] = None,
) -> Tuple[torch.Tensor, Tuple[int, ...]]:
    """Solve a least-squares surrogate for each annotator simultaneously."""
    train_w = _ensure_2d_tensor(train_w, "train_w")
    annotator_scores = _ensure_scores_tensor(annotator_scores)

    solution = torch.linalg.lstsq(train_w, annotator_scores).solution  # (d, m)
    if annotator_ids is None:
        annotator_ids = tuple(range(annotator_scores.shape[1]))
    else:
        annotator_ids = tuple(int(a) for a in annotator_ids)
    return solution, annotator_ids


def predict_lstsq_surrogates(
    coeffs: torch.Tensor,
    features: torch.Tensor,
) -> torch.Tensor:
    """Predict surrogate scores using precomputed least-squares coefficients."""
    features = _ensure_2d_tensor(features, "features")
    return features @ coeffs


def fit_krr_surrogates(
    train_w: torch.Tensor,
    annotator_scores: torch.Tensor,
    *,
    annotator_ids: Optional[Sequence[int]] = None,
    alpha: float = 1e-3,
    kernel: str = "rbf",
    gamma: Optional[float] = None,
    degree: int = 3,
    coef0: float = 1.0,
    jitter: float = 1e-8,
) -> Tuple[Dict[str, torch.Tensor], Tuple[int, ...]]:
    """Fit multi-target kernel ridge surrogates and return packed parameters."""
    train_w = _ensure_2d_tensor(train_w, "train_w")
    annotator_scores = _ensure_scores_tensor(annotator_scores)

    X = train_w
    y = annotator_scores

    if annotator_ids is None:
        annotator_ids = tuple(range(y.shape[1]))
    else:
        annotator_ids = tuple(int(a) for a in annotator_ids)

    if alpha < 0:
        raise ValueError("alpha must be non-negative")

    if kernel not in {"linear", "rbf", "poly"}:
        raise ValueError(f"Unsupported kernel '{kernel}'")

    if kernel == "rbf" and gamma is None:
        gamma = 1.0 / max(1, X.shape[1])

    def _linear(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return A @ B.T

    def _rbf(A: torch.Tensor, B: torch.Tensor, gamma_: float) -> torch.Tensor:
        A_norm = (A * A).sum(dim=1, keepdim=True)
        B_norm = (B * B).sum(dim=1, keepdim=True).T
        sqdist = (A_norm + B_norm - 2.0 * (A @ B.T)).clamp_min(0.0)
        return torch.exp(-gamma_ * sqdist)

    def _poly(A: torch.Tensor, B: torch.Tensor, gamma_: float, degree_: int, coef0_: float) -> torch.Tensor:
        return (gamma_ * (A @ B.T) + coef0_) ** degree_

    if kernel == "linear":
        gram = _linear(X, X)
    elif kernel == "rbf":
        gram = _rbf(X, X, gamma)
    else:
        gamma_val = gamma if gamma is not None else 1.0
        gram = _poly(X, X, gamma_val, degree, coef0)

    n = X.shape[0]
    eye = torch.eye(n, device=X.device, dtype=X.dtype)
    gram_reg = gram + (alpha + jitter) * eye
    chol = torch.linalg.cholesky(gram_reg)
    alpha_vec = torch.cholesky_solve(y, chol)

    packed = {
        "train_w": X.detach().clone(),
        "alpha_vec": alpha_vec.detach().clone(),
        "kernel": torch.tensor(0) if kernel == "linear" else torch.tensor(1) if kernel == "rbf" else torch.tensor(2),
        "kernel_name": kernel,
        "gamma": torch.tensor(0.0 if gamma is None else gamma),
        "degree": torch.tensor(degree),
        "coef0": torch.tensor(coef0),
        "alpha": torch.tensor(alpha),
        "jitter": torch.tensor(jitter),
    }
    return packed, annotator_ids


def predict_krr_surrogates(
    packed: Mapping[str, torch.Tensor],
    features: torch.Tensor,
) -> torch.Tensor:
    """Predict using packed kernel ridge surrogate parameters."""
    features = _ensure_2d_tensor(features, "features")
    train_w = packed["train_w"]
    alpha_vec = packed["alpha_vec"]
    kernel_name = packed.get("kernel_name")
    if kernel_name is None:
        kernel_idx = int(packed["kernel"].item())
        kernel_name = {0: "linear", 1: "rbf", 2: "poly"}.get(kernel_idx, "rbf")
    gamma = float(packed.get("gamma", torch.tensor(0.0)).item())
    degree = int(packed.get("degree", torch.tensor(3)).item())
    coef0 = float(packed.get("coef0", torch.tensor(1.0)).item())

    def _linear(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return A @ B.T

    def _rbf(A: torch.Tensor, B: torch.Tensor, gamma_: float) -> torch.Tensor:
        A_norm = (A * A).sum(dim=1, keepdim=True)
        B_norm = (B * B).sum(dim=1, keepdim=True).T
        sqdist = (A_norm + B_norm - 2.0 * (A @ B.T)).clamp_min(0.0)
        return torch.exp(-gamma_ * sqdist)

    def _poly(A: torch.Tensor, B: torch.Tensor, gamma_: float, degree_: int, coef0_: float) -> torch.Tensor:
        return (gamma_ * (A @ B.T) + coef0_) ** degree_

    if kernel_name == "linear":
        K_test = _linear(features, train_w)
    elif kernel_name == "rbf":
        gamma_val = gamma if gamma > 0 else 1.0 / max(1, train_w.shape[1])
        K_test = _rbf(features, train_w, gamma_val)
    else:
        gamma_val = gamma if gamma > 0 else 1.0
        K_test = _poly(features, train_w, gamma_val, degree, coef0)

    preds = K_test @ alpha_vec
    return preds
