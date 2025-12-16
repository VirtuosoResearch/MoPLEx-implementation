"""
Kernel-based surrogate model for reward estimation.

According to the paper Section 3.1:
- f_i(S): test result for preference i when reward model is trained on subset S
- g_i(S): surrogate model that predicts f_i(S) using kernel method
- g_i(S) = Σ_{j=1}^n α_{ij} K(1_S, 1_{S_j})
  where K(u, v) = exp(-||u-v||^2 / (2σ^2)) is RBF kernel
"""

import os
import sys
import numpy as np
import torch
from typing import List, Set, Optional, Union, Iterable, Tuple, Tuple

# Add notebooks directory to path for imports
script_dir = os.path.dirname(os.path.abspath(__file__))
notebooks_dir = os.path.join(script_dir, "..", "notebooks")
sys.path.insert(0, notebooks_dir)

from logistic_regression import (
    solve_logistic_regression,
    compute_dpo_loss,
    load_precomputed_gradients_b,
)


def indicator_vector(S: Set[int], n: int) -> np.ndarray:
    """
    Convert subset S to indicator vector 1_S ∈ {0, 1}^n.
    
    Args:
        S: set of indices (subset of {0, 1, ..., n-1})
        n: total number of samples
        
    Returns:
        indicator: numpy array of shape [n] with 1s at positions in S, 0s elsewhere
    """
    indicator = np.zeros(n, dtype=np.float32)
    for idx in S:
        if 0 <= idx < n:
            indicator[idx] = 1.0
    return indicator


def rbf_kernel(u: np.ndarray, v: np.ndarray, sigma: float = 1.0) -> float:
    """
    Radial Basis Function (RBF) kernel: K(u, v) = exp(-||u-v||^2 / (2σ^2)).
    
    Args:
        u: numpy array of shape [n]
        v: numpy array of shape [n]
        sigma: bandwidth parameter (default 1.0)
        
    Returns:
        kernel value: scalar
    """
    diff = u - v
    squared_norm = np.sum(diff ** 2)
    return np.exp(-squared_norm / (2 * sigma ** 2))


def compute_f_i(
    S: Set[int],
    test_idx: int,
    gradients: List[np.ndarray],
    z: Optional[np.ndarray] = None,
    max_iters: int = 1000,
    lr: float = 0.1,
    tol: float = 1e-6,
    verbose: bool = False,
) -> float:
    """
    Compute f_i(S): test loss for preference i when logistic regression is trained on subset S.
    
    This is the actual reward modeling performance:
    1. Train logistic regression on subset S
    2. Evaluate the trained model on test preference i
    
    Args:
        S: training subset (set of indices)
        test_idx: index of test preference i
        gradients: list of gradient vectors g_j for all samples
        z: optional labels z_j for all samples. If None, all +1.
        max_iters: maximum iterations for logistic regression
        lr: learning rate for logistic regression
        tol: tolerance for convergence
        verbose: whether to print progress
        
    Returns:
        loss: DPO loss on test preference i using model trained on S
    """
    if len(S) == 0:
        raise ValueError("Subset S cannot be empty.")
    
    if test_idx < 0 or test_idx >= len(gradients):
        raise IndexError(f"test_idx {test_idx} out of range [0, {len(gradients)}).")
    
    # Train logistic regression on subset S
    S_list = sorted(S)
    gradients_S = [gradients[j] for j in S_list]
    b_values_S = np.zeros(len(S_list), dtype=np.float32)  # b = 0
    
    z_S = None
    if z is not None:
        z_S = z[S_list]
    
    # Note: If gradients were projected during precomputation (e.g., from millions to 200 dims),
    # then theta_S will also be in the projected space. This is correct because:
    # - g_i (projected) @ theta_S (projected) gives the same result as
    # - g_i (full) @ theta_S (full) in terms of the logistic regression loss
    # No inverse projection is needed since all operations stay in the projected space.
    theta_S = solve_logistic_regression(
        gradients=gradients_S,
        b_values=b_values_S,
        z=z_S,
        max_iters=max_iters,
        lr=lr,
        tol=tol,
        verbose=verbose,
    )
    
    # Evaluate on test preference i
    g_i = gradients[test_idx]
    b_i = 0.0  # b = 0
    z_i = 1.0 if z is None else z[test_idx]
    
    # Compute loss: log(1 + exp(b_i - z_i * g_i^T theta_S))
    # Both g_i and theta_S are in the same space (projected if projection was used)
    g_dot_theta = np.dot(g_i, theta_S)
    loss_i = np.log(1 + np.exp(b_i - z_i * g_dot_theta))
    
    return float(loss_i)


def unproject_theta(
    theta_projected: np.ndarray,
    projection_matrix: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Unproject theta from projected space back to full parameter space using pseudo-inverse.
    
    If projection_matrix is None, returns theta_projected as-is (no projection was used).
    
    During precomputation, projection is: g_projected = projection_matrix.T @ g_full
    So unprojection uses pseudo-inverse: g_full = (projection_matrix.T)^+ @ g_projected
    
    Args:
        theta_projected: theta in projected space [projection_dim]
        projection_matrix: projection matrix [num_params, projection_dim] or None
        
    Returns:
        theta_full: theta in full parameter space [num_params]
    """
    if projection_matrix is None:
        return theta_projected
    
    # projection_matrix: [num_params, projection_dim]
    # projection_matrix.T: [projection_dim, num_params]
    # Pseudo-inverse: (projection_matrix.T)^+ = projection_matrix @ (projection_matrix.T @ projection_matrix)^(-1)
    # This is more accurate than just projection_matrix @ theta_projected
    
    # Compute (projection_matrix.T @ projection_matrix)^(-1)
    # This is [projection_dim, projection_dim]
    PTP = projection_matrix.T @ projection_matrix  # [projection_dim, projection_dim]
    PTP_inv = np.linalg.pinv(PTP)  # Use pseudo-inverse for numerical stability
    
    # Compute pseudo-inverse of projection_matrix.T
    # (projection_matrix.T)^+ = projection_matrix @ PTP_inv
    proj_T_pinv = projection_matrix @ PTP_inv  # [num_params, projection_dim]
    
    # Unproject: theta_full = (projection_matrix.T)^+ @ theta_projected
    theta_full = proj_T_pinv @ theta_projected  # [num_params]
    
    return theta_full


def evaluate_model_on_new_data(
    theta_full: np.ndarray,
    gradients_new: List[np.ndarray],
    z_new: Optional[np.ndarray] = None,
    projection_matrix: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, float]:
    losses = []
    
    for idx, g_new in enumerate(gradients_new):
        if projection_matrix is not None:
            theta_projected = projection_matrix.T @ theta_full
            g_dot_theta = np.dot(g_new, theta_projected)
        else:
            # New gradients are in full space, use theta_full directly
            g_dot_theta = np.dot(g_new, theta_full)
        
        b_i = 0.0  # b = 0
        z_i = 1.0 if z_new is None else z_new[idx]
        loss_i = np.log(1 + np.exp(b_i - z_i * g_dot_theta))
        losses.append(loss_i)
    
    losses = np.array(losses, dtype=np.float32)
    mean_loss = float(np.mean(losses))
    
    return losses, mean_loss


def train_and_evaluate_model(
    training_subset: Set[int],
    gradients: List[np.ndarray],
    z: Optional[np.ndarray] = None,
    projection_matrix: Optional[np.ndarray] = None,
    test_gradients: Optional[List[np.ndarray]] = None,
    test_z: Optional[np.ndarray] = None,
    max_iters: int = 1000,
    lr: float = 0.1,
    tol: float = 1e-6,
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray, Optional[Tuple[np.ndarray, float]]]:

    if len(training_subset) == 0:
        raise ValueError("Training subset cannot be empty.")
    
    # Train logistic regression
    S_list = sorted(training_subset)
    gradients_S = [gradients[j] for j in S_list]
    b_values_S = np.zeros(len(S_list), dtype=np.float32)  # b = 0
    
    z_S = None
    if z is not None:
        z_S = z[S_list]
    
    theta_projected = solve_logistic_regression(
        gradients=gradients_S,
        b_values=b_values_S,
        z=z_S,
        max_iters=max_iters,
        lr=lr,
        tol=tol,
        verbose=verbose,
    )
    
    # Unproject to full parameter space
    theta_full = unproject_theta(theta_projected, projection_matrix)
    
    # Optionally evaluate on test data
    test_results = None
    if test_gradients is not None:
        test_results = evaluate_model_on_new_data(
            theta_full=theta_full,
            gradients_new=test_gradients,
            z_new=test_z,
            projection_matrix=projection_matrix,
        )
    
    return theta_projected, theta_full, test_results


def compute_kernel_matrix(
    subsets: List[Set[int]],
    n: int,
    sigma: float = 1.0,
) -> np.ndarray:
    """
    Compute kernel matrix K where K[i, j] = K(1_{S_i}, 1_{S_j}).
    
    Args:
        subsets: list of subsets S_i
        n: total number of samples
        sigma: RBF kernel bandwidth
        
    Returns:
        K: kernel matrix of shape [len(subsets), len(subsets)]
    """
    m = len(subsets)
    K = np.zeros((m, m), dtype=np.float32)
    
    indicators = [indicator_vector(S, n) for S in subsets]
    
    for i in range(m):
        for j in range(m):
            K[i, j] = rbf_kernel(indicators[i], indicators[j], sigma)
    
    return K


def predict_surrogate(
    S: Set[int],
    basis_subsets: List[Set[int]],
    alpha: np.ndarray,
    n: int,
    sigma: float = 1.0,
) -> float:
    indicator_S = indicator_vector(S, n)
    prediction = 0.0
    
    for j, S_j in enumerate(basis_subsets):
        indicator_Sj = indicator_vector(S_j, n)
        k_val = rbf_kernel(indicator_S, indicator_Sj, sigma)
        prediction += alpha[j] * k_val
    
    return float(prediction)



def estimate_affinity_scores(
    gradients: List[np.ndarray],
    z: Optional[np.ndarray] = None,
    projection_matrix: Optional[np.ndarray] = None,
    num_training_subsets: int = 1000,
    training_subset_size: int = 10,
    num_evaluation_subsets: Optional[int] = None,
    sigma: float = 1.0,
    reg: float = 1e-6,
    max_iters: int = 1000,
    lr: float = 0.1,
    tol: float = 1e-6,
    verbose: bool = True,
) -> np.ndarray:

    import random
    
    n = len(gradients)
    T = np.zeros((n, n), dtype=np.float32)
    
    # Set random seeds for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    # Use single-element subsets as basis: S_j = {j}
    # These are used in the kernel representation g_i(S) = Σ_j α_ij K(1_S, 1_{S_j})
    basis_subsets = [{j} for j in range(n)]
    
    # Sample diverse training subsets for learning surrogate coefficients
    # This fixes Problem 1: training on diverse S, not just singletons
    if verbose:
        print(f"Sampling {num_training_subsets} diverse training subsets of size {training_subset_size}...")
    
    training_subsets = []
    effective_subset_size = min(training_subset_size, n)
    for _ in range(num_training_subsets):
        if n < effective_subset_size:
            subset = set(range(n))
        else:
            subset = set(np.random.choice(n, size=effective_subset_size, replace=False))
        training_subsets.append(subset)
    
    # For each test preference i, compute f_i(S) for all training subsets S
    # This fixes Problem 3: f_i(S) correctly represents training on S, testing on i
    if verbose:
        print(f"Computing f_i(S) for all preferences i and training subsets S...")
        print(f"  This will compute {n * num_training_subsets} values (may take time)...")
    
    f_values = {}  # Dict: (i, tuple(sorted(S))) -> f_i(S)
    
    for i in range(n):
        if verbose and i % max(1, n // 10) == 0:
            print(f"  Processing preference {i}/{n}...")
        
        for S in training_subsets:
            S_key = tuple(sorted(S))
            if (i, S_key) not in f_values:
                try:
                    f_val = compute_f_i(
                        S=S,
                        test_idx=i,
                        gradients=gradients,
                        z=z,
                        max_iters=max_iters,
                        lr=lr,
                        tol=tol,
                        verbose=False,
                    )
                    f_values[(i, S_key)] = f_val
                except Exception as e:
                    if verbose:
                        print(f"    Warning: f_{i}(S) failed for S={S_key}: {e}")
                    f_values[(i, S_key)] = 0.0
    
    # Learn surrogate coefficients α_ij for each test preference i
    # This fixes Problem 2: training on diverse S provides non-degenerate kernel matrix
    if verbose:
        print(f"Learning surrogate coefficients α_ij for each preference...")
    
    alphas = np.zeros((n, n), dtype=np.float32)
    
    for i in range(n):
        if verbose and i % max(1, n // 10) == 0:
            print(f"  Learning coefficients for preference {i}/{n}...")
        
        # Collect training data: (S, f_i(S)) pairs
        training_targets = []
        training_subset_list = []
        for S in training_subsets:
            S_key = tuple(sorted(S))
            f_val = f_values.get((i, S_key), 0.0)
            training_targets.append(f_val)
            training_subset_list.append(S)
        
        training_targets = np.array(training_targets, dtype=np.float32)
        
        K_train_basis = np.zeros((len(training_subsets), len(basis_subsets)), dtype=np.float32)
        for idx, S in enumerate(training_subsets):
            indicator_S = indicator_vector(S, n)
            for j, S_j in enumerate(basis_subsets):
                indicator_Sj = indicator_vector(S_j, n)
                K_train_basis[idx, j] = rbf_kernel(indicator_S, indicator_Sj, sigma)
        
        # Solve for basis coefficients: K_train_basis @ alpha_basis ≈ training_targets
        # With regularization: (K_train_basis^T @ K_train_basis + reg*I) @ alpha_basis = K_train_basis^T @ training_targets
        KtK = K_train_basis.T @ K_train_basis + reg * np.eye(len(basis_subsets), dtype=np.float32)
        Kt_f = K_train_basis.T @ training_targets
        alpha_basis = np.linalg.solve(KtK, Kt_f)
        
        alphas[i, :] = alpha_basis
    
    # Estimate T_{i,j} using surrogate predictions
    if verbose:
        print(f"Estimating affinity scores T_{i,j} using surrogate model...")
    
    # Use evaluation subsets (can be same as training or different)
    if num_evaluation_subsets is None:
        evaluation_subsets = training_subsets
    else:
        evaluation_subsets = []
        for _ in range(num_evaluation_subsets):
            if n < effective_subset_size:
                subset = set(range(n))
            else:
                subset = set(np.random.choice(n, size=effective_subset_size, replace=False))
            evaluation_subsets.append(subset)
    
    if verbose:
        print(f"  Using {len(evaluation_subsets)} subsets for evaluation")
    
    counts = np.zeros((n, n), dtype=np.int32)
    
    for S_t in evaluation_subsets:
        for i in S_t:
            # Predict f_i(S_t) using surrogate
            pred = predict_surrogate(
                S=S_t,
                basis_subsets=basis_subsets,
                alpha=alphas[i, :],
                n=n,
                sigma=sigma,
            )
            
            for j in S_t:
                T[i, j] += pred
                counts[i, j] += 1
    
    # Normalize by counts
    for i in range(n):
        for j in range(n):
            if counts[i, j] > 0:
                T[i, j] /= counts[i, j]
    
    return T


if __name__ == "__main__":
    """
    Example usage:
    
    python trainers/kernel_estimation.py
    
    This will:
    1. Load precomputed gradients
    2. Compute affinity scores T_{i,j}
    3. Print summary statistics
    """
    import sys
    import os
    
    # Load precomputed gradients
    precompute_filename = "Llama-3.2-1B_200.pt"
    script_dir = os.path.dirname(os.path.abspath(__file__))
    precompute_path = os.path.join(script_dir, "..", "pre_compute", precompute_filename)
    
    print(f"Loading precomputed gradients from: {precompute_path}")
    gradients_np, b_values_np, z_values_np, projection_matrix, projection_dim = load_precomputed_gradients_b(
        precompute_filename
    )
    
    print(f"Loaded {len(gradients_np)} gradients")
    
    # Estimate affinity scores
    # Uses 1000 random subsets of size 10 for training surrogate
    T = estimate_affinity_scores(
        gradients=gradients_np,
        z=z_values_np,
        projection_matrix=projection_matrix,  # Pass projection matrix for potential unprojection
        num_training_subsets=1000,  # Diverse subsets for training surrogate
        training_subset_size=10,    # Size of each training subset
        num_evaluation_subsets=None,  # None = reuse training subsets for evaluation
        sigma=1.0,
        reg=1e-6,
        verbose=True,
    )
    
    print(f"\nAffinity matrix T shape: {T.shape}")
    print(f"T statistics:")
    print(f"  Mean: {np.mean(T):.6f}")
    print(f"  Std: {np.std(T):.6f}")
    print(f"  Min: {np.min(T):.6f}")
    print(f"  Max: {np.max(T):.6f}")
    print(f"  Diagonal mean: {np.mean(np.diag(T)):.6f}")
    
    # Save affinity matrix
    output_path = os.path.join(script_dir, "..", "pre_compute", f"affinity_matrix_{precompute_filename.split('.')[0]}.npy")
    np.save(output_path, T)
    print(f"\nSaved affinity matrix to: {output_path}.npy")

