"""
Direct computation of affinity scores for reward estimation.

According to the paper Section A.2:
- f_i(S): test result for preference i when reward model is trained on subset S
- T_{i,j}: affinity score = (1 / n_{i,j}) * Σ_{t: i in S_t, j in S_t} f_i(S_t)

We compute T_{i,j} directly by:
1. Sampling random subsets S_t
2. Computing f_i(S_t) for each subset and preference i
3. Accumulating and averaging according to the formula
"""

import os
import sys
import numpy as np
import torch
from typing import List, Set, Optional, Union, Iterable, Tuple

# Add notebooks directory to path for imports
script_dir = os.path.dirname(os.path.abspath(__file__))
notebooks_dir = os.path.join(script_dir, "..", "notebooks")
sys.path.insert(0, notebooks_dir)

from logistic_regression import (
    solve_logistic_regression,
    compute_dpo_loss,
    load_precomputed_gradients_b,
)


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


def estimate_affinity_scores(
    gradients: List[np.ndarray],
    z: Optional[np.ndarray] = None,
    projection_matrix: Optional[np.ndarray] = None,
    num_subsets: int = 1000,
    subset_size: int = 10,
    max_iters: int = 1000,
    lr: float = 0.1,
    tol: float = 1e-6,
    verbose: bool = True,
) -> np.ndarray:
    """
    Estimate affinity scores T_{i,j} directly without kernel/surrogate model.
    
    According to paper Section A.2:
    T_{i,j} = (1 / n_{i,j}) * Σ_{t: i in S_t, j in S_t} f_i(S_t)
    
    where:
    - n_{i,j} is the number of sampled subsets containing both i and j
    - f_i(S_t) is the test loss on preference i when model is trained on subset S_t
    
    Args:
        gradients: list of gradient vectors for all samples
        z: optional labels for all samples
        projection_matrix: projection matrix if gradients are projected
        num_subsets: number of random subsets to sample
        subset_size: size of each subset
        max_iters: max iterations for logistic regression
        lr: learning rate for logistic regression
        tol: tolerance for convergence
        verbose: whether to print progress
        
    Returns:
        T: affinity matrix of shape [n, n] where T[i, j] estimates how task j affects task i
    """
    import random
    
    n = len(gradients)
    T = np.zeros((n, n), dtype=np.float32)
    counts = np.zeros((n, n), dtype=np.int32)
    
    # Set random seeds for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    # Sample random subsets
    if verbose:
        print(f"Sampling {num_subsets} random subsets of size {subset_size}...")
    
    effective_subset_size = min(subset_size, n)
    sampled_subsets = []
    for _ in range(num_subsets):
        if n < effective_subset_size:
            subset = set(range(n))
        else:
            subset = set(np.random.choice(n, size=effective_subset_size, replace=False))
        sampled_subsets.append(subset)
    
    # Compute f_i(S_t) for each subset S_t and each preference i in S_t
    if verbose:
        print(f"Computing f_i(S_t) for all subsets and preferences...")
        print(f"  Total computations: ~{sum(len(S) for S in sampled_subsets)} (may take time)...")
    
    f_values = {}  # Cache: (i, tuple(sorted(S))) -> f_i(S)
    total_computed = 0
    
    for subset_idx, S_t in enumerate(sampled_subsets):
        if verbose and subset_idx % max(1, num_subsets // 10) == 0:
            print(f"  Processing subset {subset_idx}/{num_subsets}...")
        
        S_key = tuple(sorted(S_t))
        
        # For each preference i in S_t, compute f_i(S_t)
        for i in S_t:
            if (i, S_key) not in f_values:
                try:
                    f_val = compute_f_i(
                        S=S_t,
                        test_idx=i,
                        gradients=gradients,
                        z=z,
                        max_iters=max_iters,
                        lr=lr,
                        tol=tol,
                        verbose=False,
                    )
                    f_values[(i, S_key)] = f_val
                    total_computed += 1
                except Exception as e:
                    if verbose:
                        print(f"    Warning: f_{i}(S_t) failed for S_t={S_key}: {e}")
                    f_values[(i, S_key)] = 0.0
        
        # Accumulate T_{i,j} for all pairs (i, j) in S_t
        for i in S_t:
            f_i_St = f_values.get((i, S_key), 0.0)
            for j in S_t:
                T[i, j] += f_i_St
                counts[i, j] += 1
    
    if verbose:
        print(f"  Computed {total_computed} unique f_i(S) values")
        print(f"  (Reused {sum(len(S) for S in sampled_subsets) - total_computed} cached values)")
    
    # Normalize by counts: T_{i,j} = (1 / n_{i,j}) * Σ f_i(S_t)
    if verbose:
        print(f"Normalizing affinity scores...")
    
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
    
    # Estimate affinity scores directly without kernel/surrogate
    # Uses 1000 random subsets of size 10
    T = estimate_affinity_scores(
        gradients=gradients_np,
        z=z_values_np,
        projection_matrix=projection_matrix,
        num_subsets=1000,  # Number of random subsets to sample
        subset_size=10,    # Size of each subset
        max_iters=1000,
        lr=0.1,
        tol=1e-6,
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

