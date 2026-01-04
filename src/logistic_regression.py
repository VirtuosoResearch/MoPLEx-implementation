# %%
import os
import sys
from typing import Optional, Iterable, List, Tuple

import numpy as np
import torch


def load_precomputed_gradients_b(precompute_filename: str):
    """
    Load precomputed gradients from ./pre_compute.

    The expected file is created by scripts/precompute_gradients_for_approx_dpo.py
    and should contain at least:
        - "gradients": list/array of gradient vectors (optionally projected)
    Optionally:
        - "z_values": numpy array or tensor of shape [N] with labels in {+1, -1}
        - "projection_matrix": [num_params, projection_dim]
        - "projection_dim": int
    
    Note: b_values are not stored (b = 0 since model == ref_model at θ0).
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    precompute_path = os.path.join(script_dir, "..", "pre_compute", precompute_filename)
    precompute_path = os.path.normpath(precompute_path)

    print(f"Loading precomputed data from: {precompute_path}")
    data = torch.load(precompute_path, map_location="cpu", weights_only=False)

    if "gradients" not in data:
        raise KeyError(
            "Precomputed file must contain key 'gradients'. "
            f"Found keys: {list(data.keys())}"
        )

    gradients = data["gradients"]
    # b_values are not stored (b = 0), so create zeros array
    num_samples = len(gradients)
    b_values = np.zeros(num_samples, dtype=np.float32)
    
    z_values = data.get("z_values", None)

    # Ensure numpy arrays
    gradients_np: List[np.ndarray] = []
    for g in gradients:
        if isinstance(g, np.ndarray):
            gradients_np.append(g.astype(np.float32))
        else:
            gradients_np.append(np.asarray(g, dtype=np.float32))

    b_values_np = (
        b_values.astype(np.float32)
        if isinstance(b_values, np.ndarray)
        else np.asarray(b_values, dtype=np.float32)
    )

    if z_values is not None:
        z_values_np = (
            z_values.astype(np.float32)
            if isinstance(z_values, np.ndarray)
            else np.asarray(z_values, dtype=np.float32)
        )
    else:
        z_values_np = None

    projection_matrix = data.get("projection_matrix", None)
    projection_dim = data.get("projection_dim", None)

    if projection_matrix is not None:
        print(f"Loaded projection matrix with shape: {projection_matrix.shape}")
    if projection_dim is not None:
        print(f"Projection dimension: {projection_dim}")

    return gradients_np, b_values_np, z_values_np, projection_matrix, projection_dim


def solve_logistic_regression(
    gradients: Iterable[np.ndarray],
    b_values: np.ndarray,
    z: Optional[np.ndarray] = None,
    max_iters: int = 1000,
    lr: float = 0.1,
    tol: float = 1e-6,
    verbose: bool = True,
) -> np.ndarray:
    """
    Solve logistic regression over theta using precomputed g, b, z.

    We minimize:
        L(theta) = 1/N * sum_i log(1 + exp(b_i - z_i * g_i^T theta))

    Args:
        gradients: iterable of numpy arrays g_i with shape [d]
        b_values: numpy array of shape [N]
        z: numpy array of shape [N] with labels in {+1, -1}. If None, all +1.
        max_iters: maximum number of optimization iterations
        lr: learning rate for Adam
        tol: early stopping tolerance on loss improvement
        verbose: whether to print progress

    Returns:
        theta: numpy array of shape [d]
    """
    gradients_list = list(gradients)
    if len(gradients_list) == 0:
        raise ValueError("No gradients provided for logistic regression.")

    N = len(gradients_list)
    b_values = np.asarray(b_values, dtype=np.float32)
    if b_values.shape[0] != N:
        raise ValueError(
            f"b_values has length {b_values.shape[0]}, "
            f"but gradients list has length {N}."
        )

    G = np.stack(gradients_list, axis=0).astype(np.float32)  # [N, d]
    d = G.shape[1]

    if z is None:
        # Default to all +1 if z is not provided (standard DPO assumption)
        z_vec = np.ones(N, dtype=np.float32)
    else:
        z_vec = np.asarray(z, dtype=np.float32)
        if z_vec.shape[0] != N:
            raise ValueError(
                f"z has length {z_vec.shape[0]}, but number of samples is {N}."
            )

    device = torch.device("cpu")
    G_t = torch.from_numpy(G).to(device)  # [N, d]
    b_t = torch.from_numpy(b_values).to(device)  # [N]
    z_t = torch.from_numpy(z_vec).to(device)  # [N]

    theta = torch.zeros(d, device=device, requires_grad=True)
    optimizer = torch.optim.Adam([theta], lr=lr)

    prev_loss = float("inf")
    for it in range(max_iters):
        optimizer.zero_grad()

        logits = b_t - z_t * (G_t @ theta)  # [N]
        # Use numerically stable log(1+exp(x)) = max(x,0) + log(1+exp(-|x|))
        # This prevents overflow when x is large
        loss = (torch.clamp(logits, min=0) + torch.log1p(torch.exp(-torch.abs(logits)))).mean()

        loss.backward()
        
        # Clip gradients to prevent explosion
        torch.nn.utils.clip_grad_norm_([theta], max_norm=1.0)
        
        optimizer.step()

        current_loss = float(loss.item())
        if verbose and (it % 100 == 0 or it == max_iters - 1):
            print(f"[LogReg] iter={it}, loss={current_loss:.6f}")

        if abs(prev_loss - current_loss) < tol:
            if verbose:
                print(f"[LogReg] converged at iter={it}, loss={current_loss:.6f}")
            break
        prev_loss = current_loss

    return theta.detach().cpu().numpy()


def compute_dpo_loss(
    theta: np.ndarray,
    gradients: Iterable[np.ndarray],
    b_values: np.ndarray,
    z: Optional[np.ndarray] = None,
) -> Tuple[float, np.ndarray]:
    """
    Compute DPO loss using the logistic regression form from Appendix A.1.

    For each sample i:
        loss_i = log(1 + exp(b_i - z_i * g_i^T theta))

    Args:
        theta: numpy array of shape [d] - the solution from logistic regression
        gradients: iterable of numpy arrays g_i with shape [d]
        b_values: numpy array of shape [N]
        z: numpy array of shape [N] with labels in {+1, -1}. If None, all +1.

    Returns:
        mean_loss: scalar, average loss over all samples
        per_sample_loss: numpy array of shape [N], loss for each sample
    """
    gradients_list = list(gradients)
    N = len(gradients_list)
    b_values = np.asarray(b_values, dtype=np.float32)

    if z is None:
        z_vec = np.ones(N, dtype=np.float32)
    else:
        z_vec = np.asarray(z, dtype=np.float32)

    G = np.stack(gradients_list, axis=0).astype(np.float32)  # [N, d]
    theta_vec = np.asarray(theta, dtype=np.float32)  # [d]

    # Compute g_i^T theta for all samples
    g_dot_theta = G @ theta_vec  # [N]

    # Compute per-sample loss: log(1 + exp(b_i - z_i * g_i^T theta))
    logits = b_values - z_vec * g_dot_theta  # [N]
    per_sample_loss = np.log1p(np.exp(logits))  # [N]

    mean_loss = float(np.mean(per_sample_loss))

    return mean_loss, per_sample_loss


if __name__ == "__main__":

    # You can either hard-code the precompute file or pass it via CLI.
    # Here we keep the default path for convenience, but still support z_values if present.
    precompute_filename = "../pre_compute/Llama-3.2-1B_200.pt"
    gradients_np, b_values_np, z_values_np, projection_matrix, projection_dim = load_precomputed_gradients_b(
        precompute_filename
    )

    theta_hat = solve_logistic_regression(
        gradients=gradients_np,
        b_values=b_values_np,
        z=z_values_np,  # if None, solve_logistic_regression will default to all +1
        max_iters=1000,
        lr=0.1,
        tol=1e-6,
        verbose=True,
    )

    print(f"theta_hat shape: {theta_hat.shape}")
    print("theta_hat:", theta_hat)

    # Compute DPO loss at theta_0 (theta = 0, i.e., reference model)
    theta_0 = np.zeros_like(theta_hat)
    mean_loss_theta0, per_sample_loss_theta0 = compute_dpo_loss(
        theta=theta_0,
        gradients=gradients_np,
        b_values=b_values_np,
        z=z_values_np,  # if None, compute_dpo_loss will default to all +1
    )
    total_loss_theta0 = np.sum(per_sample_loss_theta0)

    # Compute DPO loss using the solved theta_hat
    mean_loss_theta_hat, per_sample_loss_theta_hat = compute_dpo_loss(
        theta=theta_hat,
        gradients=gradients_np,
        b_values=b_values_np,
        z=z_values_np,  # if None, compute_dpo_loss will default to all +1
    )
    total_loss_theta_hat = np.sum(per_sample_loss_theta_hat)

    print(f"\n{'='*60}")
    print(f"DPO Loss Comparison:")
    print(f"{'='*60}")
    print(f"\nAt theta_0 (reference model, theta = 0):")
    print(f"  Mean loss: {mean_loss_theta0:.6f}")
    print(f"  Total loss (sum): {total_loss_theta0:.6f}")
    print(f"  Per-sample loss (first 10): {per_sample_loss_theta0[:10]}")
    print(f"  Per-sample loss (min): {np.min(per_sample_loss_theta0):.6f}")
    print(f"  Per-sample loss (max): {np.max(per_sample_loss_theta0):.6f}")
    print(f"  Per-sample loss (std): {np.std(per_sample_loss_theta0):.6f}")

    print(f"\nAt theta_hat (estimated from logistic regression):")
    print(f"  Mean loss: {mean_loss_theta_hat:.6f}")
    print(f"  Total loss (sum): {total_loss_theta_hat:.6f}")
    print(f"  Per-sample loss (first 10): {per_sample_loss_theta_hat[:10]}")
    print(f"  Per-sample loss (min): {np.min(per_sample_loss_theta_hat):.6f}")
    print(f"  Per-sample loss (max): {np.max(per_sample_loss_theta_hat):.6f}")
    print(f"  Per-sample loss (std): {np.std(per_sample_loss_theta_hat):.6f}")

    print(f"\n{'='*60}")
    print(f"Loss Reduction:")
    print(f"{'='*60}")
    loss_reduction = total_loss_theta0 - total_loss_theta_hat
    loss_reduction_percent = (loss_reduction / total_loss_theta0) * 100.0 if total_loss_theta0 > 0 else 0.0
    print(f"  Total loss reduction: {loss_reduction:.6f}")
    print(f"  Percentage reduction: {loss_reduction_percent:.2f}%")
    print(f"  Ratio (theta_hat / theta_0): {total_loss_theta_hat / total_loss_theta0:.6f}")
