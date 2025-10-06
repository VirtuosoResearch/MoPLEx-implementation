import torch
import numpy as np
import json
import os
import re
from scipy.stats import spearmanr
from typing import Callable, Literal, Optional

KernelName = Literal["rbf", "linear", "poly"]

def make_krr_predictor(
    train_w: torch.Tensor,
    train_subset_scores: torch.Tensor,
    *,
    alpha: float = 1e-3,                # ridge λ (noise variance in GPR view)
    kernel: KernelName = "rbf",
    gamma: Optional[float] = None,      # for RBF / poly; if None and RBF, defaults to 1 / n_features
    degree: int = 3,                    # for poly kernel
    coef0: float = 1.0,                 # for poly kernel
    jitter: float = 1e-8,               # numerical stabilizer added to diagonal
) -> Callable[[torch.Tensor], torch.Tensor]:
    """
    Fit Kernel Ridge Regression on (train_w, train_subset_scores) and return a predictor.

    Shapes
    -------
    train_w:             (n_train, d)
    train_subset_scores: (n_train,) or (n_train, m)   # supports multi-target
    Returned predictor expects test_w with shape (n_test, d).

    Usage
    -----
    predict = make_krr_predictor(train_w, train_subset_scores, alpha=1e-2, kernel="rbf")
    pred_test_scores = predict(test_w)
    """

    X = train_w
    y = train_subset_scores

    if y.dim() == 1:
        y = y.unsqueeze(1)
        squeeze_out = True
    else:
        squeeze_out = False

    device = X.device
    dtype = X.dtype
    n = X.shape[0]

    # -------------------- kernels --------------------
    def _linear(A, B):
        return A @ B.T

    def _rbf(A, B, gamma_):
        # ||A - B||^2 = ||A||^2 + ||B||^2 - 2 A B^T
        A_norm = (A * A).sum(dim=1, keepdim=True)        # (na, 1)
        B_norm = (B * B).sum(dim=1, keepdim=True).T      # (1, nb)
        sqdist = (A_norm + B_norm - 2.0 * (A @ B.T)).clamp_min(0.0)
        return torch.exp(-gamma_ * sqdist)

    def _poly(A, B, gamma_, degree_, coef0_):
        return (gamma_ * (A @ B.T) + coef0_)**degree_

    # -------------------- choose gamma --------------------
    if kernel == "rbf" and gamma is None:
        # scikit-learn default: gamma = 1 / n_features
        gamma = 1.0 / max(1, X.shape[1])

    # -------------------- Gram matrix --------------------
    if kernel == "linear":
        K = _linear(X, X)
    elif kernel == "rbf":
        K = _rbf(X, X, gamma)
    elif kernel == "poly":
        K = _poly(X, X, gamma if gamma is not None else 1.0, degree, coef0)
    else:
        raise ValueError(f"Unknown kernel: {kernel}")

    I = torch.eye(n, device=device, dtype=dtype)
    K_reg = K + (alpha + jitter) * I

    # Solve (K + αI) α_vec = y  → α_vec
    # Cholesky is stable for SPD matrices
    L = torch.linalg.cholesky(K_reg)
    # torch.cholesky_solve solves A x = b given Cholesky factor of A
    alpha_vec = torch.cholesky_solve(y, L)  # (n, m)

    # Closure that predicts on new points
    def predict(test_w: torch.Tensor) -> torch.Tensor:
        Z = test_w
        if kernel == "linear":
            K_test = _linear(Z, X)
        elif kernel == "rbf":
            K_test = _rbf(Z, X, gamma)
        else:  # poly
            K_test = _poly(Z, X, gamma if gamma is not None else 1.0, degree, coef0)

        preds = K_test @ alpha_vec  # (n_test, m)
        return preds.squeeze(1) if squeeze_out else preds

    return predict

def filter_insensitive_samples(
    subset_scores: torch.Tensor,
    variance_threshold: float = 1e-6,
    datamodels_num: int = 0
) -> dict:

    test_subset_scores = subset_scores[datamodels_num:, :]
    if subset_scores.dim() != 2:
        raise ValueError("The input must be a two-dimensional matrix.")

    variances_per_sample = torch.var(test_subset_scores, dim=0)

    sensitive_mask = variances_per_sample > variance_threshold
    sensitive_indices = torch.where(sensitive_mask)[0]
    
    insensitive_indices = torch.where(~sensitive_mask)[0]

    filtered_scores = subset_scores[:, sensitive_indices]

    return {
        "filtered_scores": filtered_scores,
        "sensitive_indices": sensitive_indices,
        "insensitive_indices": insensitive_indices,
    }

def calculate_normalized_error(
    pred_scores: torch.Tensor,
    true_scores: torch.Tensor,
    threshold: float = 1e-1
):
    if pred_scores.shape != true_scores.shape:
        raise ValueError("The shapes of the predicted and true values must be the same.")

    # 1. Calculate absolute residual error (MSE)
    residual_error_mse = torch.mean((pred_scores - true_scores) ** 2)

    # 2. Calculate the total variance of the target
    total_variance = torch.var(true_scores)

    # Prevent division by zero
    if total_variance < threshold:
        #print("Warning: The variance of the true values is close to zero.")
        return {
            "use": False,
            "mse": residual_error_mse.item(),
            "total_variance": total_variance.item(),
            'residual_error_mse': residual_error_mse.item(),
            "normalized_error": float('inf'),
            "r_squared": float('-inf')
        }

    # 3. Calculate normalized error
    normalized_error = residual_error_mse / total_variance
    #normalized_error = residual_error_mse
    
    # 4. Calculate R^2
    r_squared = 1.0 - normalized_error

    return {
        "use": True,
        "mse": residual_error_mse.item(),
        "total_variance": total_variance.item(),
        'residual_error_mse': residual_error_mse.item(),
        "normalized_error": normalized_error.item(),
        "r_squared": r_squared.item()
    }

def solve_ridge_regression(
    train_w: torch.Tensor,
    train_subset_scores: torch.Tensor,
    alpha: float = 1.0,
    driver: str = 'gelss'
) -> torch.Tensor:

    if alpha < 0:
        raise ValueError("The regularization strength alpha must be non-negative.")

    if train_w.shape[0] != train_subset_scores.shape[0]:
        raise ValueError(
            f"The number of samples in train_w and train_subset_scores must be the same, "
            f"but now they are {train_w.shape[0]} and {train_subset_scores.shape[0]}."
        )
    N = train_w.shape[1]  
    K = train_subset_scores.shape[1] if train_subset_scores.dim() > 1 else 1 

    identity_matrix = torch.eye(N, device=train_w.device, dtype=train_w.dtype)
    
    augmented_w = torch.vstack([
        train_w,
        torch.sqrt(torch.tensor(alpha)) * identity_matrix
    ])

    zeros_for_scores = torch.zeros(N, K, device=train_subset_scores.device, dtype=train_subset_scores.dtype)

    augmented_scores = torch.vstack([
        train_subset_scores,
        zeros_for_scores
    ])

    phi_solution = torch.linalg.lstsq(augmented_w, augmented_scores, driver=driver).solution

    return phi_solution

def solve_lasso_regression(
    train_w: torch.Tensor,
    train_subset_scores: torch.Tensor,
    alpha: float = 0.1,
    max_iter: int = 1000,
    tol: float = 1e-4
) -> torch.Tensor:
    # --- 1. Input validation ---
    if alpha < 0:
        raise ValueError("The regularization strength alpha must be non-negative.")
    if train_w.shape[0] != train_subset_scores.shape[0]:
        raise ValueError(
            f"The number of samples in train_w and train_subset_scores must be the same, "
            f"but now they are {train_w.shape[0]} and {train_subset_scores.shape[0]}."
        )

    # --- 2. Initialize and pre-calculate ---
    M, N = train_w.shape
    _ , K = train_subset_scores.shape

    # Move the data to GPU (if available)
    device = train_w.device
    
    # Initialize the weight Phi
    phi = torch.zeros(N, K, device=device, dtype=train_w.dtype)
    
    # Pre-calculate W^T * Y, this part remains constant in the loop
    w_t_y = train_w.T @ train_subset_scores

    # Pre-calculate the L2 norm squared of each column of W, for standardization
    w_col_norm_sq = torch.sum(train_w ** 2, dim=0)

    # --- 3. Coordinate descent iteration ---
    for i in range(max_iter):
        phi_old = phi.clone()

        for j in range(N):
            # Remove the contribution of feature j
            w_t_w_phi_j = (train_w.T @ (train_w @ phi)) - (w_col_norm_sq[j] * phi[j, :])
            
            # Calculate ρ_j = W_j^T (Y - WΦ_{-j})
            rho_j = w_t_y[j, :] - w_t_w_phi_j[j, :]
            
            # Apply the soft thresholding function
            # S(ρ, λ) = sign(ρ) * max(|ρ| - λ, 0)
            phi[j, :] = torch.sign(rho_j) * torch.maximum(
                torch.abs(rho_j) - alpha, 
                torch.tensor(0.0, device=device, dtype=train_w.dtype)
            ) / (w_col_norm_sq[j] + 1e-8) # Add a small number to prevent division by zero
        
        # --- 4. Check convergence ---
        max_change = torch.max(torch.abs(phi - phi_old))
        if max_change < tol:
            print(f"Converged in {i+1} iterations.")
            break
            
    if i == max_iter - 1:
        print("Reached the maximum number of iterations.")

    return phi

def compute_margin(outputs, targets):
    true_logits = outputs[torch.arange(outputs.size(0)), targets]
    
    # mask out the true class to find the highest logit among others
    masked_outputs = outputs.clone()
    masked_outputs[torch.arange(outputs.size(0)), targets] = float('-inf')
    max_other_logits = masked_outputs.max(dim=1).values
    
    return true_logits - max_other_logits

def eval_test_margin(model, loss_fn, load_batch_fn, test_dataset, batch_size, batch_num, device='cuda'):

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=torch.cuda.is_available())
    batch_margin_list = []
    total_margin = 0.0
    with torch.no_grad():
        for i, (batch) in enumerate(test_loader):
            inputs, targets, batch_size = load_batch_fn(batch, device)
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs).detach()
            margin = compute_margin(outputs[:, -1], targets[:, -1])
            batch_margin_list.append(margin.item())
            if (i >= batch_num - 1) or (i >= len(test_dataset) - 1):
                break
    return batch_margin_list

def eval_test_loss(model, loss_fn, load_batch_fn, test_dataset, batch_size, batch_num, device='cuda'):
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=torch.cuda.is_available())

    batch_loss_list = []
    total_loss = 0.0
    with torch.no_grad():
        for i, (batch) in enumerate(test_loader):
            inputs, targets, batch_size = load_batch_fn(batch, device)
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            total_loss += loss.item() * inputs.size(0)
            batch_loss_list.append(loss.item())
            if (i >= batch_num - 1) or (i >= len(test_dataset) - 1):
                break
    return batch_loss_list

def get_all_scores(model, model_path, use_nso, loss_fn, load_batch_fn, test_dataset, test_batch_size=1, test_batch_num=50, metric='loss', device='cuda'):
    if use_nso:
        model_file = os.path.join(model_path, f"nso_model_all_1.pth")
    else:
        model_file = os.path.join(model_path, f"model_all_1.pth")
    model.load_state_dict(torch.load(model_file, map_location='cpu'))
    model.eval()
    if metric == 'loss':
        scores = eval_test_loss(model, loss_fn, load_batch_fn, test_dataset, test_batch_size, test_batch_num, device)
    elif metric == 'margin':
        scores = eval_test_margin(model, loss_fn, load_batch_fn, test_dataset, test_batch_size, test_batch_num, device)
    scores = torch.tensor(scores)
    print("All scores shape:", scores.shape)
    return scores

def get_subset_scores(model, model_path, model_name, run_name, use_nso, use_sam, train_ids_list, loss_fn, load_batch_fn, test_dataset, test_batch_size=1, test_batch_num=50, metric='loss', device='cuda'):
    if metric == 'loss':
        score_path = './results/scores'
    elif metric == 'margin':
        score_path = './results/margin_scores'
    if not os.path.exists(score_path):
        os.makedirs(score_path)
    if run_name != '':
        score_file = os.path.join(score_path, f"{run_name}.json")
        if os.path.exists(score_file):
            with open(score_file, 'r') as f:
                subset_scores = json.load(f)
            subset_scores = torch.tensor(subset_scores)
            print("Subset scores shape:", subset_scores.shape)
            return subset_scores
    pattern = re.compile(rf"^{re.escape(model_name)}_\w+\.pth$")
    model_file_list = []
    # for file in os.listdir(model_path):
    #     if pattern.match(file):
    #         model_file = os.path.join(model_path, file)
    #         model_file_list.append(model_file)
    num_models = 0
    for file in os.listdir(model_path):
        if pattern.match(file):
            num_models += 1
    for i in range(num_models):
        if use_nso:
            model_file = os.path.join(model_path, f"nso_model_{i}.pth")
        elif use_sam:
            model_file = os.path.join(model_path, f"sam_model_{i}.pth")
        else:
            model_file = os.path.join(model_path, f"model_{i}.pth")
        model_file_list.append(model_file)

    print(f"Found {num_models} models matching the pattern '{model_name}' in '{model_path}'.")

    subset_scores = []
    for i in range(num_models):
        model_file = model_file_list[i]
        model.load_state_dict(torch.load(model_file, map_location='cpu'))
        model.eval()
        if metric == 'loss':
            scores = eval_test_loss(model, loss_fn, load_batch_fn, test_dataset, test_batch_size, test_batch_num, device)
        elif metric == 'margin':
            scores = eval_test_margin(model, loss_fn, load_batch_fn, test_dataset, test_batch_size, test_batch_num, device)
        subset_scores.append(scores)
    subset_scores = torch.tensor(subset_scores)
    print("Subset scores shape:", subset_scores.shape)
    with open(score_file, 'w') as f:
        json.dump(subset_scores.tolist(), f, indent=4)
    return subset_scores

def _flatten_ids(ids):
    # ids can be a list of ints, list of arrays/lists, or a ragged np.array(object)
    if isinstance(ids, np.ndarray) and ids.dtype != object:
        return ids.ravel()
    # list/tuple or object array -> concatenate each sub-sequence
    parts = []
    for x in ids:
        parts.append(np.asarray(x).ravel())
    return np.concatenate(parts) if parts else np.array([], dtype=int)

def datamodels(subset_scores, train_ids_list, test_batch_num=50, num_train=0, solver='lstsq'):
    #num_samples = np.max(train_ids_list) + 1
    num_samples = np.max(_flatten_ids(train_ids_list)) + 1
    num_subset = subset_scores.shape[0]
    if num_train <= 0 or num_train > num_subset:
        num_train = num_subset
    print(f"Total number of models: {num_subset}")
    train_subset_scores = subset_scores[:num_train]
    test_subset_scores = subset_scores[num_train:]
    print(f"Using only the first {num_train} models for training.")

    w_list = []
    for subset_index in range(num_subset):
        w_i = torch.zeros(num_samples)
        w_i[train_ids_list[subset_index]] = 1
        w_list.append(w_i)
    w = torch.stack(w_list, dim=0)
    train_w = w[:num_train]
    test_w = w[num_train:]

    train_w = train_w
    test_w = test_w
    train_subset_scores = train_subset_scores
    test_subset_scores = test_subset_scores

    results_list = []
    test_batch_num = min(test_batch_num, test_subset_scores.shape[1])

    # data affinity to train_w
    # aff_item = train_w.T @ data_affinity_matrix @ train_w
    # train_subset_scores = train_subset_scores - aff_item
    #Phi = solve_ridge_regression(train_w, train_subset_scores, alpha=1e10, driver='gelss')
    #Phi = solve_lasso_regression(train_w, train_subset_scores, alpha=1)
    if solver == 'lstsq':
        Phi = torch.linalg.lstsq(train_w, train_subset_scores, driver='gelss').solution
        pred_test_scores = test_w @ Phi
    elif solver == 'krr':
        predict = make_krr_predictor(train_w, train_subset_scores, alpha=1e-1, kernel="rbf")
        #predict = make_krr_predictor(train_w, train_subset_scores, alpha=1e-2, kernel="poly", degree=3, gamma=1e-1)
        pred_test_scores = predict(test_w)
    else:
        raise ValueError(f"Unknown solver: {solver}")

    
    residuals_list = []
    errors_list = []
    spearmans_list = []

    for j in range(pred_test_scores.shape[1]):
        true_scores = test_subset_scores[:, j]
        pred_scores = pred_test_scores[:, j]
        result = calculate_normalized_error(pred_scores, true_scores, threshold=1e-1)
        if result['use']:
            errors_list.append(float(result['normalized_error']))
            #residual_error_mse_list.append(float(result['residual_error_mse']))
            rho, _ = spearmanr(
                true_scores.detach().cpu().numpy(),
                pred_scores.detach().cpu().numpy()
            )
            spearmans_list.append(float(rho))
    results = {
        'error': errors_list,
        #'phi': Phi.tolist(),
        'spearman_corr': spearmans_list,
    }

    # residuals_list = []
    # errors_list = []
    # spearmans_list = []
    # for j in range(test_batch_num):
    #     train_scores = train_subset_scores[:, j]
    #     test_scores = test_subset_scores[:, j]
    #     phi_col = torch.linalg.lstsq(train_w, train_scores, driver='gelss').solution

    #     # print(f"Phi column for batch {j}:", phi_col)

    #     residual = test_w @ phi_col - test_scores
    #     # print("||residual|| =", residual.norm().item())

    #     # Compute Spearman correlation
    #     #train_spearman_corr, _ = spearmanr(train_scores.cpu().numpy(), (train_w @ phi_col).cpu().numpy())
    #     spearman_corr, _ = spearmanr(test_scores.cpu().numpy(), (test_w @ phi_col).cpu().numpy())
    #     #spearman_corr = train_spearman_corr
    #     #print("Spearman correlation =", spearman_corr)

    #     errors_list.append(float(residual.norm().item()))
    #     spearmans_list.append(float(spearman_corr))
    # results = {
    #     'error': errors_list,
    #     'phi': Phi.tolist(),
    #     'spearman_corr': spearmans_list,
    # }
    return results

def epoch_datamodels(subset_scores, model_path, model_name, epoch, train_ids_list, data_affinity_matrix=None, test_batch_num=50, num_train=0, solver='lstsq'):
    #num_samples = np.max(train_ids_list) + 1
    num_samples = np.max(_flatten_ids(train_ids_list)) + 1
    model_file_list = []
    # for file in os.listdir(model_path):
    #     if pattern.match(file):
    #         model_file = os.path.join(model_path, file)
    #         model_file_list.append(model_file)
    # model_file_list.sort()
    for i in range(50):
        model_file = os.path.join(model_path, f"{model_name}_{i}_{epoch}.pth")
        if os.path.exists(model_file):
            model_file_list.append(model_file)
        else:
            break
    num_models = len(model_file_list)
    print(f"Found {num_models} models matching the pattern '{model_name}' in '{model_path}'.")

    if num_train <= 0 or num_train > num_models:
        num_train = num_models
    print(f"Total number of models: {num_models}")
    train_subset_scores = subset_scores[:num_train]
    test_subset_scores = subset_scores[num_train:]
    print(f"Using only the first {num_train} models for training.")

    w_list = []
    for subset_index in range(num_models):
        w_i = torch.zeros(num_samples)
        w_i[train_ids_list[subset_index]] = 1
        w_list.append(w_i)
    w = torch.stack(w_list, dim=0)
    train_w = w[:num_train]
    test_w = w[num_train:]

    train_w = train_w
    test_w = test_w
    train_subset_scores = train_subset_scores
    test_subset_scores = test_subset_scores

    results_list = []
    test_batch_num = min(test_batch_num, test_subset_scores.shape[1])

    # data affinity to train_w
    # aff_item = train_w.T @ data_affinity_matrix @ train_w
    # train_subset_scores = train_subset_scores - aff_item
    #Phi = solve_ridge_regression(train_w, train_subset_scores, alpha=1e10, driver='gelss')
    #Phi = solve_lasso_regression(train_w, train_subset_scores, alpha=1)
    if solver == 'lstsq':
        Phi = torch.linalg.lstsq(train_w, train_subset_scores, driver='gelss').solution
        pred_test_scores = test_w @ Phi
    elif solver == 'krr':
        predict = make_krr_predictor(train_w, train_subset_scores, alpha=1, kernel="rbf")
        pred_test_scores = predict(test_w)
    else:
        raise ValueError(f"Unknown solver: {solver}")

    # print(f"Predicted test scores shape: {pred_test_scores.shape}")
    # print(f"Test subset scores shape: {test_subset_scores.shape}")
    # print(f"Train subset scores shape: {train_subset_scores.shape}")
    # print(f"Train weights shape: {train_w.shape}")
    # print(f"Test weights shape: {test_w.shape}")
    # print(f"Phi shape: {Phi.shape}")

    residual_error_mse_list = []
    errors_list = []
    spearmans_list = []

    for j in range(pred_test_scores.shape[1]):
        true_scores = test_subset_scores[:, j]
        pred_scores = pred_test_scores[:, j]
        result = calculate_normalized_error(pred_scores, true_scores)
        if result['use']:
            errors_list.append(float(result['normalized_error']))
            residual_error_mse_list.append(float(result['residual_error_mse']))
            rho, _ = spearmanr(
                true_scores.detach().cpu().numpy(),
                pred_scores.detach().cpu().numpy()
            )
            spearmans_list.append(float(rho))
    results = {
        'error': errors_list,
        'residual_error_mse': residual_error_mse_list,
        #'phi': Phi.tolist(),
        'spearman_corr': spearmans_list,
    }

    return results

def datamodels_split(train_subset_scores, test_subset_scores, train_ids_list, test_ids_list, test_batch_num=1000, solver='lstsq'):
    #num_samples = np.max(train_ids_list) + 1
    num_samples = np.max(_flatten_ids(train_ids_list)) + 1
    num_train_subset = train_subset_scores.shape[0]
    num_test_subset = test_subset_scores.shape[0]
    if num_train_subset <= 0 or num_train_subset > num_train_subset:
        num_train_subset = num_train_subset
    print(f"Total number of models: {num_train_subset}")
    train_subset_scores = train_subset_scores
    test_subset_scores = test_subset_scores
    print(f"Using only the first {num_train_subset} models for training.")

    train_w_list = []
    for subset_index in range(num_train_subset):
        w_i = torch.zeros(num_samples)
        w_i[train_ids_list[subset_index]] = 1
        train_w_list.append(w_i)
    train_w = torch.stack(train_w_list, dim=0)

    test_w_list = []
    for subset_index in range(num_test_subset):
        w_i = torch.zeros(num_samples)
        w_i[test_ids_list[subset_index]] = 1
        test_w_list.append(w_i)
    test_w = torch.stack(test_w_list, dim=0)

    test_batch_num = min(test_batch_num, test_subset_scores.shape[1])
    if solver == 'lstsq':
        Phi = torch.linalg.lstsq(train_w, train_subset_scores).solution
        pred_test_scores = test_w @ Phi
    elif solver == 'krr':
        predict = make_krr_predictor(train_w, train_subset_scores, alpha=1e-2, kernel="rbf")
        pred_test_scores = predict(test_w)
    else:
        raise ValueError(f"Unknown solver: {solver}")

    
    errors_list = []
    spearmans_list = []
    valid_column_indices = []

    for j in range(test_subset_scores.shape[1]):
        true_scores = test_subset_scores[:, j]
        pred_scores = pred_test_scores[:, j]
        result = calculate_normalized_error(pred_scores, true_scores, threshold=-1)
        if result['use']:
            valid_column_indices.append(j)
            errors_list.append(float(result['normalized_error']))
            rho, _ = spearmanr(
                true_scores.detach().cpu().numpy(),
                pred_scores.detach().cpu().numpy()
            )
            spearmans_list.append(float(rho))

    errors_array = np.asarray(errors_list, dtype=float)
    spearman_array = np.asarray(spearmans_list, dtype=float)
    valid_indices_array = np.asarray(valid_column_indices, dtype=int)

    if spearman_array.size > 0:
        top_k = min(500, spearman_array.size)
        top_order = np.argsort(spearman_array)[-top_k:]
        top_indices = valid_indices_array[top_order]
        top_errors = errors_array[top_order]
        top_spearmans = spearman_array[top_order]
    else:
        top_order = np.array([], dtype=int)
        top_indices = np.array([], dtype=int)
        top_errors = np.array([], dtype=float)
        top_spearmans = np.array([], dtype=float)

    results = {
        'error': top_errors.tolist(),
        'spearman_corr': top_spearmans.tolist(),
        'top_indices': top_indices.tolist(),
        'valid_indices': valid_indices_array.tolist(),
    }

    metadata = {
        'train_w': train_w,
        'test_w': test_w,
        'train_subset_scores': train_subset_scores,
        'test_subset_scores': test_subset_scores,
        'valid_indices': valid_indices_array,
        'top_indices': top_indices,
        'top_order': top_order,
        'errors': errors_array,
        'spearman': spearman_array,
    }

    return results, metadata

