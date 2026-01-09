import torch
import torchcde
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def inverse_variance(V, mask, T):
    # --- . MASK THE COVARIANCE MATRIX V ---
    # Create a broadcastable mask for the matrix
    mask_f = mask.float()
    mask_matrix = mask_f.unsqueeze(2) * mask_f.unsqueeze(1) # (N, T, T)

    # Zero-out rows/columns for padded time points
    V_masked = V * mask_matrix
    
    # Add a small identity matrix for numerical stability. This makes the
    # padded block of the matrix invertible (it becomes an identity block).
    eye = torch.eye(T, device=V.device).unsqueeze(0)
    V_masked = V_masked + 1e-6 * eye

    # --- 3. COMPUTE BLUPs USING MASKED MATRICES ---
    # Invert the masked V matrix
    V_inv = torch.linalg.inv(V_masked)              # (N, T, T)

    return V_inv

def calculate_prediction_mse_with_blup(model, data_loader, device):
    model.eval()
    total_squared_error = 0.0
    total_valid_points = 0.0

    with torch.no_grad():
        for batch in data_loader:
            t, y, s_i, mask, _, t_i, x_aug = [d.to(device) for d in batch.values()]
            N, T, D = x_aug.shape

            blup_adjusted = torch.zeros_like(y)

            for j in range(1, T):
                # build batch of partial input paths up to time j
                x_aug_hist = x_aug[:, :j+1, :]                         # (N, j+1, D)
                t_hist = t[:j+1]                                       # (j+1,)

                # batch build linear splines (much faster than cubic)
                coeffs = torchcde.hermite_cubic_coefficients_with_backward_differences(x_aug_hist)
                X = torchcde.CubicSpline(coeffs)

                # predict mean and components up to time j
                pred_mean_hist, V_hist, Z_hist, G = model(s_i, X, j+1, return_components=True)

                for i in range(N):
                    if mask[i, j] == 0:
                        continue

                    current_pred = pred_mean_hist[i, -1]

                    valid_obs = mask[i, :j] == 1
                    if valid_obs.sum() == 0:
                        blup_adjusted[i, j] = current_pred
                        continue

                    y_obs = y[i, :j][valid_obs]
                    mu_obs = pred_mean_hist[i, :j][valid_obs]
                    Z_obs = Z_hist[i, :j, :][valid_obs]
                    V_obs = V_hist[i, :j, :j][valid_obs][:, valid_obs]

                    # BLUP: b̂ = G Zᵗ V⁻¹ (y - μ)
                    residual = (y_obs - mu_obs).unsqueeze(-1)
                    V_inv_r = torch.linalg.solve(V_obs, residual)     # (T_obs, 1)
                    GZ_T = torch.matmul(G, Z_obs.T)                   # (q, T_obs)
                    b_hat = torch.matmul(GZ_T, V_inv_r).squeeze(-1)   # (q,)

                    blup_adjusted[i, j] = current_pred + torch.dot(Z_hist[i, -1, :], b_hat)

            # MSE over valid points (excluding j=0)
            squared_error = (blup_adjusted[:, 1:] - y[:, 1:]) ** 2 * mask[:, 1:]
            total_squared_error += squared_error.sum().item()
            total_valid_points += mask[:, 1:].sum().item()

    mse = total_squared_error / total_valid_points
    return mse

def calculate_fit_mse_with_blup(model, data_loader, device):
    """
    Computes MSE using BLUP (Best Linear Unbiased Prediction) in fitted mode.
    Vectorized over batch.
    """
    model.eval()
    total_squared_error = 0
    total_valid_points = 0
    total_population_se = 0

    with torch.no_grad():
        for batch in data_loader:
            if 'metabolic_baseline' in batch.keys():
                t, y, s_i, mask, _, metabaseline = [d.to(device) for d in batch.values()]
                pred_mean, V, Z, D = model(s_i, t, metabolic_baseline=metabaseline, return_components=True)
            elif 'x_aug' in batch.keys():
                t, y, s_i, mask, id, t_i, x_aug = [d.to(device) for d in batch.values()]
                N = x_aug.shape[0]

                coeffs = torchcde.hermite_cubic_coefficients_with_backward_differences(x_aug)
                X = torchcde.CubicSpline(coeffs)
            
                pred_mean, V, Z, D = model(s_i, X, len(t), return_components=True)
            else:
                t, y, s_i, mask, _ = [d.to(device) for d in batch.values()]
                pred_mean, V, Z, D = model(s_i, t, return_components=True)
            
            N, T = y.shape

            # --- MASK THE RESIDUAL ---
            # Ensure padded values do not contribute to the residual.
            residual = (y - pred_mean) * mask  # (N, T)
            residual = residual.unsqueeze(-1) # (N, T, 1)

            # --- COMPUTE BLUPs USING MASKED MATRICES ---
            # Invert the masked V matrix
            V_inv = inverse_variance(V, mask, T)          # (N, T, T)

            # Solve V_inv * r (padded entries in r are 0, so they won't contribute)
            V_inv_r = torch.matmul(V_inv, residual)         # (N, T, 1)

            # G @ Z^T @ V_inv @ r
            Z_T = Z.transpose(1, 2)                         # (N, q, T)
            DZ_T = torch.matmul(D.unsqueeze(0), Z_T)        # (N, q, T)
            b_hat = torch.matmul(DZ_T, V_inv_r).squeeze(-1) # (N, q)

            # Z @ b_hat
            blup_adjustment = torch.matmul(Z, b_hat.unsqueeze(-1)).squeeze(-1) # (N, T)

            # Final adjusted predictions
            blup_adjusted = pred_mean + blup_adjustment     # (N, T)

            # Compute squared error (the mask here ensures padded predictions are ignored)
            squared_error = (blup_adjusted - y) ** 2 * mask
            population_squared_error = (pred_mean - y)** 2 * mask
            total_population_se += torch.sum(population_squared_error)
            total_squared_error += torch.sum(squared_error)
            total_valid_points += torch.sum(mask)

    mse = total_squared_error / total_valid_points
    population_mse = total_population_se / total_valid_points
    return mse.item(), population_mse.item()

def lme_log_likelihood(model, data_loader, device):
    """
    Computes the marginal log-likelihood using efficient, batched operations.
    Calculates log p(y | θ) = log N(y | μ, ZGZᵗ + R)
    """
    model.eval()
    total_log_lik = 0.0
    
    # Pre-calculate the log(2π) constant once
    LOG_2PI = np.log(2 * np.pi)
    use_x_aug = False
    use_metabolic_baseline = False

    with torch.no_grad():
        for batch in data_loader:
            # Assuming 'metabolic_baseline' is handled inside the model if present
            # We only need t, y, s_i, mask for the likelihood calculation itself
            if 'x_aug' in batch.keys():
                t, y, s_i, mask, _, t_i, x_aug = [d.to(device) for d in batch.values()]
                metabaseline = None
                use_x_aug=True
            elif 'metabolic_baseline' in batch.keys():
                t, y, s_i, _, mask, metabolic_baseline = [d.to(device) for d in batch.values()]
                use_metabolic_baseline = True
            else:
                t, y, s_i, _, mask = [d.to(device) for d in batch.values()]
            
            N, T = y.shape

            if use_x_aug:
                N = x_aug.shape[0]
                # coeffs = torchcde.natural_cubic_spline_coeffs(x_aug, t=t)
                coeffs = torchcde.hermite_cubic_coefficients_with_backward_differences(x_aug)
                X = torchcde.CubicSpline(coeffs)
            
                mu, V, Z, D = model(s_i, X, len(t), return_components=True)
            elif use_metabolic_baseline:
                mu, V, Z, D = model(s_i, t, metabolic_baseline=metabolic_baseline, return_components=True)
            else:
                mu, V, Z, D = model(s_i, t, return_components=True)

            # --- 1. Mask the Residual ---
            residual = (y - mu) * mask
            residual = residual.unsqueeze(-1)  # Shape: (N, T, 1)

            # --- 3. Mask the Full Covariance V ---
            mask_f = mask.float()
            mask_matrix = mask_f.unsqueeze(2) * mask_f.unsqueeze(1) # (N, T, T)
            V_masked = V * mask_matrix
            
            # Add a small identity for numerical stability, making the padded
            # block of the matrix invertible.
            eye = torch.eye(T, device=device).unsqueeze(0)
            V_masked = V_masked + 1e-6 * eye

            try:
                # --- 4. Compute NLL Components (Batched) ---
                # Batched Cholesky decomposition
                L = torch.linalg.cholesky(V_masked) # (N, T, T)

                # a) Log-determinant term
                # We only sum the log of the diagonal for the valid (unmasked) time points
                diag_L = torch.diagonal(L, dim1=-2, dim2=-1) # (N, T)
                log_det = 2 * torch.sum(torch.log(diag_L) * mask_f, dim=1) # (N,)

                # b) Mahalanobis term: (y-μ)'Σ⁻¹(y-μ)
                Linv_resid = torch.cholesky_solve(residual, L) # (N, T, 1)
                mahalanobis = torch.bmm(residual.transpose(1, 2), Linv_resid).squeeze() # (N,)

                # c) log(2π) term
                T_valid = mask.sum(dim=1).float() # (N,)
                log_2pi_term = T_valid * LOG_2PI
                
                # --- 5. Compute Final Log-Likelihood for the Batch ---
                log_prob_batch = -0.5 * (log_2pi_term + log_det + mahalanobis)
                
                total_log_lik += log_prob_batch.sum().item()

            except RuntimeError as e:
                # If Cholesky fails for any item in the batch, penalize the whole batch
                print(f"Batched Cholesky failed: {e}")
                total_log_lik += (-1e6 * N)

    return total_log_lik

def fitted_trajectory(model, data, device):
    """
    BLUP-style forecasting in fit mode.
    """
    model.eval()
    with torch.no_grad():
        if 'metabolic_baseline' in data.keys():
            t, y, s_i, mask, _, metabaseline = [d.to(device) for d in data.values()]
            pred_mean, V, Z, D = model(s_i, t, metabolic_baseline=metabaseline, return_components=True)
        
        elif 'x_aug' in data.keys():

            t = torch.tensor(data["t"], dtype=torch.float32).to(device).squeeze()
            y = torch.tensor(data["y"], dtype=torch.float32).to(device).squeeze()
            x_aug = torch.tensor(data['x_aug'], dtype=torch.float32).to(device)
            s_i = torch.tensor(data['s_i'], dtype=torch.float32).unsqueeze(0).to(device)
            mask = torch.tensor(data["target_mask"], dtype=torch.float32).to(device)

            # --- Add batch dimension for single subject ---
            # x_aug: (T, input_channels) → (1, T, input_channels)
            x_aug = x_aug.unsqueeze(0)
            mask = mask.unsqueeze(0)
            coeffs = torchcde.hermite_cubic_coefficients_with_backward_differences(x_aug)
            X = torchcde.CubicSpline(coeffs)
        
            pred_mean, V, Z, D = model(s_i, X, len(t), return_components=True)

        else:
            t, y, s_i, mask, _ = [d.to(device) for d in data.values()]
            pred_mean, V, Z, D = model(s_i, t, return_components=True)
        
        T = len(y)

        # --- MASK THE RESIDUAL ---
        # Ensure padded values do not contribute to the residual.
        residual = (y - pred_mean) * mask  # (N, T)
        residual = residual.unsqueeze(-1) # (N, T, 1)

        # --- COMPUTE BLUPs USING MASKED MATRICES ---
        # Invert the masked V matrix
        V_inv = inverse_variance(V, mask, T)          # (N, T, T)

        # Solve V_inv * r (padded entries in r are 0, so they won't contribute)
        V_inv_r = torch.matmul(V_inv, residual)         # (N, T, 1)

        # D @ Z^T @ V_inv @ r
        Z_T = Z.transpose(1, 2)                         # (N, q, T)
        DZ_T = torch.matmul(D.unsqueeze(0), Z_T)        # (N, q, T)
        b_hat = torch.matmul(DZ_T, V_inv_r).squeeze(-1) # (N, q)

        # Z @ b_hat
        blup_adjustment = torch.matmul(Z, b_hat.unsqueeze(-1)).squeeze() # (N, T)

        # Final adjusted predictions
        blup_adjusted = pred_mean + blup_adjustment 
        valid = mask.squeeze().bool()    # (N, T)
        t_out = t[valid].detach().cpu().numpy()
        blup_out = blup_adjusted[valid].detach().cpu().numpy()
        y_out = y[valid].detach().cpu().numpy()
        return t_out, blup_out, y_out


def calculate_sequential_blup_forecasting(model, patient_data, device, eps=1e-5):

    t_points = torch.tensor(patient_data["t"], dtype=torch.float32, device=device).squeeze()
    y = torch.tensor(patient_data["y"], dtype=torch.float32, device=device).squeeze()  # (T,)
    x_aug = torch.tensor(patient_data["x_aug"], dtype=torch.float32, device=device)   # (T,D)
    s_i = torch.tensor(patient_data["s_i"], dtype=torch.float32, device=device).unsqueeze(0)
    target_mask = torch.tensor(patient_data["target_mask"], dtype=torch.float32, device=device).squeeze()  # (T,)

    x_aug = x_aug.unsqueeze(0)  # (1,T,D)

    seq_preds, actual_y, pred_means, real_t_points = [], [], [], [0]

    model.eval()
    with torch.no_grad():
        T = t_points.numel()
        for j in range(T):
            if target_mask[j] == 0:
                continue

            if j == 0:
                # no past y -> just population prediction at j
                seq_preds.append(y[0])
                actual_y.append(y[0])
                continue

            # Build covariate path up to and including time j (so we can get mu_j, Z_j)
            x_hist = x_aug[:, :j+1, :]  # (1, j+1, D)
            coeffs = torchcde.hermite_cubic_coefficients_with_backward_differences(x_hist)
            X_hist = torchcde.CubicSpline(coeffs)

            pred_mean_hist, V_hist, Z_hist, D = model(s_i, X_hist, j+1, return_components=True)
            # pred_mean_hist: (j+1,), V_hist: (1, j+1, j+1), Z_hist: (1, j+1, q), D: (q,q)
            mu = pred_mean_hist      # (j+1,)
            V  = V_hist[0]              # (j+1,j+1)
            Z  = Z_hist[0]          # (j+1,q)

            # Store target/time
            real_t_points.append(t_points[j].item())
            actual_y.append(y[j].item())

            # Past observed points among 0..j-1
            past_valid = target_mask[:j].bool()
            if past_valid.sum() == 0:
                seq_preds.append(mu[j].item())
                pred_means.append(mu[j].item())
                continue

            y_obs  = y[:j][past_valid]                 # (Tobs,)
            mu_obs = mu[:j][past_valid]                # (Tobs,)
            Z_obs  = Z[:j, :][past_valid, :]           # (Tobs,q)
            V_obs  = V[:j, :j][past_valid][:, past_valid]  # (Tobs,Tobs)

            resid = (y_obs - mu_obs).unsqueeze(-1)     # (Tobs,1)

            # BLUP: b = D Z^T V^{-1} r  (V is already marginal)
            V_obs = V_obs + eps * torch.eye(V_obs.shape[0], device=device)
            V_inv_r = torch.linalg.solve(V_obs, resid)       # (Tobs,1)
            b_blup = (D @ Z_obs.T @ V_inv_r).squeeze(-1)      # (q,)

            # Predict at time j
            y_pred_j = mu[j] + (Z[j, :] @ b_blup)
            seq_preds.append(y_pred_j.item())
            pred_means.append(mu[j].item())
    return np.array(real_t_points), np.array(seq_preds), np.array(actual_y), np.array(pred_means)

def filter_patient_with_id(id, dataset):
    for patient in dataset:
        if id == np.unique(patient['patient_id']):
            return patient
