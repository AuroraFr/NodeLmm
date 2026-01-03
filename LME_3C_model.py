import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset
import torchcde
import pandas as pd

def masked_NLL(predicted_mean, targets, V, mask):
    N, T = predicted_mean.shape

    # --- Mask residuals ---
    residual = (targets - predicted_mean) * mask  # (N, T)
    residual = residual.unsqueeze(-1)  # (N, T, 1)

    # --- Mask covariance matrix V ---
    # We zero-out rows and columns corresponding to padded time points
    mask_f = mask.float()
    mask_matrix = mask_f.unsqueeze(2) * mask_f.unsqueeze(1)  # (N, T, T)

    V_masked = V * mask_matrix  # (N, T, T)

    # Add small noise for numerical stability
    eye = torch.eye(T, device=V.device).unsqueeze(0).expand(N, T, T)
    V_masked = V_masked + 1e-4 * eye

    # --- Cholesky decomposition ---
    try:
        L = torch.linalg.cholesky(V_masked)  # (N, T, T)
    except RuntimeError as e:
        print("Cholesky failed in batch.")
        return torch.tensor(1e6, device=V.device, requires_grad=True)

    # --- Log determinant ---
    diag_L = torch.diagonal(L, dim1=-2, dim2=-1)  # (N, T)
    logdet_V = 2 * torch.sum(torch.log(diag_L + 1e-6) * mask, dim=1)  # (N,)

    # --- Quadratic term ---
    V_inv_residual = torch.cholesky_solve(residual, L)  # (N, T, 1)
    quad_term = torch.bmm(residual.transpose(1, 2), V_inv_residual).squeeze(-1).squeeze(-1)  # (N,)

    # --- Log(2π) term ---
    T_valid = mask.sum(dim=1).float()  # (N,)
    log_2pi_term = T_valid * np.log(2 * np.pi)

    loss = 0.5 * (logdet_V + quad_term + log_2pi_term)
    return loss.mean()

class Decoder_with_static(nn.Module):
    def __init__(self, latent_dim, static_dim, response_dim, device, fullG=False):
        super().__init__()
        self.device = device

        # Non-linear decoder for fixed effects
        # self.fixed_effects_decoder = nn.Sequential(
        #     nn.Linear(latent_dim, latent_dim * 2),
        #     nn.ReLU(),
        #     nn.Dropout(p=0.3),
        #     nn.Linear(latent_dim * 2, response_dim)
        # )

        self.decoder_gru = nn.GRU(
            input_size=latent_dim,
            hidden_size=latent_dim*2,
            batch_first=True,
            num_layers=2
        )
        self.output_layer = nn.Linear(latent_dim *2, response_dim)
        self.fullG = fullG

        # A single linear layer decoder
        # self.fixed_effects_decoder = nn.Linear(latent_dim, response_dim)

        if fullG:
            # Random effects: full covariance via Cholesky
            self.num_random_effects = latent_dim + 1  # intercept + latent dims
            L_init = 0.01 * torch.randn(self.num_random_effects, self.num_random_effects)
            self.L = nn.Parameter(torch.tril(L_init))  # lower-triangular Cholesky factor
        else:
            num_random_effects = latent_dim + 1
            self.log_std_devs = nn.Parameter(torch.randn(num_random_effects))
        
        self.log_residual_var = nn.Parameter(torch.tensor(1.0))
        # self._apply_weights_init()

    def _apply_weights_init(self):
        for layer in self.fixed_effects_decoder.modules():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, z_t, s_i, CDE=True, return_components=False):
        N, T, _ = z_t.shape

        if CDE and T > 1:
            z_t_norm = (z_t - z_t.mean(dim=1, keepdim=True)) / (z_t.std(dim=1, keepdim=True) + 1e-8)
            # z_t_static = torch.cat([z_t_norm, s_i.unsqueeze(1).expand(-1, T, -1)], dim=-1)
            # predicted_mean = self.fixed_effects_decoder(z_t_norm)
            h_out, _ = self.decoder_gru(z_t_norm)
        else:
            z_t_static = torch.cat([z_t, s_i.unsqueeze(1).expand(-1, T, -1)], dim=-1)
            # predicted_mean = self.fixed_effects_decoder(z_t)
            h_out, _ = self.decoder_gru(z_t)

        predicted_mean = self.output_layer(h_out).squeeze()      # (N, T, response_dim)

        random_intercepts = torch.ones((N, T, 1), device=self.device)
        Z = torch.cat([random_intercepts, z_t], dim=-1) # Use original z_t for random effects

        if not self.fullG:
            random_effect_vars = torch.exp(self.log_std_devs)**2
            D = torch.diag(random_effect_vars)
        else:
            # Build full random effect covariance G = L L^T
            D = self.L @ self.L.T + 1e-4 * torch.eye(self.num_random_effects, device=self.device)

        V_random = (Z @ D) @ Z.permute(0, 2, 1)

        residual_variance = torch.exp(self.log_residual_var)
        R = residual_variance * torch.eye(T, device=self.device).unsqueeze(0).repeat(N, 1, 1)

        V = V_random + R #marginal variance

        if return_components:
            return predicted_mean, V, Z, D
        else:
            return predicted_mean, V

class ODEFunc(nn.Module):
    def __init__(self, hidden_dim, static_dim):
        super(ODEFunc, self).__init__()

        # The sequential model defines the dynamics dy/dt = f(t, y)
        self.dynamics_func = nn.Sequential(
            nn.Linear(hidden_dim + static_dim, 16),
            nn.Tanh(),
            nn.Linear(16, hidden_dim)
        )

    def forward(self, t, state, static_features):
        combined_input = torch.cat([state, static_features], dim=1)
        return self.dynamics_func(combined_input)

class VectorField_with_static(nn.Module):
    def __init__(self, hidden_dim, input_dim, static_dim):
        super().__init__()

        # hidden_dim is the full augmented dimension (dynamic_dim + static_dim)
        self.hidden_dim_augmented = hidden_dim
        self.static_dim = static_dim
        self.hidden_dim_dynamic = hidden_dim - static_dim

        # A non-linear network that takes the full augmented state
        self.network = nn.Sequential(
            nn.Linear(self.hidden_dim_augmented, (self.hidden_dim_augmented) * 2),
            nn.ReLU(),
            nn.Dropout(p=0),
            nn.Linear((self.hidden_dim_augmented) * 2, self.hidden_dim_dynamic * input_dim)
        )
        self._apply_weights_init()

    def _apply_weights_init(self):
        for layer in self.network.modules():
            if isinstance(layer, nn.Linear):
                nn.init.uniform_(layer.weight, -0.01, 0.01)
                nn.init.zeros_(layer.bias)

    def forward(self, t, z):
        # z is the augmented state
        dz_dynamic_dt_times_dX = self.network(z)
        dz_dynamic_dt = dz_dynamic_dt_times_dX.view(z.size(0), self.hidden_dim_dynamic, -1)
        dz_static_dt = torch.zeros(z.size(0), self.static_dim, dz_dynamic_dt.shape[-1], device=z.device)
        dz_augmented_dt = torch.cat([dz_dynamic_dt, dz_static_dt], dim=1)
        return dz_augmented_dt

    
class ODENet(nn.Module):
    def __init__(self, scripted_ode_func, static_feature_dim, hidden_dim, device, fullG=False):
        super(ODENet, self).__init__()

        # 1. The Encoder: Maps static features to the initial hidden state z0
        self.encoder = nn.Sequential(
            nn.Linear(static_feature_dim, 16),
            nn.ReLU(),
            nn.Linear(16, hidden_dim) # Output is z0
        )
        
        # 2. The Dynamics Function: The core of the Neural ODE
        # This defines dz/dt = f(t, z)
        self.dynamics_func = scripted_ode_func
        self.fullG = fullG

        # 3. The Decoder: Maps the final hidden state to a prediction
        self.decoder = Decoder_with_static(
            latent_dim=hidden_dim,
            response_dim=1,
            device=device,
            fullG=fullG)

        self._apply_weights_init()
    
    def _apply_weights_init(self):
        for layer in self.encoder.modules():
             if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, s_i, t, CDE = False, metabolic_baseline=None, return_components=False):

        if isinstance(metabolic_baseline, torch.Tensor):
            s_i = torch.cat([s_i, metabolic_baseline], dim=1)

        # Use the encoder to get the initial state z0 from static features
        z0 = self.encoder(s_i)
        
        # Use an ODE solver to integrate the dynamics from z0 over time t
        # This requires a library like torchdiffeq
        from torchdiffeq import odeint
        func = lambda t, z: self.dynamics_func(t, z, s_i)
        z_t = odeint(func, z0, t, method='rk4')
        
        # The solver returns the hidden state at each time point
        # Shape: (num_time_points, batch_size, hidden_dim)
        # let's reorder to (batch, time, channels)
        z_t = z_t.permute(1, 0, 2)
        
        # Decode the trajectory to get the final output
        return self.decoder(z_t, CDE=CDE, return_components=return_components)

class CDEModel(nn.Module):
    def __init__(self, input_dim, static_dim, hidden_dim, device, fullG=False):
        super().__init__()

        # self.encoder = nn.GRU(input_size=input_dim + static_dim,
        #               hidden_size=hidden_dim,
        #               batch_first=True)

        self.encoder = nn.Sequential(
            nn.Linear(input_dim + static_dim, 16),
            nn.ReLU(),
            nn.Linear(16, hidden_dim) # Output is z0
        )

        self.func = VectorField_with_static(
            hidden_dim=(hidden_dim + static_dim),
            input_dim=input_dim,
            static_dim=static_dim

        )
        self.fullG = fullG

        # if not fullG:
        self.decoder = Decoder_with_static(
            latent_dim=hidden_dim,
            static_dim=static_dim,
            response_dim=1,
            device=device,
            fullG=fullG
        )
        self._apply_weights_init()

    def _apply_weights_init(self):
        for layer in self.encoder.modules():
             if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, s_i, X, N_timepoints, return_components=False):
        # We also need to add the static features at every time step
        # To create the input for the RNN, we evaluate the spline at all time points
        t = torch.linspace(X.interval[0], X.interval[-1], N_timepoints)
        # rnn_input_x = X.evaluate(t)
        # s_i_repeated = s_i.unsqueeze(1).repeat(1, rnn_input_x.size(1), 1)
        # rnn_input = torch.cat([rnn_input_x, s_i_repeated], dim=-1)
        # _, z0 = self.encoder(rnn_input)
        # z0 = z0.squeeze(0) # Remove the num_layers dimension -> (batch_size, hidden_size)

        rnn_input_x = X.evaluate(t[0])
        rnn_input = torch.cat([rnn_input_x, s_i], dim=-1)
        z0 = self.encoder(rnn_input)
        if N_timepoints > 1:
            z0_augmented = torch.cat([z0, s_i], dim=-1)
            z_t_augmented = torchcde.cdeint(X=X, z0=z0_augmented, func=self.func, t=t, method='rk4', adjoint=False)

            dynamic_hidden_dim = z0.shape[-1]
            dynamic_z_t = z_t_augmented[..., :dynamic_hidden_dim]
            return self.decoder(dynamic_z_t, s_i, return_components=return_components)
        else:
            z0 = z0.unsqueeze(1)
            return self.decoder(z0, s_i, return_components=return_components)
    
# --- DATA HANDLING ---
class PatientDataset(Dataset):
    def __init__(self, data_list):
        self.data = data_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def process_data(df, id_col, time_varying_features, static_features, target_col, with_only_static_features=False, scaler=None, metabolic_baseline=True):
    
    all_patient_data = []
    # df['GLUC'] = np.log(df['GLUC'])
    # df['HDL'] = np.log(df['HDL'])
    df_encoded = pd.get_dummies(df, columns=['DIPNIV','SEX'], dtype=float)
    metabolic_baseline_features = ["HDL0", "GLUC0", "BMI0", "CHOL0"]
    expected_times = np.array([0, 2, 4, 7, 10, 12])

    # map observed time -> nearest slot
    def map_to_expected(t):
        return expected_times[np.argmin(np.abs(expected_times - t))]
    
    def assign_time_slots_one_patient(g: pd.DataFrame) -> pd.DataFrame:

        tvals = g["time"].to_numpy()

        # nearest slot index for each time
        idx0 = np.array([np.argmin(np.abs(expected_times - t)) for t in tvals], dtype=int)

        used = set()
        final_idx = []

        for idx in idx0:
            j = idx
            # push forward until a free slot is found
            while j in used and j < len(expected_times) - 1:
                j += 1

            # if running out of slots, keep the last slot (or set NaN instead)
            if j in used:              # all remaining slots are already used
                j = len(expected_times) - 1
                # alternatively: final_idx.append(np.nan); continue

            used.add(j)
            final_idx.append(j)

        final_idx = np.array(final_idx, dtype=int)
        g["time_slot_idx"] = final_idx
        g["time_slot"] = expected_times[final_idx]
        return g
    
    for _, group in df_encoded.groupby(id_col):
        
        patient_df = group.copy().sort_values(by="SUIVI")
        
        TO_visit_data = patient_df.sort_values('SUIVI').iloc[0]
        patient_df["time"] = (patient_df["SUIVI"] - patient_df["SUIVI"].min()).dt.total_seconds() / (60 * 60 * 24 * 365)
        patient_df["HDL0"] = TO_visit_data["HDL"] 
        patient_df["GLUC0"] = TO_visit_data["GLUC"]
        patient_df["BMI0"] = TO_visit_data["BMI"]
        patient_df["CHOL0"] = TO_visit_data["CHOL"]

        patient_df = assign_time_slots_one_patient(patient_df)
        y = np.full(len(expected_times), fill_value=0.0)   # or np.nan
        target_mask = np.zeros(len(expected_times), dtype=np.float32)
        padded_time = np.zeros(len(expected_times), dtype=np.float32)
        
        for i, row in patient_df.iterrows():
            slot = np.where(row["time_slot"] == expected_times)[0]
            y[slot] = row[target_col]
            target_mask[slot] = 1.0
            padded_time[slot] = row['time']
        
        # masks_target.append(torch.tensor(m, dtype=torch.float32))
        # targets.append(torch.tensor(y, dtype=torch.float32))
        padded_time = np.where((padded_time == 0) & (np.arange(len(padded_time)) != 0),np.nan,padded_time)
        filled_time_values = torch.tensor(pd.Series(padded_time).ffill().to_numpy())

        if not with_only_static_features:

            padded_dynamic_features = np.zeros((len(expected_times), len(time_varying_features)), dtype=np.float32)
            
            for i, row in patient_df.iterrows():
                for feature_i, feature in enumerate(time_varying_features):
                    slot = np.where(row["time_slot"] == expected_times)[0]
                    padded_dynamic_features[slot, feature_i] = row[feature]
            
            padded_dynamic_features = torch.tensor(padded_dynamic_features)
            padded_dynamic_features[padded_dynamic_features == 0] = np.nan
            mask = (~torch.isnan(padded_dynamic_features)).cumsum(dim=0)

            if scaler:
                x_scaled_filled = scaler.transform(padded_dynamic_features)
                x_scaled = np.where(mask, x_scaled_filled, np.nan).astype(np.float32)

                x_filled = pd.DataFrame(x_scaled).ffill().values
                # delta_t = np.zeros_like(x_filled)
            else:
                x_filled = pd.DataFrame(padded_dynamic_features).values
                # delta_t = np.zeros_like(x_filled)
            
            # # Compute delta_t
            # delta_t = np.zeros_like(x_filled)
            # for d in range(x_filled.shape[1]):
            #     last_obs_time = 0
            #     for t in range(x_filled.shape[0]):
            #         current_time = expected_times[t]
            #         if mask[t, d]:
            #             delta_t[t, d] = current_time - last_obs_time
            #             last_obs_time = current_time
            #         else:
            #             delta_t[t, d] = current_time - last_obs_time

            # Augment the data with values, mask, delta_t
            filled_time_values = torch.tensor(filled_time_values, dtype=torch.float32).unsqueeze(1)
            x_filled =  torch.tensor(x_filled, dtype=torch.float32)

            def columnwise_tail_forward_fill(x):
                x = x.clone()
                for j in range(x.shape[1]):
                    col = x[:, j]
                    isnan = torch.isnan(col)
                    if not torch.any(~isnan):
                        continue  # skip if entire column is NaN
                    last_valid_idx = torch.where(~isnan)[0][-1]
                    last_val = col[last_valid_idx]
                    # Fill after last valid with last_val
                    tail_mask = torch.arange(x.shape[0], device=x.device) > last_valid_idx
                    col[tail_mask & isnan] = last_val
                    x[:, j] = col
                return x

            x_filled = columnwise_tail_forward_fill(x_filled)
            x_augmented = torch.cat([filled_time_values, x_filled, mask], dim=1)

            all_patient_data.append({
                'x_aug': torch.tensor(x_augmented, dtype=torch.float32),
                't': torch.tensor(torch.tensor(filled_time_values, dtype=torch.float32)),
                'y': torch.tensor(y, dtype=torch.float32),
                's_i': torch.tensor(patient_df[static_features].iloc[0].values.astype(np.float32), dtype=torch.float32),
                'target_mask': torch.tensor(target_mask, dtype=torch.float32),
                'patient_id': torch.tensor(np.unique(patient_df[id_col])[0])
            })
        elif metabolic_baseline and with_only_static_features:
            all_patient_data.append({
                't': torch.tensor(filled_time_values, dtype=torch.float32),
                'y': torch.tensor(y, dtype=torch.float32),
                's_i': torch.tensor(patient_df[static_features].iloc[0].values.astype(np.float32), dtype=torch.float32),
                'target_mask': torch.tensor(target_mask, dtype=torch.float32),
                'metabolic_baseline': torch.tensor(patient_df[metabolic_baseline_features].iloc[0].values.astype(np.float32), dtype=torch.float32),
                'patient_id': torch.tensor(np.unique(patient_df[id_col])[0])
            })
        else:
            all_patient_data.append({
                't': torch.tensor(filled_time_values, dtype=torch.float32),
                'y': torch.tensor(y, dtype=torch.float32),
                's_i': torch.tensor(patient_df[static_features].iloc[0].values.astype(np.float32), dtype=torch.float32),
                'target_mask': torch.tensor(target_mask, dtype=torch.float32),
                'patient_id': torch.tensor(np.unique(patient_df[id_col])[0])
            })
        
    return all_patient_data

def collate_fn(batch):
    """
    Collate Function
    - It now finds the time vector 't' from the longest sequence in the batch
      to use for the spline interpolation.
    """
    # Find the patient with the longest sequence
    only_static_feature = True
    use_aug = 'x_aug' in batch[0]
    expected_times = torch.Tensor(np.array([0, 2, 4, 7, 10, 12]))

    if use_aug:
        only_static_feature = False
        x_dim = batch[0]['x_aug'].shape[1]

    if 'metabolic_baseline' in batch[0].keys():
        metabolic_baseline_dim = batch[0]['metabolic_baseline'].shape[0]
    
    if 'metabolic_baseline' in batch[0].keys():
        metabolic_i_batch = torch.zeros(len(batch), metabolic_baseline_dim)

    s_i = torch.stack([p['s_i'] for p in batch])
    mask = torch.stack([p['target_mask'] for p in batch])
    t_i = torch.stack([p['t'] for p in batch])
    y = torch.stack([p['y'] for p in batch])

    if only_static_feature and 'metabolic_baseline' in batch[0].keys():
        metabolic_i_batch = torch.stack([p['metabolic_baseline'] for p in batch])

    # Return collated batch
    output = {
        't': expected_times,
        'y': y,                 
        's_i': s_i ,
        'mask': mask,
        'id':   torch.stack([p['patient_id'] for p in batch])
    }

    if 'metabolic_baseline' in batch[0].keys():
        output['metabolic_baseline'] = metabolic_i_batch
    if use_aug:
        output['t_i'] = t_i
        output['x_aug'] = torch.stack([p['x_aug'] for p in batch])
    
    return output
