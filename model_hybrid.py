from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

try:
    import torchcde
except ImportError as e:
    raise ImportError(
        "torchcde is required. Install with: pip install torchcde"
    ) from e

class MLP(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        depth: int = 2,
        dropout: float = 0.0,
        activation: nn.Module = nn.ReLU(),
    ):
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be >= 1")

        layers = []
        d = in_dim
        for _ in range(depth - 1):
            layers.append(nn.Linear(d, hidden_dim))
            layers.append(activation)
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            d = hidden_dim
        layers.append(nn.Linear(d, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class BaselineICEncoder(nn.Module):
    """
    Encode initial hidden state z0 using:
      - baseline static covariates s (N, Cs)  [optional]
      - first observation (time + x0)         (N, Cin) where Cin = 1 + Cx
    """

    def __init__(
        self,
        input_dim_x0: int,          # Cin = Cx
        static_dim: int,            # Cs (0 if none)
        hidden_dim: int,            # latent dimension Hz
        mlp_hidden: int = 128,
        depth: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.static_dim = int(static_dim)

        self.mlp = MLP(input_dim_x0 + static_dim, mlp_hidden, hidden_dim, depth=depth, dropout=dropout)

    def forward(self, x0: torch.Tensor) -> torch.Tensor:
 
        z = self.mlp(x0)
        return z

class CDEFunc(nn.Module):
    def __init__(self, hidden_channels, input_channels,
                 covariate_dim, static_dim,
                 mlp_hidden=64, depth=2, dropout=0.0):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.input_channels = input_channels

        # Input: z(t) + x(t) + static covariates
        self.net = MLP(
            in_dim=hidden_channels + covariate_dim + static_dim + 1,
            hidden_dim=mlp_hidden,
            out_dim=hidden_channels * input_channels,
            depth=depth, dropout=dropout, activation=nn.ReLU(),
        )
        self.X_spline = None
        self.static = None

    def set_context(self, X, static):
        """Store spline and static covariates before cdeint."""
        self.X_spline = X
        self.static = static      # (N, static_dim)

    def forward(self, t, z):
        x_t = self.X_spline.evaluate(t)[:, :-1]                    # (N, input_channels)
        inp = torch.cat([z, x_t, self.static], dim=-1)     # (N, H + Cx + Cs)
        out = self.net(inp)
        out = torch.tanh(out)
        return out.view(z.size(0), self.hidden_channels, self.input_channels)

def natural_spline_basis(t, knots, boundary_knots, intercept=False):
    """
    Replicate R's splines::ns() exactly using B-spline basis + natural constraints.
    
    R's algorithm:
      1. Build augmented knot vector with boundaries repeated (degree+1) times
      2. Evaluate cubic B-spline basis
      3. Compute 2nd derivative constraint matrix at boundaries
      4. QR-project out the constrained directions
      5. Drop first column if intercept=False (R default)
    
    Parameters:
        t:               (N, T) or (T,) tensor of time values
        knots:           array of INTERNAL knots
        boundary_knots:  array [t_min, t_max]
        intercept:       if False (default), drop first column (matches R's ns())
    
    Returns:
        basis: (..., df) tensor where df = len(knots)+1 if intercept=False,
               len(knots)+2 if intercept=True
    """
    from scipy.interpolate import BSpline

    t_np = t.detach().cpu().numpy()
    original_shape = t_np.shape
    x = t_np.ravel()
    
    degree = 3
    knots = np.asarray(knots)
    boundary_knots = np.asarray(boundary_knots)
    
    # Step 1: Augmented knot vector (same as R)
    Aknots = np.sort(np.concatenate([
        np.repeat(boundary_knots[0], degree + 1),
        knots,
        np.repeat(boundary_knots[1], degree + 1),
    ]))
    n_basis = len(Aknots) - degree - 1
    
    # Step 2: Evaluate B-spline basis at data points
    basis = np.zeros((len(x), n_basis))
    for j in range(n_basis):
        c = np.zeros(n_basis)
        c[j] = 1.0
        spl = BSpline(Aknots, c, degree, extrapolate=False)
        vals = spl(x)
        vals[np.isnan(vals)] = 0.0
        basis[:, j] = vals
    
    # Step 3: 2nd derivative constraint at boundary knots
    eps = 1e-6
    const = np.zeros((2, n_basis))
    for j in range(n_basis):
        c = np.zeros(n_basis)
        c[j] = 1.0
        spl = BSpline(Aknots, c, degree, extrapolate=True)
        dspl = spl.derivative(2)
        const[0, j] = dspl(boundary_knots[0] + eps)
        const[1, j] = dspl(boundary_knots[1] - eps)
    
    # Step 4: QR on constraint^T, rotate basis, drop first 2 cols
    Q, _ = np.linalg.qr(const.T, mode='complete')
    basis = basis @ Q[:, 2:]
    
    # Step 5: Drop first column if no intercept (R's default)
    if not intercept:
        basis = basis[:, 1:]
    
    # Reshape back to original shape + (df,)
    df = basis.shape[-1]
    basis = basis.reshape(*original_shape, df)
    
    return torch.tensor(basis, dtype=t.dtype, device=t.device)


class Decoder(nn.Module):
    def __init__(self, latent_dim, p, q=3,
                 fullD=True, jitter=1e-6, rho_hidden=64, D_diag_min=0.1,
                 n_static=3, n_tv=2,
                 fe_spline_knots=None, fe_spline_boundary=None,
                 re_spline_knots=None, re_spline_boundary=None,
                 interaction_pairs=None,
                 precomputed_splines=False):
        """
        Parametric fixed effects (matching HLME structure):
            mu_param = beta_0 + beta_ns1*ns1(t) + ... + beta_nsK*nsK(t)
                     + beta_tv * x(t) + beta_static * s
                     + beta_int * (x_tv_j * static_k) for each interaction pair

        Neural fixed effects (residual):
            mu_neural = h_perp(z(t))   projected ⊥ col(W)

        Random effects:
            Z = [1, rs1(t), rs2(t)]    spline basis with q columns

        Args:
            fe_spline_knots:    internal knots for FE time spline (len = df - 1)
            fe_spline_boundary: boundary knots for FE time spline [t_min, t_max]
            re_spline_knots:    internal knots for RE time spline
            re_spline_boundary: boundary knots for RE time spline
            interaction_pairs:  list of (tv_idx, static_idx) tuples for interaction terms
                                e.g. [(1, 1)] for BMI_t × AGEc if BMI is x_pad[:,1]
                                and AGEc is static[:,1]. None = no interactions.
            precomputed_splines: if True, x_pad contains:
                                 [tv_covs (n_tv), fe_splines (fe_spline_df), re_splines (q-1)]
                                 and the decoder slices them directly instead of recomputing.
        """
        super().__init__()
        self.p, self.q = p, q
        self.fullD = fullD
        self.jitter = jitter
        self.D_diag_min = D_diag_min
        self.n_static = n_static
        self.n_tv = n_tv
        self.precomputed_splines = precomputed_splines

        # --- Fixed-effect time spline ---
        if precomputed_splines:
            # Spline columns come from x_pad; knots only needed for df computation
            fe_spline_knots = np.asarray(fe_spline_knots) if fe_spline_knots is not None else np.zeros(0)
            self.fe_spline_df = len(fe_spline_knots) + 1
            self.register_buffer('fe_spline_knots',
                                 torch.tensor(fe_spline_knots, dtype=torch.float32))
            _fe_bnd = np.asarray(fe_spline_boundary) if fe_spline_boundary is not None else np.array([0.0, 1.0])
            self.register_buffer('fe_spline_boundary',
                                 torch.tensor(_fe_bnd, dtype=torch.float32))
            re_spline_knots = np.asarray(re_spline_knots) if re_spline_knots is not None else np.zeros(0)
            self.register_buffer('re_spline_knots',
                                 torch.tensor(re_spline_knots, dtype=torch.float32))
            _re_bnd = np.asarray(re_spline_boundary) if re_spline_boundary is not None else np.array([0.0, 1.0])
            self.register_buffer('re_spline_boundary',
                                 torch.tensor(_re_bnd, dtype=torch.float32))
            # Verify q matches
            re_q = 1 + len(re_spline_knots) + 1
            if re_q != q:
                raise ValueError(
                    f"q={q} doesn't match re_spline_knots length {len(re_spline_knots)} "
                    f"→ expected q={re_q}."
                )
        else:
            # Compute spline basis from knots at runtime
            if fe_spline_knots is None or fe_spline_boundary is None:
                raise ValueError(
                    "fe_spline_knots and fe_spline_boundary must be computed from "
                    "the observed times in the data and passed explicitly."
                )
            fe_spline_knots = np.asarray(fe_spline_knots)
            self.fe_spline_df = len(fe_spline_knots) + 1

            self.register_buffer('fe_spline_knots',
                                 torch.tensor(fe_spline_knots, dtype=torch.float32))
            self.register_buffer('fe_spline_boundary',
                                 torch.tensor(np.asarray(fe_spline_boundary), dtype=torch.float32))

            if re_spline_knots is None or re_spline_boundary is None:
                raise ValueError(
                    "re_spline_knots and re_spline_boundary must be computed from "
                    "the observed times in the data and passed explicitly."
                )
            re_spline_knots = np.asarray(re_spline_knots)
            re_q = 1 + len(re_spline_knots) + 1
            if re_q != q:
                raise ValueError(
                    f"q={q} doesn't match re_spline_knots: {len(re_spline_knots)} internal knots "
                    f"→ expected q={re_q}."
                )
            self.register_buffer('re_spline_knots',
                                 torch.tensor(re_spline_knots, dtype=torch.float32))
            self.register_buffer('re_spline_boundary',
                                 torch.tensor(np.asarray(re_spline_boundary), dtype=torch.float32))

        # --- Interactions (e.g. BMI×AGE) ---
        self.interaction_pairs = interaction_pairs or []
        n_interactions = len(self.interaction_pairs)

        # --- Parametric beta: computed analytically via GLS, NOT an nn.Parameter ---
        #   dim = 1 + fe_spline_df + n_tv + n_static + n_interactions
        self.n_W = 1 + self.fe_spline_df + n_tv + n_static + n_interactions
        # Store last computed beta for prediction mode (when Y is not available)
        self.register_buffer('_last_beta', torch.zeros(self.n_W))

        # --- Neural fixed effects ---
        self.rho_net = MLP(latent_dim, rho_hidden, p, depth=2, dropout=0.0)
        self.rho_norm = nn.LayerNorm(p)
        self.beta_neural = nn.Parameter(0.1 * torch.randn(p))

        # --- D and sigma2 ---
        if fullD:
            L_init = torch.eye(q) * 0.7
            L_init[0, 0] = 1.0
            self.L_unconstrained = nn.Parameter(L_init)
            self.log_std = None
        else:
            self.log_std = nn.Parameter(torch.zeros(q))
            self.L_unconstrained = None

        self.log_residual_var = nn.Parameter(torch.tensor(1.0))

    def _build_D(self, device, dtype):
        if self.fullD:
            L = torch.tril(self.L_unconstrained).to(device=device, dtype=dtype)
            diag = torch.diagonal(L, 0)
            diag_pos = F.softplus(diag) + self.D_diag_min
            L = L - torch.diag(torch.diagonal(L, 0)) + torch.diag(diag_pos)
            return L @ L.t()
        else:
            std = torch.exp(self.log_std).to(device=device, dtype=dtype)
            std = torch.clamp(std, min=self.D_diag_min ** 0.5)
            return torch.diag(std * std)

    def _build_Z(self, t_pad, x_pad=None):
        """
        Random-effect design matrix: Z = [1, rs1(t), rs2(t)]
        If precomputed_splines: slices RE spline columns from x_pad.
        Otherwise: computes from RE knots.
        """
        N, T = t_pad.shape

        if self.precomputed_splines and x_pad is not None:
            # x_pad layout: [tv_covs(n_tv), fe_splines(fe_spline_df), re_splines(q-1)]
            re_start = self.n_tv + self.fe_spline_df
            spline_cols = x_pad[:, :, re_start:]             # (N, T, q-1)
        else:
            knots = self.re_spline_knots.cpu().numpy()
            boundary = self.re_spline_boundary.cpu().numpy()
            spline_cols = natural_spline_basis(t_pad, knots, boundary)  # (N, T, q-1)

        ones = torch.ones(N, T, 1, device=t_pad.device, dtype=t_pad.dtype)
        Z = torch.cat([ones, spline_cols], dim=-1)  # (N, T, q)
        return Z

    def _build_W(self, t_pad, x_pad, static):
        """
        Build full parametric design matrix matching HLME structure:
            W = [1, ns1(t), ns2(t), ns3(t), GLUC(t), BMI(t), SEX, AGE, DIPNIV2, DIPNIV3, BMI×AGE, ...]

        If precomputed_splines: FE spline columns sliced from x_pad.
        Otherwise: computed from FE knots.

        Returns W: (N, T, n_W)
        """
        N, T = t_pad.shape
        device, dtype = t_pad.device, t_pad.dtype

        # Intercept
        ones = torch.ones(N, T, 1, device=device, dtype=dtype)

        # Time-varying covariates (first n_tv columns of x_pad)
        tv_covs = x_pad[:, :, :self.n_tv]                   # (N, T, n_tv)

        # Fixed-effect time spline basis
        if self.precomputed_splines:
            fe_spline_cols = x_pad[:, :, self.n_tv : self.n_tv + self.fe_spline_df]  # (N, T, fe_spline_df)
        else:
            knots = self.fe_spline_knots.cpu().numpy()
            boundary = self.fe_spline_boundary.cpu().numpy()
            fe_spline_cols = natural_spline_basis(t_pad, knots, boundary)  # (N, T, fe_spline_df)

        # Static covariates expanded to (N, T, n_static)
        static_exp = static.unsqueeze(1).expand(N, T, -1)

        # W = [1, ns1..nsK, tv_covs, static]
        parts = [ones, fe_spline_cols, tv_covs, static_exp]

        # Interaction columns: tv_covs[:,:,tv_idx] * static[:,static_idx]
        for tv_idx, static_idx in self.interaction_pairs:
            inter = tv_covs[:, :, tv_idx] * static[:, static_idx].unsqueeze(1)  # (N, T)
            parts.append(inter.unsqueeze(-1))                                    # (N, T, 1)

        W = torch.cat(parts, dim=-1)  # (N, T, n_W)
        return W

    def _compute_analytical_beta(self, W, h, Y, V, obs_mask):
        """
        Accumulate GLS sufficient statistics (AtA, Atb) for this batch.
        Call solve_beta() after processing all batches to get the full-data estimate.

        Uses Cholesky: L_i = chol(V_i), A_i = L⁻¹W_i, b_i = L⁻¹(Y_i - h_i)
        """
        N, T, p = W.shape
        device, dtype = W.device, W.dtype

        AtA = torch.zeros(p, p, device=device, dtype=dtype)
        Atb = torch.zeros(p, device=device, dtype=dtype)

        for i in range(N):
            obs_idx = obs_mask[i].bool()
            n_i = obs_idx.sum()
            if n_i < 1:
                continue

            W_i = W[i, obs_idx]                     # (n_i, p)
            r_i = Y[i, obs_idx] - h[i, obs_idx]     # (n_i,)

            V_i = V[i][obs_idx][:, obs_idx]          # (n_i, n_i)

            L_i = torch.linalg.cholesky(V_i)
            A_i = torch.linalg.solve_triangular(L_i, W_i, upper=False)
            b_i = torch.linalg.solve_triangular(
                L_i, r_i.unsqueeze(-1), upper=False
            ).squeeze(-1)

            AtA += A_i.t() @ A_i
            Atb += A_i.t() @ b_i

        return AtA.detach(), Atb.detach()

    def solve_beta(self, AtA, Atb):
        """Solve for beta from accumulated sufficient statistics."""
        p = AtA.shape[0]
        AtA = AtA + 1e-6 * torch.eye(p, device=AtA.device, dtype=AtA.dtype)
        beta = torch.linalg.solve(AtA, Atb)
        self._last_beta.copy_(beta)
        return beta

    def forward(self, z_t, t_pad, x_pad, static, y_pad=None, obs_mask=None,
                return_components=False):
        """
        Args:
            z_t:      (N, T, H)  latent states from CDE
            t_pad:    (N, T)     padded observation times
            x_pad:    (N, T, Cx) padded covariates (may include spline cols)
            static:   (N, Cs)    static covariates
            y_pad:    (N, T)     padded outcomes — needed for AtA/Atb accumulation
            obs_mask: (N, T)     binary mask (1=observed, 0=padded)

        When y_pad is provided: accumulates AtA/Atb (returned as extra outputs)
            and uses _last_beta (from previous epoch) for mu.
        When y_pad is None: uses stored _last_beta for prediction.
        """
        N, T, H = z_t.shape
        device, dtype = z_t.device, z_t.dtype

        # 1. Parametric design matrix
        W = self._build_W(t_pad, x_pad, static)                # (N, T, n_W)

        # 2. Neural flexible fixed effects (raw, no projection)
        rho = self.rho_net(z_t)                                 # (N, T, p)
        rho = self.rho_norm(rho)
        h = (rho * self.beta_neural).sum(dim=-1)               # (N, T)

        # 3. Random effects
        Z = self._build_Z(t_pad, x_pad)                        # (N, T, q)
        if obs_mask is not None:
            Z = Z * obs_mask.unsqueeze(-1)

        D = self._build_D(device, dtype)
        V_re = (Z @ D) @ Z.transpose(1, 2)

        sig2 = torch.exp(self.log_residual_var).to(device=device, dtype=dtype)
        eye = torch.eye(T, device=device, dtype=dtype).unsqueeze(0).expand(N, T, T)
        V = V_re + (sig2 + self.jitter) * eye

        # 4. Beta: use previous epoch's estimate (detached, no gradient through beta)
        beta = self._last_beta                                  # (n_W,)
        mu = (W * beta).sum(dim=-1) + h                        # (N, T)

        # 5. Accumulate AtA/Atb for end-of-epoch beta update
        AtA_batch, Atb_batch = None, None
        if y_pad is not None and obs_mask is not None:
            AtA_batch, Atb_batch = self._compute_analytical_beta(W, h, y_pad, V, obs_mask)

        if return_components:
            return mu, V, (W * beta).sum(dim=-1), h, Z, D, sig2, beta, AtA_batch, Atb_batch
        return mu, V, AtA_batch, Atb_batch, h

@torch.no_grad()
def probe_latent_space(model, loader, device, x_cols, static_cols):
    """
    Regress z(t) on real covariates to see what each latent
    dimension has learned to encode.
    
    Returns regression coefficients: which covariates explain
    which latent dimensions.
    """
    all_z, all_x, all_t, all_s = [], [], [], []
    
    model.eval()
    for batch in loader:
        _, t_pad, x_pad, y_pad, c_mask, mask, s_pad = batch
        t_pad = t_pad.to(device)
        x_pad = x_pad.to(device)
        c_mask = c_mask.to(device)
        s_pad = s_pad.to(device)
        mask = mask.to(device)
        
        mu, V, zt = model(t_pad, x_pad, c_mask, s_pad, return_hidden=True)
        
        # Collect only observed (non-padded) time points
        for i in range(t_pad.shape[0]):
            obs = mask[i].bool()
            all_z.append(zt[i, obs].cpu())          # (n_i, H)
            all_x.append(x_pad[i, obs].cpu())       # (n_i, Cx)
            all_t.append(t_pad[i, obs].cpu())       # (n_i,)
            all_s.append(s_pad[i].unsqueeze(0).expand(obs.sum(), -1).cpu())
    
    Z = torch.cat(all_z, dim=0).numpy()     # (total_obs, H)
    X = torch.cat(all_x, dim=0).numpy()     # (total_obs, Cx)
    T = torch.cat(all_t, dim=0).numpy()     # (total_obs,)
    S = torch.cat(all_s, dim=0).numpy()     # (total_obs, Cs)
    
    # Regress each z_h on [time, BMI, GLUC, SEX, AGE, DIPNIV]
    import numpy as np
    from sklearn.linear_model import LinearRegression
    
    features = np.column_stack([T, X, S])
    feature_names = ["time"] + x_cols + static_cols
    
    H = Z.shape[1]
    print(f"{'':>12s}", "  ".join(f"{'z'+str(h):>8s}" for h in range(H)))
    print("-" * (12 + 10 * H))
    
    r2_scores = []
    for h in range(H):
        reg = LinearRegression().fit(features, Z[:, h])
        r2_scores.append(reg.score(features, Z[:, h]))
        
    # Print R² for each latent dimension
    print(f"{'R²':>12s}", "  ".join(f"{r2:8.3f}" for r2 in r2_scores))
    print()
    
    # Print coefficients
    for j, name in enumerate(feature_names):
        coeffs = []
        for h in range(H):
            reg = LinearRegression().fit(features, Z[:, h])
            coeffs.append(reg.coef_[j])
        print(f"{name:>12s}", "  ".join(f"{c:8.4f}" for c in coeffs))
    
    return r2_scores

# -----------------------------
# Full model
# -----------------------------
@dataclass
class NeuralCDEConfig:
    hidden_channels: int = 8      # <<< was 4; need enough dims to encode BMI signal
    enc_mlp_hidden: int = 32       # <<< was 16
    func_mlp_hidden: int = 32      # 
    dec_rho_hidden: int = 16       # NEW: decoder rho capacity
    dec_g_hidden: int = 16         # NEW: decoder g capacity
    dec_p: int = 4                 # fixed-effect basis dimension
    dec_q: int = 3                 # random-effect basis dimension
    depth: int = 2
    dropout: float = 0.0
    solver: str = "rk4"
    step_size: Optional[float] = None
    atol: float = 1e-6
    rtol: float = 1e-6


class NeuralCDEModel(nn.Module):
    """
    Forward returns:
      predicted_mean: (N, T)
      hidden_path:    (N, T, H)
    """

    def __init__(self, x_dim: int, static_dim: int = 0, cfg: NeuralCDEConfig = NeuralCDEConfig(),
                 fe_spline_knots=None, fe_spline_boundary=None,
                 re_spline_knots=None, re_spline_boundary=None,
                 interaction_pairs=None,
                 precomputed_splines=False,
                 n_tv=None):
        """
        x_dim: total number of time-varying columns in x_pad (including spline cols if precomputed)
        static_dim: number of baseline covariates
        n_tv: number of actual time-varying covariates (GLUC, BMI). 
              If None, defaults to x_dim (no precomputed splines).
              When precomputed_splines=True, must be set explicitly.
        """
        super().__init__()
        self.cfg = cfg
        self.x_dim = int(x_dim)
        self.static_dim = int(static_dim)
        self.n_tv = n_tv if n_tv is not None else x_dim

        # CDE control path uses only real time-varying covariates (n_tv) + time
        self.input_channels = 1 + self.n_tv   # [time, GLUC, BMI]

        self.encoder = BaselineICEncoder(
            input_dim_x0=self.input_channels,
            static_dim=self.static_dim,
            hidden_dim=cfg.hidden_channels,
            mlp_hidden=cfg.enc_mlp_hidden,
            depth=1,
            dropout=cfg.dropout,
        )

        self.func = CDEFunc(
            hidden_channels=cfg.hidden_channels,
            input_channels=self.input_channels + 1,   # +1 for cumulative mask channel
            covariate_dim=self.n_tv,
            static_dim=self.static_dim,
            mlp_hidden=cfg.func_mlp_hidden,
            depth=cfg.depth,
            dropout=cfg.dropout,
        )
        self.z_norm = nn.LayerNorm(cfg.hidden_channels)

        self.decoder = Decoder(
            latent_dim=cfg.hidden_channels,
            p=cfg.dec_p,
            q=cfg.dec_q,
            rho_hidden=cfg.dec_rho_hidden,
            fullD=True,
            n_static=static_dim,
            n_tv=self.n_tv,
            fe_spline_knots=fe_spline_knots,
            fe_spline_boundary=fe_spline_boundary,
            re_spline_knots=re_spline_knots,
            re_spline_boundary=re_spline_boundary,
            interaction_pairs=interaction_pairs,
            precomputed_splines=precomputed_splines,
        )

    def forward(
        self,
        t_pad: torch.Tensor,                 # (N, T)
        x_pad: torch.Tensor,                 # (N, T, Cx) — may include spline cols
        masks: torch.Tensor,                 # (N, T) cumulative mask
        static_covariates: Optional[torch.Tensor] = None,# (N, Cs) or None
        obs_mask: Optional[torch.Tensor] = None,         # (N, T) outcome observation mask
        y_pad: Optional[torch.Tensor] = None,            # (N, T) outcomes for analytical beta
        return_hidden: bool = False,
        interp: str = "cubic",
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:

        N, T = t_pad.shape
        device, dtype = t_pad.device, t_pad.dtype

        # CDE only sees real time-varying covariates (first n_tv columns)
        x_cde = x_pad[:, :, :self.n_tv]                     # (N, T, n_tv)

        # Build control path: X_in = [time, x_cde, cumulative_mask]
        X_in = torch.cat([t_pad[..., None], x_cde, masks[..., None]], dim=-1)

        grid = torch.arange(T, device=device, dtype=dtype)
        coeffs = torchcde.natural_cubic_coeffs(X_in) 
        X = torchcde.CubicSpline(coeffs)

        # Interpolation method:
        #   cubic: non-causal (future points affect spline at current time) — better for training
        #   linear: causal (value at t only depends on t and t-1) — correct for PDP/prediction
        if interp == "cubic":
            coeffs = torchcde.natural_cubic_coeffs(X_in)
            X = torchcde.CubicSpline(coeffs)
        elif interp == "linear":
            coeffs = torchcde.linear_interpolation_coeffs(X_in, rectilinear=1)
            X = torchcde.LinearInterpolation(coeffs)
        else:
            raise ValueError(f"interp must be 'cubic' or 'linear', got '{interp}'")

        # Initial state: first observation (time + x_cde) + static covariates
        x0 = X.evaluate(grid[0])  # (N, input_channels + 1)
        encoder_in = torch.cat([x0[:, :-1], static_covariates], dim=-1)  # drop mask channel
        z0 = self.encoder(encoder_in)  # (N, H)

        self.func.set_context(X, static_covariates)
        zt = torchcde.cdeint(
            X=X,
            z0=z0,
            func=self.func,
            t=grid,
            method=self.cfg.solver,
            options={"step_size": 1.0},
            atol=self.cfg.atol,
            rtol=self.cfg.rtol,
            adjoint=False
        )  # (N, T, H)

        zt = self.z_norm(zt)

        # Decoder gets full x_pad (including spline cols if precomputed)
        # and y_pad for analytical beta computation
        return self.decoder(zt, t_pad, x_pad, static_covariates,
                           y_pad=y_pad, obs_mask=obs_mask)