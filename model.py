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

class Decoder(nn.Module):
    def __init__(self, latent_dim, p, q=3,
                 fullD=True, jitter=1e-6, rho_hidden=64, D_diag_min=0.1,
                 n_static=3, n_tv=2,
                 use_rho_net=True,
                 use_neural_re=True,
                 g_hidden=16,
                 skip_bmi_to_decoder=False,
                 bmi_col=1,
                 re_spline_cols=None, bmi_mean=None, bmi_std=None):
        """
        Args:
            re_spline_cols: list of column indices in x_pad for RE spline basis.
                            Used when use_neural_re=False to build Z = [1, rs1(t), rs2(t)].
                            e.g. [4, 5] if x_cols = ["BMI_t","ns1","ns2","ns3","rs1","rs2"].
                            q must equal len(re_spline_cols) + 1 (for the intercept column).
        """
        super().__init__()
        self.p, self.q = p, q
        self.fullD = fullD
        self.jitter = jitter
        self.D_diag_min = D_diag_min
        self.n_static = n_static
        self.n_tv = n_tv
        self.skip_bmi_to_decoder = skip_bmi_to_decoder
        self.bmi_col = bmi_col

        self.register_buffer('bmi_mean', torch.tensor(bmi_mean))
        self.register_buffer('bmi_std', torch.tensor(bmi_std))

        # --- Neural fixed effects ---
        self.use_rho_net = use_rho_net
        # Extra input dims for skip connection: current BMI value
        skip_dim = 1 if skip_bmi_to_decoder else 0
        if use_rho_net:
            # MLP: [z(t), BMI(t)] → R^p → weighted sum → scalar
            self.rho_net = MLP(latent_dim + skip_dim, rho_hidden, p, depth=2, dropout=0.0)
            self.rho_norm = nn.LayerNorm(p)
            self.beta_neural = nn.Parameter(0.1 * torch.randn(p))
        else:
            # Linear projection: h(z(t)) = w' z(t), just H parameters
            self.rho_net = None
            # self.rho_norm = None
            self.w_neural = nn.Parameter(0.01 * torch.randn(latent_dim))

        # --- Random effects ---
        self.use_neural_re = use_neural_re
        if use_neural_re:
            # g(z(t)): learned RE design matrix, z(t) → R^q
            self.g_net = MLP(latent_dim, g_hidden, q, depth=2, dropout=0.0)
            self.re_spline_cols = None
        else:
            # Classical spline RE: Z = [1, rs1(t), rs2(t)] from precomputed columns
            self.g_net = None
            self.re_spline_cols = re_spline_cols
            if re_spline_cols is not None:
                assert q == len(re_spline_cols) + 1, (
                    f"q={q} must equal len(re_spline_cols)+1={len(re_spline_cols)+1} "
                    f"(intercept + {len(re_spline_cols)} spline columns)"
                )

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

    def forward(self, z_t, t_pad, x_pad, static, y_pad=None, obs_mask=None,
                return_components=True):
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

        # 2. Neural flexible fixed effects
        if self.skip_bmi_to_decoder:
            # Concatenate current BMI to z(t) → decoder sees instantaneous BMI directly
            bmi_t = x_pad[:, :, self.bmi_col:self.bmi_col+1]  # (N, T, 1)
            bmi_t = (bmi_t - self.bmi_mean) / self.bmi_std 
            rho_input = torch.cat([z_t, bmi_t], dim=-1)        # (N, T, H+1)
        else:
            rho_input = z_t

        if self.use_rho_net:
            rho = self.rho_net(rho_input)                               # (N, T, p)
            rho = self.rho_norm(rho)
            mu = (rho * self.beta_neural).sum(dim=-1)               # (N, T)
        else:
            mu = (z_t * self.w_neural).sum(dim=-1)                  # (N, T)

        # 3. Random effects design matrix Z
        if self.use_neural_re:
            # Learned RE design: g(z(t)) → R^q per time point
            Z = self.g_net(z_t)                                    # (N, T, q)
        else:
            # Classical spline RE: Z = [1, rs1(t), rs2(t)] from precomputed x_pad columns
            ones = torch.ones(N, T, 1, device=device, dtype=dtype)
            if self.re_spline_cols is not None:
                rs_cols = x_pad[:, :, self.re_spline_cols]         # (N, T, q-1)
                Z = torch.cat([ones, rs_cols], dim=-1)             # (N, T, q)
            else:
                # Fallback: intercept-only RE (q should be 1)
                Z = ones                                           # (N, T, 1)

        if obs_mask is not None:
            Z = Z * obs_mask.unsqueeze(-1)

        D = self._build_D(device, dtype)
        V_re = (Z @ D) @ Z.transpose(1, 2)

        sig2 = torch.exp(self.log_residual_var).to(device=device, dtype=dtype)
        eye = torch.eye(T, device=device, dtype=dtype).unsqueeze(0).expand(N, T, T)
        V = V_re + (sig2 + self.jitter) * eye

        if return_components:
            return mu, V, Z, D, sig2
        return mu, V

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
                 n_tv=None,
                 use_rho_net=True,
                 use_neural_re=False,
                 g_hidden=16,
                 skip_bmi_to_decoder=False,
                 bmi_col=1,
                 re_spline_cols=None,
                 fullD=True,bmi_mean=None, bmi_std=None):
        """
        x_dim: total number of time-varying columns in x_pad (including spline cols if precomputed)
        static_dim: number of baseline covariates
        n_tv: number of actual time-varying covariates (GLUC, BMI). 
              If None, defaults to x_dim (no precomputed splines).
              When precomputed_splines=True, must be set explicitly.
        use_neural_re: if True, Z = g(z(t)) learned from latent space (paper eq. 6-8)
                       if False, Z = [1, rs1(t), rs2(t)] fixed spline basis from re_spline_cols
        re_spline_cols: list of column indices in x_pad for RE spline basis.
                        e.g. [4, 5] for rs1, rs2 when x_cols=["BMI_t","ns1","ns2","ns3","rs1","rs2"]
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
            fullD=fullD,
            n_static=static_dim,
            n_tv=self.n_tv,
            use_rho_net=use_rho_net,
            use_neural_re=use_neural_re,
            g_hidden=g_hidden,
            skip_bmi_to_decoder=skip_bmi_to_decoder,
            bmi_col=bmi_col,
            re_spline_cols=re_spline_cols,bmi_mean=bmi_mean, bmi_std=bmi_std
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
        interp: str = "cubic",               # "cubic" (training) or "linear" (causal PDP)
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:

        N, T = t_pad.shape
        device, dtype = t_pad.device, t_pad.dtype

        # CDE only sees real time-varying covariates (first n_tv columns)
        x_cde = x_pad[:, :, :self.n_tv]                     # (N, T, n_tv)

        # Build control path: X_in = [time, x_cde, cumulative_mask]
        X_in = torch.cat([t_pad[..., None], x_cde, masks[..., None]], dim=-1)

        grid = torch.arange(T, device=device, dtype=dtype)

        # Interpolation method:
        #   cubic: non-causal (future points affect spline at current time) — better for training
        #   linear: causal (value at t only depends on t and t-1) — correct for PDP/prediction
        if interp == "cubic":
            coeffs = torchcde.natural_cubic_coeffs(X_in)
            X = torchcde.CubicSpline(coeffs)
        elif interp == "linear":
            coeffs = torchcde.linear_interpolation_coeffs(X_in, rectilinear=True)
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