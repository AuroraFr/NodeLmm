"""
Neural CDE-LMM with:
  - inject_x:       controls whether vector field sees covariate levels
  - augment_order:   path augmentation (rough path features)
  - generalized decoder skip connections + regularization

Architecture:
  - Encoder:  z(0) = Enc(x0, static)
  - Dynamics: dz = f(z, [x(t)], static) · dX̃     (inject_x controls [x(t)])
  - Decoder:  mu(t) = rho(z(t), [skip(t)]) @ beta
  - RE:       Z = g(z(t)) or [1, rs1(t), rs2(t)]

CDE control path X̃:
  augment_order=1: [time, x_1..x_n, mask_1..mask_n]
  augment_order=2: [time, x_1..x_n, cum_abs_1..n, cum_sq_1..n,
                     mask_x_1..n, mask_aug_1..n, mask_aug_1..n]

Vector field modes:
  inject_x=False: f(z, static) · dX̃             — pure CDE, paper eq.(3)
  inject_x=True:  f(z, X̃(t), static) · dX̃      — sees covariate levels

Encoder modes:
  encoder_sees_covariates=True:  Enc(time_0, covs_0, static)
  encoder_sees_covariates=False: Enc(time_0, static)  — prevents leakage

Example configurations:
  # S2: inject BMI levels, no augmentation
  NeuralCDEModel(..., inject_x=True, augment_order=1)

  # S6 pure: no injection, no augmentation
  NeuralCDEModel(..., inject_x=False, augment_order=1)

  # S6 with rough path features: no injection, augmented path
  NeuralCDEModel(..., inject_x=False, augment_order=2)

  # S6 strict: no injection, no augmentation, encoder blind to covariates
  NeuralCDEModel(..., inject_x=False, augment_order=1, encoder_sees_covariates=False)
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, List

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


# ─────────────────────────────────────────────
# Building blocks
# ─────────────────────────────────────────────

class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, depth=2, dropout=0.0,
                 activation=None):
        super().__init__()
        if activation is None:
            activation = nn.ReLU()
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

    def forward(self, x):
        return self.net(x)


# ─────────────────────────────────────────────
# Path augmentation (rough path features)
# ─────────────────────────────────────────────

def augment_path(x_pad, c_mask):
    """
    General path augmentation for all time-varying covariates.

    For each covariate channel, adds:
      - cumulative absolute increments (total variation process)
      - cumulative squared increments (quadratic variation process)

    These are model-free path summary statistics from rough path theory.

    Input:
        x_pad:  (N, T, n_tv)    time-varying covariates
        c_mask: (N, T, n_tv)    cumulative observation masks

    Output:
        x_aug:      (N, T, 3*n_tv)  [original, cum_abs, cum_sq]
        c_mask_aug: (N, T, 3*n_tv)  masks for all channels
    """
    diffs = torch.zeros_like(x_pad)
    diffs[:, 1:] = x_pad[:, 1:] - x_pad[:, :-1]

    cum_abs = diffs.abs().cumsum(dim=1)     # total variation process
    cum_sq = diffs.pow(2).cumsum(dim=1)     # quadratic variation process

    x_aug = torch.cat([x_pad, cum_abs, cum_sq], dim=-1)

    # Augmented channels inherit the mask of their source covariate
    if c_mask.dim() == 3:
        c_mask_aug = torch.cat([c_mask, c_mask, c_mask], dim=-1)
    else:
        # c_mask is (N, T) scalar — expand
        n_tv = x_pad.shape[-1]
        cm = c_mask.unsqueeze(-1).expand(-1, -1, n_tv)
        c_mask_aug = torch.cat([cm, cm, cm], dim=-1)

    return x_aug, c_mask_aug


# ─────────────────────────────────────────────
# Encoder
# ─────────────────────────────────────────────

class BaselineICEncoder(nn.Module):
    """
    Encode z(0) from first observation + static covariates.
    Input dimension is set by the model based on encoder_sees_covariates.
    """
    def __init__(self, input_dim, static_dim, hidden_dim,
                 mlp_hidden=128, depth=2, dropout=0.0):
        super().__init__()
        self.mlp = MLP(input_dim + static_dim, mlp_hidden, hidden_dim,
                       depth=depth, dropout=dropout)

    def forward(self, x0):
        return self.mlp(x0)


# ─────────────────────────────────────────────
# CDE vector field
# ─────────────────────────────────────────────

class CDEFunc(nn.Module):
    """
    Neural vector field: dz = f(...) · dX̃

    inject_x=True:  f(z, X̃(t), static) — sees all interpolated channels
    inject_x=False: f(z, static)        — pure CDE, paper eq.(3)

    Output: (N, H, C) matrix that multiplies dX̃.
    """
    def __init__(self, hidden_channels, input_channels,
                 static_dim, inject_x=False, inject_dim=0,
                 mlp_hidden=64, depth=2, dropout=0.0):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.input_channels = input_channels
        self.inject_x = inject_x

        net_in = hidden_channels + static_dim       # z + static (always)
        if inject_x:
            net_in += inject_dim                     # + X̃(t) all channels

        self.net = MLP(
            in_dim=net_in,
            hidden_dim=mlp_hidden,
            out_dim=hidden_channels * input_channels,
            depth=depth, dropout=dropout, activation=nn.ReLU(),
        )
        self.X_spline = None
        self.static = None

    def set_context(self, X, static):
        self.X_spline = X
        self.static = static

    def forward(self, t, z):
        if self.inject_x:
            x_t = self.X_spline.evaluate(t)                    # (N, C_total)
            inp = torch.cat([z, x_t, self.static], dim=-1)
        else:
            inp = torch.cat([z, self.static], dim=-1)

        out = self.net(inp)
        out = torch.tanh(out)
        return out.view(z.size(0), self.hidden_channels, self.input_channels)


# ─────────────────────────────────────────────
# Generalized Decoder
# ─────────────────────────────────────────────

class Decoder(nn.Module):
    """
    mu(t) = rho(z(t), [skip(t)]) @ beta

    Skip connections (operate on ORIGINAL x_pad, not augmented):
      tv_skip_cols:     list of (col_idx, mean, std) in original x_pad
      static_skip_dims: list of indices in static vector
      reg_mode:         None, "l1_skip", or "ortho"
    """
    def __init__(self, latent_dim, p, q=3,
                 fullD=True, jitter=1e-6, rho_hidden=64, D_diag_min=0.1,
                 use_rho_net=True,
                 use_neural_re=True,
                 g_hidden=16,
                 re_spline_cols=None,
                 tv_skip_cols=None,
                 static_skip_dims=None,
                 reg_mode=None,
                 p_skip=None):
        super().__init__()
        self.p, self.q = p, q
        self.fullD = fullD
        self.jitter = jitter
        self.D_diag_min = D_diag_min
        self.reg_mode = reg_mode

        # ── Skip configuration ──
        self.static_skip_dims = static_skip_dims
        n_static_skip = len(static_skip_dims) if static_skip_dims else 0

        if tv_skip_cols is not None and len(tv_skip_cols) > 0:
            self.tv_skip_col_indices = [c[0] for c in tv_skip_cols]
            tv_means = torch.tensor([c[1] for c in tv_skip_cols], dtype=torch.float32)
            tv_stds = torch.tensor([c[2] for c in tv_skip_cols], dtype=torch.float32)
            self.register_buffer('tv_skip_means', tv_means)
            self.register_buffer('tv_skip_stds', tv_stds)
            n_tv_skip = len(tv_skip_cols)
        else:
            self.tv_skip_col_indices = None
            self.register_buffer('tv_skip_means', torch.empty(0))
            self.register_buffer('tv_skip_stds', torch.empty(0))
            n_tv_skip = 0

        self.n_tv_skip = n_tv_skip
        self.n_static_skip = n_static_skip
        skip_dim = n_tv_skip + n_static_skip
        self.skip_dim = skip_dim

        if p_skip is None:
            p_skip = p
        self.p_skip = p_skip

        if reg_mode == "ortho" and skip_dim == 0:
            raise ValueError("reg_mode='ortho' requires at least one skip covariate")

        # ── Fixed effects ──
        self.use_rho_net = use_rho_net

        if reg_mode == "ortho":
            self.rho_latent_net = MLP(latent_dim, rho_hidden, p, depth=2)
            self.rho_latent_norm = nn.LayerNorm(p)
            self.beta_latent = nn.Parameter(0.1 * torch.randn(p))
            self.rho_skip_net = MLP(skip_dim, rho_hidden, p_skip, depth=2)
            self.rho_skip_norm = nn.LayerNorm(p_skip)
            self.beta_skip = nn.Parameter(0.1 * torch.randn(p_skip))
            self.beta_neural = self.beta_latent
            self.rho_net = None
        elif use_rho_net:
            self.rho_net = MLP(latent_dim + skip_dim, rho_hidden, p, depth=2)
            self.rho_norm = nn.LayerNorm(p)
            self.beta_neural = nn.Parameter(0.1 * torch.randn(p))
            self.rho_latent_net = None
            self.rho_skip_net = None
        else:
            self.rho_net = None
            self.rho_latent_net = None
            self.rho_skip_net = None
            self.w_neural = nn.Parameter(0.01 * torch.randn(latent_dim + skip_dim))
            self.beta_neural = None

        # ── Random effects ──
        self.use_neural_re = use_neural_re
        if use_neural_re:
            self.g_net = MLP(latent_dim, g_hidden, q, depth=2)
            self.g_norm = nn.LayerNorm(q)
            self.re_spline_cols = None
        else:
            self.g_net = None
            self.g_norm = None
            self.re_spline_cols = re_spline_cols
            if re_spline_cols is not None:
                assert q == len(re_spline_cols) + 1

        # ── D matrix ──
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

    def _build_skip_input(self, x_pad, static, N, T):
        skip_parts = []
        if self.tv_skip_col_indices is not None:
            for i, col_idx in enumerate(self.tv_skip_col_indices):
                tv_raw = x_pad[:, :, col_idx:col_idx + 1]
                tv_std = (tv_raw - self.tv_skip_means[i]) / self.tv_skip_stds[i]
                skip_parts.append(tv_std)
        if self.static_skip_dims:
            static_vals = static[:, self.static_skip_dims]
            static_expanded = static_vals.unsqueeze(1).expand(-1, T, -1)
            skip_parts.append(static_expanded)
        if skip_parts:
            return torch.cat(skip_parts, dim=-1)
        return None

    def forward(self, z_t, t_pad, x_pad, static, y_pad=None, obs_mask=None,
                return_components=True):
        N, T, H = z_t.shape
        device, dtype = z_t.device, z_t.dtype

        skip_input = self._build_skip_input(x_pad, static, N, T)
        reg_dict = {}

        if self.reg_mode == "ortho":
            rho_lat = self.rho_latent_norm(self.rho_latent_net(z_t))
            mu_latent = (rho_lat * self.beta_latent).sum(dim=-1)
            rho_skip = self.rho_skip_norm(self.rho_skip_net(skip_input))
            mu_skip = (rho_skip * self.beta_skip).sum(dim=-1)
            mu = mu_latent + mu_skip

            if obs_mask is not None:
                m = obs_mask.reshape(-1) > 0.5
                lat_f, skip_f = mu_latent.reshape(-1)[m], mu_skip.reshape(-1)[m]
            else:
                lat_f, skip_f = mu_latent.reshape(-1), mu_skip.reshape(-1)
            lat_c, skip_c = lat_f - lat_f.mean(), skip_f - skip_f.mean()
            corr = (lat_c * skip_c).sum() / (lat_c.norm() * skip_c.norm() + 1e-8)
            reg_dict.update(reg_term=corr**2, mu_latent=mu_latent.detach(),
                            mu_skip=mu_skip.detach(), corr=corr.detach())

        elif self.reg_mode == "l1_skip" and self.use_rho_net:
            rho_in = torch.cat([z_t, skip_input], dim=-1) if skip_input is not None else z_t
            rho = self.rho_norm(self.rho_net(rho_in))
            mu = (rho * self.beta_neural).sum(dim=-1)
            if skip_input is not None:
                rho_z = self.rho_norm(self.rho_net(torch.cat([z_t, torch.zeros_like(skip_input)], dim=-1)))
                mu_no = (rho_z * self.beta_neural).sum(dim=-1)
                sc = (mu - mu_no).abs()
                if obs_mask is not None:
                    reg_dict["reg_term"] = (sc * obs_mask).sum() / obs_mask.sum().clamp(min=1)
                else:
                    reg_dict["reg_term"] = sc.mean()
                reg_dict["skip_contrib_mean"] = sc.detach().mean()
            else:
                reg_dict["reg_term"] = torch.tensor(0.0, device=device, dtype=dtype)
        else:
            rho_in = torch.cat([z_t, skip_input], dim=-1) if skip_input is not None else z_t
            if self.use_rho_net:
                rho = self.rho_norm(self.rho_net(rho_in))
                mu = (rho * self.beta_neural).sum(dim=-1)
            else:
                mu = (rho_in * self.w_neural).sum(dim=-1)

        # ── Random effects ──
        if self.use_neural_re:
            Z = self.g_norm(self.g_net(z_t))
        else:
            ones = torch.ones(N, T, 1, device=device, dtype=dtype)
            if self.re_spline_cols is not None:
                Z = torch.cat([ones, x_pad[:, :, self.re_spline_cols]], dim=-1)
            else:
                Z = ones

        if obs_mask is not None:
            Z = Z * obs_mask.unsqueeze(-1)

        D = self._build_D(device, dtype)
        V_re = (Z @ D) @ Z.transpose(1, 2)
        sig2 = torch.exp(self.log_residual_var).to(device=device, dtype=dtype)
        eye = torch.eye(T, device=device, dtype=dtype).unsqueeze(0).expand(N, T, T)
        V = V_re + (sig2 + self.jitter) * eye

        if return_components:
            return mu, V, Z, D, sig2, reg_dict
        return mu, V

    def describe_skip(self):
        parts = []
        if self.tv_skip_col_indices:
            for i, col in enumerate(self.tv_skip_col_indices):
                m, s = self.tv_skip_means[i].item(), self.tv_skip_stds[i].item()
                parts.append(f"x_pad[:,:,{col}] (mean={m:.2f}, std={s:.2f})")
        if self.static_skip_dims:
            for dim in self.static_skip_dims:
                parts.append(f"static[:, {dim}]")
        if not parts:
            parts.append("(none — pure latent)")
        print(f"Skip connections ({self.skip_dim} dims):")
        for s in parts:
            print(f"  - {s}")
        print(f"Regularization: {self.reg_mode or 'none'}")


# ─────────────────────────────────────────────
# Latent space diagnostics
# ─────────────────────────────────────────────

@torch.no_grad()
def probe_latent_space(model, loader, device, x_cols, static_cols):
    """Regress z(t) on real covariates to interpret latent dimensions."""
    all_z, all_x, all_t, all_s = [], [], [], []
    model.eval()
    for batch in loader:
        _, t_pad, x_pad, y_pad, c_mask, mask, s_pad = batch
        t_pad = t_pad.to(device)
        x_pad = x_pad.to(device)
        c_mask = c_mask.to(device)
        s_pad = s_pad.to(device)
        mask = mask.to(device)
        out = model(t_pad, x_pad, c_mask, s_pad, obs_mask=mask, return_hidden=True)
        zt = out[-1] if isinstance(out, tuple) else out
        for i in range(t_pad.shape[0]):
            obs = mask[i].bool()
            all_z.append(zt[i, obs].cpu())
            all_x.append(x_pad[i, obs].cpu())
            all_t.append(t_pad[i, obs].cpu())
            all_s.append(s_pad[i].unsqueeze(0).expand(obs.sum(), -1).cpu())

    Z = torch.cat(all_z, dim=0).numpy()
    X = torch.cat(all_x, dim=0).numpy()
    T_arr = torch.cat(all_t, dim=0).numpy()
    S = torch.cat(all_s, dim=0).numpy()

    from sklearn.linear_model import LinearRegression
    features = np.column_stack([T_arr, X, S])
    feature_names = ["time"] + list(x_cols) + list(static_cols)
    H = Z.shape[1]
    print(f"{'':>12s}", "  ".join(f"{'z'+str(h):>8s}" for h in range(H)))
    print("-" * (12 + 10 * H))
    all_regs, r2_scores = [], []
    for h in range(H):
        reg = LinearRegression().fit(features, Z[:, h])
        r2_scores.append(reg.score(features, Z[:, h]))
        all_regs.append(reg)
    print(f"{'R2':>12s}", "  ".join(f"{r2:8.3f}" for r2 in r2_scores))
    print()
    for j, name in enumerate(feature_names):
        coeffs = [all_regs[h].coef_[j] for h in range(H)]
        print(f"{name:>12s}", "  ".join(f"{c:8.4f}" for c in coeffs))
    return r2_scores


# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────

@dataclass
class NeuralCDEConfig:
    hidden_channels: int = 8
    enc_mlp_hidden: int = 32
    func_mlp_hidden: int = 32
    dec_rho_hidden: int = 16
    dec_g_hidden: int = 16
    dec_p: int = 4
    dec_q: int = 3
    depth: int = 2
    dropout: float = 0.0
    solver: str = "rk4"
    step_size: Optional[float] = None
    atol: float = 1e-6
    rtol: float = 1e-6


# ─────────────────────────────────────────────
# Full CDE model
# ─────────────────────────────────────────────

class NeuralCDEModel(nn.Module):
    """
    Neural CDE-LMM with inject_x, path augmentation, and generalized decoder.

    CDE control path X̃:
      augment_order=1: [time, x_1..x_n, mask_1..mask_n]
      augment_order=2: [time, x_1..x_n, cum_abs_1..n, cum_sq_1..n,
                         mask_x_1..n, mask_aug_1..n, mask_aug_1..n]

    Vector field:
      inject_x=False: f(z, static) · dX̃             — pure CDE
      inject_x=True:  f(z, X̃(t), static) · dX̃      — sees levels
    """

    def __init__(self, x_dim: int, static_dim: int = 0,
                 cfg: NeuralCDEConfig = NeuralCDEConfig(),
                 n_tv=None,
                 inject_x=False,
                 augment_order=1,
                 encoder_sees_covariates=True,
                 use_rho_net=True,
                 use_neural_re=True,
                 g_hidden=16,
                 re_spline_cols=None,
                 fullD=True,
                 tv_skip_cols=None,
                 static_skip_dims=None,
                 reg_mode=None,
                 p_skip=None):
        super().__init__()
        self.cfg = cfg
        self.x_dim = int(x_dim)
        self.static_dim = int(static_dim)
        self.inject_x = inject_x
        self.augment_order = augment_order
        self.encoder_sees_covariates = encoder_sees_covariates

        # Original number of time-varying covariates
        n_tv_orig = n_tv if n_tv is not None else x_dim
        self.n_tv_orig = n_tv_orig

        # Effective n_tv after augmentation
        if augment_order >= 2:
            n_tv_eff = n_tv_orig * 3
        else:
            n_tv_eff = n_tv_orig
        self.n_tv = n_tv_eff

        # CDE control path: [time(1), covariates(n_tv_eff), masks(n_tv_eff)]
        self.input_channels = 1 + n_tv_eff + n_tv_eff

        # Encoder
        if encoder_sees_covariates:
            encoder_input_dim = 1 + n_tv_eff     # time_0 + covariate values
        else:
            encoder_input_dim = 1                  # time_0 only

        self.encoder = BaselineICEncoder(
            input_dim=encoder_input_dim,
            static_dim=self.static_dim,
            hidden_dim=cfg.hidden_channels,
            mlp_hidden=cfg.enc_mlp_hidden,
            depth=1,
            dropout=cfg.dropout,
        )

        # CDE vector field
        inject_dim = self.input_channels if inject_x else 0

        self.func = CDEFunc(
            hidden_channels=cfg.hidden_channels,
            input_channels=self.input_channels,
            static_dim=self.static_dim,
            inject_x=inject_x,
            inject_dim=inject_dim,
            mlp_hidden=cfg.func_mlp_hidden,
            depth=cfg.depth,
            dropout=cfg.dropout,
        )

        self.z_norm = nn.LayerNorm(cfg.hidden_channels)

        # Decoder (operates on ORIGINAL x_pad)
        self.decoder = Decoder(
            latent_dim=cfg.hidden_channels,
            p=cfg.dec_p,
            q=cfg.dec_q,
            rho_hidden=cfg.dec_rho_hidden,
            fullD=fullD,
            use_rho_net=use_rho_net,
            use_neural_re=use_neural_re,
            g_hidden=g_hidden,
            re_spline_cols=re_spline_cols,
            tv_skip_cols=tv_skip_cols,
            static_skip_dims=static_skip_dims,
            reg_mode=reg_mode,
            p_skip=p_skip,
        )

    def forward(
        self,
        t_pad: torch.Tensor,           # (N, T)
        x_pad: torch.Tensor,           # (N, T, x_dim) ORIGINAL covariates
        masks: torch.Tensor,           # (N, T) or (N, T, n_tv) cumulative masks
        static_covariates: Optional[torch.Tensor] = None,
        obs_mask: Optional[torch.Tensor] = None,
        y_pad: Optional[torch.Tensor] = None,
        return_hidden: bool = False,
        interp: str = "cubic",
    ):
        N, T = t_pad.shape
        device, dtype = t_pad.device, t_pad.dtype

        # ── Extract original time-varying covariates ──
        x_tv = x_pad[:, :, :self.n_tv_orig]

        # ── Ensure masks are (N, T, n_tv_orig) ──
        if masks.dim() == 2:
            c_mask = masks.unsqueeze(-1).expand(-1, -1, self.n_tv_orig)
        else:
            c_mask = masks[:, :, :self.n_tv_orig]

        # ── Path augmentation ──
        if self.augment_order >= 2:
            x_cde, c_mask_cde = augment_path(x_tv, c_mask)
        else:
            x_cde = x_tv
            c_mask_cde = c_mask

        # ── Build CDE control path: [time, covariates, masks] ──
        X_in = torch.cat([
            t_pad[..., None],        # (N, T, 1)
            x_cde,                   # (N, T, n_tv_eff)
            c_mask_cde,              # (N, T, n_tv_eff)
        ], dim=-1)                   # (N, T, 1 + 2*n_tv_eff)

        grid = torch.arange(T, device=device, dtype=dtype)

        # ── Interpolation ──
        if interp == "cubic":
            coeffs = torchcde.natural_cubic_coeffs(X_in)
            X = torchcde.CubicSpline(coeffs)
        elif interp == "linear":
            coeffs = torchcde.linear_interpolation_coeffs(X_in, rectilinear=True)
            X = torchcde.LinearInterpolation(coeffs)
        else:
            raise ValueError(f"interp must be 'cubic' or 'linear', got '{interp}'")

        # ── Encoder ──
        x0 = X.evaluate(grid[0])                          # (N, 1 + 2*n_tv_eff)

        if self.encoder_sees_covariates:
            # [time_0, covariate_values_0] — drop mask channels
            x0_no_mask = x0[:, :1 + self.n_tv]
            encoder_in = torch.cat([x0_no_mask, static_covariates], dim=-1)
        else:
            # [time_0] only
            time_0 = x0[:, 0:1]
            encoder_in = torch.cat([time_0, static_covariates], dim=-1)

        z0 = self.encoder(encoder_in)

        # ── CDE integration ──
        self.func.set_context(X, static_covariates)
        zt = torchcde.cdeint(
            X=X, z0=z0, func=self.func, t=grid,
            method=self.cfg.solver,
            options={"step_size": 1.0},
            atol=self.cfg.atol, rtol=self.cfg.rtol,
            adjoint=False,
        )

        zt = self.z_norm(zt)

        # ── Decoder (uses ORIGINAL x_pad) ──
        result = self.decoder(zt, t_pad, x_pad, static_covariates,
                              y_pad=y_pad, obs_mask=obs_mask)

        if return_hidden:
            if isinstance(result, tuple):
                return result + (zt,)
            return result, zt
        return result

    def describe(self):
        """Print full model configuration."""
        print(f"NeuralCDEModel:")
        print(f"  x_dim={self.x_dim}, n_tv_orig={self.n_tv_orig}, n_tv_eff={self.n_tv}")
        aug_str = "+ cum_abs + cum_sq" if self.augment_order >= 2 else "none"
        print(f"  augment_order={self.augment_order} ({aug_str})")
        inj_str = "f(z, X(t), static)" if self.inject_x else "f(z, static)"
        print(f"  inject_x={self.inject_x} ({inj_str})")
        print(f"  encoder_sees_covariates={self.encoder_sees_covariates}")
        print(f"  input_channels={self.input_channels} = 1 + {self.n_tv} + {self.n_tv}")
        n_params = sum(p.numel() for p in self.parameters())
        print(f"  total parameters: {n_params}")
        self.decoder.describe_skip()