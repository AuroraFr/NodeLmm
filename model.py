"""
Neural ODE-LMM model.

Architecture
------------
  Encoder:  z(0) = Enc(t0, [x_baseline], static)
  Dynamics: dz/dt = f(z, [x_interp(t)], [mask(t)], t) ← Neural ODE
  Skip:     skip(t) = gate ⊙ [x_interp_std(t), (mask(t)), static_skip]
  Decoder:  mu(t)   = rho(z(t), skip(t))^T beta
  RE:       Z       = g(z(t))   or   1  (random intercept)
  Marginal: y(t) ~ N(mu(t), Z D Z^T + sigma^2 I)

Canonical input (packed)
------------------------
  x_aug : (N, T, 1 + K + K*include_mask)
          layout = [ time(1) | x_interp(K) | mask(K) if include_mask ]
  static_covariates : (N, Ks)

Regularisation groups (data-independent ⇒ clean M-estimator inference,
Commenges et al. 2014)
  - Dynamic group k : columns (x_interp_k, mask_k) if include_mask else (x_interp_k,)
  - Static  group j : one column per skip-connected static covariate
  reg_mode:
    None           no penalty
    "group_lasso"  Frobenius norm of first-layer rho weights per group;
                   penalty = λ · Σ_g ||W_g||_F   (Yuan & Lin, 2006)

----------------------------
  baseline (Euler, single BMI, rho noLayerNorm):
      n_tv=1, include_mask=False, use_dynamic_in_ode=False,
      encode_baseline_cov=False,
      cfg.ode_solver="euler", cfg.use_rho_norm=False
  cumulative (multi-solver + torchdiffeq, single BMI, no rho norm):
      n_tv=1, include_mask=False, use_dynamic_in_ode=True,
      encode_baseline_cov=False,
      cfg.ode_solver=<solver>, cfg.use_rho_norm=True
  real (multi-covariate + mask, grouped reg):
      n_tv=K, include_mask=True, use_dynamic_in_ode=True,
      encode_baseline_cov=True,
      cfg.ode_solver=<solver>, cfg.use_rho_norm=<flag>
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────
# Building blocks
# ─────────────────────────────────────────────

class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, depth=2, dropout=0.0,
                 activation=None):
        super().__init__()
        if activation is None:
            activation = nn.SiLU()
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
# Encoder
# ─────────────────────────────────────────────

class BaselineEncoder(nn.Module):
    """
    z(0) = Enc(t0, [x_baseline(K)], static(Ks)).

    x_baseline is included only if encode_baseline_cov=True.
    Input dim: 1 + (K if encode_baseline_cov else 0) + Ks.
    """
    def __init__(self, n_tv, static_dim, hidden_dim,
                 encode_baseline_cov=True,
                 mlp_hidden=128, depth=2, dropout=0.0):
        super().__init__()
        self.encode_baseline_cov = encode_baseline_cov
        n_cov = n_tv if encode_baseline_cov else 0
        input_dim = 1 + n_cov + static_dim
        self.mlp = MLP(input_dim, mlp_hidden, hidden_dim,
                       depth=depth, dropout=dropout)

    def forward(self, x0_and_static):
        return self.mlp(x0_and_static)


# ─────────────────────────────────────────────
# ODE vector field
# ─────────────────────────────────────────────

class ODEFunc(nn.Module):
    """
    dz/dt = f(z, [x_interp(t)], [mask(t)], t).

    Static covariates are excluded from the field so that dz/dt does not
    depend directly on demographics; static info enters only through z(0).

    Input dim: H + 1 (z, t)
             + K if use_dynamic_in_ode           (x_interp)
             + K if use_dynamic_in_ode & mask     (mask)
    """
    def __init__(self, hidden_channels, n_tv,
                 use_dynamic_in_ode=True, include_mask=True,
                 mlp_hidden=64, depth=2, dropout=0.0, activation=None):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.n_tv = n_tv
        self.use_dynamic_in_ode = use_dynamic_in_ode
        self.include_mask = include_mask

        n_dyn = n_tv if use_dynamic_in_ode else 0
        n_msk = n_tv if (use_dynamic_in_ode and include_mask) else 0
        in_dim = hidden_channels + 1 + n_dyn + n_msk

        self.net = MLP(
            in_dim=in_dim,
            hidden_dim=mlp_hidden,
            out_dim=hidden_channels,
            depth=depth, dropout=dropout, activation=activation,
        )

    def forward(self, z, t_scalar, x_interp_t=None, mask_t=None):
        """
        z:          (N, H)
        t_scalar:   scalar / (N,) / (N, 1)
        x_interp_t: (N, K) interpolated covariates at t (may be None)
        mask_t:     (N, K) observation mask at t (may be None)
        returns dz/dt : (N, H)
        """
        if t_scalar.dim() == 0:
            t_expanded = t_scalar.unsqueeze(0).expand(z.size(0), 1)
        elif t_scalar.dim() == 1:
            t_expanded = t_scalar.unsqueeze(-1)
        else:
            t_expanded = t_scalar

        parts = [z]
        if self.use_dynamic_in_ode and x_interp_t is not None:
            parts.append(x_interp_t)
            if self.include_mask and mask_t is not None:
                parts.append(mask_t)
        parts.append(t_expanded)

        inp = torch.cat(parts, dim=-1)
        return torch.tanh(self.net(inp))


# ─────────────────────────────────────────────
# Decoder with grouped, regularised skip connections
# ─────────────────────────────────────────────

class Decoder(nn.Module):
    """
    Mixed-effects decoder.

    Skip layout: [ x_interp_std(K), (mask(K)), static_skip(Ks_skip) ]
    rho input:   [ z(t)(H), skip ]

    Regularisation groups:
      - Dynamic group k : (x_interp_k [, mask_k])   → 1 or 2 columns
      - Static  group j : (static_j)                → 1 column
    """
    def __init__(self, latent_dim, n_tv, p, q=3,
                 fullD=True, jitter=1e-6, rho_hidden=64, D_diag_min=0.1,
                 use_rho_net=True,
                 use_neural_re=True,
                 g_hidden=16,
                 cov_means=None, cov_stds=None,
                 static_skip_dims=None,
                 use_dynamic_skip=True,
                 include_mask=True,
                 reg_mode=None,
                 use_rho_norm=True,
                 beta_init_scale=0.01):
        super().__init__()
        self.p, self.q = p, q
        self.fullD = fullD
        self.jitter = jitter
        self.D_diag_min = D_diag_min
        self.reg_mode = reg_mode
        self.latent_dim = latent_dim
        self.n_tv = n_tv
        self.include_mask = include_mask

        # ── Standardization buffers for dynamic covariates ──────────────
        if cov_means is not None:
            self.register_buffer('cov_means',
                                 torch.as_tensor(cov_means, dtype=torch.float32))
        else:
            self.register_buffer('cov_means', torch.zeros(n_tv))
        if cov_stds is not None:
            self.register_buffer('cov_stds',
                                 torch.as_tensor(cov_stds, dtype=torch.float32))
        else:
            self.register_buffer('cov_stds', torch.ones(n_tv))

        # ── Skip connection layout ──────────────────────────────────────
        self.use_dynamic_skip = use_dynamic_skip
        self.static_skip_dims = list(static_skip_dims) if static_skip_dims else []
        n_static_skip = len(self.static_skip_dims)

        cols_per_dyn = 2 if include_mask else 1
        n_dynamic_groups = n_tv if use_dynamic_skip else 0
        self.n_groups = n_dynamic_groups + n_static_skip

        skip_dim_dynamic = cols_per_dyn * n_tv if use_dynamic_skip else 0
        self.skip_dim = skip_dim_dynamic + n_static_skip

        # Group → column-index mapping (into the skip tensor)
        # skip layout: [ x_std(K), (mask(K)), static(Ks_skip) ]
        self._group_col_indices: List[List[int]] = []
        if use_dynamic_skip:
            for k in range(n_tv):
                if include_mask:
                    self._group_col_indices.append([k, n_tv + k])
                else:
                    self._group_col_indices.append([k])
            col_offset = skip_dim_dynamic
        else:
            col_offset = 0
        for j in range(n_static_skip):
            self._group_col_indices.append([col_offset + j])

        # ── Neural fixed effects ────────────────────────────────────────
        self.use_rho_net = use_rho_net
        if use_rho_net:
            self.rho_net = MLP(latent_dim + self.skip_dim, rho_hidden, p,
                               depth=2, dropout=0.0)
            self.rho_norm = nn.LayerNorm(p) if use_rho_norm else nn.Identity()
            self.beta_neural = nn.Parameter(beta_init_scale * torch.randn(p))
        else:
            self.rho_net = None
            self.rho_norm = None
            self.w_neural = nn.Parameter(
                0.01 * torch.randn(latent_dim + self.skip_dim))
            self.beta_neural = None

        # ── Random effects ──────────────────────────────────────────────
        # Two options: neural RE (g_net) or intercept-only.
        # Intercept-only is a scalar random intercept ⇒ q is forced to 1.
        self.use_neural_re = use_neural_re
        if use_neural_re:
            self.g_net = MLP(latent_dim, g_hidden, q, depth=2, dropout=0.0)
            self.g_norm = nn.LayerNorm(q)
        else:
            self.g_net = None
            self.q = 1

        # ── D matrix ────────────────────────────────────────────────────
        qD = self.q
        if fullD:
            L_init = torch.eye(qD) * 0.7
            L_init[0, 0] = 1.0
            self.L_unconstrained = nn.Parameter(L_init)
            self.log_std = None
        else:
            self.log_std = nn.Parameter(torch.zeros(qD))
            self.L_unconstrained = None

        self.log_residual_var = nn.Parameter(torch.tensor(1.0))

    # ── D ───────────────────────────────────────────────────────────────
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

    # ── Skip input ───────────────────────────────────────────────────────
    def _build_skip_input(self, x_interp, mask, static, N, T):
        """Return (N, T, skip_dim) or None."""
        parts = []
        if self.use_dynamic_skip:
            x_std = (x_interp - self.cov_means) / self.cov_stds   # (N, T, K)
            parts.append(x_std)
            if self.include_mask and mask is not None:
                parts.append(mask)                                # (N, T, K)
        if self.static_skip_dims:
            s_skip = static[:, self.static_skip_dims]             # (N, n_skip)
            s_skip = s_skip.unsqueeze(1).expand(-1, T, -1)        # (N, T, n_skip)
            parts.append(s_skip)
        if parts:
            return torch.cat(parts, dim=-1)
        return None

    def _compute_reg(self):
        """Data-independent regularisation term (+ diagnostics)."""
        reg_dict = {}
        if self.reg_mode == "group_lasso" and self.use_rho_net \
                and self.n_groups > 0:
            W = self.rho_net.net[0].weight                        # (h_out, h_in)
            offset = self.latent_dim
            group_norms = []
            for cols in self._group_col_indices:
                W_cols = W[:, [offset + c for c in cols]]
                group_norms.append(W_cols.norm(p='fro'))
            group_norms = torch.stack(group_norms)
            reg_dict["reg_term"] = group_norms.sum()
            reg_dict["group_norms"] = group_norms.detach()

        else:
            reg_dict["reg_term"] = torch.tensor(0.0)
        return reg_dict

    # ── Forward ──────────────────────────────────────────────────────────
    def forward(self, z_t, x_interp, mask, static,
                obs_mask=None, return_components=True):
        """
        z_t:      (N, T, H)
        x_interp: (N, T, K)
        mask:     (N, T, K) or None
        static:   (N, Ks)
        obs_mask: (N, T) binary mask for observed outcomes
        """
        N, T, H = z_t.shape
        device, dtype = z_t.device, z_t.dtype

        skip_input = self._build_skip_input(x_interp, mask, static, N, T)

        # Fixed effects
        rho_input = torch.cat([z_t, skip_input], dim=-1) \
            if skip_input is not None else z_t
        if self.use_rho_net:
            rho = self.rho_norm(self.rho_net(rho_input))
            mu = (rho * self.beta_neural).sum(dim=-1)             # (N, T)
        else:
            mu = (rho_input * self.w_neural).sum(dim=-1)

        reg_dict = self._compute_reg()

        # Random effects design Z
        if self.use_neural_re:
            Z = self.g_norm(self.g_net(z_t))                     # (N, T, q)
        else:
            Z = torch.ones(N, T, 1, device=device, dtype=dtype)  # random intercept
        if obs_mask is not None:
            Z = Z * obs_mask.unsqueeze(-1)

        D = self._build_D(device, dtype)
        V_re = (Z @ D) @ Z.transpose(1, 2)

        sig2 = torch.exp(self.log_residual_var).to(device=device, dtype=dtype)
        eye = torch.eye(T, device=device, dtype=dtype).unsqueeze(0)
        V = V_re + (sig2 + self.jitter) * eye

        if return_components:
            return mu, V, Z, D, sig2, reg_dict
        return mu, V


# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────

@dataclass
class NeuralODEConfig:
    hidden_channels: int = 8
    enc_mlp_hidden: int = 32
    func_mlp_hidden: int = 32
    dec_rho_hidden: int = 16
    dec_p: int = 4
    dec_q: int = 3
    depth: int = 2
    dropout: float = 0.0
    euler_steps_per_interval: int = 4    # sub-steps between observation times
    ode_solver: str = "euler"            # euler/midpoint/rk4 (manual) or dopri5/... (torchdiffeq)
    use_rho_norm: bool = False
    beta_init_scale: float = 0.01


# ─────────────────────────────────────────────
# Full model
# ─────────────────────────────────────────────

class NeuralODEModel(nn.Module):
    """
    Unified Neural ODE-LMM. See module docstring for the flag→variant map.
    """

    def __init__(self, n_tv, static_dim=0, cfg=None,
                 use_rho_net=True,
                 use_neural_re=False,
                 g_hidden=16,
                 fullD=True,
                 cov_means=None, cov_stds=None,
                 static_skip_dims=None,
                 use_dynamic_skip=True,
                 reg_mode=None,
                 # ── generalising flags ────────────────────────
                 include_mask=True,
                 use_dynamic_in_ode=True,
                 encode_baseline_cov=True):
        """
        n_tv:               number of time-varying covariates K
        static_dim:         number of static covariates Ks
        include_mask:       x_aug carries a mask block; mask feeds ODE & skip.
                            False ⇒ x_aug = [time, x(K)] (single/multi cov,
                            no missingness channel — baseline/cumulative).
        use_dynamic_in_ode: dynamic covariates feed the ODE vector field
                            (generalises use_bmi_in_ode).
        encode_baseline_cov: encoder sees x_baseline (real) vs [t0, static].
        cov_means/cov_stds: (K,) standardisation for dynamic skip inputs
                            (scalar bmi_mean/std ⇒ pass length-1 vectors).
        """
        super().__init__()
        if cfg is None:
            cfg = NeuralODEConfig()
        self.cfg = cfg
        self.n_tv = n_tv
        self.static_dim = static_dim
        self.reg_mode = reg_mode
        self.include_mask = include_mask
        self.encode_baseline_cov = encode_baseline_cov

        # ── Encoder ─────────────────────────────────────────────────────
        self.encoder = BaselineEncoder(
            n_tv=n_tv,
            static_dim=static_dim,
            hidden_dim=cfg.hidden_channels,
            encode_baseline_cov=encode_baseline_cov,
            mlp_hidden=cfg.enc_mlp_hidden,
            depth=1,
            dropout=cfg.dropout,
        )

        # ── ODE dynamics ────────────────────────────────────────────────
        self.func = ODEFunc(
            hidden_channels=cfg.hidden_channels,
            n_tv=n_tv,
            use_dynamic_in_ode=use_dynamic_in_ode,
            include_mask=include_mask,
            mlp_hidden=cfg.func_mlp_hidden,
            depth=cfg.depth,
            dropout=cfg.dropout,
            activation=nn.SiLU(),
        )

        self.z_norm = nn.LayerNorm(cfg.hidden_channels)

        # ── Decoder ─────────────────────────────────────────────────────
        self.decoder = Decoder(
            latent_dim=cfg.hidden_channels,
            n_tv=n_tv,
            p=cfg.dec_p,
            q=cfg.dec_q,
            rho_hidden=cfg.dec_rho_hidden,
            fullD=fullD,
            use_rho_net=use_rho_net,
            use_neural_re=use_neural_re,
            g_hidden=g_hidden,
            cov_means=cov_means,
            cov_stds=cov_stds,
            static_skip_dims=static_skip_dims,
            use_dynamic_skip=use_dynamic_skip,
            include_mask=include_mask,
            reg_mode=reg_mode,
            use_rho_norm=cfg.use_rho_norm,
            beta_init_scale=cfg.beta_init_scale,
        )

    # ── Integration ─────────────────────────────────────────────────────

    @staticmethod
    def _interp(x_k, x_k1, alpha):
        return (1 - alpha) * x_k + alpha * x_k1

    def _integrate_manual(self, z0, times, x_interp, mask):
        """Fixed-step euler / midpoint / rk4. x_interp linearly interpolated;
        mask held at interval-start (discrete)."""
        N, T = times.shape
        H = z0.shape[1]
        device, dtype = z0.device, z0.dtype
        n_sub = self.cfg.euler_steps_per_interval
        solver = self.cfg.ode_solver

        zt = torch.zeros(N, T, H, device=device, dtype=dtype)
        z = z0
        zt[:, 0] = z

        for k in range(T - 1):
            t_start = times[:, k]
            dt_total = times[:, k + 1] - t_start
            dt_sub = dt_total / n_sub

            x_k = x_interp[:, k] if x_interp is not None else None
            x_k1 = x_interp[:, k + 1] if x_interp is not None else None
            mask_k = mask[:, k] if mask is not None else None

            for s in range(n_sub):
                alpha = s / n_sub
                t_s = t_start + s * dt_sub
                dt = dt_sub.unsqueeze(-1)

                if solver == "euler":
                    x_s = self._interp(x_k, x_k1, alpha) if x_k is not None else None
                    k1 = self.func(z, t_s, x_s, mask_k)
                    z = z + k1 * dt

                elif solver == "midpoint":
                    x_s = self._interp(x_k, x_k1, alpha) if x_k is not None else None
                    k1 = self.func(z, t_s, x_s, mask_k)
                    a_mid = min(alpha + 0.5 / n_sub, 1.0)
                    x_mid = self._interp(x_k, x_k1, a_mid) if x_k is not None else None
                    k2 = self.func(z + k1 * (0.5 * dt), t_s + 0.5 * dt_sub, x_mid, mask_k)
                    z = z + k2 * dt

                elif solver == "rk4":
                    x_s = self._interp(x_k, x_k1, alpha) if x_k is not None else None
                    k1 = self.func(z, t_s, x_s, mask_k)
                    a_mid = min(alpha + 0.5 / n_sub, 1.0)
                    x_mid = self._interp(x_k, x_k1, a_mid) if x_k is not None else None
                    t_mid = t_s + 0.5 * dt_sub
                    k2 = self.func(z + k1 * (0.5 * dt), t_mid, x_mid, mask_k)
                    k3 = self.func(z + k2 * (0.5 * dt), t_mid, x_mid, mask_k)
                    a_end = min(alpha + 1.0 / n_sub, 1.0)
                    x_end = self._interp(x_k, x_k1, a_end) if x_k is not None else None
                    k4 = self.func(z + k3 * dt, t_s + dt_sub, x_end, mask_k)
                    z = z + (k1 + 2 * k2 + 2 * k3 + k4) * (dt / 6)
                else:
                    raise ValueError(f"unknown manual solver {solver!r}")

            zt[:, k + 1] = z
        return zt

    def _integrate_torchdiffeq(self, z0, times, x_interp, mask):
        """torchdiffeq, interval-by-interval in normalised time τ∈[0,1]."""
        from torchdiffeq import odeint

        N, T = times.shape
        H = z0.shape[1]
        device, dtype = z0.device, z0.dtype
        solver = self.cfg.ode_solver

        odeint_kwargs = {"method": solver, "atol": 1e-6, "rtol": 1e-6}
        if self.cfg.euler_steps_per_interval is not None \
                and solver in ("euler", "midpoint", "rk4", "fixed_adams"):
            odeint_kwargs["options"] = {
                "step_size": 1.0 / self.cfg.euler_steps_per_interval}

        zt = torch.zeros(N, T, H, device=device, dtype=dtype)
        z = z0
        zt[:, 0] = z
        tau_eval = torch.tensor([0.0, 1.0], device=device, dtype=dtype)

        for k in range(T - 1):
            self._interval_ctx = {
                "x_k":  x_interp[:, k] if x_interp is not None else None,
                "x_k1": x_interp[:, k + 1] if x_interp is not None else None,
                "mask_k": mask[:, k] if mask is not None else None,
                "t_start": times[:, k],
                "dt_total": times[:, k + 1] - times[:, k],
            }
            z_out = odeint(self._odefunc_normalized, z, tau_eval, **odeint_kwargs)
            z = z_out[-1]
            zt[:, k + 1] = z

        self._interval_ctx = None
        return zt

    def _odefunc_normalized(self, tau, z):
        ctx = self._interval_ctx
        t_abs = ctx["t_start"] + tau * ctx["dt_total"]
        if ctx["x_k"] is not None:
            x_tau = (1 - tau) * ctx["x_k"] + tau * ctx["x_k1"]
        else:
            x_tau = None
        mask_tau = ctx["mask_k"]                                  # discrete: no interp
        dzdt = self.func(z, t_abs, x_tau, mask_tau)
        return dzdt * ctx["dt_total"].unsqueeze(-1)

    def _integrate(self, z0, times, x_interp, mask):
        if self.cfg.ode_solver in {"euler", "midpoint", "rk4"}:
            return self._integrate_manual(z0, times, x_interp, mask)
        return self._integrate_torchdiffeq(z0, times, x_interp, mask)

    # ── Forward ──────────────────────────────────────────────────────────

    def forward(self, x_aug, static_covariates=None,
                obs_mask=None, return_hidden=False):
        """
        x_aug: (N, T, 1 + K + K*include_mask)
               layout = [ time(1) | x_interp(K) | mask(K) if include_mask ]
        static_covariates: (N, Ks)
        obs_mask: (N, T) outcome observation mask

        returns mu, V, Z, D, sig2, reg_dict  (+ zt if return_hidden)
        """
        K = self.n_tv
        N, T, _ = x_aug.shape

        # ── Unpack ──────────────────────────────────────────────────────
        t_pad = x_aug[:, :, 0]                                    # (N, T)
        x_interp = x_aug[:, :, 1:1 + K]                           # (N, T, K)
        mask = x_aug[:, :, 1 + K:1 + 2 * K] if self.include_mask else None

        # ── Encoder ─────────────────────────────────────────────────────
        t0 = t_pad[:, 0:1]                                        # (N, 1)
        enc_parts = [t0]
        if self.encode_baseline_cov:
            enc_parts.append(x_interp[:, 0])                     # (N, K)
        if static_covariates is not None:
            enc_parts.append(static_covariates)
        z0 = self.encoder(torch.cat(enc_parts, dim=-1))          # (N, H)

        # ── ODE integration ─────────────────────────────────────────────
        zt = self._integrate(z0, t_pad, x_interp, mask)
        zt = self.z_norm(zt)

        # ── Decoder ─────────────────────────────────────────────────────
        result = self.decoder(zt, x_interp, mask, static_covariates,
                              obs_mask=obs_mask)

        if return_hidden:
            return (result + (zt,)) if isinstance(result, tuple) else (result, zt)
        return result