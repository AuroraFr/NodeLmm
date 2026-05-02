"""
Neural ODE-LMM: multi-covariate version for the 3C cohort.

Architecture:
  - Encoder:  z(0) = Enc(t0, x_baseline, static)
  - Dynamics: dz/dt = f(z, x_interp(t), mask(t), t)     ← no static
  - Skip:     skip(t) = gate ⊙ [x_interp_std(t), mask(t), static]
  - Decoder:  mu(t) = rho(z(t), skip(t))^T beta
  - RE:       Z = g(z(t))

The vector field sees only the latent state, interpolated covariates, mask,
and time — static covariates are excluded so that dz/dt does not depend
directly on demographics. Static information enters z(t) only through z(0).

Skip connection groups for regularisation:
  - Dynamic group k (k=1..K):  (x_interp_k, mask_k) — paired
  - Static group j  (j=1..Ks): one scalar per static covariate
Total n_groups = K + Ks.  The actual skip input dimension = 2K + Ks.

Regularisation modes:
  - None:          no penalty on skip.
  - "skip_gate":   one sigmoid gate per group.
                   Penalty: λ · Σ_g σ(α_g)  (data-independent, smooth).
  - "group_lasso": group L2 on first-layer rho weights per group.
                   Penalty: λ · Σ_g ||W_g||_F  (data-independent, smooth when >0).

Compatible with preprocessed x_aug from Preprocess_3C.py:
  x_aug layout: [time (1) | x_interp (K) | mask (K)]
  Total: 1 + 2K columns.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


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
    Encode initial hidden state z(0) from:
      - baseline time t0          (1)
      - baseline dynamic covs     (K)
      - static covariates          (Ks)
    Total input: 1 + K + Ks
    """
    def __init__(self, n_tv, static_dim, hidden_dim,
                 mlp_hidden=128, depth=2, dropout=0.0):
        super().__init__()
        # input: [t0, x_baseline(K), static(Ks)]
        input_dim = 1 + n_tv + static_dim
        self.mlp = MLP(input_dim, mlp_hidden, hidden_dim,
                       depth=depth, dropout=dropout)

    def forward(self, x0_and_static):
        return self.mlp(x0_and_static)


# ─────────────────────────────────────────────
# ODE vector field
# ─────────────────────────────────────────────

class ODEFunc(nn.Module):
    """
    dz/dt = f(z, x_interp(t), mask(t), t)

    No static covariates — accumulation dynamics do not depend directly
    on demographics.  Static information enters only through z(0).

    Input: [z(H), x_interp(K), mask(K), t(1)]  →  dimension H + 2K + 1
    """
    def __init__(self, hidden_channels, n_tv, 
                 mlp_hidden=64, depth=2, dropout=0.0):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.n_tv = n_tv

        # in_dim = H + K + K + 1 = H + 2K + 1
        self.net = MLP(
            # in_dim=hidden_channels + 2 * n_tv + 1 + static_dim,
            in_dim=hidden_channels + 2 * n_tv + 1,
            hidden_dim=mlp_hidden,
            out_dim=hidden_channels,
            depth=depth, dropout=dropout, activation=nn.SiLU(),
        )

    def forward(self, z, t_scalar, x_interp_t, mask_t):
        """
        Args:
            z:           (N, H)
            t_scalar:    scalar or (N,) or (N, 1) — current time
            x_interp_t:  (N, K) — interpolated covariate values at time t
            mask_t:      (N, K) — observation mask at time t
        Returns:
            dz/dt:       (N, H)
        """
        if t_scalar.dim() == 0:
            t_expanded = t_scalar.unsqueeze(0).expand(z.size(0), 1)
        elif t_scalar.dim() == 1:
            t_expanded = t_scalar.unsqueeze(-1)
        else:
            t_expanded = t_scalar

        inp = torch.cat([z, x_interp_t, mask_t, t_expanded], dim=-1)
        return torch.tanh(self.net(inp))


# ─────────────────────────────────────────────
# Decoder with multi-covariate skip connection
# ─────────────────────────────────────────────

class Decoder(nn.Module):
    """
    Mixed-effects decoder with regularised skip connections.

    Skip input: [x_interp_std(t) (K), mask(t) (K), static_skip (Ks_skip)]
    Actual input to rho_net: [z(t) (H), skip (2K + Ks_skip)]

    Regularisation groups:
      - Dynamic group k (k=0..K-1): 2 columns (x_interp_k, mask_k)
      - Static group j  (j=0..Ks_skip-1): 1 column (static_j)

    For skip_gate:  one scalar gate α_g per group, applied multiplicatively.
    For group_lasso: L2 norm of first-layer weight columns per group.
    """
    def __init__(self, latent_dim, n_tv, p, q=3,
                 fullD=True, jitter=1e-6, rho_hidden=64, D_diag_min=0.1,
                 use_rho_net=True,
                 use_neural_re=True,
                 g_hidden=16,
                 re_spline_cols=None,
                 cov_means=None, cov_stds=None,
                 static_skip_dims=None,
                 use_dynamic_skip=True,
                 reg_mode=None):
        """
        Args:
            latent_dim:   dimension of z(t)
            n_tv:         number of time-varying covariates K
            p:            fixed-effect basis dimension (rho output)
            q:            random-effect basis dimension
            cov_means:    (K,) means for standardizing dynamic skip inputs
            cov_stds:     (K,) stds for standardizing dynamic skip inputs
            static_skip_dims: indices of static covariates to skip-connect.
                              None or [] = no static skip.
            use_dynamic_skip: if False, no dynamic covariates in skip path
                              (all signal must go through the ODE).
            reg_mode:     None, "skip_gate", or "group_lasso".
        """
        super().__init__()
        self.p, self.q = p, q
        self.fullD = fullD
        self.jitter = jitter
        self.D_diag_min = D_diag_min
        self.reg_mode = reg_mode
        self.latent_dim = latent_dim
        self.n_tv = n_tv

        # ── Standardization buffers for dynamic covariates ──────────────
        if cov_means is not None:
            self.register_buffer('cov_means', torch.tensor(cov_means, dtype=torch.float32))
        else:
            self.register_buffer('cov_means', torch.zeros(n_tv))
        if cov_stds is not None:
            self.register_buffer('cov_stds', torch.tensor(cov_stds, dtype=torch.float32))
        else:
            self.register_buffer('cov_stds', torch.ones(n_tv))

        # ── Skip connection layout ──────────────────────────────────────
        self.use_dynamic_skip = use_dynamic_skip
        self.static_skip_dims = static_skip_dims if static_skip_dims else []
        n_static_skip = len(self.static_skip_dims)

        # Number of regularisation groups
        n_dynamic_groups = n_tv if use_dynamic_skip else 0
        self.n_groups = n_dynamic_groups + n_static_skip

        # Actual skip input dimension (each dynamic group = 2 cols: x_k + mask_k)
        skip_dim_dynamic = 2 * n_tv if use_dynamic_skip else 0
        self.skip_dim = skip_dim_dynamic + n_static_skip

        # Group → column index mapping for regularisation
        # Each entry: list of column indices in the skip input tensor
        self._group_col_indices = []
        if use_dynamic_skip:
            for k in range(n_tv):
                # x_interp_k at position k, mask_k at position n_tv + k
                # skip layout: [x_std(K), mask(K), static(Ks)]
                self._group_col_indices.append([k, n_tv + k])
            col_offset = 2 * n_tv
        else:
            col_offset = 0
        for j in range(n_static_skip):
            self._group_col_indices.append([col_offset + j])

        # Pre-build column → group mapping for skip_gate (autograd-safe)
        col_to_group = torch.zeros(self.skip_dim, dtype=torch.long)
        for g, cols in enumerate(self._group_col_indices):
            for c in cols:
                col_to_group[c] = g
        self.register_buffer('_gate_col_to_group', col_to_group)

        # ── Neural fixed effects ────────────────────────────────────────
        self.use_rho_net = use_rho_net

        if use_rho_net:
            self.rho_net = MLP(latent_dim + self.skip_dim, rho_hidden, p,
                               depth=2, dropout=0.0)
            self.rho_norm = nn.LayerNorm(p)
            # self.rho_norm = nn.Identity(p)
            self.beta_neural = nn.Parameter(0.1 * torch.randn(p))
        else:
            self.rho_net = None
            self.w_neural = nn.Parameter(
                0.01 * torch.randn(latent_dim + self.skip_dim))
            self.beta_neural = None

        # ── Skip gate (one gate per group) ──────────────────────────────
        if reg_mode == "skip_gate" and self.n_groups > 0:
            # Initialise at 2.0 → σ(2) ≈ 0.88 (gates start open)
            self.skip_gate_logit = nn.Parameter(
                2.0 * torch.ones(self.n_groups))
        else:
            self.skip_gate_logit = None

        # ── Random effects ──────────────────────────────────────────────
        self.use_neural_re = use_neural_re
        if use_neural_re:
            self.g_net = MLP(latent_dim, g_hidden, q, depth=2, dropout=0.0)
            self.g_norm = nn.LayerNorm(q)
            self.re_spline_cols = None
        else:
            self.g_net = None
            self.re_spline_cols = re_spline_cols
            if re_spline_cols is not None:
                assert q == len(re_spline_cols) + 1, (
                    f"q={q} must equal len(re_spline_cols)+1="
                    f"{len(re_spline_cols)+1}"
                )

        # ── D matrix ────────────────────────────────────────────────────
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

    def _build_skip_input(self, x_interp, mask, static, N, T):
        """
        Build skip input: [x_interp_std(t), mask(t), static_skip].

        Args:
            x_interp: (N, T, K) interpolated dynamic covariates
            mask:     (N, T, K) observation mask (binary or cumulative)
            static:   (N, Ks) static covariates

        Returns:
            skip_input: (N, T, skip_dim) or None if skip_dim == 0
        """
        parts = []

        if self.use_dynamic_skip:
            # Standardize dynamic covariates
            x_std = (x_interp - self.cov_means) / self.cov_stds  # (N, T, K)
            parts.append(x_std)
            parts.append(mask)                                     # (N, T, K)

        if self.static_skip_dims:
            s_skip = static[:, self.static_skip_dims]              # (N, n_skip)
            s_skip = s_skip.unsqueeze(1).expand(-1, T, -1)         # (N, T, n_skip)
            parts.append(s_skip)

        if parts:
            return torch.cat(parts, dim=-1)                        # (N, T, skip_dim)
        return None

    def _apply_gate(self, skip_input):
        """
        Apply per-group sigmoid gates to skip input (autograd-safe).

        Builds a (skip_dim,) gate vector where each position has the
        gate value of its group, then broadcasts a single multiply.
        No in-place operations.
        """
        if self.skip_gate_logit is None or skip_input is None:
            return skip_input

        gate_per_group = torch.sigmoid(self.skip_gate_logit)  # (n_groups,)

        # Build (skip_dim,) gate vector: each col gets its group's gate
        # _gate_col_to_group is registered in __init__ as (skip_dim,) LongTensor
        gate_expanded = gate_per_group[self._gate_col_to_group]  # (skip_dim,)

        # Broadcast: (N, T, skip_dim) * (skip_dim,)
        return skip_input * gate_expanded

    def _compute_reg(self):
        """
        Compute regularisation term (data-independent).

        skip_gate:   λ · Σ_g σ(α_g)
        group_lasso: λ · Σ_g ||W_g||_F  where W_g are the first-layer
                     rho_net columns corresponding to group g's skip inputs.
        """
        reg_dict = {}

        if self.reg_mode == "skip_gate" and self.skip_gate_logit is not None:
            gate = torch.sigmoid(self.skip_gate_logit)
            reg_dict["reg_term"] = gate.sum()
            reg_dict["gate_values"] = gate.detach()

        elif self.reg_mode == "group_lasso" and self.use_rho_net:
            W = self.rho_net.net[0].weight                   # (h_out, h_in)
            # Skip columns start after latent_dim
            offset = self.latent_dim
            group_norms = []
            for cols in self._group_col_indices:
                W_cols = W[:, [offset + c for c in cols]]    # (h_out, |group|)
                group_norms.append(W_cols.norm(p='fro'))
            group_norms = torch.stack(group_norms)
            reg_dict["reg_term"] = group_norms.sum()
            reg_dict["group_norms"] = group_norms.detach()

        else:
            reg_dict["reg_term"] = torch.tensor(0.0)

        return reg_dict

    def forward(self, z_t, x_interp, mask, static,
                obs_mask=None, re_basis=None, return_components=True):
        """
        Args:
            z_t:       (N, T, H)  latent states from ODE
            x_interp:  (N, T, K)  interpolated dynamic covariates
            mask:      (N, T, K)  observation mask
            static:    (N, Ks)    static covariates
            obs_mask:  (N, T)     binary mask for observed outcomes
            re_basis:  (N, T, q-1) optional precomputed RE spline basis

        Returns:
            mu:       (N, T)     population mean prediction
            V:        (N, T, T)  marginal covariance
            Z:        (N, T, q)  RE design matrix
            D:        (q, q)     RE covariance
            sig2:     scalar     residual variance
            reg_dict: dict with regularisation info
        """
        N, T, H = z_t.shape
        device, dtype = z_t.device, z_t.dtype

        # ── Build and gate skip input ───────────────────────────────────
        skip_input = self._build_skip_input(x_interp, mask, static, N, T)

        if self.reg_mode == "skip_gate":
            skip_input = self._apply_gate(skip_input)

        # ── Fixed effects ───────────────────────────────────────────────
        if skip_input is not None:
            rho_input = torch.cat([z_t, skip_input], dim=-1)
        else:
            rho_input = z_t

        if self.use_rho_net:
            rho = self.rho_net(rho_input)
            rho = self.rho_norm(rho)
            mu = (rho * self.beta_neural).sum(dim=-1)         # (N, T)
        else:
            mu = (rho_input * self.w_neural).sum(dim=-1)

        # ── Regularisation (data-independent) ───────────────────────────
        reg_dict = self._compute_reg()

        # ── Random effects design matrix Z ──────────────────────────────
        if self.use_neural_re:
            Z = self.g_net(z_t)
            Z = self.g_norm(Z)                                 # (N, T, q)
        else:
            ones = torch.ones(N, T, 1, device=device, dtype=dtype)
            if self.re_spline_cols is not None and re_basis is not None:
                Z = torch.cat([ones, re_basis], dim=-1)        # (N, T, q)
            else:
                Z = ones

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
    euler_steps_per_interval: int = 4
    ode_solver: str = "euler"
    # use_rho_norm: False


# ─────────────────────────────────────────────
# Full model
# ─────────────────────────────────────────────

class NeuralODEModel(nn.Module):
    """
    Neural ODE-LMM: multi-covariate version.

    Data flow:
      x_aug from Preprocess_3C → split into x_interp (K), mask (K), time
      Encoder:  z(0) = Enc(t0, x_baseline, static)
      ODE:      dz/dt = f(z, x_interp(t), mask(t), t)
      Skip:     gate ⊙ [x_interp_std(t), mask(t), static_skip]
      Decoder:  mu = rho(z(t), skip(t))^T beta + g(z(t))^T b_i + eps

    Regularisation modes (reg_mode):
      None:          no skip penalty
      "skip_gate":   sigmoid gate per group, penalty = λ·Σ σ(α_g)
      "group_lasso": group L2 on rho first-layer, penalty = λ·Σ||W_g||_F

    Both penalties are data-independent and smooth, enabling
    M-estimator inference (Commenges et al., 2014).
    """

    def __init__(self, n_tv, static_dim=0, cfg=None,
                 use_rho_net=True,
                 use_neural_re=False,
                 g_hidden=16,
                 re_spline_cols=None,
                 fullD=True,
                 cov_means=None, cov_stds=None,
                 static_skip_dims=None,
                 use_dynamic_skip=True,
                 reg_mode=None):
        """
        Args:
            n_tv:             number of time-varying covariates K
            static_dim:       total number of static covariates Ks
            cov_means:        (K,) means for standardizing skip inputs
            cov_stds:         (K,) stds for standardizing skip inputs
            static_skip_dims: indices of static covariates to skip-connect
                              to decoder (e.g. [0] for AGEc). None = none.
            use_dynamic_skip: if False, decoder sees only z(t) for dynamics.
            reg_mode:         None, "skip_gate", or "group_lasso".
        """
        super().__init__()
        if cfg is None:
            cfg = NeuralODEConfig()
        self.cfg = cfg
        self.n_tv = n_tv
        self.static_dim = static_dim
        self.reg_mode = reg_mode

        # ── Encoder: z(0) = Enc(t0, x_baseline, static) ────────────────
        self.encoder = BaselineEncoder(
            n_tv=n_tv,
            static_dim=static_dim,
            hidden_dim=cfg.hidden_channels,
            mlp_hidden=cfg.enc_mlp_hidden,
            depth=1,
            dropout=cfg.dropout,
        )

        # ── ODE dynamics: f(z, x_interp, mask, t) ──────────────────────
        self.func = ODEFunc(
            hidden_channels=cfg.hidden_channels,
            n_tv=n_tv,
            mlp_hidden=cfg.func_mlp_hidden,
            depth=cfg.depth,
            dropout=cfg.dropout,
        )

        self.z_norm = nn.LayerNorm(cfg.hidden_channels)

        # ── Decoder with skip ───────────────────────────────────────────
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
            re_spline_cols=re_spline_cols,
            cov_means=cov_means,
            cov_stds=cov_stds,
            static_skip_dims=static_skip_dims,
            use_dynamic_skip=use_dynamic_skip,
            reg_mode=reg_mode,
        )

    # ── Sub-step interpolation for ODE integration ──────────────────────

    def _interp_substep(self, x_k, x_k1, alpha):
        """
        Linearly interpolate x_interp between grid points k and k+1.

        Args:
            x_k:   (N, K) covariate values at grid point k
            x_k1:  (N, K) covariate values at grid point k+1
            alpha:  float in [0, 1], interpolation fraction
        Returns:
            x_sub: (N, K) interpolated values
        """
        return (1 - alpha) * x_k + alpha * x_k1

    # ── Manual integration ──────────────────────────────────────────────

    def _integrate_manual(self, z0, times, x_interp, mask):
        """
        Manual fixed-step integration: euler, midpoint, or rk4.

        x_interp is linearly interpolated between grid points.
        mask is held constant at the interval-start value within each
        interval (it is discrete and should not be interpolated).
        """
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

            # Covariates at grid endpoints
            x_k = x_interp[:, k]                               # (N, K)
            x_k1 = x_interp[:, k + 1]                          # (N, K)
            mask_k = mask[:, k]                                 # (N, K) held constant

            for s in range(n_sub):
                alpha = s / n_sub
                t_s = t_start + s * dt_sub
                dt = dt_sub.unsqueeze(-1)                       # (N, 1)

                if solver == "euler":
                    x_s = self._interp_substep(x_k, x_k1, alpha)
                    k1 = self.func(z, t_s, x_s, mask_k)
                    z = z + k1 * dt

                elif solver == "midpoint":
                    x_s = self._interp_substep(x_k, x_k1, alpha)
                    k1 = self.func(z, t_s, x_s, mask_k)
                    alpha_mid = min(alpha + 0.5 / n_sub, 1.0)
                    x_mid = self._interp_substep(x_k, x_k1, alpha_mid)
                    k2 = self.func(
                        z + k1 * (0.5 * dt),
                        t_s + 0.5 * dt_sub,
                        x_mid, mask_k)
                    z = z + k2 * dt

                elif solver == "rk4":
                    x_s = self._interp_substep(x_k, x_k1, alpha)
                    k1 = self.func(z, t_s, x_s, mask_k)

                    alpha_mid = min(alpha + 0.5 / n_sub, 1.0)
                    x_mid = self._interp_substep(x_k, x_k1, alpha_mid)
                    t_mid = t_s + 0.5 * dt_sub
                    k2 = self.func(z + k1 * (0.5 * dt), t_mid, x_mid, mask_k)
                    k3 = self.func(z + k2 * (0.5 * dt), t_mid, x_mid, mask_k)

                    alpha_end = min(alpha + 1.0 / n_sub, 1.0)
                    x_end = self._interp_substep(x_k, x_k1, alpha_end)
                    k4 = self.func(z + k3 * dt, t_s + dt_sub, x_end, mask_k)

                    z = z + (k1 + 2*k2 + 2*k3 + k4) * (dt / 6)

            zt[:, k + 1] = z

        return zt

    # ── torchdiffeq integration ─────────────────────────────────────────

    def _integrate_torchdiffeq(self, z0, times, x_interp, mask):
        """
        Integration via torchdiffeq, interval-by-interval in normalised
        time τ ∈ [0, 1].

        x_interp linearly interpolated; mask held at interval start.
        """
        from torchdiffeq import odeint

        N, T = times.shape
        H = z0.shape[1]
        device, dtype = z0.device, z0.dtype
        solver = self.cfg.ode_solver

        odeint_kwargs = {
            "method": solver,
            "atol": 1e-6,
            "rtol": 1e-6,
        }
        if self.cfg.euler_steps_per_interval is not None:
            odeint_kwargs["options"] = {
                "step_size": 1.0 / self.cfg.euler_steps_per_interval
            }

        zt = torch.zeros(N, T, H, device=device, dtype=dtype)
        z = z0
        zt[:, 0] = z

        tau_eval = torch.tensor([0.0, 1.0], device=device, dtype=dtype)

        for k in range(T - 1):
            t_start = times[:, k]
            dt_total = times[:, k + 1] - t_start

            self._interval_ctx = {
                "x_k": x_interp[:, k],                         # (N, K)
                "x_k1": x_interp[:, k + 1],                    # (N, K)
                "mask_k": mask[:, k],                           # (N, K)
                "t_start": t_start,
                "dt_total": dt_total,
            }

            z_out = odeint(self._odefunc_normalized, z, tau_eval,
                           **odeint_kwargs)
            z = z_out[-1]
            zt[:, k + 1] = z

        self._interval_ctx = None
        return zt

    def _odefunc_normalized(self, tau, z):
        """
        ODE in normalised time: dz/dτ = f(z, x(τ), mask, t(τ)) · dt_total.
        """
        ctx = self._interval_ctx
        t_abs = ctx["t_start"] + tau * ctx["dt_total"]

        # Linear interpolation of covariates; mask held constant
        x_tau = (1 - tau) * ctx["x_k"] + tau * ctx["x_k1"]
        mask_tau = ctx["mask_k"]                                # discrete: no interp

        dzdt = self.func(z, t_abs, x_tau, mask_tau)
        return dzdt * ctx["dt_total"].unsqueeze(-1)

    def _integrate(self, z0, times, x_interp, mask):
        """Dispatch to manual or torchdiffeq integration."""
        manual_solvers = {"euler", "midpoint", "rk4"}
        if self.cfg.ode_solver in manual_solvers:
            return self._integrate_manual(z0, times, x_interp, mask)
        else:
            return self._integrate_torchdiffeq(z0, times, x_interp, mask)

    # ── Forward pass ────────────────────────────────────────────────────

    def forward(
        self,
        x_aug,                          # (N, T, 1+2K) from Preprocess_3C
        static_covariates=None,         # (N, Ks)
        obs_mask=None,                  # (N, T) outcome observation mask
        re_basis=None,                  # (N, T, q-1) optional RE spline basis
        return_hidden=False,
    ):
        """
        Args:
            x_aug:  (N, T, 1+2K)  preprocessed augmented input.
                    Layout: [time(1), x_interp(K), mask(K)]
            static_covariates: (N, Ks) static covariates
            obs_mask: (N, T) binary mask for observed outcomes
            re_basis: (N, T, q-1) optional precomputed RE spline basis
            return_hidden: if True, also return z(t)

        Returns:
            mu, V, Z, D, sig2, reg_dict  (and optionally zt)
        """
        K = self.n_tv
        N, T, _ = x_aug.shape

        # ── Unpack x_aug ────────────────────────────────────────────────
        t_pad = x_aug[:, :, 0]                                 # (N, T)
        x_interp = x_aug[:, :, 1:1+K]                          # (N, T, K)
        mask = x_aug[:, :, 1+K:1+2*K]                          # (N, T, K)

        # ── Encoder ─────────────────────────────────────────────────────
        t0 = t_pad[:, 0:1]                                     # (N, 1)
        x_baseline = x_interp[:, 0]                             # (N, K)
        encoder_in = torch.cat([t0, x_baseline, static_covariates], dim=-1)
        z0 = self.encoder(encoder_in)                           # (N, H)

        # ── ODE integration ─────────────────────────────────────────────
        zt = self._integrate(z0, t_pad, x_interp, mask)
        zt = self.z_norm(zt)

        # ── Decoder ─────────────────────────────────────────────────────
        result = self.decoder(
            zt, x_interp, mask, static_covariates,
            obs_mask=obs_mask, re_basis=re_basis)

        if return_hidden:
            if isinstance(result, tuple):
                return result + (zt,)
            return result, zt
        return result