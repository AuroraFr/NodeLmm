"""
Neural ODE-LMM with BMI Skip Connection.

Architecture:
  - Encoder:  z(0) = Enc(t0, BMI0, static)
  - Dynamics: dz/dt = f(z, t, static)          ← Neural ODE (no BMI in dynamics)
  - Decoder:  mu(t) = rho(z(t), BMI_std(t)) @ beta_neural   ← BMI only here via skip
  - RE:       Z = [1, rs1(t), rs2(t)]  (classical spline basis)

Key insight: BMI is completely removed from the ODE dynamics.
The CDE/control path machinery is unnecessary — we use simple Euler integration.
BMI enters the model ONLY through the decoder skip connection (standardized).
This guarantees PDP separation at all times: the decoder always sees the intervention value directly.

For path-dependent scenarios (5, 6), switch back to the full CDE model.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple

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
# Encoder
# ─────────────────────────────────────────────

class BaselineEncoder(nn.Module):
    """
    Encode initial hidden state z(0) from:
      - baseline time t0
      - baseline BMI (first observation)
      - static covariates (AGEc, SEX, DIPNIV)
    """
    def __init__(self, input_dim, static_dim, hidden_dim,
                 mlp_hidden=128, depth=2, dropout=0.0):
        super().__init__()
        # input: [t0, BMI0, static]
        self.mlp = MLP(input_dim + static_dim, mlp_hidden, hidden_dim,
                       depth=depth, dropout=dropout)

    def forward(self, x0_and_static):
        return self.mlp(x0_and_static)


# ─────────────────────────────────────────────
# ODE vector field (no BMI — autonomous dynamics)
# ─────────────────────────────────────────────

class ODEFunc(nn.Module):
    """
    dz/dt = f(z, t, static)

    No covariate path, no BMI — the ODE encodes time structure
    and static-covariate-dependent dynamics only.
    """
    def __init__(self, hidden_channels, static_dim,
                 mlp_hidden=64, depth=2, dropout=0.0):
        super().__init__()
        self.hidden_channels = hidden_channels
        # Input: [z(t), t, static]
        self.net = MLP(
            in_dim=hidden_channels + 1 + static_dim + 1,
            hidden_dim=mlp_hidden,
            out_dim=hidden_channels,
            depth=depth, dropout=dropout, activation=nn.ReLU(),
        )

    def forward(self, z, t_scalar, static, bmi_t):
        """
        Args:
            z:       (N, H)
            t_scalar: scalar or (N, 1) — current time
            static:  (N, Cs)
        Returns:
            dz/dt:   (N, H)
        """
        if t_scalar.dim() == 0:
            t_expanded = t_scalar.unsqueeze(0).expand(z.size(0), 1)
        elif t_scalar.dim() == 1:
            t_expanded = t_scalar.unsqueeze(-1)
        else:
            t_expanded = t_scalar
        # inp = torch.cat([z, t_expanded, static], dim=-1)
        inp = torch.cat([z, t_expanded, bmi_t, static], dim=-1)
        return torch.tanh(self.net(inp))


# ─────────────────────────────────────────────
# Decoder with BMI skip connection
# ─────────────────────────────────────────────

class Decoder(nn.Module):
    def __init__(self, latent_dim, p, q=3,
                 fullD=True, jitter=1e-6, rho_hidden=64, D_diag_min=0.1,
                 n_static=4,
                 use_rho_net=True,
                 use_neural_re=True,
                 g_hidden=16,
                 re_spline_cols=None,
                 bmi_mean=0.0, bmi_std=1.0,
                 static_skip_dims=None):
        """
        Args:
            latent_dim: dimension of z(t)
            p: fixed-effect basis dimension (rho output)
            q: random-effect basis dimension
            bmi_mean, bmi_std: for standardizing BMI skip input
            static_skip_dims: list of static covariate dimensions to skip-connect
                              to decoder (e.g., [1] for AGEc). None = no static skip.
            re_spline_cols: column indices in x_pad for RE spline basis
        """
        super().__init__()
        self.p, self.q = p, q
        self.fullD = fullD
        self.jitter = jitter
        self.D_diag_min = D_diag_min
        self.n_static = n_static

        # BMI standardization buffers
        self.register_buffer('bmi_mean', torch.tensor(float(bmi_mean)))
        self.register_buffer('bmi_std', torch.tensor(float(bmi_std)))

        # Static skip connection
        self.static_skip_dims = static_skip_dims
        n_static_skip = len(static_skip_dims) if static_skip_dims else 0

        # Skip dims: 1 (BMI) + n_static_skip
        skip_dim = 1 + n_static_skip

        # --- Neural fixed effects ---
        self.use_rho_net = use_rho_net
        if use_rho_net:
            # MLP: [z(t), BMI_std(t), static_skip...] → R^p
            self.rho_net = MLP(latent_dim + skip_dim, rho_hidden, p, depth=2, dropout=0.0)
            self.rho_norm = nn.LayerNorm(p)
            self.beta_neural = nn.Parameter(0.1 * torch.randn(p))
        else:
            self.rho_net = None
            self.w_neural = nn.Parameter(0.01 * torch.randn(latent_dim + skip_dim))

        # --- Random effects ---
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
                    f"q={q} must equal len(re_spline_cols)+1={len(re_spline_cols)+1}"
                )

        # --- D matrix ---
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
            z_t:      (N, T, H)  latent states from ODE
            t_pad:    (N, T)     padded observation times
            x_pad:    (N, T, Cx) padded covariates (BMI_t at col 0, rs1/rs2 for RE)
            static:   (N, Cs)    static covariates
            obs_mask: (N, T)     binary mask (1=observed, 0=padded)
        """
        N, T, H = z_t.shape
        device, dtype = z_t.device, z_t.dtype

        # --- Build skip connection input ---
        # BMI: always col 0 in x_pad, standardized
        bmi_t = x_pad[:, :, 0:1]                                    # (N, T, 1)
        bmi_std = (bmi_t - self.bmi_mean) / self.bmi_std             # (N, T, 1)

        # Static skip (e.g., AGEc)
        if self.static_skip_dims:
            static_expanded = static[:, self.static_skip_dims]       # (N, n_skip)
            static_expanded = static_expanded.unsqueeze(1).expand(-1, T, -1)  # (N, T, n_skip)
            skip_input = torch.cat([bmi_std, static_expanded], dim=-1)  # (N, T, 1+n_skip)
        else:
            skip_input = bmi_std                                     # (N, T, 1)

        rho_input = torch.cat([z_t, skip_input], dim=-1)            # (N, T, H+skip_dim)

        # --- Fixed effects ---
        if self.use_rho_net:
            rho = self.rho_net(rho_input)                            # (N, T, p)
            rho = self.rho_norm(rho)
            mu = (rho * self.beta_neural).sum(dim=-1)                # (N, T)
        else:
            mu = (rho_input * self.w_neural).sum(dim=-1)             # (N, T)

        # --- Random effects design matrix Z ---
        if self.use_neural_re:
            Z = self.g_net(z_t)     
            Z = self.g_norm(Z)                                 # (N, T, q)
        else:
            ones = torch.ones(N, T, 1, device=device, dtype=dtype)
            if self.re_spline_cols is not None:
                rs_cols = x_pad[:, :, self.re_spline_cols]           # (N, T, q-1)
                Z = torch.cat([ones, rs_cols], dim=-1)               # (N, T, q)
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
            return mu, V, Z, D, sig2
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
    euler_steps_per_interval: int = 4   # sub-steps between observation times


# ─────────────────────────────────────────────
# Full model: Neural ODE + BMI skip
# ─────────────────────────────────────────────

class NeuralODEModel(nn.Module):
    """
    Neural ODE-LMM with BMI skip connection.

    Architecture:
      Encoder:  z(0) = Enc(t0, BMI0, static)
      ODE:      dz/dt = f(z, t, static)       — no BMI in dynamics
      Decoder:  mu = rho(z(t), BMI_std(t)) @ beta_neural — BMI only here

    Forward signature is kept compatible with the CDE model's dataloader.
    The c_mask (cumulative mask) argument is accepted but ignored.
    """

    def __init__(self, x_dim, static_dim=0, cfg=None,
                 n_tv=1,
                 use_rho_net=True,
                 use_neural_re=False,
                 g_hidden=16,
                 re_spline_cols=None,
                 fullD=True,
                 bmi_mean=0.0, bmi_std=1.0,
                 static_skip_dims=None):
        """
        Args:
            x_dim: total columns in x_pad (e.g. 3 for [BMI_t, rs1, rs2])
            static_dim: number of static covariates
            n_tv: number of time-varying covariates (only BMI_t = 1)
            static_skip_dims: indices of static covariates to skip-connect to decoder
                              e.g. [1] to pass AGEc directly. None = no static skip.
        """
        super().__init__()
        if cfg is None:
            cfg = NeuralODEConfig()
        self.cfg = cfg
        self.x_dim = x_dim
        self.static_dim = static_dim
        self.n_tv = n_tv

        # Encoder: sees [t0, BMI0] + static
        # BMI0 is column 0 of x_pad at time 0
        encoder_input_dim = 1 + n_tv   # t0 + BMI0
        self.encoder = BaselineEncoder(
            input_dim=encoder_input_dim,
            static_dim=static_dim,
            hidden_dim=cfg.hidden_channels,
            mlp_hidden=cfg.enc_mlp_hidden,
            depth=1,
            dropout=cfg.dropout,
        )

        # ODE dynamics: f(z, t, static) → dz/dt
        self.func = ODEFunc(
            hidden_channels=cfg.hidden_channels,
            static_dim=static_dim,
            mlp_hidden=cfg.func_mlp_hidden,
            depth=cfg.depth,
            dropout=cfg.dropout,
        )

        self.z_norm = nn.LayerNorm(cfg.hidden_channels)

        # Decoder with BMI skip
        self.decoder = Decoder(
            latent_dim=cfg.hidden_channels,
            p=cfg.dec_p,
            q=cfg.dec_q,
            rho_hidden=cfg.dec_rho_hidden,
            fullD=fullD,
            n_static=static_dim,
            use_rho_net=use_rho_net,
            use_neural_re=use_neural_re,
            g_hidden=g_hidden,
            re_spline_cols=re_spline_cols,
            bmi_mean=bmi_mean,
            bmi_std=bmi_std,
            static_skip_dims=static_skip_dims,
        )

    def _euler_integrate(self, z0, times, static, bmi_t=None):
        """
        Euler integration of dz/dt = f(z, t, static) on the padded time grid.

        Args:
            z0:     (N, H) initial state
            times:  (N, T) observation times
            static: (N, Cs) static covariates
        Returns:
            zt:     (N, T, H) latent states at each time
        """
        N, T = times.shape
        H = z0.shape[1]
        device, dtype = z0.device, z0.dtype
        n_sub = self.cfg.euler_steps_per_interval

        zt = torch.zeros(N, T, H, device=device, dtype=dtype)
        z = z0
        zt[:, 0] = z

        for k in range(T - 1):
            t_start = times[:, k]       # (N,)
            t_end = times[:, k + 1]     # (N,)
            dt_total = t_end - t_start  # (N,)
            dt_sub = dt_total / n_sub   # (N,)

            for s in range(n_sub):
                t_current = t_start + s * dt_sub   # (N,)
                alpha = s / n_sub
                if bmi_t is not None:
                    bmi_current = (1 - alpha) * bmi_t[:, k] + alpha * bmi_t[:, k + 1]  # (N, 1)
                    dzdt = self.func(z, t_current, static, bmi_current)  # (N, H)
                else:
                    dzdt = self.func(z, t_current, static)  # (N, H)
                z = z + dzdt * dt_sub.unsqueeze(-1)      # (N, H)

            zt[:, k + 1] = z

        return zt

    def forward(
        self,
        t_pad,                                    # (N, T)
        x_pad,                                    # (N, T, Cx)
        masks=None,                               # (N, T) cumulative mask — IGNORED
        static_covariates=None,  
        bmi_t = None,                 # (N, Cs)
        obs_mask=None,                            # (N, T) outcome observation mask
        y_pad=None,                               # (N, T)
        return_hidden=False,
        interp=None,                              # ignored (no spline interpolation needed)
    ):
        N, T = t_pad.shape
        device, dtype = t_pad.device, t_pad.dtype

        # --- Encoder ---
        # Input: [t0, BMI0, static]
        t0 = t_pad[:, 0:1]                                  # (N, 1)
        bmi0 = x_pad[:, 0, 0:self.n_tv]                     # (N, n_tv) — baseline BMI
        encoder_in = torch.cat([t0, bmi0, static_covariates], dim=-1)
        z0 = self.encoder(encoder_in)                        # (N, H)

        # --- ODE integration ---
        zt = self._euler_integrate(z0, t_pad, static_covariates, bmi_t)  # (N, T, H)
        zt = self.z_norm(zt)

        # --- Decoder (with BMI skip) ---
        result = self.decoder(zt, t_pad, x_pad, static_covariates,
                              y_pad=y_pad, obs_mask=obs_mask)

        if return_hidden:
            if isinstance(result, tuple):
                return result + (zt,)
            return result, zt
        return result