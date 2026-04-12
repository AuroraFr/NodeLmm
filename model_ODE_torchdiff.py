"""
Neural ODE-LMM with BMI Skip Connection.

Architecture:
  - Encoder:  z(0) = Enc(t0, static)
  - Dynamics: dz/dt = f(z, t, BMI(t))          ← Neural ODE
  - Decoder:  mu(t) = rho(z(t), BMI_std(t)) @ beta_neural 
  - RE:       Z = g(z(t))

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
# ODE vector field
# ─────────────────────────────────────────────

class ODEFunc(nn.Module):
    """
    dz/dt = f(z, t, static, bmi(t))
    """
    def __init__(self, hidden_channels, static_dim,
                 mlp_hidden=64, depth=2, dropout=0.0):
        super().__init__()
        self.hidden_channels = hidden_channels
        
        self.net = MLP(
            # in_dim=hidden_channels + 1 + static_dim + 1,
            in_dim=hidden_channels + 1 + 1,
            hidden_dim=mlp_hidden,
            out_dim=hidden_channels,
            depth=depth, dropout=dropout, activation=nn.SiLU(),
        )

    def forward(self, z, t_scalar, static, bmi_t):
        """
        Args:
            z:       (N, H)
            t_scalar: scalar or (N, 1) — current time
            static:  (N, Cs)  — currently unused (commented out in inp)
            bmi_t:   (N, 1) — current BMI value
        Returns:
            dz/dt:   (N, H)
        """
        if t_scalar.dim() == 0:
            t_expanded = t_scalar.unsqueeze(0).expand(z.size(0), 1)
        elif t_scalar.dim() == 1:
            t_expanded = t_scalar.unsqueeze(-1)
        else:
            t_expanded = t_scalar

        # inp = torch.cat([z, t_expanded, bmi_t, static], dim=-1)
        inp = torch.cat([z, t_expanded, bmi_t], dim=-1)
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
                 static_skip_dims=None,
                 use_bmi_skip=True,
                 reg_mode=None,
                 p_skip=None):
        """
        Args:
            latent_dim: dimension of z(t)
            p: fixed-effect basis dimension (rho output for ODE pathway)
            q: random-effect basis dimension
            bmi_mean, bmi_std: for standardizing BMI skip input
            static_skip_dims: list of static covariate dimensions to skip-connect
                              to decoder (e.g., [1] for AGEc). None = no static skip.
            re_spline_cols: column indices in x_pad for RE spline basis
            use_bmi_skip: if False, decoder sees only z(t) — no direct BMI input.
                          Use False for path-dependent scenarios (S5, S6).
            reg_mode: None, "l1_skip", or "grad_penalty".
                None:           standard decoder, single rho network.
                "l1_skip":      single rho network, penalty on |mu_full - mu_no_skip|.
                                Penalizes marginal skip contribution → parsimony.
                "grad_penalty": two separate networks (ODE + skip pathways):
                                  mu = mu_ode(z(t)) + mu_skip(BMI, AGEc)
                                Gradient penalty ∂mu_ode/∂BMI(t) is computed at the
                                NeuralODEModel level to penalize BMI signal leaking
                                into the ODE pathway.
            p_skip: basis dimension for the skip pathway in grad_penalty mode.
                    Defaults to p if not specified.

        Theory:
            reg_mode="l1_skip":
                L = NLL + lambda * E[|mu(z,s) - mu(z,0)|]
                Encourages the optimizer to route signal through z(t) unless
                the skip genuinely improves fit. The L1 penalty induces sparsity
                in the skip's functional contribution.

            reg_mode="grad_penalty":
                mu(t) = mu_ode(z(t)) + mu_skip(BMI(t), AGEc)
                L = NLL + lambda * E[(∂mu_ode/∂BMI(t))²]
                Penalizes instantaneous sensitivity of the ODE pathway to BMI,
                pushing BMI effects into the skip pathway where they belong
                for instantaneous-effect scenarios.
        """
        super().__init__()
        self.p, self.q = p, q
        self.fullD = fullD
        self.jitter = jitter
        self.D_diag_min = D_diag_min
        self.n_static = n_static
        self.reg_mode = reg_mode

        # BMI standardization buffers
        self.register_buffer('bmi_mean', torch.tensor(float(bmi_mean)))
        self.register_buffer('bmi_std', torch.tensor(float(bmi_std)))

        # Skip connections
        self.use_bmi_skip = use_bmi_skip
        self.static_skip_dims = static_skip_dims
        n_static_skip = len(static_skip_dims) if static_skip_dims else 0

        # Skip dims: BMI (if enabled) + static skips
        n_bmi_skip = 1 if use_bmi_skip else 0
        skip_dim = n_bmi_skip + n_static_skip
        self.skip_dim = skip_dim

        if reg_mode == "grad_penalty" and skip_dim == 0:
            raise ValueError("reg_mode='grad_penalty' requires use_bmi_skip=True or static_skip_dims")

        if p_skip is None:
            p_skip = p
        self.p_skip = p_skip

        # --- Neural fixed effects ---
        self.use_rho_net = use_rho_net

        if reg_mode == "grad_penalty":
            # Two separate pathways: mu = mu_ode(z(t)) + mu_skip(BMI, AGEc)
            # Gradient penalty on ∂mu_ode/∂BMI is computed at NeuralODEModel level.
            # ODE pathway: z(t) → R^p → scalar
            self.rho_ode_net = MLP(latent_dim, rho_hidden, p, depth=2, dropout=0.0)
            self.rho_ode_norm = nn.LayerNorm(p)
            self.beta_ode = nn.Parameter(0.1 * torch.randn(p))

            # Skip pathway: [BMI_std, AGEc, ...] → R^p_skip → scalar
            self.rho_skip_net = MLP(skip_dim, rho_hidden, p_skip, depth=2, dropout=0.0)
            self.rho_skip_norm = nn.LayerNorm(p_skip)
            self.beta_skip = nn.Parameter(0.1 * torch.randn(p_skip))

            # For backward compat: expose beta_neural as reference
            self.beta_neural = self.beta_ode

            self.rho_net = None  # not used in grad_penalty mode

        elif use_rho_net:
            # Standard single network: [z(t), skip] → R^p → scalar
            self.rho_net = MLP(latent_dim + skip_dim, rho_hidden, p, depth=2, dropout=0.0)
            self.rho_norm = nn.LayerNorm(p)
            self.beta_neural = nn.Parameter(0.1 * torch.randn(p))

            # Not used in standard mode
            self.rho_ode_net = None
            self.rho_skip_net = None
        else:
            self.rho_net = None
            self.rho_ode_net = None
            self.rho_skip_net = None
            self.w_neural = nn.Parameter(0.01 * torch.randn(latent_dim + skip_dim))
            self.beta_neural = None

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

    def _build_skip_input(self, x_pad, static, N, T):
        """Build the skip connection input tensor. Returns (N, T, skip_dim) or None."""
        skip_parts = []

        if self.use_bmi_skip:
            bmi_t = x_pad[:, :, 0:1]                                    # (N, T, 1)
            bmi_std = (bmi_t - self.bmi_mean) / self.bmi_std             # (N, T, 1)
            skip_parts.append(bmi_std)

        if self.static_skip_dims:
            static_expanded = static[:, self.static_skip_dims]           # (N, n_skip)
            static_expanded = static_expanded.unsqueeze(1).expand(-1, T, -1)
            skip_parts.append(static_expanded)

        if skip_parts:
            return torch.cat(skip_parts, dim=-1)                         # (N, T, skip_dim)
        return None

    def forward(self, z_t, x_pad, static, obs_mask=None,
                return_components=True):
        """
        Args:
            z_t:      (N, T, H)  latent states from ODE
            x_pad:    (N, T, Cx) padded covariates (BMI_t at col 0, rs1/rs2 for RE)
            static:   (N, Cs)    static covariates
            obs_mask: (N, T)     binary mask (1=observed, 0=padded)

        Returns:
            mu:       (N, T)     population mean prediction
            V:        (N, T, T)  marginal covariance
            Z:        (N, T, q)  RE design matrix
            D:        (q, q)     RE covariance
            sig2:     scalar     residual variance
            reg_dict: dict with regularization info (only when reg_mode is set)
                      Keys: "reg_term" (scalar loss), "mu_ode" (N,T), "mu_skip" (N,T)
        """
        N, T, H = z_t.shape
        device, dtype = z_t.device, z_t.dtype

        # --- Build skip input ---
        skip_input = self._build_skip_input(x_pad, static, N, T)

        # --- Fixed effects (depends on reg_mode) ---
        reg_dict = {}

        if self.reg_mode == "grad_penalty":
            # ============================================
            # Gradient penalty: mu = mu_ode + mu_skip
            # Penalty is computed at the NeuralODEModel level
            # (requires differentiating through Euler integration)
            # ============================================
            # ODE pathway
            rho_ode = self.rho_ode_net(z_t)                              # (N, T, p)
            rho_ode = self.rho_ode_norm(rho_ode)
            mu_ode = (rho_ode * self.beta_ode).sum(dim=-1)               # (N, T)
 
            # Skip pathway
            rho_skip = self.rho_skip_net(skip_input)                     # (N, T, p_skip)
            rho_skip = self.rho_skip_norm(rho_skip)
            mu_skip = (rho_skip * self.beta_skip).sum(dim=-1)            # (N, T)
 
            mu = mu_ode + mu_skip
 
            # No penalty here — grad penalty computed at model level
            # Return mu_ode so model can differentiate it w.r.t. BMI
            reg_dict["mu_ode"] = mu_ode                                  # keep graph!
            reg_dict["mu_skip"] = mu_skip.detach()

        elif self.reg_mode == "l1_skip" and self.use_rho_net:
            # ============================================
            # L1 skip: penalize |mu_full - mu_no_skip|
            # ============================================
            # Full prediction (with skip)
            if skip_input is not None:
                rho_input = torch.cat([z_t, skip_input], dim=-1)
            else:
                rho_input = z_t

            rho = self.rho_net(rho_input)
            rho = self.rho_norm(rho)
            mu = (rho * self.beta_neural).sum(dim=-1)

            # Zero-skip prediction (skip input replaced with zeros)
            if skip_input is not None:
                zero_skip = torch.zeros_like(skip_input)
                rho_input_zero = torch.cat([z_t, zero_skip], dim=-1)
                rho_zero = self.rho_net(rho_input_zero)
                rho_zero = self.rho_norm(rho_zero)
                mu_no_skip = (rho_zero * self.beta_neural).sum(dim=-1)

                # L1 penalty on marginal skip contribution
                skip_contrib = (mu - mu_no_skip).abs()
                if obs_mask is not None:
                    # Average over observed positions only
                    n_obs = obs_mask.sum().clamp(min=1)
                    reg_dict["reg_term"] = (skip_contrib * obs_mask).sum() / n_obs
                else:
                    reg_dict["reg_term"] = skip_contrib.mean()

                reg_dict["mu_full"] = mu.detach()
                reg_dict["mu_no_skip"] = mu_no_skip.detach()
                reg_dict["skip_contrib_mean"] = skip_contrib.detach().mean()
            else:
                reg_dict["reg_term"] = torch.tensor(0.0, device=device, dtype=dtype)

        else:
            # ============================================
            # Standard: single network, no regularization
            # ============================================
            if skip_input is not None:
                rho_input = torch.cat([z_t, skip_input], dim=-1)
            else:
                rho_input = z_t

            if self.use_rho_net:
                rho = self.rho_net(rho_input)
                rho = self.rho_norm(rho)
                mu = (rho * self.beta_neural).sum(dim=-1)
            else:
                mu = (rho_input * self.w_neural).sum(dim=-1)

        # --- Random effects design matrix Z ---
        if self.use_neural_re:
            Z = self.g_net(z_t)
            Z = self.g_norm(Z)                                           # (N, T, q)
        else:
            ones = torch.ones(N, T, 1, device=device, dtype=dtype)
            if self.re_spline_cols is not None:
                rs_cols = x_pad[:, :, self.re_spline_cols]               # (N, T, q-1)
                Z = torch.cat([ones, rs_cols], dim=-1)                   # (N, T, q)
            else:
                Z = ones

        if obs_mask is not None:
            Z = Z * obs_mask.unsqueeze(-1)

        D = self._build_D(device, dtype)
        V_re = (Z @ D) @ Z.transpose(1, 2)

        sig2 = torch.exp(self.log_residual_var).to(device=device, dtype=dtype)
        eye = torch.eye(T, device=device, dtype=dtype).unsqueeze(0).expand(N, T, T)
        V = V_re + (sig2 + self.jitter) * eye #TODO: really need self.jitter here?

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
    euler_steps_per_interval: int = 4   # sub-steps between observation times
    ode_solver: str = "euler"            # "euler"/"midpoint"/"rk4" (manual), or "dopri5"/"adams" (torchdiffeq)


# ─────────────────────────────────────────────
# Full model: Neural ODE + BMI skip
# ─────────────────────────────────────────────

class NeuralODEModel(nn.Module):
    """
    Neural ODE-LMM with BMI skip connection.

    Architecture:
      Encoder:  z(0) = Enc(t0, BMI0, static)
      ODE:      dz/dt = f(z, t, BMI(t))            — BMI in dynamics
      Decoder:  mu = rho(z(t), BMI_std(t)) @ beta   — BMI also in skip

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
                 static_skip_dims=None,
                 use_bmi_skip=True,
                 reg_mode=None,
                 p_skip=None):
        """
        Args:
            x_dim: total columns in x_pad (e.g. 3 for [BMI_t, rs1, rs2])
            static_dim: number of static covariates
            n_tv: number of time-varying covariates (only BMI_t = 1)
            static_skip_dims: indices of static covariates to skip-connect to decoder
                              e.g. [1] to pass AGEc directly. None = no static skip.
            use_bmi_skip: if False, decoder sees only z(t) — no direct BMI input.
                          Use False for path-dependent scenarios (S5, S6).
            reg_mode: None, "l1_skip", or "grad_penalty". See Decoder docstring.
            p_skip: basis dimension for skip pathway in grad_penalty mode.
        """
        super().__init__()
        if cfg is None:
            cfg = NeuralODEConfig()
        self.cfg = cfg
        self.x_dim = x_dim
        self.static_dim = static_dim
        self.n_tv = n_tv
        self.reg_mode = reg_mode

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

        # ODE dynamics: f(z, t, bmi(t)) → dz/dt
        self.func = ODEFunc(
            hidden_channels=cfg.hidden_channels,
            static_dim=static_dim,
            mlp_hidden=cfg.func_mlp_hidden,
            depth=cfg.depth,
            dropout=cfg.dropout,
        )

        self.z_norm = nn.LayerNorm(cfg.hidden_channels)

        # Decoder with optional BMI skip
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
            use_bmi_skip=use_bmi_skip,
            reg_mode=reg_mode,
            p_skip=p_skip,
        )

    def _interpolate_bmi(self, bmi_t, k, alpha):
        """Linearly interpolate BMI between observation times k and k+1."""
        if bmi_t is None:
            return None
        return (1 - alpha) * bmi_t[:, k] + alpha * bmi_t[:, k + 1]

    def _integrate_manual(self, z0, times, static, bmi_t=None):
        """
        Manual fixed-step integration: euler, midpoint, or rk4.
        Handles per-subject time grids natively.
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

            for s in range(n_sub):
                alpha = s / n_sub
                t_s = t_start + s * dt_sub
                dt = dt_sub.unsqueeze(-1)

                if solver == "euler":
                    bmi_s = self._interpolate_bmi(bmi_t, k, alpha)
                    k1 = self.func(z, t_s, static, bmi_s)
                    z = z + k1 * dt

                elif solver == "midpoint":
                    bmi_s = self._interpolate_bmi(bmi_t, k, alpha)
                    k1 = self.func(z, t_s, static, bmi_s)
                    alpha_mid = alpha + 0.5 / n_sub
                    bmi_mid = self._interpolate_bmi(bmi_t, k, min(alpha_mid, 1.0))
                    k2 = self.func(z + k1 * (0.5 * dt), t_s + 0.5 * dt_sub, static, bmi_mid)
                    z = z + k2 * dt

                elif solver == "rk4":
                    bmi_s = self._interpolate_bmi(bmi_t, k, alpha)
                    k1 = self.func(z, t_s, static, bmi_s)
                    alpha_mid = alpha + 0.5 / n_sub
                    bmi_mid = self._interpolate_bmi(bmi_t, k, min(alpha_mid, 1.0))
                    t_mid = t_s + 0.5 * dt_sub
                    k2 = self.func(z + k1 * (0.5 * dt), t_mid, static, bmi_mid)
                    k3 = self.func(z + k2 * (0.5 * dt), t_mid, static, bmi_mid)
                    alpha_end = alpha + 1.0 / n_sub
                    bmi_end = self._interpolate_bmi(bmi_t, k, min(alpha_end, 1.0))
                    k4 = self.func(z + k3 * dt, t_s + dt_sub, static, bmi_end)
                    z = z + (k1 + 2*k2 + 2*k3 + k4) * (dt / 6)

            zt[:, k + 1] = z

        return zt

    def _integrate_torchdiffeq(self, z0, times, static, bmi_t=None):
        """
        Integration via torchdiffeq. Supports adaptive solvers (dopri5, etc.)
        and fixed-step solvers (euler, rk4, midpoint, etc.).

        Integrates interval-by-interval, reparameterized to τ ∈ [0, 1],
        to handle per-subject time grids.
        """
        from torchdiffeq import odeint

        N, T = times.shape
        H = z0.shape[1]
        device, dtype = z0.device, z0.dtype
        solver = self.cfg.ode_solver

        # Build odeint options
        odeint_kwargs = {"method": solver}
        if self.cfg.euler_steps_per_interval is not None:
            # For fixed-step methods, set step_size in normalized [0,1] coords
            odeint_kwargs["options"] = {
                "step_size": 1.0 / self.cfg.euler_steps_per_interval
            }

        zt = torch.zeros(N, T, H, device=device, dtype=dtype)
        z = z0
        zt[:, 0] = z

        # Evaluation points in normalized time
        tau_eval = torch.tensor([0.0, 1.0], device=device, dtype=dtype)

        for k in range(T - 1):
            t_start = times[:, k]                  # (N,)
            dt_total = times[:, k + 1] - t_start   # (N,)

            # Store interval context for the wrapper
            self._interval_ctx = {
                "static": static,
                "bmi_k": bmi_t[:, k] if bmi_t is not None else None,
                "bmi_k1": bmi_t[:, k + 1] if bmi_t is not None else None,
                "t_start": t_start,
                "dt_total": dt_total,
            }

            # Integrate in normalized time τ ∈ [0, 1]
            # dz/dτ = f(z, t(τ), static, bmi(τ)) * dt_total
            z_out = odeint(self._odefunc_normalized, z, tau_eval,
                           **odeint_kwargs)
            # z_out: (2, N, H) — [z(τ=0), z(τ=1)]
            z = z_out[-1]
            zt[:, k + 1] = z

        # Clean up
        self._interval_ctx = None
        return zt

    def _odefunc_normalized(self, tau, z):
        """
        ODE function in normalized time τ ∈ [0, 1].
        dz/dτ = f(z, t(τ), static, bmi(τ)) * dt_total

        Called by torchdiffeq.odeint.
        """
        ctx = self._interval_ctx
        static = ctx["static"]
        t_start = ctx["t_start"]
        dt_total = ctx["dt_total"]

        # Map τ → absolute time
        t_abs = t_start + tau * dt_total   # (N,)

        # Interpolate BMI
        if ctx["bmi_k"] is not None:
            bmi_current = (1 - tau) * ctx["bmi_k"] + tau * ctx["bmi_k1"]
        else:
            bmi_current = None

        # Evaluate vector field and scale by dt
        dzdt = self.func(z, t_abs, static, bmi_current)   # (N, H)
        return dzdt * dt_total.unsqueeze(-1)               # dz/dτ = f * dt

    def _integrate(self, z0, times, static, bmi_t=None):
        """
        Dispatch to manual or torchdiffeq integration based on cfg.ode_solver.

        Manual solvers:     "euler", "midpoint", "rk4"
        torchdiffeq solvers: "dopri5", "adams", "adaptive_heun",
                             "scipy_solver", or any method supported
                             by torchdiffeq.odeint.

        Set cfg.ode_solver to the desired method name.
        For torchdiffeq fixed-step methods, cfg.euler_steps_per_interval
        controls the step size (step_size = 1/n_sub in normalized coords).
        """
        manual_solvers = {"euler", "midpoint", "rk4"}
        if self.cfg.ode_solver in manual_solvers:
            return self._integrate_manual(z0, times, static, bmi_t)
        else:
            return self._integrate_torchdiffeq(z0, times, static, bmi_t)

    def forward(
        self,
        t_pad,                                    # (N, T)
        x_pad,                                    # (N, T, Cx)
        masks=None,                               # (N, T) cumulative mask — IGNORED
        static_covariates=None,  
        bmi_t = None,                 # (N, Cs)
        obs_mask=None,                            # (N, T) outcome observation mask
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

        # --- Gradient penalty: make bmi_t a differentiable leaf ---
        # We need autograd to track ∂mu_ode/∂bmi_t through ODE integration
        if self.reg_mode == "grad_penalty" and bmi_t is not None:
            bmi_t_leaf = bmi_t.detach().requires_grad_(True)
        else:
            bmi_t_leaf = bmi_t

        # --- ODE integration ---
        zt = self._integrate(z0, t_pad, static_covariates, bmi_t_leaf)  # (N, T, H)
        zt = self.z_norm(zt)

        # --- Decoder (with BMI skip) ---
        result = self.decoder(zt, x_pad, static_covariates,
                              obs_mask=obs_mask)
        
        # --- Compute gradient penalty at model level ---
        if self.reg_mode == "grad_penalty" and bmi_t_leaf is not None:
            mu, V, Z, D, sig2, reg_dict = result
 
            mu_ode = reg_dict["mu_ode"]  # (N, T) — with grad graph intact
 
            # Compute ∂mu_ode(t_k)/∂BMI(t_k) for each observation time k.
            # This is the INSTANTANEOUS sensitivity only — accumulated effects
            # ∂mu_ode(t_j)/∂BMI(t_k) for j>k are NOT penalized.
            grad_penalty = torch.tensor(0.0, device=device, dtype=dtype)
            n_valid = 0
 
            for k in range(T):
                # Skip padded time steps
                if obs_mask is not None:
                    if obs_mask[:, k].sum() < 0.5:
                        continue
 
                grads = torch.autograd.grad(
                    mu_ode[:, k].sum(),
                    bmi_t_leaf,
                    create_graph=True,
                    retain_graph=True,
                )[0]                                         # (N, T) or (N, T, 1)
 
                # Extract instantaneous component: ∂mu_ode(t_k)/∂BMI(t_k)
                g_k = grads[:, k]                            # (N,) or (N, 1)
 
                if obs_mask is not None:
                    mask_k = obs_mask[:, k]                   # (N,)
                    grad_penalty = grad_penalty + ((g_k ** 2) * mask_k).sum() / mask_k.sum().clamp(min=1)
                else:
                    grad_penalty = grad_penalty + (g_k ** 2).mean()
                n_valid += 1
 
            if n_valid > 0:
                grad_penalty = grad_penalty / n_valid
 
            reg_dict["reg_term"] = grad_penalty
            reg_dict["grad_penalty_per_t"] = grad_penalty.detach()
 
            result = (mu, V, Z, D, sig2, reg_dict)

        if return_hidden:
            if isinstance(result, tuple):
                return result + (zt,)
            return result, zt
        return result