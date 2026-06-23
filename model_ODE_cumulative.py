"""
Neural ODE-LMM with BMI Skip Connection (torchdiffeq version).

Architecture:
  - Encoder:  z(0) = Enc(t0, static)
  - Dynamics: dz/dt = f(z, t, BMI(t))          ← Neural ODE
  - Decoder:  mu(t) = rho(z(t), BMI_std(t)) @ beta_neural 
  - RE:       Z = g(z(t))

Regularisation modes for skip connections:
  - None:          standard single rho network, no penalty.
  - "skip_gate":   learnable sigmoid gate on skip inputs.
                   Penalty: λ · Σ sigmoid(α_k)  (data-independent, smooth).
  - "group_lasso": group lasso on first-layer weights connecting to skip inputs.
                   Penalty: λ · Σ_k ||W_skip_k||_F  (data-independent, smooth when >0).
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
                 mlp_hidden=64, depth=2, dropout=0.0, use_bmi_in_ode=True):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.use_bmi_in_ode = use_bmi_in_ode

        in_dim = hidden_channels + 1 + (1 if use_bmi_in_ode else 0)

        self.net = MLP(
            in_dim=in_dim,
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

        if self.use_bmi_in_ode and bmi_t is not None:
            inp = torch.cat([z, t_expanded, bmi_t], dim=-1)
        else:
            inp = torch.cat([z, t_expanded], dim=-1)
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
                 reg_mode=None):
        """
        Args:
            latent_dim: dimension of z(t)
            p: fixed-effect basis dimension (rho output)
            q: random-effect basis dimension
            bmi_mean, bmi_std: for standardizing BMI skip input
            static_skip_dims: list of static covariate dimensions to skip-connect
                              to decoder (e.g., [1] for AGEc). None = no static skip.
            re_spline_cols: column indices in x_pad for RE spline basis
            use_bmi_skip: if False, decoder sees only z(t) — no direct BMI input.
            reg_mode: None, "skip_gate", or "group_lasso".

                None:
                    Standard decoder, single rho network, no regularisation.

                "skip_gate":
                    Learnable sigmoid gate on skip inputs:
                        gate_k = sigmoid(alpha_k)
                        skip_gated = [gate_1 * BMI_std, gate_2 * AGEc, ...]
                    Penalty: lambda * sum(gate_k)
                    Properties:
                        - Smooth everywhere (sigmoid), M-estimator theory applies.
                        - Data-independent penalty → Commenges framework directly.
                        - gate → 0 ≡ no skip; gate → 1 ≡ full skip.
                        - Interactions between z(t) and skip preserved when gate open.

                "group_lasso":
                    Group L2 penalty on first-layer rho_net weights for skip columns:
                        W = rho_net.net[0].weight  (h × (d + skip_dim))
                        W_skip_k = W[:, latent_dim + k]  (column for skip input k)
                    Penalty: lambda * sum_k ||W_skip_k||_2
                    Properties:
                        - Smooth when ||W_k|| > 0 (generic), M-estimator applies.
                        - Data-independent → Commenges framework directly.
                        - ||W_k|| → 0 means input k cannot enter the network.
                        - Well-established theory (Yuan & Lin, 2006).
        """
        super().__init__()
        self.p, self.q = p, q
        self.fullD = fullD
        self.jitter = jitter
        self.D_diag_min = D_diag_min
        self.n_static = n_static
        self.reg_mode = reg_mode
        self.latent_dim = latent_dim

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

        # --- Neural fixed effects ---
        self.use_rho_net = use_rho_net

        if use_rho_net:
            # Single network: [z(t), skip (possibly gated)] → R^p → scalar
            self.rho_net = MLP(latent_dim + skip_dim, rho_hidden, p,
                               depth=2, dropout=0.0)
            # self.rho_norm = nn.LayerNorm(p)
            self.beta_neural = nn.Parameter(0.1 * torch.randn(p))
        else:
            self.rho_net = None
            self.w_neural = nn.Parameter(0.01 * torch.randn(latent_dim + skip_dim))
            self.beta_neural = None

        # --- Skip gate parameters (only for skip_gate mode) ---
        if reg_mode == "skip_gate" and skip_dim > 0:
            # alpha_k initialised at 2.0 → sigmoid(2) ≈ 0.88 (gate starts open)
            self.skip_gate_logit = nn.Parameter(1.0 * torch.ones(skip_dim))
        else:
            self.skip_gate_logit = None

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

    def _compute_reg(self, skip_input):
        """
        Compute the regularisation term and return it in reg_dict.
        Data-independent for both modes → clean M-estimator inference.
        """
        reg_dict = {}

        if self.reg_mode == "skip_gate":
            gate = torch.sigmoid(self.skip_gate_logit)
            reg_dict["reg_term"] = gate.sum()
            reg_dict["gate_values"] = gate.detach()

        elif self.reg_mode == "group_lasso":
            W = self.rho_net.net[0].weight
            W_skip = W[:, self.latent_dim:]
            channel_norms = W_skip.norm(p=2, dim=0)
            reg_dict["reg_term"] = channel_norms.sum()
            reg_dict["channel_norms"] = channel_norms.detach()

        else:
            reg_dict["reg_term"] = torch.tensor(0.0)

        return reg_dict

    def forward(self, z_t, x_pad, static, obs_mask=None,
                return_components=True):
        """
        Args:
            z_t:      (N, T, H)  latent states from ODE
            x_pad:    (N, T, Cx) padded covariates (BMI_t at col 0)
            static:   (N, Cs)    static covariates
            obs_mask: (N, T)     binary mask (1=observed, 0=padded)

        Returns:
            mu:       (N, T)     population mean prediction
            V:        (N, T, T)  marginal covariance
            Z:        (N, T, q)  RE design matrix
            D:        (q, q)     RE covariance
            sig2:     scalar     residual variance
            reg_dict: dict with regularization info
        """
        N, T, H = z_t.shape
        device, dtype = z_t.device, z_t.dtype

        # --- Build skip input ---
        skip_input = self._build_skip_input(x_pad, static, N, T)

        # --- Apply skip gate if enabled ---
        if self.reg_mode == "skip_gate" and skip_input is not None:
            gate = torch.sigmoid(self.skip_gate_logit)
            skip_input = skip_input * gate

        # --- Fixed effects ---
        if skip_input is not None:
            rho_input = torch.cat([z_t, skip_input], dim=-1)
        else:
            rho_input = z_t

        if self.use_rho_net:
            rho = self.rho_net(rho_input)
            # rho = self.rho_norm(rho)
            mu = (rho * self.beta_neural).sum(dim=-1)
        else:
            mu = (rho_input * self.w_neural).sum(dim=-1)

        # --- Regularisation (data-independent) ---
        reg_dict = self._compute_reg(skip_input)

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
    Neural ODE-LMM with BMI skip connection (torchdiffeq version).

    Architecture:
      Encoder:  z(0) = Enc(t0, BMI0, static)
      ODE:      dz/dt = f(z, t, BMI(t))
      Decoder:  mu = rho(z(t), skip_input) @ beta_neural

    Regularisation modes (reg_mode):
      None:          no skip penalty
      "skip_gate":   sigmoid gate on skip inputs, penalty = λ·Σ gate_k
      "group_lasso": group L2 on first-layer skip weights, penalty = λ·Σ||W_k||

    Both penalties are data-independent and smooth, enabling clean
    M-estimator inference (Commenges et al., 2014).
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
                 use_bmi_in_ode=True):
        """
        Args:
            x_dim: total columns in x_pad (e.g. 3 for [BMI_t, rs1, rs2])
            static_dim: number of static covariates
            n_tv: number of time-varying covariates (only BMI_t = 1)
            static_skip_dims: indices of static covariates to skip-connect to decoder
                              e.g. [1] to pass AGEc directly. None = no static skip.
            use_bmi_skip: if False, decoder sees only z(t) — no direct BMI input.
            reg_mode: None, "skip_gate", or "group_lasso". See Decoder docstring.
            use_bmi_in_ode: if True, BMI(t) is fed to the ODE vector field.
        """
        super().__init__()
        if cfg is None:
            cfg = NeuralODEConfig()
        self.cfg = cfg
        self.static_dim = static_dim
        self.n_tv = n_tv
        self.reg_mode = reg_mode

        # Encoder
        encoder_input_dim = 1   # t0
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
            use_bmi_in_ode=use_bmi_in_ode,
        )

        self.z_norm = nn.LayerNorm(cfg.hidden_channels)
        # self.z_norm = nn.Identity()

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
        # from torchdiffeq import odeint_adjoint as odeint

        N, T = times.shape
        H = z0.shape[1]
        device, dtype = z0.device, z0.dtype
        solver = self.cfg.ode_solver

        # Build odeint options
        odeint_kwargs = {
            "method": solver,
            "atol": 1e-6,
            "rtol": 1e-6,
        }
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
        bmi_t=None,                               # (N, T, 1)
        obs_mask=None,                            # (N, T) outcome observation mask
        return_hidden=False,
        interp=None,                              # ignored (no spline interpolation needed)
    ):
        N, T = t_pad.shape
        device, dtype = t_pad.device, t_pad.dtype

        # --- Encoder ---
        t0 = t_pad[:, 0:1]                                  # (N, 1)
        bmi0 = x_pad[:, 0, 0:self.n_tv]                     # (N, n_tv)
        encoder_in = torch.cat([t0, static_covariates], dim=-1)
        z0 = self.encoder(encoder_in)                        # (N, H)

        # --- ODE integration ---
        zt = self._integrate(z0, t_pad, static_covariates, bmi_t)
        zt = self.z_norm(zt)

        # --- Decoder (with skip) ---
        result = self.decoder(zt, x_pad, static_covariates,
                              obs_mask=obs_mask)

        if return_hidden:
            if isinstance(result, tuple):
                return result + (zt,)
            return result, zt
        return result