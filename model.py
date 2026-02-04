# model.py
# Neural CDE model with:
#   - Encoder: uses baseline (static) covariates + first observation to set initial hidden state
#   - NeuralCDE: latent dynamics driven by an interpolated control path
#   - Decoder: maps hidden trajectory -> predicted mean at each timepoint
#
# Works with padded batches:
#   t_pad: (N, T)             subject-specific times (padded but strictly increasing after valid)
#   x_pad: (N, T, Cx)         longitudinal covariates (NOT including time)
#   mask : (N, T) {0,1}       valid points mask (optional, only for loss)
#   s_pad: (N, Cs) or None    baseline covariates (e.g., SEX_code, AGE0, etc.)
#
# Important note about irregular time grids:
# We append the *actual* time as an input channel, and let the CDE be parameterized
# by a common dummy grid (0..1). The "time" channel carries the real timing information.

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn

try:
    import torchcde
except ImportError as e:
    raise ImportError(
        "torchcde is required. Install with: pip install torchcde"
    ) from e


# -----------------------------
# Small building blocks
# -----------------------------
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
        input_dim_x0: int,          # Cin = 1 + Cx
        static_dim: int,            # Cs (0 if none)
        hidden_dim: int,            # latent dimension Hz
        mlp_hidden: int = 128,
        depth: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.static_dim = int(static_dim)

        self.x0_mlp = MLP(input_dim_x0, mlp_hidden, hidden_dim, depth=depth, dropout=dropout)

        if self.static_dim > 0:
            self.s_mlp = MLP(static_dim, mlp_hidden, hidden_dim, depth=depth, dropout=dropout)
        else:
            self.s_mlp = None

        # Optional fusion layer (gives model a bit more flexibility than pure sum)
        self.fuse = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )

    def forward(self, x0: torch.Tensor, s: Optional[torch.Tensor]) -> torch.Tensor:
        """
        x0: (N, Cin)
        s : (N, Cs) or None
        returns z0: (N, hidden_dim)
        """
        z = self.x0_mlp(x0)
        if self.s_mlp is not None and s is not None:
            z = z + self.s_mlp(s)
        return self.fuse(z)


class CDEFunc(nn.Module):
    """
    Defines f(z) so that:
        dz = f(z) dX
    where X has input_channels channels.
    torchcde expects func(t, z) -> (N, hidden_channels, input_channels)
    """

    def __init__(
        self,
        hidden_channels: int,
        input_channels: int,
        mlp_hidden: int = 128,
        depth: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.input_channels = input_channels

        # Output is hidden_channels * input_channels, then reshape.
        self.net = MLP(
            in_dim=hidden_channels,
            hidden_dim=mlp_hidden,
            out_dim=hidden_channels * input_channels,
            depth=depth,
            dropout=dropout,
            activation=nn.ReLU(),
        )

    def forward(self, t: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        t: scalar tensor (ignored; dynamics are autonomous in z, but kept for API)
        z: (N, hidden_channels)
        returns: (N, hidden_channels, input_channels)
        """
        out = self.net(z)
        return out.view(z.size(0), self.hidden_channels, self.input_channels)


import torch
import torch.nn as nn
import torch.nn.functional as F

class Decoder(nn.Module):
    """
    Mean:     mu_it = g(z_it, s_i) -> (N,T)
    Random:   Z_it = [1, u_it] with u_it scalar -> (N,T,2)
    Cov:      V = Z D Z^T + sigma^2 I  -> (N,T,T)

    Choose u_it as:
      - "latent": u_it = proj(z_it) (learned 1D projection)
      - "time":   u_it = t_it
    """

    def __init__(
        self,
        latent_dim: int,          # H
        static_dim: int = 0,      # Cs
        mean_hidden: int = 8,
        mean_depth: int = 2,
        fullD: bool = False,      # full 2x2 D or diagonal
        u_mode: str = "latent",   # "latent" or "time"
        jitter: float = 1e-6,
    ):
        super().__init__()
        assert u_mode in ("latent", "time")
        self.latent_dim = latent_dim
        self.static_dim = static_dim
        self.fullD = fullD
        self.u_mode = u_mode
        self.jitter = jitter
        self.q = 2  # random effects dimension

        # ---- Mean network g(z,s) ----
        in_mean = latent_dim + static_dim
        layers = []
        d = in_mean
        for _ in range(mean_depth - 1):
            layers += [nn.Linear(d, mean_hidden), nn.ReLU()]
            d = mean_hidden
        layers += [nn.Linear(d, 1)]
        self.mean_net = nn.Sequential(*layers)

        # ---- u_it scalar definition ----
        if self.u_mode == "latent":
            self.u_proj = nn.Linear(latent_dim, 1, bias=False)  # u_it = a^T z_it
        else:
            self.u_proj = None  # use time channel

        # ---- D (2x2) ----
        if not fullD:
            # diagonal variances
            self.log_std = nn.Parameter(torch.zeros(self.q))  # std positive via exp
            self.L_unconstrained = None
        else:
            # full SPD: D = L L^T, L lower-triangular with positive diag
            self.L_unconstrained = nn.Parameter(torch.zeros(self.q, self.q))
            self.log_std = None

        # ---- residual variance ----
        self.log_residual_var = nn.Parameter(torch.tensor(-2.0))  # exp(-2) ~ 0.135, adjust as you like

    def _build_D(self, device, dtype):
        if not self.fullD:
            std = torch.exp(self.log_std).to(device=device, dtype=dtype)  # (2,)
            return torch.diag(std * std)                                   # (2,2)
        else:
            L = torch.tril(self.L_unconstrained).to(device=device, dtype=dtype)
            # force positive diag
            diag = torch.diagonal(L, 0)
            diag_pos = F.softplus(diag) + 1e-6
            L = L - torch.diag(diag) + torch.diag(diag_pos)
            return L @ L.t()                                               # (2,2)

    def forward(
        self,
        z_t: torch.Tensor,                 # (N,T,H)
        s_i: torch.Tensor | None = None,   # (N,Cs)
        t_pad: torch.Tensor | None = None, # (N,T) required if u_mode="time"
        return_components: bool = False,
    ):
        assert z_t.ndim == 3, z_t.shape
        N, T, H = z_t.shape
        device, dtype = z_t.device, z_t.dtype

        # ----- mean -----
        if s_i is not None and self.static_dim > 0:
            s_rep = s_i.unsqueeze(1).expand(-1, T, -1)          # (N,T,Cs)
            mean_in = torch.cat([z_t, s_rep], dim=-1)           # (N,T,H+Cs)
        else:
            mean_in = z_t                                       # (N,T,H)

        mu = self.mean_net(mean_in).squeeze(-1)                 # (N,T)  <- safe squeeze

        # ----- build u_it scalar for random slope -----
        if self.u_mode == "latent":
            u = self.u_proj(z_t).squeeze(-1)                    # (N,T)
        else:
            if t_pad is None:
                raise ValueError("t_pad is required when u_mode='time'")
            u = t_pad.to(device=device, dtype=dtype)            # (N,T)

        # ----- design matrix Z = [1, u] -----
        ones = torch.ones((N, T), device=device, dtype=dtype)
        Z = torch.stack([ones, u], dim=-1)                      # (N,T,2)

        # ----- covariance V = Z D Z^T + sigma^2 I -----
        D = self._build_D(device, dtype)                        # (2,2)
        ZD = Z @ D                                              # (N,T,2)
        V_re = ZD @ Z.transpose(1, 2)                           # (N,T,T)

        sig2 = torch.exp(self.log_residual_var).to(device=device, dtype=dtype)
        eye = torch.eye(T, device=device, dtype=dtype).unsqueeze(0).expand(N, T, T)
        V = V_re + sig2 * eye + self.jitter * eye

        if return_components:
            return mu, V, Z, D, sig2
        return mu, V



# -----------------------------
# Full model
# -----------------------------
@dataclass
class NeuralCDEConfig:
    hidden_channels: int = 4       # latent dimension
    enc_mlp_hidden: int = 16
    func_mlp_hidden: int = 3
    depth: int = 2
    dropout: float = 0.0
    solver: str = "rk4"               # common choices: "rk4", "dopri5", "euler"
    step_size: Optional[float] = None # for rk4/euler; if None, torchcde default
    atol: float = 1e-6
    rtol: float = 1e-6


class NeuralCDEModel(nn.Module):
    """
    Forward returns:
      predicted_mean: (N, T)
      hidden_path:    (N, T, H)
    """

    def __init__(self, x_dim: int, static_dim: int = 0, cfg: NeuralCDEConfig = NeuralCDEConfig()):
        """
        x_dim: number of longitudinal covariates (excluding time)
        static_dim: number of baseline covariates (e.g., SEX_code, AGE0). 0 if none.
        """
        super().__init__()
        self.cfg = cfg
        self.x_dim = int(x_dim)
        self.static_dim = int(static_dim)

        # We always append time as a channel:
        self.input_channels = 1 + self.x_dim

        self.encoder = BaselineICEncoder(
            input_dim_x0=self.input_channels,
            static_dim=self.static_dim,
            hidden_dim=cfg.hidden_channels,
            mlp_hidden=cfg.enc_mlp_hidden,
            depth=cfg.depth,
            dropout=cfg.dropout,
        )

        self.func = CDEFunc(
            hidden_channels=cfg.hidden_channels,
            input_channels=self.input_channels,
            mlp_hidden=cfg.func_mlp_hidden,
            depth=cfg.depth,
            dropout=cfg.dropout,
        )

        self.decoder = Decoder(
            latent_dim=cfg.hidden_channels,
            static_dim=self.static_dim,
            mean_hidden=8,
            mean_depth=2,
            fullD=False,          # start diag; can switch to fullD=True later
            u_mode="latent",      # or "time"
        )

    @staticmethod
    def _make_common_grid(T: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        # Common parameterization grid; real timing is carried by the time channel.
        return torch.linspace(0.0, 1.0, T, device=device, dtype=dtype)

    def forward(
        self,
        t_pad: torch.Tensor,                 # (N, T)
        x_pad: torch.Tensor,                 # (N, T, Cx)
        s_pad: Optional[torch.Tensor] = None,# (N, Cs) or None
        return_hidden: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Returns:
          predicted_mean: (N, T)
          hidden_path: (N, T, H) if return_hidden else None
        """
        if t_pad.ndim != 2:
            raise ValueError(f"t_pad must be (N,T), got {t_pad.shape}")
        if x_pad.ndim != 3:
            raise ValueError(f"x_pad must be (N,T,Cx), got {x_pad.shape}")
        if x_pad.size(-1) != self.x_dim:
            raise ValueError(f"x_pad last dim must be x_dim={self.x_dim}, got {x_pad.size(-1)}")
        if self.static_dim > 0 and s_pad is None:
            raise ValueError("static_dim > 0 but s_pad is None")
        if s_pad is not None and s_pad.size(-1) != self.static_dim:
            raise ValueError(f"s_pad last dim must be static_dim={self.static_dim}, got {s_pad.size(-1)}")

        N, T = t_pad.shape
        device, dtype = t_pad.device, t_pad.dtype

        # Build control path with time as first channel: X_in = [t, x]
        X_in = torch.cat([t_pad.unsqueeze(-1), x_pad], dim=-1)  # (N,T,1+Cx)

        # Natural cubic spline coefficients on a common parameter grid (0..1),
        # while actual times are embedded in the first channel.
        coeffs = torchcde.natural_cubic_coeffs(X_in) 
        X = torchcde.CubicSpline(coeffs)

        # Initial state uses first observation (including time channel) + baseline covariates.
        x0 = X.evaluate(torch.zeros((), device=device, dtype=dtype))  # (N, input_channels)
        z0 = self.encoder(x0=x0, s=s_pad)                              # (N, H)

        # Integrate over common grid
        grid = self._make_common_grid(T, device=device, dtype=dtype)   # (T,)
        zt = torchcde.cdeint(
            X=X,
            z0=z0,
            func=self.func,
            t=grid,
            method=self.cfg.solver,
            options={"step_size": self.cfg.step_size} if self.cfg.step_size is not None else None,
            atol=self.cfg.atol,
            rtol=self.cfg.rtol,
        )  # (T, N, H)

        return self.decoder(zt, s_pad, t_pad) # (N, T)

