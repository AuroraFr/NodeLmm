"""
Takeuchi Information Criterion (TIC) for Neural ODE-LMM model selection.

TIC = -2 * ell(theta_hat) + 2 * tr(J^{-1} K)

where:
  J = -(1/n) sum_i nabla^2 ell_i(theta_hat)    (observed Hessian)
  K = (1/n) sum_i s_i s_i^T                     (outer product of scores)
  s_i = nabla ell_i(theta_hat)                   (per-subject score)

Under correct specification J = K and tr(J^{-1}K) = p (= AIC).
Under misspecification J != K and TIC captures true effective complexity.

Computation uses:
  - Per-subject gradients for K
  - Hessian-vector products (no explicit Hessian) for J
  - Conjugate gradient to solve J^{-1} w
  - Hutchinson's trace estimator for tr(J^{-1} K)

Usage:
    from tic import compute_tic

    tic_value, info = compute_tic(
        model=model,
        dataset=dataset,          # list of per-subject dicts
        nll_fn=compute_nll,       # fn(model, batch) -> scalar NLL
        total_nll_fn=compute_total_nll,  # fn(model, dataset) -> scalar, with graph
        n_probes=30,
        cg_max_iter=50,
        damping=1e-4,
    )
"""
from __future__ import annotations
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


# ─────────────────────────────────────────────
# Conjugate Gradient solver
# ─────────────────────────────────────────────

def conjugate_gradient(
    mvp: Callable[[torch.Tensor], torch.Tensor],
    b: torch.Tensor,
    max_iter: int = 50,
    tol: float = 1e-5,
    damping: float = 0.0,
) -> torch.Tensor:
    """
    Solve (A + damping * I) x = b via CG, where mvp computes A @ v.

    Args:
        mvp:      function v -> A @ v  (Hessian-vector product)
        b:        right-hand side, shape (p,)
        max_iter: maximum CG iterations
        tol:      convergence tolerance on residual norm
        damping:  Tikhonov regularization for numerical stability

    Returns:
        x: approximate solution, shape (p,)
    """
    x = torch.zeros_like(b)
    r = b.clone()
    p = r.clone()
    rs_old = r.dot(r)

    for _ in range(max_iter):
        Ap = mvp(p)
        if damping > 0:
            Ap = Ap + damping * p
        pAp = p.dot(Ap)
        if pAp.abs() < 1e-12:
            break
        alpha = rs_old / (pAp + 1e-10)
        x = x + alpha * p
        r = r - alpha * Ap
        rs_new = r.dot(r)
        if rs_new.sqrt().item() < tol:
            break
        p = r + (rs_new / (rs_old + 1e-10)) * p
        rs_old = rs_new

    return x


# ─────────────────────────────────────────────
# Hessian-vector product (flat)
# ─────────────────────────────────────────────

def _flatten_grad(grads):
    """Flatten a list of gradient tensors into a single vector."""
    return torch.cat([g.contiguous().flatten() for g in grads])


def hvp_flat(
    total_nll: torch.Tensor,
    params: List[torch.Tensor],
    v: torch.Tensor,
) -> torch.Tensor:
    """
    Compute (Hessian of total_nll) @ v using two backward passes.

    Args:
        total_nll: scalar loss computed with create_graph=True
        params:    list of parameter tensors
        v:         vector to multiply, shape (p,)

    Returns:
        Hv: shape (p,)
    """
    # First backward: get gradient (with graph retained)
    grads = torch.autograd.grad(
        total_nll, params, create_graph=True, retain_graph=True,
    )
    flat_grad = _flatten_grad(grads)

    # Second backward: Hessian-vector product
    # d/d(params) [v^T @ flat_grad] = H @ v
    hvp_grads = torch.autograd.grad(
        flat_grad, params, grad_outputs=v, retain_graph=True,
    )
    return _flatten_grad([h.detach() for h in hvp_grads])


# ─────────────────────────────────────────────
# Per-subject score computation
# ─────────────────────────────────────────────

@torch.no_grad()
def _get_param_list(model: nn.Module) -> List[torch.Tensor]:
    """Get list of parameters that require grad."""
    return [p for p in model.parameters() if p.requires_grad]


def compute_per_subject_scores(
    model: nn.Module,
    dataset: List[Dict],
    nll_fn: Callable,
) -> torch.Tensor:
    """
    Compute per-subject score vectors s_i = nabla_theta ell_i(theta_hat).

    The sign convention: nll_fn returns the NEGATIVE log-likelihood
    (i.e., the loss to minimize). The score is the gradient of the
    log-likelihood, so s_i = -grad(nll_i).

    Args:
        model:   trained model (parameters at theta_hat)
        dataset: list of per-subject data dicts, each passable to nll_fn
        nll_fn:  fn(model, subject_data) -> scalar NLL for one subject.
                 Must be differentiable w.r.t. model parameters.

    Returns:
        S: (n_subjects, p) matrix of score vectors
    """
    params = _get_param_list(model)
    scores = []

    for subject_data in dataset:
        model.zero_grad()
        nll_i = nll_fn(model, subject_data)
        nll_i.backward()

        # s_i = -grad(nll_i) = grad(ell_i)
        s_i = torch.cat([
            -p.grad.detach().flatten() for p in params
        ])
        scores.append(s_i)

    return torch.stack(scores)  # (n, p)


# ─────────────────────────────────────────────
# Hutchinson trace estimator
# ─────────────────────────────────────────────

def estimate_trace_jinv_k(
    model: nn.Module,
    S: torch.Tensor,
    total_nll_with_graph: torch.Tensor,
    n_probes: int = 30,
    cg_max_iter: int = 50,
    cg_tol: float = 1e-5,
    damping: float = 1e-4,
    probe_type: str = "rademacher",
) -> Tuple[float, Dict]:
    """
    Estimate tr(J^{-1} K) using Hutchinson's stochastic trace estimator.

    For each probe vector v:
      1. w = K v = (1/n) S^T (S v)
      2. u = J^{-1} w  via CG (using Hessian-vector products)
      3. trace_sample = v^T u

    Args:
        model:                trained model
        S:                    (n, p) per-subject score matrix
        total_nll_with_graph: scalar NLL over all subjects, computed with
                              create_graph=True so Hessian-vector products work
        n_probes:             number of Hutchinson probe vectors
        cg_max_iter:          max CG iterations per solve
        cg_tol:               CG convergence tolerance
        damping:              Tikhonov damping J + delta*I for stability
        probe_type:           "rademacher" or "gaussian"

    Returns:
        trace_estimate: float, estimated tr(J^{-1} K)
        info: dict with per-probe trace samples and diagnostics
    """
    params = _get_param_list(model)
    n, p = S.shape
    device = S.device

    # Define Hessian-vector product function
    # J = -(1/n) sum_i H_i = (1/n) Hessian(total_nll)
    # since total_nll = sum_i nll_i and J = -(1/n) H(sum -ell_i) = (1/n) H(sum nll_i)
    def mvp(v):
        return hvp_flat(total_nll_with_graph, params, v) / n

    trace_samples = []

    for probe_idx in range(n_probes):
        # Generate probe vector
        if probe_type == "rademacher":
            v = torch.randint(0, 2, (p,), device=device, dtype=S.dtype) * 2 - 1
        else:
            v = torch.randn(p, device=device, dtype=S.dtype)

        # w = K @ v = (1/n) S^T (S @ v)
        Sv = S @ v                     # (n,)
        w = S.t() @ Sv / n             # (p,)

        # Solve (J + damping*I) u = w
        u = conjugate_gradient(
            mvp=mvp,
            b=w,
            max_iter=cg_max_iter,
            tol=cg_tol,
            damping=damping,
        )

        trace_sample = v.dot(u).item()
        trace_samples.append(trace_sample)

    trace_estimate = float(np.mean(trace_samples))
    trace_std = float(np.std(trace_samples) / np.sqrt(n_probes))

    info = {
        "trace_samples": trace_samples,
        "trace_mean": trace_estimate,
        "trace_std": trace_std,
        "n_probes": n_probes,
        "n_params": p,
        "n_subjects": n,
        "damping": damping,
    }

    return trace_estimate, info


# ─────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────

def compute_tic(
    model: nn.Module,
    dataset: List[Dict],
    nll_fn: Callable,
    total_nll_fn: Callable,
    n_probes: int = 30,
    cg_max_iter: int = 50,
    cg_tol: float = 1e-5,
    damping: float = 1e-4,
    probe_type: str = "rademacher",
) -> Tuple[float, Dict]:
    """
    Compute the Takeuchi Information Criterion.

    TIC = -2 * ell(theta_hat) + 2 * tr(J^{-1} K)
        = 2 * total_nll + 2 * tr(J^{-1} K)

    Args:
        model:        trained model at theta_hat
        dataset:      list of per-subject data dicts
        nll_fn:       fn(model, subject_data) -> scalar NLL for ONE subject.
                      Must support backprop. Called once per subject.
        total_nll_fn: fn(model, dataset) -> scalar NLL over ALL subjects.
                      Must be called with create_graph=True internally
                      (or return a tensor with grad graph intact) so that
                      Hessian-vector products can be computed.
        n_probes:     Hutchinson probe count (20-50 typical)
        cg_max_iter:  CG iterations (50-100)
        cg_tol:       CG tolerance
        damping:      Tikhonov damping for J (1e-4 to 1e-3 typical)
        probe_type:   "rademacher" or "gaussian"

    Returns:
        tic:  float, the TIC value (lower is better)
        info: dict with diagnostics:
              - total_nll: -ell(theta_hat)
              - trace_jinv_k: estimated tr(J^{-1} K)
              - effective_p: trace_jinv_k (interpretation: effective parameters)
              - aic_p: raw parameter count (for comparison)
              - trace_info: detailed Hutchinson diagnostics

    Example:
        def nll_fn(model, subject_data):
            mu, V = model(subject_data['t'], subject_data['x'], ...)[:2]
            return gaussian_nll(mu, V, subject_data['y'])

        def total_nll_fn(model, dataset):
            # Must keep computation graph for Hessian
            total = 0
            for s in dataset:
                total = total + nll_fn(model, s)
            return total

        tic, info = compute_tic(model, dataset, nll_fn, total_nll_fn)
    """
    model.eval()
    device = next(model.parameters()).device

    # --- Step 1: Per-subject scores → K ---
    print(f"[TIC] Computing per-subject scores for {len(dataset)} subjects...")
    S = compute_per_subject_scores(model, dataset, nll_fn)
    print(f"[TIC] Score matrix: {S.shape}, norm: {S.norm():.4f}")

    # --- Step 2: Total NLL with computation graph for Hessian ---
    print("[TIC] Computing total NLL with graph...")
    model.zero_grad()
    total_nll = total_nll_fn(model, dataset)
    total_nll_value = total_nll.item()
    print(f"[TIC] Total NLL: {total_nll_value:.4f}")

    # --- Step 3: Hutchinson trace estimate ---
    print(f"[TIC] Estimating tr(J^{{-1}}K) with {n_probes} probes, "
          f"CG({cg_max_iter}), damping={damping}...")
    trace_estimate, trace_info = estimate_trace_jinv_k(
        model=model,
        S=S,
        total_nll_with_graph=total_nll,
        n_probes=n_probes,
        cg_max_iter=cg_max_iter,
        cg_tol=cg_tol,
        damping=damping,
        probe_type=probe_type,
    )
    print(f"[TIC] tr(J^{{-1}}K) = {trace_estimate:.2f} "
          f"± {trace_info['trace_std']:.2f}")

    # --- Step 4: TIC ---
    tic = 2 * total_nll_value + 2 * trace_estimate

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    aic = 2 * total_nll_value + 2 * n_params

    print(f"[TIC] TIC = {tic:.2f}  (AIC = {aic:.2f}, "
          f"effective p = {trace_estimate:.1f}, raw p = {n_params})")

    info = {
        "tic": tic,
        "aic": aic,
        "total_nll": total_nll_value,
        "log_likelihood": -total_nll_value,
        "trace_jinv_k": trace_estimate,
        "effective_p": trace_estimate,
        "raw_p": n_params,
        "trace_info": trace_info,
    }

    return tic, info


# ─────────────────────────────────────────────
# Comparison utility
# ─────────────────────────────────────────────

def compare_models(
    results: Dict[str, Dict],
    sort_by: str = "tic",
) -> None:
    """
    Print a comparison table from a dict of {model_name: info_dict}.

    Args:
        results: dict mapping model names to info dicts from compute_tic
        sort_by: "tic" or "aic"

    Example:
        results = {}
        for name, model in models.items():
            _, info = compute_tic(model, dataset, nll_fn, total_nll_fn)
            results[name] = info
        compare_models(results)
    """
    sorted_models = sorted(results.items(), key=lambda x: x[1][sort_by])

    best_val = sorted_models[0][1][sort_by]

    print(f"\n{'Model':<40} {'NLL':>10} {'eff_p':>8} {'raw_p':>8} "
          f"{'TIC':>12} {'AIC':>12} {'ΔTIC':>8}")
    print("─" * 100)

    for name, info in sorted_models:
        delta = info["tic"] - best_val
        marker = " ←" if delta == 0 else ""
        print(f"{name:<40} {info['total_nll']:>10.1f} "
              f"{info['effective_p']:>8.1f} {info['raw_p']:>8d} "
              f"{info['tic']:>12.1f} {info['aic']:>12.1f} "
              f"{delta:>+8.1f}{marker}")