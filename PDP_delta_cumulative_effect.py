"""
Full-parameter delta method for ∆PDP variance — Neural ODE-LMM.
Adapted for CUMULATIVE EFFECT scenario (S5).

Key differences from baseline version:
  - Oracle ∆PDP is TIME-DEPENDENT: ∆PDP(t) = coeff × (bmi_hi − bmi_lo) × t̄
    where t̄ is the mean actual observation time of subjects in the time bin.
  - Time-windowed matching: subjects contribute to a target time only if
    their closest observation is within max_dist years.

Architecture and inference are unchanged:
  - skip_gate / group_lasso regularisation (data-independent penalty)
  - M-estimator inference (Commenges et al., 2014)

Three variance estimators (selected via --sandwich):

  Default:  Cov(θ̂) = F⁻¹                 (F = Σ φ_i φ_iᵀ, penalised scores)
  Bayesian: Cov(θ̂) = J⁻¹                 (O'Sullivan 1988)
  Sandwich: Cov(θ̂) = J⁻¹ F J⁻¹           (robust, Commenges et al. 2014)
"""
from __future__ import annotations
import math, os, csv, time
import torch
import torch.nn.functional as F_torch
import numpy as np
from typing import Dict, List, Tuple


# ─────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────

def _param_list(model):
    """Ordered list of parameters that require grad."""
    return [p for p in model.parameters() if p.requires_grad]


def _param_count(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _cat_grads(grads):
    """Flatten and concatenate a tuple of gradient tensors."""
    return torch.cat([g.reshape(-1) for g in grads])


# ─────────────────────────────────────────────────────────
# 1. Per-subject NLL (differentiable)
# ─────────────────────────────────────────────────────────

def _per_subject_nll(mu, V, y_pad, mask, jitter=1e-4):
    """
    Masked Gaussian NLL for ONE subject (B=1).

    Args:
        mu:    (1, T)
        V:     (1, T, T)
        y_pad: (1, T)
        mask:  (1, T)

    Returns:
        scalar NLL (differentiable)
    """
    mu = mu.squeeze(0)
    V = V.squeeze(0)
    y = y_pad.squeeze(0)
    m = mask.squeeze(0)

    idx = m.bool()
    n_i = idx.sum()
    if n_i == 0:
        return torch.tensor(0.0, device=mu.device, requires_grad=True)

    mu_obs = mu[idx]
    y_obs = y[idx]
    V_obs = V[idx][:, idx] + jitter * torch.eye(n_i, device=mu.device, dtype=mu.dtype)

    residual = y_obs - mu_obs
    L = torch.linalg.cholesky(V_obs)
    Vinv_r = torch.cholesky_solve(residual.unsqueeze(-1), L).squeeze(-1)
    log_det = 2.0 * torch.sum(torch.log(torch.diagonal(L)))

    nll = 0.5 * (log_det + residual @ Vinv_r + n_i * math.log(2 * math.pi))
    return nll


# ─────────────────────────────────────────────────────────
# 2. Empirical Fisher  F = Σ_i  s_i  s_iᵀ
# ─────────────────────────────────────────────────────────

def _compute_penalty_gradient(model, lambda_reg, weight_decay):
    """
    Compute the data-independent penalty gradient:

        c = λ_reg · ∇_θ reg_term  +  λ_wd · θ

    This constant is added to each NLL score to form the penalized score
    φ_i = ∇nll_i + c  (Commenges et al., 2014, eq. 8).

    Returns:
        c: (P,) tensor on CPU, or None if no penalty
    """
    params = _param_list(model)
    P = sum(p.numel() for p in params)
    c = torch.zeros(P)

    has_penalty = False

    # --- reg_term gradient (skip_gate or group_lasso) ---
    if lambda_reg > 0:
        reg_mode = getattr(model, 'reg_mode', None)
        if reg_mode is not None:
            reg_dict = model.decoder._compute_reg(None)
            reg_term = reg_dict["reg_term"]
            if reg_term.requires_grad:
                grads = torch.autograd.grad(reg_term, params,
                                            allow_unused=True)
                grads = [g if g is not None else torch.zeros_like(p)
                         for g, p in zip(grads, params)]
                c += lambda_reg * _cat_grads(grads).cpu()
                has_penalty = True

    # --- Weight decay gradient: λ_wd · θ ---
    if weight_decay > 0:
        theta_flat = torch.cat([p.detach().reshape(-1)
                                for p in params]).cpu()
        c += weight_decay * theta_flat
        has_penalty = True

    return c if has_penalty else None


def compute_empirical_fisher(model, dataset, device, collate_fn,
                              lambda_reg=0.0, weight_decay=0.0,
                              verbose=True):
    """
    Compute F = Σ_i  φ_i  φ_iᵀ   where  φ_i = ∇_θ nll_i + c  (penalised score).

    c = λ_reg · ∇_θ reg_term + λ_wd · θ  is a data-independent constant
    computed once and added to each NLL score (Commenges et al., 2014, eq. 8-9).

    At the penalised MLE: Σ φ_i = 0, so mean(φ_i) ≈ 0 (stationarity check).

    Uses batch_size=1 so each forward/backward is one subject.

    Returns:
        F: (P, P) tensor on CPU
        scores: (N, P) tensor on CPU  (penalised scores φ_i)
    """
    from torch.utils.data import DataLoader

    model.eval()
    params = _param_list(model)
    P = sum(p.numel() for p in params)
    N = len(dataset)

    # --- Compute penalty gradient (data-independent constant) ---
    c = _compute_penalty_gradient(model, lambda_reg, weight_decay)

    if verbose:
        if c is not None:
            parts = ["∇nll_i"]
            if lambda_reg > 0:
                reg_mode = getattr(model, 'reg_mode', 'unknown')
                parts.append(f"λ_reg·∇reg ({reg_mode})")
            if weight_decay > 0:
                parts.append(f"λ_wd·θ")
            print(f"    Score: φ_i = {' + '.join(parts)}")
            print(f"    Penalty gradient ||c|| = {c.norm().item():.4f}")
        else:
            print(f"    Score: φ_i = ∇nll_i  (no penalty)")

    fisher = torch.zeros(P, P)
    score_list = []

    loader = DataLoader(dataset, batch_size=1, shuffle=False,
                        collate_fn=collate_fn)

    t0 = time.time()
    for i, batch in enumerate(loader):
        _, t_pad, x_pad, y_pad, c_mask, mask, s = batch
        t_pad = t_pad.to(device)
        x_pad = x_pad.to(device)
        y_pad = y_pad.to(device)
        mask = mask.to(device)
        s = s.to(device)
        bmi_t = x_pad[:, :, 0:1]

        if mask.sum() == 0:
            continue

        mu, V, Z, D, sig2, _ = model(
            t_pad, x_pad, masks=None,
            static_covariates=s, bmi_t=bmi_t, obs_mask=mask
        )

        nll_i = _per_subject_nll(mu, V, y_pad, mask)

        grads = torch.autograd.grad(nll_i, params, retain_graph=False,
                                    allow_unused=True)
        grads = [g if g is not None else torch.zeros_like(p)
                 for g, p in zip(grads, params)]
        phi_i = _cat_grads(grads).cpu()

        # Add data-independent penalty gradient
        if c is not None:
            phi_i = phi_i + c

        fisher += torch.outer(phi_i, phi_i)
        score_list.append(phi_i)

        if verbose and (i + 1) % 500 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (N - i - 1) / rate
            print(f"    Fisher: {i+1}/{N} subjects "
                  f"({elapsed:.0f}s elapsed, ~{eta:.0f}s remaining)")

    scores = torch.stack(score_list)
    if verbose:
        print(f"    Fisher: done ({time.time()-t0:.1f}s), "
              f"cond = {torch.linalg.cond(fisher).item():.2e}")

    return fisher, scores

import torch.func as TF


def _compute_penalty_hessian(model, device, lambda_reg, verbose=True):
    """
    Compute ∇²(reg_term) for skip_gate or group_lasso via autograd.

    The reg_term is data-independent (depends only on model params),
    so this is a single forward+backward, no data loop.

    Returns:
        H_pen: (P, P) tensor on CPU, or None if no penalty
    """
    reg_mode = getattr(model, 'reg_mode', None)
    if reg_mode is None or lambda_reg <= 0:
        return None

    params = _param_list(model)
    P = sum(p.numel() for p in params)

    # Compute reg_term with create_graph
    reg_dict = model.decoder._compute_reg(None)
    reg_term = reg_dict["reg_term"]

    if not reg_term.requires_grad:
        if verbose:
            print(f"    Penalty Hessian: reg_term has no grad (penalty=0)")
        return None

    # First derivatives
    grad1 = torch.autograd.grad(reg_term, params, create_graph=True,
                                 allow_unused=True)
    grad1 = [g if g is not None else torch.zeros_like(p)
             for g, p in zip(grad1, params)]
    grad_flat = torch.cat([g.reshape(-1) for g in grad1])

    # Second derivatives: row by row
    H_pen = torch.zeros(P, P)
    for j in range(P):
        if grad_flat[j].requires_grad:
            row = torch.autograd.grad(
                grad_flat[j], params,
                retain_graph=(j < P - 1),
                allow_unused=True,
            )
            row = [g if g is not None else torch.zeros_like(p)
                   for g, p in zip(row, params)]
            H_pen[j, :] = _cat_grads(row).cpu()

    H_pen = 0.5 * (H_pen + H_pen.T)  # symmetrise

    if verbose:
        nnz = (H_pen.abs() > 1e-12).sum().item()
        print(f"    Penalty Hessian ({reg_mode}): "
              f"{nnz}/{P*P} nonzero entries, "
              f"||H_pen|| = {H_pen.norm().item():.4e}")

    return H_pen


def compute_hessian_explicit(model, dataset, device, collate_fn,
                             n_subsample=None, weight_decay=0.0,
                             lambda_reg=0.0,
                             verbose=True):
    from torch.utils.data import DataLoader, Subset

    model.eval()
    params = _param_list(model)
    P = sum(p.numel() for p in params)
    N = len(dataset)

    # --- subsampling logic (unchanged) ---
    if n_subsample is not None and n_subsample < N:
        indices = torch.randperm(N)[:n_subsample].tolist()
        subset = Subset(dataset, indices)
        scale = N / n_subsample
        M = n_subsample
        if verbose:
            print(f"    Hessian: subsampling {M}/{N} subjects (scale={scale:.2f})")
    else:
        subset = dataset
        scale = 1.0
        M = N

    loader = DataLoader(subset, batch_size=1, shuffle=False,
                        collate_fn=collate_fn)

    # --- Build functional version of per-subject NLL ---
    param_names, param_shapes, param_numels = [], [], []
    flat_params = []
    for name, p in model.named_parameters():
        if p.requires_grad:
            param_names.append(name)
            param_shapes.append(p.shape)
            param_numels.append(p.numel())
            flat_params.append(p.detach().reshape(-1))
    theta0 = torch.cat(flat_params).to(device)  # (P,)

    def _nll_from_flat(theta, batch_data):
        """Functional: flat θ → per-subject NLL (no mutation)."""
        t_pad, x_pad, y_pad, mask, s = batch_data
        # Load params into model via func stateless
        offset = 0
        param_dict = {}
        for name, shape, numel in zip(param_names, param_shapes, param_numels):
            param_dict[name] = theta[offset:offset+numel].view(shape)
            offset += numel

        # Use torch.func.functional_call
        bmi_t = x_pad[:, :, 0:1]
        mu, V, Z, D_mat, sig2, _ = TF.functional_call(
            model, param_dict,
            args=(t_pad, x_pad),
            kwargs=dict(masks=None, static_covariates=s,
                        bmi_t=bmi_t, obs_mask=mask)
        )
        return _per_subject_nll(mu, V, y_pad, mask)

    hessian = torch.zeros(P, P)
    t0 = time.time()
    n_done = 0

    for i, batch in enumerate(loader):
        _, t_pad, x_pad, y_pad, c_mask, mask, s = batch
        t_pad, x_pad = t_pad.to(device), x_pad.to(device)
        y_pad, mask, s = y_pad.to(device), mask.to(device), s.to(device)

        if mask.sum() == 0:
            continue

        batch_data = (t_pad, x_pad, y_pad, mask, s)

        # jacrev(jacrev(f)) computes full P×P Hessian in ~2 backward passes
        H_i = torch.func.jacrev(torch.func.jacrev(
            lambda th: _nll_from_flat(th, batch_data)
        ))(theta0)

        hessian += H_i.cpu()
        n_done += 1

        if verbose and n_done % 10 == 0:
            elapsed = time.time() - t0
            rate = n_done / elapsed
            eta = (M - n_done) / rate
            print(f"    Hessian: {n_done}/{M} subjects "
                  f"({elapsed:.0f}s elapsed, ~{eta:.0f}s remaining)")

    hessian *= scale
    hessian = 0.5 * (hessian + hessian.T)

    # --- Penalty Hessians ---
    # Training loss: (1/N)Σ nll_i + λ_reg·reg_term + (λ_wd/2)||θ||²
    # Stationarity: Σ ∇nll_i + N·λ_reg·∇reg_term + N·λ_wd·θ = 0
    # So: J = Σ ∇²nll_i + N·λ_reg·∇²reg_term + N·λ_wd·I
    N_full = len(dataset)

    # Weight decay: +N·λ_wd·I
    if weight_decay > 0:
        hessian += N_full * weight_decay * torch.eye(P)
        if verbose:
            print(f"    Added weight decay to J: +{N_full * weight_decay:.2e} on diag")

    # Skip gate / group lasso: +N·λ_reg·∇²reg_term
    if lambda_reg > 0:
        H_pen = _compute_penalty_hessian(model, device, lambda_reg, verbose)
        if H_pen is not None:
            hessian += N_full * lambda_reg * H_pen
            if verbose:
                print(f"    Added penalty Hessian to J: "
                      f"N·λ_reg = {N_full * lambda_reg:.2e}")

    if verbose:
        eigvals = torch.linalg.eigvalsh(hessian)
        print(f"    Hessian done ({time.time()-t0:.1f}s), "
              f"used {n_done} subjects")
        print(f"    J eigenvalue range: [{eigvals.min().item():.2e}, "
              f"{eigvals.max().item():.2e}]")
        n_neg = (eigvals < 0).sum().item()
        print(f"    J negative eigenvalues: {n_neg}/{P}")
        print(f"    J cond = {torch.linalg.cond(hessian).item():.2e}")
    return hessian



# ─────────────────────────────────────────────────────────
# 2b. Sandwich via HVP + MINRES (matrix-free, for large P)
# ─────────────────────────────────────────────────────────

def _hvp_dataset(model, dataset, device, collate_fn, v,
                 n_subsample=None, fixed_loader=None, scale=1.0,
                 weight_decay=0.0, lambda_reg=0.0, penalty_hessian=None):
    """
    Hessian-vector product:  Jv = ∇²(NLL) · v  in one dataset pass.

    For each subject i:
      1. forward  →  nll_i
      2. backward with create_graph  →  ∇nll_i
      3. compute  (∇nll_i)ᵀv  (scalar, differentiable)
      4. backward  →  ∇[(∇nll_i)ᵀv] = (∇²nll_i) · v
      5. accumulate  →  Jv = Σ (∇²nll_i) · v = ∇²(NLL) · v

    J = ∇²(NLL) is the observed information (PSD at the MLE).

    Args:
        v: (P,) tensor on device — the vector to multiply
        n_subsample: if set AND fixed_loader is None, subsample
                     (creates a NEW random subset each call — only for
                     one-shot use, not inside iterative solvers)
        fixed_loader: pre-built DataLoader with fixed subset
                      (use this inside MINRES/CG to keep the operator
                      deterministic across iterations)
        scale: rescaling factor (N/M) for subsampled Hessian

    Returns:
        Jv: (P,) tensor on CPU
    """
    from torch.utils.data import DataLoader, Subset

    model.eval()
    params = _param_list(model)
    P = sum(p.numel() for p in params)

    if fixed_loader is not None:
        loader = fixed_loader
    else:
        N = len(dataset)
        if n_subsample is not None and n_subsample < N:
            indices = torch.randperm(N)[:n_subsample].tolist()
            subset = Subset(dataset, indices)
            scale = N / n_subsample
        else:
            subset = dataset
            scale = 1.0
        loader = DataLoader(subset, batch_size=1, shuffle=False,
                            collate_fn=collate_fn)

    # Distribute v into parameter shapes (on device) for the dot product
    v_dev = v.to(device) if v.device != device else v
    v_parts = []
    offset = 0
    for p in params:
        n = p.numel()
        v_parts.append(v_dev[offset:offset + n].reshape(p.shape))
        offset += n

    Jv = torch.zeros(P)

    for batch in loader:
        _, t_pad, x_pad, y_pad, c_mask, mask, s = batch
        t_pad = t_pad.to(device)
        x_pad = x_pad.to(device)
        y_pad = y_pad.to(device)
        mask = mask.to(device)
        s = s.to(device)
        bmi_t = x_pad[:, :, 0:1]

        if mask.sum() == 0:
            continue

        mu, V, Z, D_mat, sig2, _ = model(
            t_pad, x_pad, masks=None,
            static_covariates=s, bmi_t=bmi_t, obs_mask=mask
        )
        nll_i = _per_subject_nll(mu, V, y_pad, mask)

        # Score with create_graph
        score_tuple = torch.autograd.grad(nll_i, params,
                                           create_graph=True,
                                           allow_unused=True)
        score_tuple = [g if g is not None else torch.zeros_like(p)
                       for g, p in zip(score_tuple, params)]

        # sᵀv — scalar, stays in graph
        sv = sum((s_j * v_j).sum()
                 for s_j, v_j in zip(score_tuple, v_parts))

        # ∇_θ (sᵀv) = H_i · v
        hvp = torch.autograd.grad(sv, params,
                                   retain_graph=False,
                                   allow_unused=True)
        hvp = [g if g is not None else torch.zeros_like(p)
               for g, p in zip(hvp, params)]
        Jv += _cat_grads(hvp).cpu()

    # J = ∇²(NLL) = observed information (PSD at the MLE)
    # The accumulator already contains Σ ∇²(nll_i) · v = J · v
    Jv = Jv * scale
    N_full = len(dataset)
    # Weight decay: +N·λ_wd·v
    if weight_decay > 0:
        Jv += N_full * weight_decay * v.cpu()
    # Skip gate / group lasso: +N·λ_reg·H_pen·v
    if lambda_reg > 0 and penalty_hessian is not None:
        Jv += N_full * lambda_reg * (penalty_hessian @ v.cpu())
    return Jv


def _minres_solve(matvec_fn, b, precond_fn=None,
                  maxiter=100, tol=1e-6, verbose=True):
    """
    MINRES for symmetric indefinite systems, via scipy.

    Minimises ||Ax - b||₂ over the Krylov subspace K_k(A, b).
    Works for any symmetric A — does NOT require positive definiteness.

    Args:
        matvec_fn:  callable  v → Av   (P,) torch → (P,) torch
        b:          (P,) torch tensor
        precond_fn: optional callable  v → M⁻¹v  (SPD preconditioner)
        maxiter:    max iterations
        tol:        relative residual tolerance

    Returns:
        x:    (P,) torch tensor — approximate solution
        info: dict with convergence diagnostics
    """
    from scipy.sparse.linalg import minres, LinearOperator
    import numpy as np

    P = b.shape[0]
    b_np = b.numpy().astype(np.float64)
    b_norm = np.linalg.norm(b_np)

    if b_norm < 1e-15:
        return torch.zeros_like(b), {
            'converged': True, 'iters': 0, 'rel_residual': 0.0}

    # Track iterations and residuals
    callback_info = {'iters': 0, 'residuals': []}
    t0 = time.time()

    def _callback(xk):
        callback_info['iters'] += 1
        k = callback_info['iters']
        if verbose and k % 10 == 0:
            # xk is the current solution; compute residual
            elapsed = time.time() - t0
            print(f"    MINRES iter {k} ({elapsed:.0f}s)")

    # Wrap torch matvec for scipy
    def _matvec_np(v_np):
        v_t = torch.from_numpy(v_np.astype(np.float64)).float()
        Av = matvec_fn(v_t)
        return Av.numpy().astype(np.float64)

    A_op = LinearOperator((P, P), matvec=_matvec_np, dtype=np.float64)

    # Wrap preconditioner
    M_op = None
    if precond_fn is not None:
        def _precond_np(v_np):
            v_t = torch.from_numpy(v_np.astype(np.float64)).float()
            Mv = precond_fn(v_t)
            return Mv.numpy().astype(np.float64)
        M_op = LinearOperator((P, P), matvec=_precond_np, dtype=np.float64)

    # scipy < 1.12 uses 'tol', >= 1.12 uses 'rtol'
    try:
        x_np, info_flag = minres(A_op, b_np, M=M_op,
                                  rtol=tol, maxiter=maxiter,
                                  callback=_callback)
    except TypeError:
        x_np, info_flag = minres(A_op, b_np, M=M_op,
                                  tol=tol, maxiter=maxiter,
                                  callback=_callback)

    x = torch.from_numpy(x_np).float()

    # Compute actual residual
    Ax = matvec_fn(x)
    res = (b - Ax).norm().item()
    rel_res = res / b_norm

    converged = (info_flag == 0)
    iters = callback_info['iters']

    if verbose:
        elapsed = time.time() - t0
        status = "converged" if converged else f"flag={info_flag}"
        print(f"    MINRES {status} at iter {iters}: "
              f"||r||/||b|| = {rel_res:.2e} ({elapsed:.0f}s)")

    return x, {'converged': converged, 'iters': iters,
                'rel_residual': rel_res, 'scipy_flag': info_flag}


def compute_sandwich_variance_cg(
    model, dataset, device, collate_fn,
    fisher, gradients, visit_times,
    fisher_inv=None,
    n_subsample=None,
    weight_decay=0.0,
    lambda_reg=0.0,
    penalty_hessian=None,
    cg_maxiter=100, cg_tol=1e-6,
    verbose=True,
):
    """
    Compute  Var(∆PDP_ℓ) = gᵀ J⁻¹ F J⁻¹ g  via MINRES, without forming J.

    For each visit time ℓ:
      1. Solve  J w_ℓ = g_ℓ  via MINRES with HVP  (handles indefinite J)
      2. var_ℓ = w_ℓᵀ F w_ℓ   (since J symmetric)

    Optionally preconditions with F⁻¹ (J ≈ F under correct spec).

    Returns:
        sandwich_vars: dict {vt: float}
        solve_info:    dict {vt: convergence info}
    """
    from torch.utils.data import DataLoader, Subset

    N = len(dataset)

    # --- Build a FIXED loader for deterministic HVP across iterations ---
    if n_subsample is not None and n_subsample < N:
        indices = torch.randperm(N)[:n_subsample].tolist()
        subset = Subset(dataset, indices)
        scale = N / n_subsample
        if verbose:
            print(f"    HVP subset: {n_subsample}/{N} subjects "
                  f"(fixed for all iterations, scale={scale:.2f})")
    else:
        subset = dataset
        scale = 1.0

    fixed_loader = DataLoader(subset, batch_size=1, shuffle=False,
                              collate_fn=collate_fn)

    # HVP closure with FIXED loader (deterministic operator)
    def hvp_raw(v):
        return _hvp_dataset(model, dataset, device, collate_fn, v,
                            fixed_loader=fixed_loader, scale=scale,
                            weight_decay=weight_decay,
                            lambda_reg=lambda_reg,
                            penalty_hessian=penalty_hessian)

    # Preconditioner: F⁻¹  (already computed, cheap matmul)
    precond_fn = None
    if fisher_inv is not None:
        def precond_fn(v):
            return fisher_inv @ v

    sandwich_vars = {}
    solve_info = {}

    # --- Diagnostic: verify HVP operator is working ---
    if verbose:
        v_test = torch.randn(gradients[visit_times[0]].shape[0])
        Jv_test = hvp_raw(v_test)
        print(f"    HVP diagnostic: ||v|| = {v_test.norm():.4f}, "
              f"||Jv|| = {Jv_test.norm():.4f}, "
              f"ratio = {Jv_test.norm() / v_test.norm():.4e}")
        print(f"    Jv nonzero entries: {(Jv_test.abs() > 1e-10).sum()}/{len(Jv_test)}")
        # Check symmetry: vᵀJw vs wᵀJv
        w_test = torch.randn_like(v_test)
        Jw_test = hvp_raw(w_test)
        sym1 = (v_test @ Jw_test).item()
        sym2 = (w_test @ Jv_test).item()
        print(f"    Symmetry check: vᵀJw = {sym1:.4e}, wᵀJv = {sym2:.4e}, "
              f"diff = {abs(sym1-sym2)/(abs(sym1)+1e-15):.2e}")

    for vt in visit_times:
        g = gradients[vt]
        if verbose:
            print(f"\n    Solving J w = g for t={int(vt)} "
                  f"(||g|| = {g.norm().item():.4f}) via MINRES ...")

        w, info = _minres_solve(
            hvp_raw, g,
            precond_fn=precond_fn,
            maxiter=cg_maxiter, tol=cg_tol,
            verbose=verbose,
        )

        # var = wᵀ F w   (= gᵀ J⁻¹ F J⁻¹ g  since J sym)
        var_sand = (w @ fisher @ w).item()
        sandwich_vars[vt] = var_sand
        solve_info[vt] = info

        if verbose:
            print(f"    t={int(vt)}: var_sandwich = {var_sand:.6e}, "
                  f"iters = {info['iters']}, "
                  f"converged = {info['converged']}")

    return sandwich_vars, solve_info


# ─────────────────────────────────────────────────────────
# 3. ∆PDP gradient  g_ℓ = ∇_θ  ∆PDP_ℓ
# ─────────────────────────────────────────────────────────

def compute_delta_pdp_gradients(model, loader, device,
                                 bmi_lo, bmi_hi,
                                 visit_times,
                                 max_dist=1.5,
                                 verbose=True):
    """
    Compute g_ℓ = ∇_θ ∆PDP_ℓ for each visit time, with time windowing.

    ∆PDP_ℓ = (1/n_ℓ) Σ_i [μ^{hi}_{i,c(i,ℓ)} − μ^{lo}_{i,c(i,ℓ)}]

    where i contributes only if |t_{i,c(i,ℓ)} − t_ℓ| ≤ max_dist.

    The gradient is accumulated across batches by linearity:
        g_ℓ = (1/n_ℓ) Σ_batch  ∇_θ  Σ_{i∈batch} δ_{i,ℓ}

    Also tracks the mean actual observation time per bin (needed for
    time-dependent oracle in cumulative scenarios).

    Returns:
        gradients:    dict {vt: g_ℓ ∈ R^P}  (CPU)
        estimates:    dict {vt: float}        ∆PDP estimates
        counts:       dict {vt: int}          subjects per visit time
        mean_times:   dict {vt: float}        mean actual obs time in bin
    """
    model.eval()
    params = _param_list(model)
    P = sum(p.numel() for p in params)

    g_accum = {vt: torch.zeros(P) for vt in visit_times}
    est_accum = {vt: 0.0 for vt in visit_times}
    n_accum = {vt: 0 for vt in visit_times}
    time_accum = {vt: 0.0 for vt in visit_times}  # sum of actual obs times

    for batch in loader:
        _, t_pad, x_pad, y_pad, c_mask, mask, s = batch
        t_pad = t_pad.to(device)
        x_pad = x_pad.to(device)
        mask = mask.to(device)
        s = s.to(device)

        B, T = t_pad.shape

        # --- Two counterfactual forward passes (with grad) ---
        x_hi = x_pad.clone()
        x_hi[:, :, 0] = bmi_hi
        mu_hi, _, _, _, _, _ = model(
            t_pad, x_hi, masks=None,
            static_covariates=s, bmi_t=x_hi[:, :, 0:1], obs_mask=mask
        )

        x_lo = x_pad.clone()
        x_lo[:, :, 0] = bmi_lo
        mu_lo, _, _, _, _, _ = model(
            t_pad, x_lo, masks=None,
            static_covariates=s, bmi_t=x_lo[:, :, 0:1], obs_mask=mask
        )

        # --- For each visit time, build differentiable sum ---
        for vt_idx, vt in enumerate(visit_times):
            delta_sum = torch.tensor(0.0, device=device)
            n_vt = 0

            for i in range(B):
                obs_idx = torch.where(mask[i] > 0.5)[0]
                if len(obs_idx) == 0:
                    continue
                obs_times = t_pad[i, obs_idx]
                dists = torch.abs(obs_times - vt)
                best = torch.argmin(dists)

                # Time windowing: only include if within max_dist
                if dists[best].item() > max_dist:
                    continue

                closest = obs_idx[best]
                delta_sum = delta_sum + (mu_hi[i, closest] - mu_lo[i, closest])
                n_vt += 1
                time_accum[vt] += obs_times[best].item()

            if n_vt > 0:
                # Backward — keep graph for remaining visit times
                retain = (vt_idx < len(visit_times) - 1)
                grads = torch.autograd.grad(delta_sum, params,
                                            retain_graph=retain,
                                            allow_unused=True)
                grads = [g if g is not None else torch.zeros_like(p)
                         for g, p in zip(grads, params)]
                g_batch = _cat_grads(grads).cpu()

                g_accum[vt] += g_batch
                est_accum[vt] += delta_sum.item()
                n_accum[vt] += n_vt

    # Normalise
    gradients = {}
    estimates = {}
    mean_times = {}
    for vt in visit_times:
        n = n_accum[vt]
        if n > 0:
            gradients[vt] = g_accum[vt] / n
            estimates[vt] = est_accum[vt] / n
            mean_times[vt] = time_accum[vt] / n
        else:
            gradients[vt] = torch.zeros(P)
            estimates[vt] = 0.0
            mean_times[vt] = float(vt)

    if verbose:
        for vt in visit_times:
            print(f"    g_{int(vt)}: ||g|| = {gradients[vt].norm().item():.4f}, "
                  f"n = {n_accum[vt]}, ∆PDP = {estimates[vt]:.4f}, "
                  f"mean_t = {mean_times[vt]:.2f}")

    return gradients, estimates, n_accum, mean_times


# ─────────────────────────────────────────────────────────
# 4. Main: full-parameter delta method variance
# ─────────────────────────────────────────────────────────

def _regularise_and_invert(M, label, LAMBDA=1e-4, verbose=True):
    """Marquardt-damp a PSD matrix and invert it."""
    diag_M = torch.diag(M)
    diag_M = torch.clamp(diag_M, min=1e-4 * diag_M.max())
    M_reg = M + LAMBDA * torch.diag(diag_M)
    cond = torch.linalg.cond(M_reg).item()
    if verbose:
        print(f"  {label}: diag range [{torch.diag(M).min().item():.2e}, "
              f"{torch.diag(M).max().item():.2e}], cond = {cond:.2e}")
    try:
        M_inv = torch.linalg.inv(M_reg)
    except torch.linalg.LinAlgError:
        if verbose:
            print(f"  WARNING: {label} not invertible, using pseudo-inverse")
        M_inv = torch.linalg.pinv(M_reg)
    return M_reg, M_inv


def compute_full_delta_variance(
    model, dataset, loader, device, collate_fn,
    bmi_lo=20.0, bmi_hi=35.0,
    visit_times=np.array([0, 5, 10, 14]),
    true_coeff=-0.05,
    sandwich=False,
    n_hessian_subsample=None,
    weight_decay=0.0,
    lambda_reg=0.0,
    max_dist=1.5,
):
    """
    Full-parameter delta method for ∆PDP variance — cumulative scenario.

    Oracle ∆PDP is time-dependent:
        ∆PDP(t) = true_coeff × (bmi_hi − bmi_lo) × t̄
    where t̄ is the mean actual observation time in the time bin.

    If sandwich=False (default):
        Var(∆PDP_ℓ) = g_ℓᵀ  F⁻¹  g_ℓ

    If sandwich=True, reports three estimators:
        SE_F⁻¹:    g_ℓᵀ  F⁻¹  g_ℓ                  (penalised Fisher)
        SE_bayes:  g_ℓᵀ  J⁻¹  g_ℓ                  (Bayesian, O'Sullivan 1988)
        SE_sand:   g_ℓᵀ  J⁻¹ F J⁻¹  g_ℓ            (sandwich, Commenges)

    Returns dict with results per visit time + Fisher + gradients.
    """
    P = _param_count(model)
    delta_v = bmi_hi - bmi_lo
    var_method = "SANDWICH (J⁻¹FJ⁻¹)" if sandwich else "FISHER (F⁻¹)"

    print(f"\n{'='*60}")
    print(f"FULL-PARAMETER DELTA METHOD — {var_method}")
    print(f"  (Cumulative scenario)")
    print(f"{'='*60}")
    print(f"  P = {P} parameters")
    print(f"  N = {len(dataset)} subjects")
    print(f"  BMI: lo={bmi_lo}, hi={bmi_hi}, Δv={delta_v}")
    print(f"  true_coeff = {true_coeff}")
    print(f"  max_dist = {max_dist} years")
    if weight_decay > 0:
        print(f"  Weight decay: {weight_decay:.2e}")
    if lambda_reg > 0:
        reg_mode = getattr(model, 'reg_mode', 'unknown')
        print(f"  λ_reg: {lambda_reg:.2e} (mode: {reg_mode})")

    # --- Step 1: Empirical Fisher with penalised scores ---
    print(f"\nStep 1: Empirical Fisher F = Σ_i φ_i φ_iᵀ ...")
    fisher, scores = compute_empirical_fisher(
        model, dataset, device, collate_fn,
        lambda_reg=lambda_reg, weight_decay=weight_decay,
    )

    # Eigendecomposition
    eigvals, eigvecs = torch.linalg.eigh(fisher)
    print(f"  Fisher rank (>1e-6): {(eigvals > 1e-6).sum().item()} / {P}")
    print(f"  Eigenvalue range: [{eigvals.min().item():.2e}, {eigvals.max().item():.2e}]")

    # --- Diagnostic: stationarity check ---
    mean_score = scores.mean(dim=0)
    mean_abs = scores.abs().mean(dim=0)
    ratio = mean_score.norm() / mean_abs.norm()
    print(f"  Stationarity: ||mean(φ)|| / ||mean(|φ|)|| = {ratio:.4f}")
    print(f"    ||mean(φ)|| = {mean_score.norm().item():.4f}")
    print(f"    Should be ≈ 0 at penalised MLE (Commenges et al.)")

    # --- Regularise & invert Fisher ---
    LAMBDA = 1e-4
    fisher_reg, fisher_inv = _regularise_and_invert(
        fisher, "Fisher", LAMBDA)

    # --- Step 1b (optional): Sandwich ---
    sandwich_vars = {}
    sandwich_cov = None
    if sandwich:
        print(f"\nStep 1b: Hessian J = ∇²(penalised objective) ...")
        J = compute_hessian_explicit(model, dataset, device, collate_fn,
                                      n_subsample=n_hessian_subsample,
                                      weight_decay=weight_decay,
                                      lambda_reg=lambda_reg)

        J_reg, J_inv = _regularise_and_invert(J, "Hessian J", LAMBDA)

        # Sandwich: J⁻¹ F J⁻¹  (Commenges et al., 2014, eq. 9)
        sandwich_cov = J_inv @ fisher @ J_inv
        # Bayesian: J⁻¹  (O'Sullivan, 1988)
        bayes_cov = J_inv
        print(f"  Sandwich cov: J⁻¹ F J⁻¹  (Commenges)")
        print(f"  Bayesian cov: J⁻¹  (O'Sullivan)")

        # Diagnostic: J vs F agreement
        diag_ratio = torch.diag(fisher) / torch.clamp(torch.diag(J_reg), min=1e-10)
        print(f"  diag(F)/diag(J): median={diag_ratio.median().item():.3f}, "
              f"mean={diag_ratio.mean().item():.3f}, "
              f"std={diag_ratio.std().item():.3f}")

        # --- Step 2: ∆PDP gradients (windowed) ---
        print(f"\nStep 2: ∆PDP gradients g_ℓ = ∇_θ ∆PDP_ℓ (windowed, max_dist={max_dist}) ...")
        gradients, estimates, counts, mean_times = compute_delta_pdp_gradients(
            model, loader, device, bmi_lo, bmi_hi, visit_times,
            max_dist=max_dist,
        )
    else:
        # --- Step 2: ∆PDP gradients (windowed) ---
        print(f"\nStep 2: ∆PDP gradients g_ℓ = ∇_θ ∆PDP_ℓ (windowed, max_dist={max_dist}) ...")
        gradients, estimates, counts, mean_times = compute_delta_pdp_gradients(
            model, loader, device, bmi_lo, bmi_hi, visit_times,
            max_dist=max_dist,
        )

    # --- Step 3: Var = gᵀ Cov g ---
    # Oracle for cumulative scenario: ∆PDP(t) = true_coeff × Δv × t̄
    # where t̄ is the mean actual observation time in the bin
    print(f"\nStep 3: Var(∆PDP_ℓ) = g_ℓᵀ Cov g_ℓ")
    print(f"  Oracle: ∆PDP(t) = {true_coeff} × {delta_v} × t̄")

    if sandwich:
        print(f"\n  {'Time':>6s}  {'t̄':>6s}  {'∆PDP':>10s}  {'SE_F⁻¹':>10s}  "
              f"{'SE_bayes':>10s}  {'SE_sand':>10s}  {'CI_lo':>10s}  "
              f"{'CI_hi':>10s}  {'True':>10s}  {'Bias':>10s}  {'n':>5s}")
    else:
        print(f"\n  {'Time':>6s}  {'t̄':>6s}  {'∆PDP':>10s}  {'SE_F⁻¹':>10s}  "
              f"{'CI_lo':>10s}  {'CI_hi':>10s}  {'True':>10s}  {'Bias':>10s}  {'n':>5s}")
    print(f"  {'-'*90}")

    results = {}
    for vt in visit_times:
        g = gradients[vt]
        est = estimates[vt]
        n_vt = counts[vt]
        mean_t = mean_times[vt]

        # Time-dependent oracle
        true_delta = true_coeff * delta_v * mean_t

        coeffs = eigvecs.T @ g
        null_mask = eigvals < 1e-6
        frac_null = (coeffs[null_mask] ** 2).sum() / (coeffs ** 2).sum()
        print(f"  t={int(vt)}: ||g_null||² / ||g||² = {frac_null.item():.4f}")

        var_fisher = (g @ fisher_inv @ g).item()

        if sandwich:
            var_sand = (g @ sandwich_cov @ g).item()
            var_bayes = (g @ bayes_cov @ g).item()
            se_main = np.sqrt(max(var_bayes, 0.0))  # Bayesian as primary
            se_sand = np.sqrt(max(var_sand, 0.0))
            se_fisher = np.sqrt(max(var_fisher, 0.0))
        else:
            var_sand = float('nan')
            var_bayes = float('nan')
            se_main = np.sqrt(max(var_fisher, 0.0))
            se_sand = float('nan')
            se_fisher = np.sqrt(max(var_fisher, 0.0))

        ci_lo = est - 1.96 * se_main
        ci_hi = est + 1.96 * se_main
        bias = est - true_delta

        results[vt] = {
            'estimate': est,
            'se': se_main,
            'var': var_bayes if sandwich else var_fisher,
            'var_fisher': var_fisher,
            'var_sandwich': var_sand,
            'var_bayes': var_bayes,
            'ci_lo': ci_lo,
            'ci_hi': ci_hi,
            'true': true_delta,
            'bias': bias,
            'mean_t': mean_t,
            'n': n_vt,
        }

        if sandwich:
            print(f"  {vt:6.0f}  {mean_t:6.2f}  {est:+10.4f}  {se_fisher:10.4f}  "
                  f"{se_main:10.4f}  {se_sand:10.4f}  "
                  f"{ci_lo:+10.4f}  {ci_hi:+10.4f}  "
                  f"{true_delta:+10.4f}  {bias:+10.4f}  {n_vt:5d}")
        else:
            print(f"  {vt:6.0f}  {mean_t:6.2f}  {est:+10.4f}  {se_fisher:10.4f}  "
                  f"{ci_lo:+10.4f}  {ci_hi:+10.4f}  "
                  f"{true_delta:+10.4f}  {bias:+10.4f}  {n_vt:5d}")

    ret = {
        'results': results,
        'fisher': fisher.numpy(),
        'fisher_inv': fisher_inv.numpy(),
        'gradients': {vt: g.numpy() for vt, g in gradients.items()},
        'estimates': estimates,
        'counts': counts,
        'mean_times': mean_times,
        'P': P,
    }
    if sandwich:
        ret['hessian'] = J.numpy()
        ret['hessian_inv'] = J_inv.numpy()
        ret['sandwich_cov'] = sandwich_cov.numpy()
        ret['bayes_cov'] = bayes_cov.numpy()

    return ret


# ─────────────────────────────────────────────────────────
# 5. Multi-simulation aggregation
# ─────────────────────────────────────────────────────────

def aggregate_simulations(all_results, visit_times):
    """
    Aggregate across D simulations. Compute var_mc, var_est, coverage.

    For cumulative scenario, oracle varies per simulation (depends on
    mean actual observation time in each bin), so coverage uses
    per-simulation true values.
    """
    D = len(all_results)
    summary = {}

    for vt in visit_times:
        ests = np.array([r['results'][vt]['estimate'] for r in all_results])
        ses = np.array([r['results'][vt]['se'] for r in all_results])

        # Per-simulation oracle (time-dependent)
        true_vals = np.array([r['results'][vt]['true'] for r in all_results])
        ci_los = np.array([r['results'][vt]['ci_lo'] for r in all_results])
        ci_his = np.array([r['results'][vt]['ci_hi'] for r in all_results])

        mean_true = true_vals.mean()
        mean_hat = ests.mean()
        bias = mean_hat - mean_true
        var_mc = ests.var(ddof=1) if D > 1 else float('nan')
        mean_var_est = (ses ** 2).mean()
        mse = bias ** 2 + var_mc if D > 1 else float('nan')
        rmse = np.sqrt(mse) if D > 1 else float('nan')
        # Coverage: each simulation's CI vs its own oracle
        coverage = np.mean((ci_los <= true_vals) & (true_vals <= ci_his)) if D > 1 else float('nan')

        summary[vt] = {
            'mean_hat': mean_hat,
            'true': mean_true,
            'bias': bias,
            'var_mc': var_mc,
            'mean_var_est': mean_var_est,
            'mse': mse,
            'rmse': rmse,
            'coverage95': coverage,
        }

    return summary


# ─────────────────────────────────────────────────────────
# 6. Main
# ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    import pyreadr
    from torch.utils.data import DataLoader
    from dataset import LongitudinalDataset, collate_pad
    from model_ODE_torchdiff import NeuralODEModel, NeuralODEConfig

    parser = argparse.ArgumentParser(
        description="Full-parameter delta method for ∆PDP variance — cumulative scenario")
    parser.add_argument("--n_sims", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--true_coeff", type=float, default=-0.05,
                        help="True h5 coefficient: h5(t) = coeff * integral BMI(tau) dtau")
    parser.add_argument("--output_csv", type=str,
                        default="results_simu/simulation_cumulative_summary.csv")
    parser.add_argument("--data_dir", type=str, default="simu_datasets/S5_sims")
    parser.add_argument("--ckpt_dir", type=str,
                        default="checkpoints/simulation_cumulative_effect_diagoD_noBMIInEncoder_norhonorm")
    parser.add_argument("--bmi_pairs", type=str, default=None)
    parser.add_argument("--sandwich", action="store_true",
                        help="Use sandwich variance J⁻¹FJ⁻¹ instead of F⁻¹")
    parser.add_argument("--n_hessian_subsample", type=int, default=None,
                        help="Subsample N subjects for Hessian (e.g. 500).")
    parser.add_argument("--weight_decay", type=float, default=1e-5,
                        help="Weight decay used during training (added to J)")
    parser.add_argument("--lambda_reg", type=float, default=0.0,
                        help="Skip penalty coefficient (skip_gate or group_lasso)")
    parser.add_argument("--reg_mode", type=str, default=None,
                        choices=[None, "skip_gate", "group_lasso"],
                        help="Regularisation mode for skip connection")
    parser.add_argument("--max_dist", type=float, default=1.5,
                        help="Max distance (years) for time-windowed matching")
    parser.add_argument("--hidden_channels", type=int, default=8)
    parser.add_argument("--ode_solver", type=str, default="rk4")
    parser.add_argument("--euler_steps", type=int, default=4)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    visit_times = np.array([0, 4, 8, 10])

    # BMI pairs
    if args.bmi_pairs:
        bmi_pairs = [tuple(map(float, p.split(':')))
                     for p in args.bmi_pairs.split(',')]
    else:
        grid = [20, 23, 26, 29, 32, 35]
        bmi_pairs = [(grid[i], grid[i+1]) for i in range(len(grid)-1)]
        bmi_pairs.append((23, 32))

    time_col, y_col, id_col = "time", "ISA15_sim", "NUM_ID"
    x_cols = ["BMI_t"]
    static_cols = ["SEX_code", "AGEc", "DIPNIV2", "DIPNIV3"]

    all_pair_results = {pair: [] for pair in bmi_pairs}

    # --- Checkpoint: resume from last completed simulation ---
    ckpt_dir = os.path.dirname(args.output_csv) or '.'
    ckpt_file = os.path.join(ckpt_dir, "delta_method_cumulative_checkpoint.pt")
    start_sim = 0

    if os.path.exists(ckpt_file):
        print(f"Found checkpoint: {ckpt_file}")
        ckpt_data = torch.load(ckpt_file, map_location="cpu", weights_only=False)
        all_pair_results = ckpt_data['all_pair_results']
        start_sim = ckpt_data['completed_up_to'] + 1
        print(f"Resuming from simulation {start_sim} "
              f"({start_sim}/{args.n_sims} already done)")

    for sim_idx in range(start_sim, args.n_sims):
        if args.n_sims > 1:
            data_path = f"{args.data_dir}/sim_{sim_idx+1:03d}.rds"
            ckpt_path = f"{args.ckpt_dir}/best_model_ode_{sim_idx}.pt"
            print(f"\n{'#'*60}")
            print(f"# SIMULATION {sim_idx}")
            print(f"{'#'*60}")
        else:
            data_path = f"{args.data_dir}/sim_001.rds"
            ckpt_path = f"{args.ckpt_dir}/best_model_ode_0.pt"

        # --- Data ---
        df = next(iter(pyreadr.read_r(data_path).values()))
        df["SEX"] = df["SEX"].astype("category")
        df["SEX_code"] = df["SEX"].cat.codes.astype("float64")
        df["DIPNIV2"] = (df["DIPNIV"].astype(str) == "2").astype("float64")
        df["DIPNIV3"] = (df["DIPNIV"].astype(str) == "3").astype("float64")

        dataset = LongitudinalDataset(df, id_col, time_col, x_cols, y_col,
                                      static_cols=static_cols)
        loader = DataLoader(dataset, batch_size=args.batch_size,
                            shuffle=False, collate_fn=collate_pad)

        # --- Model (match training config) ---
        cfg = NeuralODEConfig(
            hidden_channels=args.hidden_channels,
            enc_mlp_hidden=32, func_mlp_hidden=32,
            dec_rho_hidden=16, dec_p=4, dec_q=3, depth=2, dropout=0.0,
            euler_steps_per_interval=args.euler_steps if args.ode_solver != 'dopri5' else None,
            ode_solver=args.ode_solver,
        )
        model = NeuralODEModel(
            x_dim=len(x_cols), static_dim=len(static_cols), cfg=cfg,
            n_tv=1, use_rho_net=True, use_neural_re=True,
            re_spline_cols=None, g_hidden=16, fullD=False,
            bmi_mean=0.0, bmi_std=1.0,
            use_bmi_skip=False, static_skip_dims=None,
            reg_mode=args.reg_mode,
        ).to(device)

        checkpoint = torch.load(ckpt_path, map_location=device,
                                weights_only=False)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)
        print(f"Loaded: {ckpt_path}")

        # --- Compute for each BMI pair ---
        for bmi_lo, bmi_hi in bmi_pairs:
            result = compute_full_delta_variance(
                model, dataset, loader, device, collate_pad,
                bmi_lo=bmi_lo, bmi_hi=bmi_hi,
                visit_times=visit_times,
                true_coeff=args.true_coeff,
                sandwich=args.sandwich,
                n_hessian_subsample=args.n_hessian_subsample,
                weight_decay=args.weight_decay,
                lambda_reg=args.lambda_reg,
                max_dist=args.max_dist,
            )
            all_pair_results[(bmi_lo, bmi_hi)].append(result)

        # --- Checkpoint: save after each simulation ---
        torch.save({
            'all_pair_results': all_pair_results,
            'completed_up_to': sim_idx,
        }, ckpt_file)
        print(f"  [Checkpoint] Saved after simulation {sim_idx} → {ckpt_file}")

    # --- Aggregate & CSV ---
    os.makedirs(os.path.dirname(args.output_csv)
                if os.path.dirname(args.output_csv) else '.', exist_ok=True)

    header = ['time', 'BMI_lo', 'BMI_hi', 'D', 'mean_hat', 'mean_true',
              'bias', 'var_mc', 'mean_var_est', 'mse', 'rmse', 'coverage95']
    csv_rows = []

    print(f"\n{'='*90}")
    print(f"AGGREGATED — Full Delta Method, Cumulative (D={args.n_sims})")
    print(f"{'='*90}")
    print(f"{'t':>4s} {'lo':>4s} {'hi':>4s} {'D':>3s}  {'mean_hat':>10s} "
          f"{'mean_true':>10s} {'bias':>8s}  {'var_mc':>10s} {'var_est':>10s} "
          f"{'mse':>10s} {'cov95':>6s}")
    print(f"{'-'*90}")

    for (bmi_lo, bmi_hi), results_list in all_pair_results.items():
        D = len(results_list)
        summary = aggregate_simulations(results_list, visit_times) if D > 0 else {}

        for vt in visit_times:
            if D == 1:
                r = results_list[0]['results'][vt]
                row = {
                    'time': int(vt), 'BMI_lo': int(bmi_lo), 'BMI_hi': int(bmi_hi),
                    'D': D, 'mean_hat': r['estimate'], 'mean_true': r['true'],
                    'bias': r['bias'], 'var_mc': float('nan'),
                    'mean_var_est': r['var'], 'mse': float('nan'),
                    'rmse': float('nan'), 'coverage95': float('nan'),
                }
            else:
                s = summary[vt]
                row = {
                    'time': int(vt), 'BMI_lo': int(bmi_lo), 'BMI_hi': int(bmi_hi),
                    'D': D, 'mean_hat': s['mean_hat'], 'mean_true': s['true'],
                    'bias': s['bias'], 'var_mc': s['var_mc'],
                    'mean_var_est': s['mean_var_est'], 'mse': s['mse'],
                    'rmse': s['rmse'], 'coverage95': s['coverage95'],
                }
            csv_rows.append(row)

            vm = f"{row['var_mc']:.6f}" if not np.isnan(row['var_mc']) else "NA"
            ms = f"{row['mse']:.6f}" if not np.isnan(row['mse']) else "NA"
            cv = f"{row['coverage95']:.2f}" if not np.isnan(row['coverage95']) else "NA"
            print(f"{int(vt):4d} {int(bmi_lo):4d} {int(bmi_hi):4d} {D:3d}  "
                  f"{row['mean_hat']:+10.4f} {row['mean_true']:+10.4f} "
                  f"{row['bias']:+8.4f}  {vm:>10s} {row['mean_var_est']:10.6f} "
                  f"{ms:>10s} {cv:>6s}")

    with open(args.output_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"\nCSV saved to {args.output_csv}")