"""
Full-parameter delta method for ∆PDP variance — Neural ODE-LMM.

Designed for skip_gate and group_lasso regularisation modes, where the
penalty is data-independent. This enables clean M-estimator inference
(Commenges et al., 2014):

  Scores: pure NLL gradients  s_i = ∇_θ nll_i(θ̂)
  Centering removes the penalty gradient (data-independent constant).
  J = ∇²(penalised objective) = Σ ∇²nll_i + N·λ_reg·∇²reg + N·λ_wd·I

Three variance estimators (selected via --sandwich):

  Default:  Cov(θ̂) = F⁻¹                 (F = Σ φ_i φ_iᵀ, penalised scores)
  Bayesian: Cov(θ̂) = J⁻¹                 (O'Sullivan 1988)
  Sandwich: Cov(θ̂) = J⁻¹ F J⁻¹           (robust, Commenges et al. 2014)
"""
from __future__ import annotations
import math, os, csv, time
import torch
import numpy as np


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

def _is_nn_param(name):
    """True for encoder/func/decoder network weights, False for β, D, σ², gates."""
    excluded = ('beta', 'log_D_diag', 'log_sigma2', 'D_off_diag',
                'skip_gate_logits', 'gate_logits')
    return not any(ex in name for ex in excluded)


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

    # --- weight decay gradient: λ_wd · θ_nn only ---
    if weight_decay > 0:
        offset = 0
        for name, p in model.named_parameters():
            numel = p.numel()
            is_nn = _is_nn_param(name)  # True for encoder/func/decoder weights
            if is_nn:
                c[offset:offset + numel] = weight_decay * p.detach().reshape(-1).cpu()
            # else: leave zeros (no weight decay on β, D, σ², gate logits)
            offset += numel
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

    # Add weight decay Hessian: λ_wd on θ_nn diagonal only
    if weight_decay > 0:
        if H_pen is None:
            H_pen = torch.zeros(P, P)
        offset = 0
        for name, p in model.named_parameters():
            numel = p.numel()
            if _is_nn_param(name):
                for j in range(numel):
                    H_pen[offset + j, offset + j] += weight_decay
            offset += numel

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



# # ─────────────────────────────────────────────────────────
# # 2b. Sandwich via HVP + MINRES (matrix-free, for large P)
# # ─────────────────────────────────────────────────────────

# def _hvp_dataset(model, dataset, device, collate_fn, v,
#                  n_subsample=None, fixed_loader=None, scale=1.0,
#                  weight_decay=0.0, lambda_reg=0.0, penalty_hessian=None):
#     """
#     Hessian-vector product:  Jv = ∇²(NLL) · v  in one dataset pass.

#     For each subject i:
#       1. forward  →  nll_i
#       2. backward with create_graph  →  ∇nll_i
#       3. compute  (∇nll_i)ᵀv  (scalar, differentiable)
#       4. backward  →  ∇[(∇nll_i)ᵀv] = (∇²nll_i) · v
#       5. accumulate  →  Jv = Σ (∇²nll_i) · v = ∇²(NLL) · v

#     J = ∇²(NLL) is the observed information (PSD at the MLE).

#     Args:
#         v: (P,) tensor on device — the vector to multiply
#         n_subsample: if set AND fixed_loader is None, subsample
#                      (creates a NEW random subset each call — only for
#                      one-shot use, not inside iterative solvers)
#         fixed_loader: pre-built DataLoader with fixed subset
#                       (use this inside MINRES/CG to keep the operator
#                       deterministic across iterations)
#         scale: rescaling factor (N/M) for subsampled Hessian

#     Returns:
#         Jv: (P,) tensor on CPU
#     """
#     from torch.utils.data import DataLoader, Subset

#     model.eval()
#     params = _param_list(model)
#     P = sum(p.numel() for p in params)

#     if fixed_loader is not None:
#         loader = fixed_loader
#     else:
#         N = len(dataset)
#         if n_subsample is not None and n_subsample < N:
#             indices = torch.randperm(N)[:n_subsample].tolist()
#             subset = Subset(dataset, indices)
#             scale = N / n_subsample
#         else:
#             subset = dataset
#             scale = 1.0
#         loader = DataLoader(subset, batch_size=1, shuffle=False,
#                             collate_fn=collate_fn)

#     # Distribute v into parameter shapes (on device) for the dot product
#     v_dev = v.to(device) if v.device != device else v
#     v_parts = []
#     offset = 0
#     for p in params:
#         n = p.numel()
#         v_parts.append(v_dev[offset:offset + n].reshape(p.shape))
#         offset += n

#     Jv = torch.zeros(P)

#     for batch in loader:
#         _, t_pad, x_pad, y_pad, c_mask, mask, s = batch
#         t_pad = t_pad.to(device)
#         x_pad = x_pad.to(device)
#         y_pad = y_pad.to(device)
#         mask = mask.to(device)
#         s = s.to(device)
#         bmi_t = x_pad[:, :, 0:1]

#         if mask.sum() == 0:
#             continue

#         mu, V, Z, D_mat, sig2, _ = model(
#             t_pad, x_pad, masks=None,
#             static_covariates=s, bmi_t=bmi_t, obs_mask=mask
#         )
#         nll_i = _per_subject_nll(mu, V, y_pad, mask)

#         # Score with create_graph
#         score_tuple = torch.autograd.grad(nll_i, params,
#                                            create_graph=True,
#                                            allow_unused=True)
#         score_tuple = [g if g is not None else torch.zeros_like(p)
#                        for g, p in zip(score_tuple, params)]

#         # sᵀv — scalar, stays in graph
#         sv = sum((s_j * v_j).sum()
#                  for s_j, v_j in zip(score_tuple, v_parts))

#         # ∇_θ (sᵀv) = H_i · v
#         hvp = torch.autograd.grad(sv, params,
#                                    retain_graph=False,
#                                    allow_unused=True)
#         hvp = [g if g is not None else torch.zeros_like(p)
#                for g, p in zip(hvp, params)]
#         Jv += _cat_grads(hvp).cpu()

#     # J = ∇²(NLL) = observed information (PSD at the MLE)
#     # The accumulator already contains Σ ∇²(nll_i) · v = J · v
#     Jv = Jv * scale
#     N_full = len(dataset)
#     # Weight decay: +N·λ_wd·v
#     if weight_decay > 0:
#         Jv += N_full * weight_decay * v.cpu()
#     # Skip gate / group lasso: +N·λ_reg·H_pen·v
#     if lambda_reg > 0 and penalty_hessian is not None:
#         Jv += N_full * lambda_reg * (penalty_hessian @ v.cpu())
#     return Jv


# def _minres_solve(matvec_fn, b, precond_fn=None,
#                   maxiter=100, tol=1e-6, verbose=True):
#     """
#     MINRES for symmetric indefinite systems, via scipy.

#     Minimises ||Ax - b||₂ over the Krylov subspace K_k(A, b).
#     Works for any symmetric A — does NOT require positive definiteness.

#     Args:
#         matvec_fn:  callable  v → Av   (P,) torch → (P,) torch
#         b:          (P,) torch tensor
#         precond_fn: optional callable  v → M⁻¹v  (SPD preconditioner)
#         maxiter:    max iterations
#         tol:        relative residual tolerance

#     Returns:
#         x:    (P,) torch tensor — approximate solution
#         info: dict with convergence diagnostics
#     """
#     from scipy.sparse.linalg import minres, LinearOperator
#     import numpy as np

#     P = b.shape[0]
#     b_np = b.numpy().astype(np.float64)
#     b_norm = np.linalg.norm(b_np)

#     if b_norm < 1e-15:
#         return torch.zeros_like(b), {
#             'converged': True, 'iters': 0, 'rel_residual': 0.0}

#     # Track iterations and residuals
#     callback_info = {'iters': 0, 'residuals': []}
#     t0 = time.time()

#     def _callback(xk):
#         callback_info['iters'] += 1
#         k = callback_info['iters']
#         if verbose and k % 10 == 0:
#             # xk is the current solution; compute residual
#             elapsed = time.time() - t0
#             print(f"    MINRES iter {k} ({elapsed:.0f}s)")

#     # Wrap torch matvec for scipy
#     def _matvec_np(v_np):
#         v_t = torch.from_numpy(v_np.astype(np.float64)).float()
#         Av = matvec_fn(v_t)
#         return Av.numpy().astype(np.float64)

#     A_op = LinearOperator((P, P), matvec=_matvec_np, dtype=np.float64)

#     # Wrap preconditioner
#     M_op = None
#     if precond_fn is not None:
#         def _precond_np(v_np):
#             v_t = torch.from_numpy(v_np.astype(np.float64)).float()
#             Mv = precond_fn(v_t)
#             return Mv.numpy().astype(np.float64)
#         M_op = LinearOperator((P, P), matvec=_precond_np, dtype=np.float64)

#     # scipy < 1.12 uses 'tol', >= 1.12 uses 'rtol'
#     try:
#         x_np, info_flag = minres(A_op, b_np, M=M_op,
#                                   rtol=tol, maxiter=maxiter,
#                                   callback=_callback)
#     except TypeError:
#         x_np, info_flag = minres(A_op, b_np, M=M_op,
#                                   tol=tol, maxiter=maxiter,
#                                   callback=_callback)

#     x = torch.from_numpy(x_np).float()

#     # Compute actual residual
#     Ax = matvec_fn(x)
#     res = (b - Ax).norm().item()
#     rel_res = res / b_norm

#     converged = (info_flag == 0)
#     iters = callback_info['iters']

#     if verbose:
#         elapsed = time.time() - t0
#         status = "converged" if converged else f"flag={info_flag}"
#         print(f"    MINRES {status} at iter {iters}: "
#               f"||r||/||b|| = {rel_res:.2e} ({elapsed:.0f}s)")

#     return x, {'converged': converged, 'iters': iters,
#                 'rel_residual': rel_res, 'scipy_flag': info_flag}


# def compute_sandwich_variance_cg(
#     model, dataset, device, collate_fn,
#     fisher, gradients, visit_times,
#     fisher_inv=None,
#     n_subsample=None,
#     weight_decay=0.0,
#     lambda_reg=0.0,
#     penalty_hessian=None,
#     cg_maxiter=100, cg_tol=1e-6,
#     verbose=True,
# ):
#     """
#     Compute  Var(∆PDP_ℓ) = gᵀ J⁻¹ F J⁻¹ g  via MINRES, without forming J.

#     For each visit time ℓ:
#       1. Solve  J w_ℓ = g_ℓ  via MINRES with HVP  (handles indefinite J)
#       2. var_ℓ = w_ℓᵀ F w_ℓ   (since J symmetric)

#     Optionally preconditions with F⁻¹ (J ≈ F under correct spec).

#     Returns:
#         sandwich_vars: dict {vt: float}
#         solve_info:    dict {vt: convergence info}
#     """
#     from torch.utils.data import DataLoader, Subset

#     N = len(dataset)

#     # --- Build a FIXED loader for deterministic HVP across iterations ---
#     if n_subsample is not None and n_subsample < N:
#         indices = torch.randperm(N)[:n_subsample].tolist()
#         subset = Subset(dataset, indices)
#         scale = N / n_subsample
#         if verbose:
#             print(f"    HVP subset: {n_subsample}/{N} subjects "
#                   f"(fixed for all iterations, scale={scale:.2f})")
#     else:
#         subset = dataset
#         scale = 1.0

#     fixed_loader = DataLoader(subset, batch_size=1, shuffle=False,
#                               collate_fn=collate_fn)

#     # HVP closure with FIXED loader (deterministic operator)
#     def hvp_raw(v):
#         return _hvp_dataset(model, dataset, device, collate_fn, v,
#                             fixed_loader=fixed_loader, scale=scale,
#                             weight_decay=weight_decay,
#                             lambda_reg=lambda_reg,
#                             penalty_hessian=penalty_hessian)

#     # Preconditioner: F⁻¹  (already computed, cheap matmul)
#     precond_fn = None
#     if fisher_inv is not None:
#         def precond_fn(v):
#             return fisher_inv @ v

#     sandwich_vars = {}
#     solve_info = {}

#     # --- Diagnostic: verify HVP operator is working ---
#     if verbose:
#         v_test = torch.randn(gradients[visit_times[0]].shape[0])
#         Jv_test = hvp_raw(v_test)
#         print(f"    HVP diagnostic: ||v|| = {v_test.norm():.4f}, "
#               f"||Jv|| = {Jv_test.norm():.4f}, "
#               f"ratio = {Jv_test.norm() / v_test.norm():.4e}")
#         print(f"    Jv nonzero entries: {(Jv_test.abs() > 1e-10).sum()}/{len(Jv_test)}")
#         # Check symmetry: vᵀJw vs wᵀJv
#         w_test = torch.randn_like(v_test)
#         Jw_test = hvp_raw(w_test)
#         sym1 = (v_test @ Jw_test).item()
#         sym2 = (w_test @ Jv_test).item()
#         print(f"    Symmetry check: vᵀJw = {sym1:.4e}, wᵀJv = {sym2:.4e}, "
#               f"diff = {abs(sym1-sym2)/(abs(sym1)+1e-15):.2e}")

#     for vt in visit_times:
#         g = gradients[vt]
#         if verbose:
#             print(f"\n    Solving J w = g for t={int(vt)} "
#                   f"(||g|| = {g.norm().item():.4f}) via MINRES ...")

#         w, info = _minres_solve(
#             hvp_raw, g,
#             precond_fn=precond_fn,
#             maxiter=cg_maxiter, tol=cg_tol,
#             verbose=verbose,
#         )

#         # var = wᵀ F w   (= gᵀ J⁻¹ F J⁻¹ g  since J sym)
#         var_sand = (w @ fisher @ w).item()
#         sandwich_vars[vt] = var_sand
#         solve_info[vt] = info

#         if verbose:
#             print(f"    t={int(vt)}: var_sandwich = {var_sand:.6e}, "
#                   f"iters = {info['iters']}, "
#                   f"converged = {info['converged']}")

#     return sandwich_vars, solve_info


# ─────────────────────────────────────────────────────────
# 3. ∆PDP gradient  g_ℓ = ∇_θ  ∆PDP_ℓ
# ─────────────────────────────────────────────────────────

from contextlib import contextmanager, nullcontext

@contextmanager
def _adaptive_solver(model):
    """
    Temporarily replace model._euler_integrate with a dopri5 solver.

    Uses torchdiffeq with adaptive stepping (atol/rtol) and normalised
    time τ ∈ [0,1] per interval — identical mathematics to the Euler
    integrator but with automatic step-size control.  The integration
    grid spacing does NOT affect accuracy.
    """
    from torchdiffeq import odeint

    orig_integrate = model._euler_integrate

    def _dopri5_integrate(z0, times, static, bmi_t=None):
        N, T = times.shape
        H = z0.shape[1]
        device, dtype = z0.device, z0.dtype

        zt = torch.zeros(N, T, H, device=device, dtype=dtype)
        z = z0
        zt[:, 0] = z
        tau_eval = torch.tensor([0.0, 1.0], device=device, dtype=dtype)

        for k in range(T - 1):
            t_start = times[:, k]                              # (N,)
            dt_total = times[:, k + 1] - t_start               # (N,)

            # Capture interval context for the normalised ODE func
            bmi_k  = bmi_t[:, k]     if bmi_t is not None else None
            bmi_k1 = bmi_t[:, k + 1] if bmi_t is not None else None

            def _odefunc(tau, z_, _ts=t_start, _dt=dt_total,
                         _bk=bmi_k, _bk1=bmi_k1, _s=static):
                """dz/dτ = f(z, x(τ), t(τ)) · Δt"""
                t_abs = _ts + tau * _dt
                if _bk is not None:
                    bmi_tau = (1 - tau) * _bk + tau * _bk1
                    dzdt = model.func(z_, t_abs, _s, bmi_tau)
                else:
                    dzdt = model.func(z_, t_abs, _s, None)
                return dzdt * _dt.unsqueeze(-1)

            z_out = odeint(_odefunc, z, tau_eval,
                           method='dopri5', atol=1e-6, rtol=1e-6)
            z = z_out[-1]
            zt[:, k + 1] = z

        return zt

    model._euler_integrate = _dopri5_integrate
    try:
        yield
    finally:
        model._euler_integrate = orig_integrate


def _build_dense_grid(visit_times, dense_step=1.0):
    """
    Build a dense integration grid that includes the eval times.

    Only needed for fixed-step solvers (euler/midpoint/rk4) where
    the step size depends on the spacing of t_pad entries.

    Args:
        visit_times: (L,) array of times to report results at
        dense_step:  max spacing between consecutive grid points (years)

    Returns:
        t_dense:     (M,) sorted numpy array  (superset of visit_times)
        eval_indices: list of int — positions of visit_times in t_dense
    """
    t_min, t_max = visit_times[0], visit_times[-1]
    t_dense = np.arange(t_min, t_max + 1e-9, dense_step)
    t_dense = np.union1d(t_dense, visit_times)
    t_dense = np.sort(t_dense)

    eval_indices = [int(np.searchsorted(t_dense, vt)) for vt in visit_times]
    return t_dense, eval_indices


def _resample_batch_to_grid(t_pad, x_pad, mask, grid):
    """
    Resample batch covariates from irregular observed times onto a fixed grid.

    For each subject, linearly interpolates each covariate channel from
    observed time points to `grid` (LOCF/FOCB beyond range via np.interp).

    Args:
        t_pad:  (B, T_orig) original time points
        x_pad:  (B, T_orig, C) original covariates
        mask:   (B, T_orig) observation mask
        grid:   (M,) numpy array of target times

    Returns:
        t_grid:    (B, M)    common time grid
        x_grid:    (B, M, C) interpolated covariates
        mask_grid: (B, M)    all-ones mask
    """
    B, T_orig, C = x_pad.shape
    M = len(grid)
    device = x_pad.device
    dtype = x_pad.dtype

    t_grid = torch.tensor(grid, device=device, dtype=dtype
                          ).unsqueeze(0).expand(B, -1).clone()
    x_grid = torch.zeros(B, M, C, device=device, dtype=dtype)
    mask_grid = torch.ones(B, M, device=device, dtype=dtype)

    t_np = t_pad.cpu().numpy()
    x_np = x_pad.cpu().numpy()
    m_np = mask.cpu().numpy()

    for i in range(B):
        obs_i = m_np[i] > 0.5
        if not obs_i.any():
            continue
        t_obs = t_np[i, obs_i]
        for c in range(C):
            vals_obs = x_np[i, obs_i, c]
            x_grid[i, :, c] = torch.tensor(
                np.interp(grid, t_obs, vals_obs),
                device=device, dtype=dtype)

    return t_grid, x_grid, mask_grid


def compute_delta_pdp_gradients(model, loader, device,
                                 bmi_lo, bmi_hi,
                                 visit_times,
                                 dense_step=1.0,
                                 use_dopri5=True,
                                 verbose=True):
    """
    Compute g_ℓ = ∇_θ ∆PDP_ℓ for each visit time.

    All subjects are resampled onto a common time grid so every
    subject contributes at every visit time — no argmin, no
    censoring filter.

    Two integration modes:

      use_dopri5=True  (default):
        Temporarily switches the model to adaptive dopri5 via
        torchdiffeq.  The solver picks its own step size, so the
        ODE is solved directly on visit_times — no dense grid.

      use_dopri5=False:
        Uses the model's training solver (euler/midpoint/rk4).
        A dense integration grid with spacing ≤ dense_step keeps
        the fixed step size small; results reported at visit_times.

    ∆PDP_ℓ = (1/N) Σ_i [μ^{hi}_{i,ℓ} − μ^{lo}_{i,ℓ}]

    Returns:
        gradients: dict {vt: g_ℓ ∈ R^P}  (CPU)
        estimates: dict {vt: float}        ∆PDP estimates
        counts:    dict {vt: int}          subjects per visit time (= N for all ℓ)
    """
    model.eval()
    params = _param_list(model)
    P = sum(p.numel() for p in params)
    L = len(visit_times)

    # --- Choose integration grid ---
    if use_dopri5:
        # Adaptive solver: eval grid = visit_times directly
        integration_grid = np.array(visit_times, dtype=np.float64)
        eval_indices = list(range(L))
        if verbose:
            print(f"    Solver: dopri5 (adaptive) → grid = visit_times directly")
    else:
        # Fixed-step solver: dense grid to keep Euler steps small
        integration_grid, eval_indices = _build_dense_grid(visit_times, dense_step)
        solver_name = getattr(model.cfg, 'ode_solver', 'euler')
        if verbose:
            M = len(integration_grid)
            print(f"    Solver: {solver_name} (fixed-step) → "
                  f"dense grid: {M} points, step ≤ {dense_step}")
            print(f"    Eval indices: {list(zip(visit_times, eval_indices))}")

    g_accum = {vt: torch.zeros(P) for vt in visit_times}
    est_accum = {vt: 0.0 for vt in visit_times}
    n_total = 0

    # --- Context: switch to dopri5 if requested ---
    ctx = _adaptive_solver(model) if use_dopri5 else nullcontext()

    with ctx:
        for batch in loader:
            _, t_pad, x_pad, y_pad, c_mask, mask, s = batch
            t_pad = t_pad.to(device)
            x_pad = x_pad.to(device)
            mask = mask.to(device)
            s = s.to(device)

            B = t_pad.shape[0]

            # --- Resample all subjects onto integration grid ---
            t_grid, x_grid, mask_grid = _resample_batch_to_grid(
                t_pad, x_pad, mask, integration_grid)

            # --- Two counterfactual forward passes ---
            x_hi = x_grid.clone()
            x_hi[:, :, 0] = bmi_hi
            mu_hi, _, _, _, _, _ = model(
                t_grid, x_hi, masks=None,
                static_covariates=s, bmi_t=x_hi[:, :, 0:1], obs_mask=mask_grid
            )

            x_lo = x_grid.clone()
            x_lo[:, :, 0] = bmi_lo
            mu_lo, _, _, _, _, _ = model(
                t_grid, x_lo, masks=None,
                static_covariates=s, bmi_t=x_lo[:, :, 0:1], obs_mask=mask_grid
            )

            # --- Gradient at eval times only ---
            for vt_idx, vt in enumerate(visit_times):
                col = eval_indices[vt_idx]
                delta_sum = (mu_hi[:, col] - mu_lo[:, col]).sum()

                retain = (vt_idx < L - 1)
                grads = torch.autograd.grad(delta_sum, params,
                                            retain_graph=retain,
                                            allow_unused=True)
                grads = [g if g is not None else torch.zeros_like(p)
                         for g, p in zip(grads, params)]
                g_batch = _cat_grads(grads).cpu()

                g_accum[vt] += g_batch
                est_accum[vt] += delta_sum.item()

            n_total += B

    # Normalise: n_total is the same for all visit times
    gradients = {}
    estimates = {}
    counts = {}
    for vt in visit_times:
        if n_total > 0:
            gradients[vt] = g_accum[vt] / n_total
            estimates[vt] = est_accum[vt] / n_total
        else:
            gradients[vt] = torch.zeros(P)
            estimates[vt] = 0.0
        counts[vt] = n_total

    if verbose:
        for vt in visit_times:
            print(f"    g_{int(vt)}: ||g|| = {gradients[vt].norm().item():.4f}, "
                  f"n = {counts[vt]}, ∆PDP = {estimates[vt]:.4f}")

    return gradients, estimates, counts


# ─────────────────────────────────────────────────────────
# 4. Main: full-parameter delta method variance
# ─────────────────────────────────────────────────────────

def _ledoit_wolf_shrink_fisher(fisher, scores, verbose=True):
    """
    Ledoit-Wolf shrinkage: F_shrunk = (1-α)F + α·(tr(F)/P)·I
    """
    N, P = scores.shape
    S = fisher / N
    mu = torch.trace(S).item() / P
    delta_sq = (S - mu * torch.eye(P)).pow(2).sum().item()
    if delta_sq < 1e-30:
        return fisher.clone(), 0.0

    norms_sq = (scores ** 2).sum(dim=1)
    sum_norms_4 = (norms_sq ** 2).sum().item()
    S_frob_sq = (S ** 2).sum().item()
    beta = max((1.0 / N**2) * (sum_norms_4 - N * S_frob_sq), 0.0)
    alpha = min(beta / delta_sq, 1.0)

    trace_F = torch.trace(fisher).item()
    F_shrunk = (1.0 - alpha) * fisher + alpha * (trace_F / P) * torch.eye(P)

    if verbose:
        eig_orig = torch.linalg.eigvalsh(fisher)
        eig_shrunk = torch.linalg.eigvalsh(F_shrunk)
        print(f"  Ledoit-Wolf shrinkage:")
        print(f"    N/P = {N}/{P} = {N/P:.1f},  α* = {alpha:.4f}")
        print(f"    F  eigenvalues: [{eig_orig.min().item():.2e}, "
              f"{eig_orig.max().item():.2e}]")
        print(f"    F* eigenvalues: [{eig_shrunk.min().item():.2e}, "
              f"{eig_shrunk.max().item():.2e}], "
              f"cond = {eig_shrunk.max().item()/max(eig_shrunk.min().item(),1e-30):.2e}")
    return F_shrunk, alpha


def _active_subspace_setup(fisher, threshold_ratio=1e-4, verbose=True):
    """
    Eigendecompose F and retain only the active subspace (λ_k > threshold).

    Returns eigvecs_active (P, K) and eigvals_active (K,) for projected
    variance: Var(h) = Σ_{k ∈ active} (gᵀv_k)² / λ_k
    """
    eigvals, eigvecs = torch.linalg.eigh(fisher)
    threshold = threshold_ratio * eigvals.max().item()
    active = eigvals > threshold
    K = active.sum().item()
    P = fisher.shape[0]

    eigvecs_active = eigvecs[:, active]
    eigvals_active = eigvals[active]

    if verbose:
        print(f"  Active subspace: {K}/{P} directions "
              f"(threshold = {threshold:.2e})")
        print(f"    Active eigenvalue range: [{eigvals_active.min().item():.2e}, "
              f"{eigvals_active.max().item():.2e}]")
        frac_trace = eigvals_active.sum().item() / max(eigvals.sum().item(), 1e-30)
        print(f"    Captures {frac_trace*100:.2f}% of tr(F)")
        # Check how much of g is in the null space (computed later per time point)
    return eigvecs_active, eigvals_active


def _variance_projected(g, eigvecs_active, eigvals_active):
    """Var = Σ_{k ∈ active} (gᵀv_k)² / λ_k"""
    coeffs = eigvecs_active.T @ g
    return (coeffs ** 2 / eigvals_active).sum().item()


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
    visit_times=np.array([0, 5, 10, 15]),
    true_beta_bmi=-0.175,
    true_beta_int=-0.015,
    sandwich=False,
    n_hessian_subsample=None,
    weight_decay=0.0,
    lambda_reg=0.0,
    dense_step=1.0,
    use_dopri5=True,
    variance_mode="marquardt",
    active_threshold=1e-10,
):
    """
    Full-parameter delta method for ∆PDP variance.

    Scores are pure NLL (penalty is data-independent for skip_gate/group_lasso).
    Centering removes the penalty gradient from F (Commenges et al., 2014).

    If sandwich=False (default):
        Var(∆PDP_ℓ) = g_ℓᵀ  F⁻¹  g_ℓ

    If sandwich=True, reports three estimators:
        SE_F⁻¹:    g_ℓᵀ  F⁻¹  g_ℓ                  (penalised Fisher)
        SE_bayes:  g_ℓᵀ  J⁻¹  g_ℓ                  (Bayesian, O'Sullivan 1988)
        SE_sand:   g_ℓᵀ  J⁻¹ F J⁻¹  g_ℓ            (sandwich, Commenges)

    where J = ∇²(penalised objective) includes weight decay + reg_term Hessian.

    Returns dict with results per visit time + Fisher + gradients.
    """
    P = _param_count(model)
    delta_v = bmi_hi - bmi_lo
    var_method = "SANDWICH (J⁻¹FJ⁻¹)" if sandwich else "FISHER (F⁻¹)"

    print(f"\n{'='*60}")
    print(f"FULL-PARAMETER DELTA METHOD — {var_method}")
    print(f"{'='*60}")
    print(f"  P = {P} parameters")
    print(f"  N = {len(dataset)} subjects")
    print(f"  BMI: lo={bmi_lo}, hi={bmi_hi}, Δv={delta_v}")
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
    use_active = (variance_mode == "active_subspace")
    use_lw = (variance_mode == "ledoit_wolf")

    if use_lw:
        print(f"\n  Applying Ledoit-Wolf shrinkage to F ...")
        fisher, lw_alpha = _ledoit_wolf_shrink_fisher(fisher, scores)

    if use_active:
        print(f"\n  Active subspace projection ...")
        eigvecs_active, eigvals_active = _active_subspace_setup(
            fisher, threshold_ratio=active_threshold)
        # Still compute fisher_inv for diagnostics / sandwich
        fisher_reg, fisher_inv = _regularise_and_invert(fisher, "Fisher", LAMBDA)
    else:
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

        # --- Step 2: ∆PDP gradients ---
        print(f"\nStep 2: ∆PDP gradients g_ℓ = ∇_θ ∆PDP_ℓ ...")
        gradients, estimates, counts = compute_delta_pdp_gradients(
            model, loader, device, bmi_lo, bmi_hi, visit_times,
            dense_step=dense_step, use_dopri5=use_dopri5,
        )
    else:
        # --- Step 2: ∆PDP gradients ---
        print(f"\nStep 2: ∆PDP gradients g_ℓ = ∇_θ ∆PDP_ℓ ...")
        gradients, estimates, counts = compute_delta_pdp_gradients(
            model, loader, device, bmi_lo, bmi_hi, visit_times,
            dense_step=dense_step, use_dopri5=use_dopri5,
        )

    # --- Step 3: Var = gᵀ Cov g ---
    all_ages = []
    for batch in loader:
        _, _, _, _, _, _, s = batch
        all_ages.append(s[:, 1])
    mean_age = torch.cat(all_ages).mean().item()
    true_delta = delta_v * (true_beta_bmi + true_beta_int * mean_age)

    print(f"\nStep 3: Var(∆PDP_ℓ) = g_ℓᵀ Cov g_ℓ")
    print(f"  mean AGEc = {mean_age:.4f}")
    print(f"  true ∆PDP = {true_delta:.4f}")

    if sandwich:
        print(f"\n  {'Time':>6s}  {'∆PDP':>10s}  {'SE_F⁻¹':>10s}  "
              f"{'SE_bayes':>10s}  {'SE_sand':>10s}  {'CI_lo':>10s}  "
              f"{'CI_hi':>10s}  {'True':>10s}  {'Bias':>10s}")
    else:
        print(f"\n  {'Time':>6s}  {'∆PDP':>10s}  {'SE_F⁻¹':>10s}  "
              f"{'CI_lo':>10s}  {'CI_hi':>10s}  {'True':>10s}  {'Bias':>10s}")
    print(f"  {'-'*80}")

    results = {}
    for vt in visit_times:
        g = gradients[vt]
        est = estimates[vt]

        coeffs = eigvecs.T @ g
        null_mask = eigvals < 1e-6
        frac_null = (coeffs[null_mask] ** 2).sum() / (coeffs ** 2).sum()
        print(f"  t={int(vt)}: ||g_null||² / ||g||² = {frac_null.item():.4f}")

        if use_active:
            var_fisher = _variance_projected(g, eigvecs_active, eigvals_active)
        else:
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
        }

        if sandwich:
            print(f"  {vt:6.0f}  {est:+10.4f}  {se_fisher:10.4f}  "
                  f"{se_main:10.4f}  {se_sand:10.4f}  "
                  f"{ci_lo:+10.4f}  {ci_hi:+10.4f}  "
                  f"{true_delta:+10.4f}  {bias:+10.4f}")
        else:
            print(f"  {vt:6.0f}  {est:+10.4f}  {se_fisher:10.4f}  "
                  f"{ci_lo:+10.4f}  {ci_hi:+10.4f}  "
                  f"{true_delta:+10.4f}  {bias:+10.4f}")

    ret = {
        'results': results,
        'fisher': fisher.numpy(),
        'fisher_inv': fisher_inv.numpy(),
        'gradients': {vt: g.numpy() for vt, g in gradients.items()},
        'estimates': estimates,
        'counts': counts,
        'P': P,
        'mean_age': mean_age,
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
    Output format matches LMM CSV.
    """
    D = len(all_results)
    summary = {}

    for vt in visit_times:
        ests = np.array([r['results'][vt]['estimate'] for r in all_results])
        ses = np.array([r['results'][vt]['se'] for r in all_results])

        # true_vals = np.array([r['results'][vt]['true'] for r in all_results])
        # coverage = np.mean((ci_los <= true_vals) & (true_vals <= ci_his))
        true_val = all_results[0]['results'][vt]['true']
        ci_los = np.array([r['results'][vt]['ci_lo'] for r in all_results])
        ci_his = np.array([r['results'][vt]['ci_hi'] for r in all_results])

        mean_hat = ests.mean()
        bias = mean_hat - true_val
        var_mc = ests.var(ddof=1) if D > 1 else float('nan')
        mean_var_est = (ses ** 2).mean()
        mse = bias ** 2 + var_mc if D > 1 else float('nan')
        rmse = np.sqrt(mse) if D > 1 else float('nan')
        coverage = np.mean((ci_los <= true_val) & (true_val <= ci_his)) if D > 1 else float('nan')

        summary[vt] = {
            'mean_hat': mean_hat,
            'true': true_val,
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
    from model_ODE_baseline import NeuralODEModel, NeuralODEConfig

    parser = argparse.ArgumentParser(
        description="Full-parameter delta method for ∆PDP variance")
    parser.add_argument("--n_sims", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--true_beta_bmi", type=float, default=-0.30)
    parser.add_argument("--true_beta_int", type=float, default=-0.05)
    parser.add_argument("--output_csv", type=str,
                        default="results_simu/simulation_baseline_seed42_norm_diagD_summary_2328_noreg.csv")
    parser.add_argument("--bmi_pairs", type=str, default=None)
    parser.add_argument("--sandwich", action="store_true",
                        help="Use sandwich variance J⁻¹FJ⁻¹ instead of F⁻¹")
    parser.add_argument("--n_hessian_subsample", type=int, default=None,
                        help="Subsample N subjects for Hessian (e.g. 500).")
    parser.add_argument("--weight_decay", type=float, default=1e-5,
                        help="Weight decay used during training (added to J)")
    parser.add_argument("--lambda_reg", type=float, default=0.1,
                        help="Skip penalty coefficient (skip_gate or group_lasso)")
    parser.add_argument("--reg_mode", type=str, default=None,
                        choices=[None, "skip_gate", "group_lasso"],
                        help="Regularisation mode for skip connection")
    parser.add_argument("--dense_step", type=float, default=0.5,
                        help="Max spacing (years) for dense ODE integration grid "
                             "(only used when --no_dopri5)")
    parser.add_argument("--no_dopri5", action="store_true",
                        help="Use training solver + dense grid instead of adaptive dopri5")
    parser.add_argument("--variance_mode", type=str, default="ledoit_wolf",
                        choices=["marquardt", "ledoit_wolf", "active_subspace"],
                        help="Variance estimation mode: "
                             "marquardt (F⁻¹ with Marquardt damping), "
                             "ledoit_wolf (Ledoit-Wolf shrinkage + invert), "
                             "active_subspace (project onto active eigenvectors)")
    parser.add_argument("--active_threshold", type=float, default=1e-8,
                        help="Eigenvalue threshold ratio for active_subspace mode "
                             "(keeps λ_k > ratio × λ_max; default 1e-10 keeps all "
                             "non-zero eigenvalues)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    visit_times = np.array([0, 2, 4, 8, 10])

    # BMI pairs
    if args.bmi_pairs:
        bmi_pairs = [tuple(map(float, p.split(':')))
                     for p in args.bmi_pairs.split(',')]
    else:
        grid = [20, 23, 26, 29, 32, 35]
        bmi_pairs = [(grid[i], grid[i+1]) for i in range(len(grid)-1)]
        bmi_pairs.append((20, 35))

    time_col, y_col, id_col = "time", "ISA15_sim", "NUM_ID"
    x_cols = ["BMI_t"]
    static_cols = ["SEX_code", "AGEc", "DIPNIV2", "DIPNIV3"]

    all_pair_results = {pair: [] for pair in bmi_pairs}

    # --- Checkpoint: resume from last completed simulation ---
    ckpt_dir = os.path.dirname(args.output_csv) or '.'
    ckpt_file = os.path.join(ckpt_dir, "delta_method_checkpoint_baseline_configD.pt")
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
            data_path = f"simu_datasets/S2a_sims/sim_{sim_idx+1:03d}.rds"
            ckpt_path = f"checkpoints/model_selection_S1/D_noskip_sim00{sim_idx}.pt"
            print(f"\n{'#'*60}")
            print(f"# SIMULATION {sim_idx}")
            print(f"{'#'*60}")
        else:
            data_path = "simu_datasets/S2a_sims/sim_001.rds"
            ckpt_path = "checkpoints/simulation_baseline_noreg_seed42_rhonorm_diagoD/best_model_ode_0.pt"

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

        # --- Model ---
        cfg = NeuralODEConfig(
            hidden_channels=4, enc_mlp_hidden=16, func_mlp_hidden=16,
            dec_rho_hidden=16, dec_p=4, dec_q=3, depth=2, dropout=0.0,
            euler_steps_per_interval=4,
        )
        model = NeuralODEModel(
            x_dim=len(x_cols), static_dim=len(static_cols), cfg=cfg,
            n_tv=1, use_rho_net=True, use_neural_re=True,
            re_spline_cols=None, g_hidden=8, fullD=False,
            bmi_mean=0.0, bmi_std=1.0, static_skip_dims=None, use_bmi_skip=False,
            reg_mode=args.reg_mode, use_learned_z0=False, use_bmi_in_ode=True
        ).to(device)

        print(model)

        checkpoint = torch.load(ckpt_path, map_location=device,
                                weights_only=False)
        
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'], strict=True)
        else:
            model.load_state_dict(checkpoint, strict=True)
        print(f"Loaded: {ckpt_path}")

        # --- Compute for each BMI pair ---
        for bmi_lo, bmi_hi in bmi_pairs:
            result = compute_full_delta_variance(
                model, dataset, loader, device, collate_pad,
                bmi_lo=bmi_lo, bmi_hi=bmi_hi,
                visit_times=visit_times,
                true_beta_bmi=args.true_beta_bmi,
                true_beta_int=args.true_beta_int,
                sandwich=args.sandwich,
                n_hessian_subsample=args.n_hessian_subsample,
                weight_decay=args.weight_decay,
                lambda_reg=args.lambda_reg,
                dense_step=args.dense_step,
                use_dopri5=not args.no_dopri5,
                variance_mode=args.variance_mode,
                active_threshold=args.active_threshold,
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
    print(f"AGGREGATED — Full Delta Method (D={args.n_sims})")
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

    # # Clean up checkpoint after successful completion
    # if os.path.exists(ckpt_file):
    #     os.remove(ckpt_file)
    #     print(f"Checkpoint removed (all simulations complete).")