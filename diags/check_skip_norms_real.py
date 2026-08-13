"""
Check group lasso skip weight norms for the Neural ODE-LMM decoder.

For each time-varying covariate group (value + mask = 2 columns) and
each static skip covariate (1 column), computes ||W_g||_F from the
first layer of rho_net.

Usage:
    python check_skip_norms.py --ckpt checkpoints/best_model.pt
"""
from model_ODE_real import NeuralODEModel, NeuralODEConfig
import torch
import argparse

def check_skip_norms(ckpt_path, model_type="cumulative"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Load checkpoint ─────────────────────────────────────────────────
    checkpoint = torch.load(ckpt_path, map_location=device,
                            weights_only=False)
    ckpt_cfg = checkpoint['config']
    print(ckpt_cfg)

    print(f"Checkpoint: {ckpt_path}")

    # ── Feature definitions ─────────────────────────────────────────────
    id_col = "NUM_ID"
    target_col = "ISA15"
    time_varying_features = ckpt_cfg.get('time_varying_features',
                                         ["BMI", "PAS", "PAD", "GLUC", "HDL"])
    static_features = ckpt_cfg.get('static_features',
                                    ["SEX_code", "AGEc", "DIPNIV_2", "DIPNIV_3"])
    K = len(time_varying_features)
    Ks = len(static_features)
    interp_method = ckpt_cfg.get('interp_method', 'linear')
    mask_type = ckpt_cfg.get('mask_type', 'binary')
    cov_means = checkpoint['cov_means']
    cov_stds = checkpoint['cov_stds']

    print(f"  Covariates: {time_varying_features}")
    print(f"  Statics:    {static_features}")
    

    # ── Rebuild model ───────────────────────────────────────────────────
    cfg = NeuralODEConfig(
        hidden_channels=ckpt_cfg['hidden_channels'],
        enc_mlp_hidden=ckpt_cfg.get('enc_mlp_hidden', 16),
        func_mlp_hidden=ckpt_cfg.get('func_mlp_hidden', 16),
        dec_rho_hidden=ckpt_cfg.get('dec_rho_hidden', 16),
        dec_p=ckpt_cfg.get('dec_p', 4),
        dec_q=ckpt_cfg.get('dec_q', 3),
        depth=ckpt_cfg.get('depth', 2),
        dropout=0.0,
        euler_steps_per_interval=ckpt_cfg.get('euler_steps', 4),
        ode_solver=ckpt_cfg.get('ode_solver', 'rk4'),
        use_rho_norm=ckpt_cfg.get('use_rho_norm', True)
    )

    model = NeuralODEModel(
        n_tv=K, static_dim=Ks, cfg=cfg,
        use_rho_net=True, use_neural_re=True,
        g_hidden=8, fullD=False,
        cov_means=cov_means, cov_stds=cov_stds,
        use_dynamic_skip=True,
        static_skip_dims=list(range(Ks)),
        reg_mode=ckpt_cfg.get('reg_mode', None),
    ).to(device)

    # --- Load checkpoint ---
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'], strict=True)
        if 'config' in checkpoint:
            print(f"Config: {checkpoint['config']}")
        if 'bmi_mean' in checkpoint:
            print(f"BMI stats: mean={checkpoint['bmi_mean']:.4f}, "
                  f"std={checkpoint['bmi_std']:.4f}")
    else:
        model.load_state_dict(checkpoint, strict=True)
    print(f"Loaded: {ckpt_path}\n")

    # --- Extract first-layer weights ---
    decoder = model.decoder
    W = decoder.rho_net.net[0].weight.detach().cpu()  # (h_out, h_in)
    d = decoder.latent_dim
    K = len(time_varying_features)
    Ks_skip = len(static_features)

    print(f"First layer shape: {list(W.shape)}")
    print(f"  Latent dim (d): {d}")
    print(f"  Dynamic groups (K): {K} × 2 columns = {K}")
    print(f"  Static skip dims: {Ks_skip}")
    print(f"  Expected input dim: {d + K * 2 + Ks_skip} "
          f"(actual: {W.shape[1]})")

    # --- Latent state norm (reference) ---
    W_z = W[:, :d]
    z_norm = W_z.norm(p='fro').item()

    # --- Dynamic covariate group norms ---
    print(f"\n{'='*60}")
    print(f"Group lasso skip weight norms")
    print(f"{'='*60}")
    print(f"  {'Group':>15s}  {'Type':>12s}  {'Cols':>6s}  "
          f"{'||W_g||_F':>10s}  {'||W_z||_F':>10s}  {'ratio':>8s}")
    print(f"  {'-'*70}")

    col_tv = d          # start of TV values
    col_static = d + K  # start of static
    col_mask = d + K + Ks_skip  # start of masks

    for k, name in enumerate(time_varying_features):
        # Group k: value column + mask column (non-adjacent)
        val_col = col_tv + k
        mask_col = col_mask + k
        W_g = torch.cat([W[:, val_col:val_col+1],
                        W[:, mask_col:mask_col+1]], dim=1)  # (h_out, 2)
        g_norm = W_g.norm(p='fro').item()
        print(f"  {name:>15s}  {'dynamic':>12s}  {val_col},{mask_col}  "
            f"{g_norm:10.4f}  {z_norm:10.4f}  {g_norm/z_norm:8.4f}")

    for s, name in enumerate(static_features):
        sc = col_static + s
        W_g = W[:, sc:sc+1]
        g_norm = W_g.norm(p='fro').item()
        print(f"  {name:>15s}  {'static':>12s}  {sc}        "
            f"{g_norm:10.4f}  {z_norm:10.4f}  {g_norm/z_norm:8.4f}")

        # --- Model parameters summary ---
        print(f"\n{'='*60}")
        print(f"Model parameters")
        print(f"{'='*60}")

    beta = decoder.beta_neural.detach().cpu()
    sig2 = torch.exp(decoder.log_residual_var).item()
    D = decoder._build_D(device, decoder.log_residual_var.dtype).detach().cpu()

    print(f"  sigma2 = {sig2:.4f}")
    print(f"  beta   = [{', '.join(f'{v:.4f}' for v in beta)}]")
    print(f"  D:")
    for i in range(D.shape[0]):
        print(f"    [{', '.join(f'{D[i,j]:.4f}' for j in range(D.shape[1]))}]")

    # --- All small parameters ---
    print(f"\n{'='*60}")
    print(f"Small parameters (numel <= 20)")
    print(f"{'='*60}")
    for name, param in model.named_parameters():
        p = param.detach().cpu()
        if p.numel() <= 20:
            print(f"  {name}: {p.data}")
        else:
            print(f"  {name}: shape={list(p.shape)}, "
                  f"norm={p.norm():.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    args = parser.parse_args()
    check_skip_norms(args.ckpt)