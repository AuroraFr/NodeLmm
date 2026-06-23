"""
Check group lasso skip weight norms for the Neural ODE-LMM decoder.

For each time-varying covariate group (value + mask = 2 columns) and
each static skip covariate (1 column), computes ||W_g||_F from the
first layer of rho_net.

Usage:
    python check_skip_norms.py --ckpt checkpoints/best_model.pt
"""
from model_ODE_cumulative import NeuralODEModel, NeuralODEConfig
import torch
import argparse

def check_skip_norms(ckpt_path, model_type="cumulative"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    time_varying_features = ["BMI_t"]
    static_features = ["SEX_code", "AGEc", "DIPNIV2", "DIPNIV3"]
    static_skip_dims = [0, 1, 2, 3]

    # --- Build model ---
    cfg = NeuralODEConfig(
        hidden_channels=8, enc_mlp_hidden=16, func_mlp_hidden=16,
        dec_rho_hidden=16, dec_p=4, dec_q=3, depth=2, dropout=0.0,
        euler_steps_per_interval=4,
    )

    model = NeuralODEModel(
        x_dim=len(time_varying_features), static_dim=len(static_features),
        cfg=cfg, n_tv=1, use_rho_net=True, use_neural_re=True,
        re_spline_cols=None, g_hidden=8, fullD=False,
        bmi_mean=0.0, bmi_std=1.0,
        static_skip_dims=static_skip_dims,
        reg_mode="group_lasso",
        use_bmi_in_ode=True,
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
    Ks_skip = len(static_skip_dims)

    print(f"First layer shape: {list(W.shape)}")
    print(f"  Latent dim (d): {d}")
    print(f"  Dynamic groups (K): {K} × 2 columns = {K*2}")
    print(f"  Static skip dims: {Ks_skip}")
    print(f"  Expected input dim: {d + K*2 + Ks_skip} "
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

    col = d  # current column offset

    for k, name in enumerate(time_varying_features):
        # Each dynamic group: 2 columns (value + mask)
        W_g = W[:, col:col+2]
        g_norm = W_g.norm(p='fro').item()
        print(f"  {name:>15s}  {'dynamic':>12s}  {col}-{col+1}  "
              f"{g_norm:10.4f}  {z_norm:10.4f}  {g_norm/z_norm:8.4f}")
        col += 2

    for idx in static_skip_dims:
        name = static_features[idx]
        W_g = W[:, col:col+1]
        g_norm = W_g.norm(p='fro').item()
        print(f"  {name:>15s}  {'static':>12s}  {col}      "
              f"{g_norm:10.4f}  {z_norm:10.4f}  {g_norm/z_norm:8.4f}")
        col += 1

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