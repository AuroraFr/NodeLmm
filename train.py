import torch
from torch.utils.data import DataLoader
from model import NeuralCDEModel
import os
import math
import pyreadr
import pandas as pd
from dataset import LongitudinalDataset, collate_pad
# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# -----------------------
# Your masked NLL
# -----------------------
def masked_NLL(predicted_mean, targets, V, mask):
    N, T = predicted_mean.shape

    residual = (targets - predicted_mean) * mask  # (N, T)
    residual = residual.unsqueeze(-1)  # (N, T, 1)

    mask_f = mask.float()
    mask_matrix = mask_f.unsqueeze(2) * mask_f.unsqueeze(1)  # (N, T, T)

    V_masked = V * mask_matrix  # (N, T, T)

    eye = torch.eye(T, device=V.device, dtype=V.dtype).unsqueeze(0).expand(N, T, T)
    V_masked = V_masked + 1e-6 * eye

    # Cholesky
    try:
        L = torch.linalg.cholesky(V_masked)  # (N,T,T)
    except RuntimeError:
        # return large loss to keep training running
        return torch.tensor(1e6, device=V.device, requires_grad=True)

    diag_L = torch.diagonal(L, dim1=-2, dim2=-1)  # (N,T)
    logdet_V = 2 * torch.sum(torch.log(diag_L + 1e-8) * mask, dim=1)  # (N,)

    V_inv_residual = torch.cholesky_solve(residual, L)  # (N,T,1)
    quad_term = torch.bmm(residual.transpose(1, 2), V_inv_residual).squeeze(-1).squeeze(-1)  # (N,)

    T_valid = mask.sum(dim=1).float()  # (N,)
    log_2pi_term = T_valid * math.log(2 * math.pi)

    loss = 0.5 * (logdet_V + quad_term + log_2pi_term)
    return loss.mean()


# -----------------------
# Build V per batch (TEMPLATE)
# -----------------------
def build_batch_V_random_intercept_slope(
    t_pad: torch.Tensor,         # (N,T)
    mask: torch.Tensor,          # (N,T)
    sigma_eps: torch.Tensor,     # scalar (or 1,)
    G: torch.Tensor,             # (q,q) here q=2 for (intercept,slope)
) -> torch.Tensor:
    """
    Example LMM covariance:
      y_i(t) = mu_i(t) + b0_i + b1_i * t + eps_it
      b_i ~ N(0, G), eps ~ N(0, sigma_eps^2 I)

    Returns:
      V: (N,T,T) on the padded grid.
    """
    N, T = t_pad.shape
    device, dtype = t_pad.device, t_pad.dtype

    ones = torch.ones((N, T), device=device, dtype=dtype)
    # Z: (N,T,2)
    Z = torch.stack([ones, t_pad], dim=-1)

    # V_re = Z G Z^T -> (N,T,T)
    # (N,T,2) @ (2,2) -> (N,T,2), then @ (N,2,T) -> (N,T,T)
    ZG = torch.matmul(Z, G)                         # (N,T,2)
    V_re = torch.matmul(ZG, Z.transpose(1, 2))      # (N,T,T)

    # iid residual
    eye = torch.eye(T, device=device, dtype=dtype).unsqueeze(0).expand(N, T, T)
    V = V_re + (sigma_eps ** 2) * eye

    # IMPORTANT: do NOT “hard mask” V here; your masked_NLL already masks rows/cols
    # But you may want to stabilize padded region:
    # - we keep V defined everywhere; mask will exclude it in NLL.

    return V


# -----------------------
# Training loop
# -----------------------
def train(
    model,
    loader,
    optimizer,
    device,
    epochs=200,
    grad_clip=1.0,
    print_every=50,
    # LMM covariance params (could be learned too)
    sigma_eps=0.2,
    G_init=((0.5, 0.0), (0.0, 0.1)),
    # checkpointing
    save_path="checkpoints/simuS3_model.pt",
):
    model.train()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    sigma_eps_t = torch.tensor(float(sigma_eps), device=device)
    G = torch.tensor(G_init, device=device)

    best_loss = float("inf")
    best_epoch = 0

    for epoch in range(1, epochs + 1):
        total = 0.0
        count = 0

        for batch in loader:
            t_pad, x_pad, y_pad, mask, s_pad, lengths = batch
            t_pad = t_pad.to(device)
            x_pad = x_pad.to(device)
            y_pad = y_pad.to(device)
            mask  = mask.to(device)
            s_pad = None if s_pad is None else s_pad.to(device)

            mu, V = model(t_pad, x_pad, s_pad)
            loss = masked_NLL(mu, y_pad, V, mask)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()

            if grad_clip is not None and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            optimizer.step()

            total += loss.item()
            count += 1

        avg = total / max(count, 1)

        # --- save best ---
        if avg < best_loss:
            best_loss = avg
            best_epoch = epoch
            torch.save(
                {
                    "epoch": epoch,
                    "best_loss": best_loss,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    # save these too so you can reproduce V-building params
                    "sigma_eps": float(sigma_eps_t.item()),
                    "G": G.detach().cpu(),
                },
                save_path,
            )

        if epoch % print_every == 0 or epoch == 1:
            print(
                f"Epoch {epoch:5d} | loss = {avg:.6f} | "
                f"best = {best_loss:.6f} (epoch {best_epoch})"
            )

    print(f"Training done. Best loss {best_loss:.6f} at epoch {best_epoch}. Saved to {save_path}")
    return model


# -----------------------
# Example main
# -----------------------
if __name__ == "__main__":
    import pyreadr
    import pandas as pd

    from dataset import LongitudinalDataset, collate_pad
    from model import NeuralCDEModel, NeuralCDEConfig

    # Hyperparams
    LR = 1e-3
    WD = 1e-4
    EPOCHS = 2000
    BATCH_SIZE = 64

    x_cols = ["GLUC_interp", "BMI_interp"]
    time_col = "time"
    static_cols = ["SEX_code", "AGE0"]
    y_col = "ISA15_sim"
    id_col = "NUM_ID"

    path = "simu_datasets/df_sim15_S3.rds"
    df = next(iter(pyreadr.read_r(path).values()))

    df["SEX"] = df["SEX"].astype("category")
    df["SEX_code"] = df["SEX"].cat.codes.astype("int64")

    dataset = LongitudinalDataset(df, id_col, time_col, x_cols, y_col, static_cols=static_cols)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_pad)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg = NeuralCDEConfig(hidden_channels=4, solver="rk4", step_size=0.05, atol=1e-6, rtol=1e-6)
    model = NeuralCDEModel(x_dim=len(x_cols), static_dim=len(static_cols), cfg=cfg).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)

    train(model, loader, optimizer, device, epochs=EPOCHS, print_every=50)
