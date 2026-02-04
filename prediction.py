import torch
from model import NeuralCDEModel, NeuralCDEConfig
from dataset import LongitudinalDataset, collate_pad

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ckpt_path = "checkpoints/simuS3_model.pt"   # <- change if needed

# Recreate the SAME config you used in training
cfg = NeuralCDEConfig(hidden_channels=4, solver="rk4", step_size=0.05, atol=1e-6, rtol=1e-6)

# Must match training: x_dim=len(x_cols)=2 and static_dim=len(static_cols)=2
model = NeuralCDEModel(x_dim=2, static_dim=2, cfg=cfg).to(device)

ckpt = torch.load(ckpt_path, map_location=device)
model.load_state_dict(ckpt["model_state_dict"], strict=True)
model.eval()

print("Loaded:", ckpt_path, "| epoch:", ckpt.get("epoch"), "| best_loss:", ckpt.get("best_loss"))

@torch.no_grad()
def predict_batch(model, batch, device):
    t_pad, x_pad, y_pad, mask, s_pad, lengths = batch
    t_pad = t_pad.to(device)
    x_pad = x_pad.to(device)
    mask  = mask.to(device)
    s_pad = None if s_pad is None else s_pad.to(device)

    mu, V = model(t_pad, x_pad, s_pad)  # mu: (N,T), V: (N,T,T)
    return mu, V, mask, y_pad



@torch.no_grad()
def predict_prefix_last(model, batch, ell, device):
    t_pad, x_pad, y_pad, mask, s_pad, lengths = batch
    t_pref = t_pad[:, :ell].to(device)
    x_pref = x_pad[:, :ell, :].to(device)
    s_pad  = None if s_pad is None else s_pad.to(device)

    mu, V = model(t_pref, x_pref, s_pad)   # mu: (N, ell)
    return mu[:, -1]     

import torch
import matplotlib.pyplot as plt

#fit mode
@torch.no_grad()
def plot_batch_predictions(model, loader, device, n_subjects=20, ncols=4):
    model.eval()

    batch = next(iter(loader))
    t_pad, x_pad, y_pad, mask, s_pad, lengths = batch

    t_pad = t_pad.to(device)
    x_pad = x_pad.to(device)
    y_pad = y_pad.to(device)
    mask  = mask.to(device)
    s_pad = None if s_pad is None else s_pad.to(device)

    mu, V = model(t_pad, x_pad, s_pad)   # mu: (N,T), V: (N,T,T)

    sig2 = torch.exp(model.decoder.log_residual_var).to(device=device, dtype=V.dtype)
    jitter = torch.tensor(getattr(model.decoder, "jitter", 0.0), device=device, dtype=V.dtype)

    N, T = mu.shape
    n_subjects = min(n_subjects, N)

    nrows = (n_subjects + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.2 * nrows), squeeze=False, sharey=True)
    axes = axes.ravel()

    for p in range(n_subjects):
        ax = axes[p]
        idx = torch.where(mask[p] > 0)[0]  # observed indices
        if idx.numel() < 2:
            ax.set_title(f"Subject {p} (too few points)")
            ax.axis("off")
            continue

        mu_i = mu[p, idx]                    # (n_i,)
        y_i  = y_pad[p, idx]                 # (n_i,)
        V_i  = V[p][idx][:, idx]             # (n_i,n_i)

        I = torch.eye(idx.numel(), device=device, dtype=V.dtype)
        Vre_i = V_i - (sig2 + jitter) * I

        r = (y_i - mu_i).unsqueeze(-1)       # (n_i,1)
        L = torch.linalg.cholesky(V_i + 1e-6 * I)
        Vinv_r = torch.cholesky_solve(r, L)  # (n_i,1)
        re_contrib = (Vre_i @ Vinv_r).squeeze(-1)  # (n_i,)
        y_blup = mu_i + re_contrib           # (n_i,)

        # CPU for plotting
        t_i = t_pad[p, idx].detach().cpu()
        y_true = y_i.detach().cpu()
        y_mu = mu_i.detach().cpu()
        y_blup = y_blup.detach().cpu()

        ax.scatter(t_i.numpy(), y_true.numpy(), marker="o", label="truth")
        ax.plot(t_i.numpy(), y_blup.numpy(), marker="x", linestyle="--", label="BLUP")
        ax.plot(t_i.numpy(), y_mu.numpy(), linestyle=":", label="mu only")
        ax.set_title(f"Subject {p} (n={idx.numel()})")
        ax.set_xlabel("time")
        ax.set_ylabel("ISA15_sim")

    # turn off unused panels
    for k in range(n_subjects, len(axes)):
        axes[k].axis("off")

    # one shared legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig('simu3_prediction.pdf')

# usage:
# plot_batch_predictions(model, test_loader, device, n_subjects=6)
                  # forecast at time ell (N,)

import pyreadr
from torch.utils.data import DataLoader
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

# ---- sanity run ----
batch = next(iter(loader))   # or test_loader
mu, V, mask, y_pad = predict_batch(model, batch, device)

print("mu:", mu.shape, "V:", V.shape, "mask:", mask.shape)
print("first subject mu (first 5):", mu[0, :5])
plot_batch_predictions(model, loader, device)