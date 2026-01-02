import matplotlib.pyplot as plt
import torch
from LME_3C_model import process_data, PatientDataset, collate_fn, CDEModel, ODENet, VectorField_with_static
import pandas as pd
from torch.utils.data import DataLoader
from LME_3C_evaluation import *

device = torch.device("cpu")
import warnings
# Add this after your imports to ignore this specific warning
warnings.filterwarnings("ignore", category=UserWarning, message="X does not have valid feature names")


static_features_dim = 6
# Load the model first
LATENT_DIM, LEARNING_RATE, WEIGHT_DECAY, EPOCHS, BATCH_SIZE = 4, 0.0001, 1e-4, 5000, 256
features = ['GLUC','HDL', 'PAD', 'BMI']
# features = ['GLUC', 'HDL', 'LDL', 'CHOL', 'TRYG', 'PAD', 'PAS', 'BMI']
static_features, target_col, id_col = ["DIPNIV_1","DIPNIV_2","DIPNIV_3", "SEX_1", "SEX_2", "AGE0"], "ISA15", "NUM_ID"
static_features, target_col, id_col = ["DIPNIV_1.0","DIPNIV_2.0","DIPNIV_3.0", "SEX_1.0", "SEX_2.0", "AGE0"], "ISA15", "NUM_ID"
# static_features, target_col, id_col = ["AGE0", "SEX_1.0", "SEX_2.0",], "ISA15", "NUM_ID"

import warnings
# Add this after your imports to ignore this specific warning
warnings.filterwarnings("ignore")
with_only_static_features = False
scale_dynamic_features = False
metabolic_features_baselines = False

train_df = pd.read_csv("3C_dataset/train_3C_data_1.csv", na_values=["NA", ""])
val_df = pd.read_csv("3C_dataset/test_3C_data.csv", na_values=["NA", ""])

train_df["SUIVI"] = pd.to_datetime(train_df["SUIVI"])
val_df["SUIVI"] = pd.to_datetime(val_df["SUIVI"])

train_data = process_data(train_df, id_col, features, static_features, target_col, with_only_static_features=with_only_static_features, scaler=None, metabolic_baseline=metabolic_features_baselines)
val_data = process_data(val_df, id_col, features, static_features, target_col, with_only_static_features=with_only_static_features, scaler=None,metabolic_baseline=metabolic_features_baselines)

train_dataset = PatientDataset(train_data)
val_dataset = PatientDataset(val_data)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

model = CDEModel(len(features)* 2 + 1, static_features_dim, LATENT_DIM, device).to(device)
print(model)
model.load_state_dict(torch.load('EXPs/LME_exp/model_latent_4_CDE_diagoG.pth', map_location='cpu')['model_state_dict'])

random_effect_std_devs = torch.exp(model.decoder.log_std_devs)
print("Learned Random Effect Standard Deviations:", random_effect_std_devs)

# Inspect the learned residual error
residual_std_dev = torch.sqrt(torch.exp(model.decoder.log_residual_var))
print(f"\nLearned Residual Standard Deviation: {residual_std_dev.item():.4f}")

# train_mse, train_pop_mse = calculate_fit_mse_with_blup(model, train_loader, device)
# print(f"train fitted MSE: {train_mse:.4f}, {train_pop_mse:.4f}")

# validation_mse, val_pop_mse = calculate_fit_mse_with_blup(model, val_loader, device)
# print(f"Validation fitted MSE: {validation_mse:.4f}, {val_pop_mse:.4f}")

# log_likelihood = lme_log_likelihood(model, val_loader, device)
# print("validation dataset likelihood", log_likelihood)

train_log_likelihood = lme_log_likelihood(model, train_loader, device)
print("train dataset likelihood", train_log_likelihood)

# validation_mse = calculate_prediction_mse_with_blup(model, val_loader, device)
# print(f"Validation pred MSE: {validation_mse:.4f}")

# train_mse = calculate_prediction_mse_with_blup(model, train_loader, device)
# print(f"train pred MSE: {train_mse:.4f}")


###random select participant to plot 
sample_ids = train_df['NUM_ID'].sample(n=45, random_state=2).tolist()
#List of 15 patient IDs (replace with actual IDs from your dataset)

hlme_predictions = pd.read_csv('results/ISA15_Model_4_train_predicted.csv', sep=',')
fig, axes = plt.subplots(5, 5, figsize=(22, 12), sharex=True, sharey=True)
axes = axes.flatten()
count = 0
for idx, patient_id in enumerate(sample_ids):
    # Get individual patient data
    sample_patient_data = filter_patient_with_id(patient_id, train_dataset)
    
    # Compute predicted trajectory
    t_points, seq_preds, actual_y, pop_preds = calculate_sequential_blup_forecasting(model, sample_patient_data, device)

    hlme_prediction = hlme_predictions[hlme_predictions.NUM_ID==patient_id]
    new_row = {
        "NUM_ID": patient_id,
        "time": 0,
        "Y_predicted": actual_y[0],
        }
    hlme_prediction = pd.concat([pd.DataFrame([new_row]), hlme_prediction], ignore_index=True)
    if len(hlme_prediction['Y_predicted']) != len(t_points):
        print(patient_id)
        continue
    ax = axes[count]
    # Plot actual vs predicted
    ax.plot(t_points, actual_y, 'o', label='Real data', color='royalblue', markersize=6, zorder=5)
    ax.plot(t_points, seq_preds, label='BLUP', color='forestgreen', linewidth=2, linestyle='--')
    ax.plot(t_points, hlme_prediction['Y_predicted'], label='HLME', color='black', linewidth=2, linestyle='--')

    ax.set_title(f'Patient ID: {patient_id}', fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.5)
    
    if count % 5 == 0:
        ax.set_ylabel("ISA15", fontsize=10)
    if count >= 10:
        ax.set_xlabel("Time (years)", fontsize=10)
    count += 1
    if count == 25:
        break

# Shared legend (outside the grid)
handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', ncol=2, fontsize=12)
plt.tight_layout(rect=[0, 0, 1, 0.95])  # Leave space for the legend
plt.suptitle("train dataset predictions", fontsize=16)
plt.savefig("figures/blup_predictions_train.pdf", format='pdf', bbox_inches='tight')
plt.close()


###random select participant to plot 
sample_ids = val_df['NUM_ID'].sample(n=45, random_state=42).tolist()
#List of 15 patient IDs (replace with actual IDs from your dataset)

hlme_predictions = pd.read_csv('results/ISA15_Model_4_val_predicted.csv', sep=',')
fig, axes = plt.subplots(5, 5, figsize=(22, 12), sharex=True, sharey=True)
axes = axes.flatten()
count = 0
for idx, patient_id in enumerate(sample_ids):
    # Get individual patient data
    sample_patient_data = filter_patient_with_id(patient_id, val_dataset)
    
    # Compute predicted trajectory
    t_points, seq_preds, actual_y, pop_preds = calculate_sequential_blup_forecasting(model, sample_patient_data, device)

    hlme_prediction = hlme_predictions[hlme_predictions.NUM_ID==patient_id]
    new_row = {
        "NUM_ID": patient_id,
        "time": 0,
        "Y_predicted": actual_y[0],
        }
    hlme_prediction = pd.concat([pd.DataFrame([new_row]), hlme_prediction], ignore_index=True)
    
    if len(hlme_prediction['Y_predicted']) != len(t_points):
        print("ERROR", patient_id)
        print(hlme_prediction["time"], t_points)
        continue
    ax = axes[count]
    # Plot actual vs predicted
    ax.plot(t_points, actual_y, 'o', label='Real data', color='royalblue', markersize=6, zorder=5)
    ax.plot(t_points, seq_preds, label='BLUP', color='forestgreen', linewidth=2, linestyle='--')
    ax.plot(t_points, hlme_prediction['Y_predicted'], label='HLME', color='black', linewidth=2, linestyle='--')

    ax.set_title(f'Patient ID: {patient_id}', fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.5)
    
    if count % 5 == 0:
        ax.set_ylabel("ISA15", fontsize=10)
    if count >= 10:
        ax.set_xlabel("Time (years)", fontsize=10)
    count += 1
    if count == 25:
        break

# Shared legend (outside the grid)
handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', ncol=2, fontsize=12)
plt.tight_layout(rect=[0, 0, 1, 0.95])  # Leave space for the legend
plt.suptitle("validation dataset predictions", fontsize=16)
plt.savefig("figures/blup_predictions_validation.pdf", format='pdf', bbox_inches='tight')
plt.close()

train_predicitons = []

for idx, patient_id in enumerate(train_df['NUM_ID'].unique().tolist()):
    # ax = axes[idx]
    # Get individual patient data
    sample_patient_data = filter_patient_with_id(patient_id, train_dataset)
    
    # Compute predicted trajectory
    t_points, seq_preds, actual_y, pop_preds = calculate_sequential_blup_forecasting(model, sample_patient_data, device)
    pred_dict = {'time':t_points, 'ISA15':seq_preds, "id": patient_id, 'pop_pred':pop_preds}
    train_predicitons.append(pred_dict)
    

np.save("results/LME_3C_train_predictions.npy", train_predicitons)
def convert_pred_list_to_df(pred_list, value_col, time_col):
     # ---- Convert list of prediction dicts → DataFrame ----
    rows = []
    for d in pred_list:
        times = d["time"]
        values = np.asarray(d[value_col])
        if "id" in d.keys():
            pid = d["id"]
        else:
            pid = d['NUM_ID']
        
        for t, v in zip(times, values):
            rows.append({time_col: t, value_col: v, "id": pid})

    df = pd.DataFrame(rows)
    print(df.columns)
    return df


def compute_global_bin_percentiles(df, time_col, value_col, percentiles=[5, 50, 95]):
    # ---- Now reuse your existing logic ----
    bins = [0, 1.5, 3.5, 5.5, 8.5, 11, 14]
    df = df.copy()
    df["bin"] = pd.cut(df[time_col], bins=bins, labels=False, include_lowest=True)

    out_rows = []
    for b in range(len(bins) - 1):
        sub = df[df["bin"] == b]

        row = {
            "segment_start": bins[b],
            "segment_end": bins[b+1],
        }

        if len(sub) == 0:
            for p in percentiles:
                row[f"p{p}"] = np.nan
        else:
            vals = sub[value_col].values
            for p in percentiles:
                row[f"p{p}"] = np.percentile(vals, p)

        out_rows.append(row)

    return pd.DataFrame(out_rows)

pop_df = convert_pred_list_to_df(train_predicitons,
    time_col="time",
    value_col="pop_pred")
fig, ax = plt.subplots(figsize=(8, 5))
for subject, df_sub in pop_df.groupby("id"):
    ax.plot(df_sub["time"], df_sub["pop_pred"], alpha=0.4)

ax.set_xlabel("Time (years)")
ax.set_ylabel("ISA15")

plt.savefig("figures/pop_prediction.pdf")
plt.close()


df = convert_pred_list_to_df(train_predicitons,
    time_col="time",
    value_col="ISA15")
summary_df = compute_global_bin_percentiles(
    df,
    time_col="time",
    value_col="ISA15"
)
summary_df["bin_center"] = (summary_df["segment_start"] + summary_df["segment_end"]) / 2
# Extract x (time) and y (percentiles)
x = summary_df["bin_center"].values
y5  = summary_df["p5"].values
y50 = summary_df["p50"].values
y95 = summary_df["p95"].values
print(summary_df)


train_df["time"] = (train_df["SUIVI"] - train_df["SUIVI"].min()).dt.total_seconds() / (60 * 60 * 24 * 365)
summary_df_real_data = compute_global_bin_percentiles(
    train_df,
    time_col="time",
    value_col="ISA15"
)
summary_df_real_data["bin_center"] = (summary_df_real_data["segment_start"] + summary_df_real_data["segment_end"]) / 2
# Extract x (time) and y (percentiles)
x_real_data = summary_df_real_data["bin_center"].values
y5_real_data  = summary_df_real_data["p5"].values
y50_real_data = summary_df_real_data["p50"].values
y95_real_data = summary_df_real_data["p95"].values

hlme_train_predictions = []
hlme_predictions = pd.read_csv('results/ISA15_Model_4_train_predicted.csv', sep=',')
for name, group in hlme_predictions.groupby("NUM_ID"):
    hlme_prediction = hlme_predictions[hlme_predictions.NUM_ID==name]
    new_row = {
        "NUM_ID": patient_id,
        "time": 0,
        "Y_predicted": actual_y[0],
        }
    hlme_prediction = pd.concat([pd.DataFrame([new_row]), group], ignore_index=True)
    df = group.reset_index(drop=True)
    hlme_train_predictions.append(df)

df = convert_pred_list_to_df(hlme_train_predictions,  time_col="time",
    value_col="Y_predicted",)
summary_df_hlme = compute_global_bin_percentiles(
    df,
    time_col="time",
    value_col="Y_predicted",
)
summary_df_hlme["bin_center"] = (summary_df_hlme["segment_start"] + summary_df_hlme["segment_end"]) / 2

# Extract x (time) and y (percentiles)
x_hlme = summary_df_hlme["bin_center"].values
y5_hlme  = summary_df_hlme["p5"].values
y50_hlme = summary_df_hlme["p50"].values
y95_hlme = summary_df_hlme["p95"].values

# Plot
plt.figure(figsize=(10, 6))

plt.plot(x, y50, label="CDE 50%", linewidth=3, color="blue")
plt.plot(x, y5,  label="CDE 5%", linestyle="--",color="blue")
plt.plot(x, y95, label="CDE 95%", linestyle="--", color="blue")

plt.plot(x_real_data, y50_real_data, label="real data 50%", linewidth=3, color="red")
plt.plot(x_real_data, y5_real_data,  label="real data CDE 5%", linestyle="--",color="red")
plt.plot(x_real_data, y95_real_data, label="real data CDE 95%", linestyle="--", color="red")

plt.plot(x_hlme, y50_hlme, label="HLME 50%", linewidth=3, color="black")
plt.plot(x_hlme, y5_hlme,  label="HLME 5%", linestyle="--",color="black")
plt.plot(x_hlme, y95_hlme, label="HLME 95%", linestyle="--", color="black")

plt.xlabel("Time")
plt.ylabel("Value")
plt.title("Percentile Trajectories Over Time")
plt.legend()
plt.savefig("figures/sequence_percentile_3C_data.pdf")

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# metabolic_baseline_features_dim = 4
# from exp.LME_3C_model import ODEFunc

# # 1. Create an instance of your original ODEFunc
# ode_dynamics = ODEFunc(hidden_dim=LATENT_DIM, static_dim=static_features_dim)

# # 2. Apply torch.jit.script to it
# print("Scripting the ODE dynamics function...")
# scripted_ode_func = torch.jit.script(ode_dynamics)
# print("Scripting complete.")

# ode_model = ODENet(scripted_ode_func, static_features_dim, LATENT_DIM, device, fullG=False).to(device)
# ode_model.load_state_dict(torch.load('LME_exp/model_latent_4_ODE_diagoG.pth', map_location='cpu')['model_state_dict'])

# # Inspect the learned random effect standard deviations
# # These correspond to the intercept and each latent dimension
# import warnings
# # Add this after your imports to ignore this specific warning
# warnings.filterwarnings("ignore", category=UserWarning)
# random_effect_std_devs = torch.exp(ode_model.decoder.log_std_devs / 2)
# print("Learned Random Effect Standard Deviations:")
# print(random_effect_std_devs)
# # Inspect the learned residual error
# residual_std_dev = torch.sqrt(torch.exp(ode_model.decoder.log_residual_var))
# print(f"\nLearned Residual Standard Deviation: {residual_std_dev.item():.4f}")

