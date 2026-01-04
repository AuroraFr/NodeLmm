import matplotlib.pyplot as plt
import torch
from LME_3C_model import process_data, PatientDataset, collate_fn, CDEModel, ODENet, VectorField_with_static
import pandas as pd
from torch.utils.data import DataLoader
from LME_3C_evaluation import *
from utils import *
device = torch.device("cpu")
import warnings
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
warnings.filterwarnings("ignore")
with_only_static_features = False
scale_dynamic_features = False
metabolic_features_baselines = False

train_df = pd.read_csv("3C_dataset/train_3C_data_1.csv", na_values=["NA", ""])
val_df = pd.read_csv("3C_dataset/test_3C_data.csv", na_values=["NA", ""])

train_df["SUIVI"] = pd.to_datetime(train_df["SUIVI"])
val_df["SUIVI"] = pd.to_datetime(val_df["SUIVI"])

##### PLOT mean predictions with CI ###############

# CDE_predictions = np.load("results/CDE_3C_train_predictions.npy", allow_pickle=True)
# CDE_predictions_list = CDE_predictions.tolist()
# predictions_df = pd.DataFrame(CDE_predictions_list)
# predictions_df = predictions_df.drop(columns=["pop_pred"])
# cols_to_explode = ["time", "ISA15"]

# df_long = (
#     predictions_df
#     .explode(cols_to_explode, ignore_index=True)
#     .assign(
#         time=lambda d: pd.to_numeric(d["time"]),
#         ISA15=lambda d: pd.to_numeric(d["ISA15"])
#     )
# )

# hlme_predictions = pd.read_csv('results/ISA15_Model_4_train_predicted.csv', sep=',')
# value_col = "ISA15"
# time_col  = "time"
# df_y0 = (df_long.sort_values([ "id", time_col ])
#                  .groupby("id", as_index=False)
#                  .first()[["id", value_col]]
#         )

# id_to_y0 = dict(zip(df_y0["id"], df_y0[value_col]))
# new_rows = pd.DataFrame({
#     "NUM_ID": list(id_to_y0.keys()),
#     "time": 0,
#     "Y_predicted": list(id_to_y0.values()),
#     "Y_observed": list(id_to_y0.values())
# })

# hlme_predictions = (
#     pd.concat([new_rows, hlme_predictions], ignore_index=True)
#       .sort_values(by=["NUM_ID", "time"], ascending=[True, True])
#       .reset_index(drop=True)
# )
# import textwrap
# train_df["time"] = (
#     (train_df["SUIVI"] - train_df.groupby("NUM_ID")["SUIVI"].transform("min"))
#       .dt.total_seconds() / (60 * 60 * 24 * 365)
# )
# value_col = "ISA15"
# binned_mean_observations = compute_global_bin_means_with_ci(train_df, time_col, value_col)
# binned_mean_predictions = compute_global_bin_means_with_ci(df_long, time_col, value_col)
# value_col = "Y_predicted"
# binned_mean_predictions_hlme = compute_global_bin_means_with_ci(hlme_predictions, time_col, value_col)
# x = (binned_mean_observations["segment_start"] + binned_mean_observations["segment_end"]) / 2
# y = binned_mean_observations["mean"]
# hat_y = binned_mean_predictions["mean"]
# hat_y_hlme = binned_mean_predictions_hlme['mean']
# yerr_lower = binned_mean_observations["mean"] - binned_mean_observations["ci_low"]
# yerr_upper = binned_mean_observations["ci_high"] - binned_mean_observations["mean"]
# yerr = np.vstack([yerr_lower, yerr_upper])

# plt.figure(figsize=(8,4))
# plt.errorbar(x, y, yerr=yerr, fmt='-',elinewidth=2, capthick=2,capsize=5, label="observations")
# plt.scatter(x, hat_y, marker="D", label="CDE conditional predictions")
# plt.scatter(x, hat_y_hlme, marker='o',label="HLME conditional predictions")
# plt.xlabel("Visit times (in years)")
# plt.ylabel("ISA15")
# plt.ylim(30, 36)
# plt.legend(loc="best")
# title = "Mean of the observations (with 95% confidence interval) and of the conditional predictions from CDE model and of the conditional" \
# " predictions from HLME by time intervals defined according to visit times"
# plt.title("\n".join(textwrap.wrap(title, width=50)))
# plt.grid(True, linestyle="--", alpha=0.5)
# plt.tight_layout()
# plt.savefig("figures/mean_trajectory.pdf",format='pdf', bbox_inches='tight')

train_data = process_data(train_df, id_col, features, static_features, target_col, with_only_static_features=with_only_static_features, scaler=None, metabolic_baseline=metabolic_features_baselines)
val_data = process_data(val_df, id_col, features, static_features, target_col, with_only_static_features=with_only_static_features, scaler=None,metabolic_baseline=metabolic_features_baselines)

train_dataset = PatientDataset(train_data)
val_dataset = PatientDataset(val_data)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

model = CDEModel(len(features)* 2 + 1, static_features_dim, LATENT_DIM, device).to(device)
model.load_state_dict(torch.load('EXPs/model_latent_4_CDE_diagoG_full_features.pth', map_location='cpu')['model_state_dict'])

random_effect_std_devs = torch.exp(model.decoder.log_std_devs)
print("Learned Random Effect Standard Deviations:", random_effect_std_devs)

# Inspect the learned residual error
residual_std_dev = torch.sqrt(torch.exp(model.decoder.log_residual_var))
print(f"\nLearned Residual Standard Deviation: {residual_std_dev.item():.4f}")

train_mse, train_pop_mse = calculate_fit_mse_with_blup(model, train_loader, device)
print(f"train fitted MSE: {train_mse:.4f}, {train_pop_mse:.4f}")

validation_mse, val_pop_mse = calculate_fit_mse_with_blup(model, val_loader, device)
print(f"Validation fitted MSE: {validation_mse:.4f}, {val_pop_mse:.4f}")

log_likelihood = lme_log_likelihood(model, val_loader, device)
print("validation dataset likelihood", log_likelihood)

train_log_likelihood = lme_log_likelihood(model, train_loader, device)
print("train dataset likelihood", train_log_likelihood)

validation_mse = calculate_prediction_mse_with_blup(model, val_loader, device)
print(f"Validation pred MSE: {validation_mse:.4f}")

train_mse = calculate_prediction_mse_with_blup(model, train_loader, device)
print(f"train pred MSE: {train_mse:.4f}")


# ##random select participant to plot 
sample_ids = train_df['NUM_ID'].sample(n=25, random_state=42).tolist()
# hlme_predictions = pd.read_csv('results/ISA15_Model_4_train_fitted.csv', sep=',')
# fig, axes = plt.subplots(5, 5, figsize=(22, 12), sharex=True, sharey=True)
# axes = axes.flatten()
# for idx, patient_id in enumerate(sample_ids):

#     sample_patient_data = filter_patient_with_id(patient_id, train_dataset)
#     t_points, trajectory, actual_y = fitted_trajectory(model, sample_patient_data, device)
#     hlme_prediction = hlme_predictions[hlme_predictions.NUM_ID==patient_id]
#     ax = axes[idx]
#     # Plot actual vs predicted
#     ax.plot(t_points, actual_y, 'o', label='Real data', color='royalblue', markersize=6, zorder=5)
#     ax.plot(t_points, trajectory, label='CDE_LMM', color='forestgreen', linewidth=2, linestyle='--')
#     ax.plot(t_points, hlme_prediction['Yfitted'], label='HLME', color='black', linewidth=2, linestyle='--')

#     ax.set_title(f'Patient ID: {patient_id}', fontsize=10)
#     ax.grid(True, linestyle='--', alpha=0.5)
    
# handles, labels = ax.get_legend_handles_labels()
# fig.legend(handles, labels, loc='upper center', ncol=2, fontsize=12)
# plt.tight_layout(rect=[0, 0, 1, 0.95])  # Leave space for the legend
# plt.suptitle("train dataset predictions", fontsize=16)
# plt.savefig("figures/train_blup_predictions_fit_mode.pdf", format='pdf', bbox_inches='tight')
# plt.close() 

# hlme_predictions = pd.read_csv('results/ISA15_Model_4_train_predicted.csv', sep=',')
# fig, axes = plt.subplots(5, 5, figsize=(22, 12), sharex=True, sharey=True)
# axes = axes.flatten()
# for idx, patient_id in enumerate(sample_ids):
#     # Get individual patient data
#     sample_patient_data = filter_patient_with_id(patient_id, train_dataset)
    
#     # Compute predicted trajectory
#     t_points, seq_preds, actual_y, pop_preds = calculate_sequential_blup_forecasting(model, sample_patient_data, device)

#     hlme_prediction = hlme_predictions[hlme_predictions.NUM_ID==patient_id]
#     new_row = {
#         "NUM_ID": patient_id,
#         "time": 0,
#         "Y_predicted": actual_y[0],
#         }
#     hlme_prediction = pd.concat([pd.DataFrame([new_row]), hlme_prediction], ignore_index=True)
    
#     ax = axes[idx]
#     # Plot actual vs predicted
#     ax.plot(t_points, actual_y, 'o', label='Real data', color='royalblue', markersize=6, zorder=5)
#     ax.plot(t_points, seq_preds, label='BLUP', color='forestgreen', linewidth=2, linestyle='--')
#     ax.plot(t_points, hlme_prediction['Y_predicted'], label='HLME', color='black', linewidth=2, linestyle='--')

#     ax.set_title(f'Patient ID: {patient_id}', fontsize=10)
#     ax.grid(True, linestyle='--', alpha=0.5)

# # Shared legend (outside the grid)
# handles, labels = ax.get_legend_handles_labels()
# fig.legend(handles, labels, loc='upper center', ncol=2, fontsize=12)
# plt.tight_layout(rect=[0, 0, 1, 0.95])  # Leave space for the legend
# plt.suptitle("validation dataset predictions", fontsize=16)
# plt.savefig("figures/train_blup_predictions_pred_mode.pdf", format='pdf', bbox_inches='tight')
# plt.close()

# ###random select participant to plot 
# sample_ids = val_df['NUM_ID'].sample(n=25, random_state=42).tolist()
# hlme_predictions = pd.read_csv('results/ISA15_Model_4_val_predicted.csv', sep=',')
# fig, axes = plt.subplots(5, 5, figsize=(22, 12), sharex=True, sharey=True)
# axes = axes.flatten()
# for idx, patient_id in enumerate(sample_ids):
#     # Get individual patient data
#     sample_patient_data = filter_patient_with_id(patient_id, val_dataset)
    
#     # Compute predicted trajectory
#     t_points, seq_preds, actual_y, pop_preds = calculate_sequential_blup_forecasting(model, sample_patient_data, device)

#     hlme_prediction = hlme_predictions[hlme_predictions.NUM_ID==patient_id]
#     new_row = {
#         "NUM_ID": patient_id,
#         "time": 0,
#         "Y_predicted": actual_y[0],
#         }
#     hlme_prediction = pd.concat([pd.DataFrame([new_row]), hlme_prediction], ignore_index=True)
    
#     ax = axes[idx]
#     # Plot actual vs predicted
#     ax.plot(t_points, actual_y, 'o', label='Real data', color='royalblue', markersize=6, zorder=5)
#     ax.plot(t_points, seq_preds, label='BLUP', color='forestgreen', linewidth=2, linestyle='--')
#     ax.plot(t_points, hlme_prediction['Y_predicted'], label='HLME', color='black', linewidth=2, linestyle='--')

#     ax.set_title(f'Patient ID: {patient_id}', fontsize=10)
#     ax.grid(True, linestyle='--', alpha=0.5)

# # Shared legend (outside the grid)
# handles, labels = ax.get_legend_handles_labels()
# fig.legend(handles, labels, loc='upper center', ncol=2, fontsize=12)
# plt.tight_layout(rect=[0, 0, 1, 0.95])  # Leave space for the legend
# plt.suptitle("validation dataset predictions", fontsize=16)
# plt.savefig("figures/blup_predictions_validation.pdf", format='pdf', bbox_inches='tight')
# plt.close()


#### CHECK ODE modes performance ####
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

