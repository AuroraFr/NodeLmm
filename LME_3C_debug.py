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
model.load_state_dict(torch.load('EXPs/model_latent_4_CDE_diagoG.pth', map_location='cpu')['model_state_dict'])

random_effect_std_devs = torch.exp(model.decoder.log_std_devs)
print("Learned Random Effect Standard Deviations:", random_effect_std_devs)

# Inspect the learned residual error
residual_std_dev = torch.sqrt(torch.exp(model.decoder.log_residual_var))
print(f"\nLearned Residual Standard Deviation: {residual_std_dev.item():.4f}")

patient = filter_patient_with_id(32074, train_dataset)

log_likelihood = lme_log_likelihood(model, val_loader, device)
print("validation dataset likelihood", log_likelihood)

train_log_likelihood = lme_log_likelihood(model, train_loader, device)
print("train dataset likelihood", train_log_likelihood)

# train_mse, train_pop_mse = calculate_fit_mse_with_blup(model, train_loader, device)
# print(f"train fitted MSE: {train_mse:.4f}, {train_pop_mse:.4f}")

# sample_ids = train_df['NUM_ID'].sample(n=25, random_state=42).tolist()

# hlme_predictions = pd.read_csv('results/ISA15_Model_4_train_fitted.csv', sep=',')
# fig, axes = plt.subplots(5, 5, figsize=(22, 12), sharex=True, sharey=True)
# axes = axes.flatten()
# for idx, patient_id in enumerate(sample_ids):

#     sample_patient_data = filter_patient_with_id(patient_id, train_dataset)
#     # t_points, trajectory, actual_y, _ = calculate_sequential_blup_forecasting(model, sample_patient_data, device)
#     t_points, trajectory, actual_y = fitted_trajectory(model, sample_patient_data, device)
#     hlme_prediction = hlme_predictions[hlme_predictions.NUM_ID==patient_id]
#     # new_row = {
#     #     "NUM_ID": patient_id,
#     #     "time": 0,
#     #     "Y_predicted": actual_y[0],
#     #     }
#     # hlme_prediction = pd.concat([pd.DataFrame([new_row]), hlme_prediction], ignore_index=True)

#     ax = axes[idx]
#     # Plot actual vs predicted
#     ax.plot(t_points, actual_y, 'o', label='Real data', color='royalblue', markersize=6, zorder=5)
#     ax.plot(t_points, trajectory, label='CDE_LMM', color='forestgreen', linewidth=2, linestyle='--')
#     # ax.plot(t_points, hlme_prediction['Y_predicted'], label='HLME', color='black', linewidth=2, linestyle='--')
#     ax.plot(t_points, hlme_prediction['Yfitted'], label='HLME', color='black', linewidth=2, linestyle='--')

#     ax.set_title(f'Patient ID: {patient_id}', fontsize=10)
#     ax.grid(True, linestyle='--', alpha=0.5)
    
# handles, labels = ax.get_legend_handles_labels()
# fig.legend(handles, labels, loc='upper center', ncol=2, fontsize=12)
# plt.tight_layout(rect=[0, 0, 1, 0.95])  # Leave space for the legend
# plt.suptitle("train dataset predictions", fontsize=16)
# plt.savefig("figures/blup_predictions_fit_debug.pdf", format='pdf', bbox_inches='tight')
# plt.close() 

# ###random select participant to plot 
# sample_ids = val_df['NUM_ID'].sample(n=25, random_state=2).tolist()

# hlme_predictions = pd.read_csv('results/ISA15_Model_4_val_predicted.csv', sep=',')
# fig, axes = plt.subplots(5, 5, figsize=(22, 12), sharex=True, sharey=True)
# axes = axes.flatten()
# count = 0
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
    
#     if len(hlme_prediction['Y_predicted']) != len(t_points):
#         print("ERROR", patient_id)
#         print(hlme_prediction["time"], t_points)
#         continue
#     ax = axes[count]
#     # Plot actual vs predicted
#     ax.plot(t_points, actual_y, 'o', label='Real data', color='royalblue', markersize=6, zorder=5)
#     ax.plot(t_points, seq_preds, label='BLUP', color='forestgreen', linewidth=2, linestyle='--')
#     ax.plot(t_points, hlme_prediction['Y_predicted'], label='HLME', color='black', linewidth=2, linestyle='--')

#     ax.set_title(f'Patient ID: {patient_id}', fontsize=10)
#     ax.grid(True, linestyle='--', alpha=0.5)
    
#     if count % 5 == 0:
#         ax.set_ylabel("ISA15", fontsize=10)
#     if count % 10 == 0:
#         ax.set_xlabel("Time (years)", fontsize=10)
#     count += 1

# # Shared legend (outside the grid)
# handles, labels = ax.get_legend_handles_labels()
# fig.legend(handles, labels, loc='upper center', ncol=2, fontsize=12)
# plt.tight_layout(rect=[0, 0, 1, 0.95])  # Leave space for the legend
# plt.suptitle("validation dataset predictions", fontsize=16)
# plt.savefig("figures/blup_predictions_validation.pdf", format='pdf', bbox_inches='tight')
# plt.close()