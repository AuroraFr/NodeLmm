import matplotlib.pyplot as plt
import torch
from LME_3C_model import process_data, PatientDataset, collate_fn, CDEModel, ODENet, VectorField_with_static
import pandas as pd
from torch.utils.data import DataLoader
from LME_3C_evaluation import *

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


