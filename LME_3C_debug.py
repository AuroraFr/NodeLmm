import pandas as pd
from LME_3C_model import process_data, PatientDataset
import warnings
# Add this after your imports to ignore this specific warning
warnings.filterwarnings("ignore", category=UserWarning)
import numpy as np


with_only_static_features = False
scale_dynamic_features = False
metabolic_features_baselines = False

static_features_dim = 6
# Load the model first
LATENT_DIM, LEARNING_RATE, WEIGHT_DECAY, EPOCHS, BATCH_SIZE = 4, 0.0001, 1e-4, 5000, 256
features = ['GLUC','HDL', 'PAD', 'BMI']
# features = ['GLUC', 'HDL', 'LDL', 'CHOL', 'TRYG', 'PAD', 'PAS', 'BMI']
static_features, target_col, id_col = ["DIPNIV_1","DIPNIV_2","DIPNIV_3", "SEX_1", "SEX_2", "AGE0"], "ISA15", "NUM_ID"
static_features, target_col, id_col = ["DIPNIV_1.0","DIPNIV_2.0","DIPNIV_3.0", "SEX_1.0", "SEX_2.0", "AGE0"], "ISA15", "NUM_ID"

train_df = pd.read_csv("3C_dataset/train_3C_data_1.csv", na_values=["NA", ""])
val_df = pd.read_csv("3C_dataset/test_3C_data.csv", na_values=["NA", ""])

print(val_df[val_df.NUM_ID == 23370])

train_df["SUIVI"] = pd.to_datetime(train_df["SUIVI"])
val_df["SUIVI"] = pd.to_datetime(val_df["SUIVI"])

train_data = process_data(train_df, id_col, features, static_features, target_col, with_only_static_features=with_only_static_features, scaler=None, metabolic_baseline=metabolic_features_baselines)
val_data = process_data(val_df, id_col, features, static_features, target_col, with_only_static_features=with_only_static_features, scaler=None,metabolic_baseline=metabolic_features_baselines)

train_dataset = PatientDataset(train_data)
val_dataset = PatientDataset(val_data)

def filter_patient_with_id(id, dataset):
    for patient in dataset:
        if id == np.unique(patient['patient_id']):
            return patient

data = filter_patient_with_id(23370, val_data)
print(data)
