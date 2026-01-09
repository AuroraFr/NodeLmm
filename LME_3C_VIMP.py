from torch.utils.data import DataLoader
from utils import *
import warnings
warnings.filterwarnings("ignore")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with_only_static_features = False
metabolic_features_baselines = False
static_features_dim = 6
# Load the model first
LATENT_DIM, BATCH_SIZE = 4, 16
features = ['GLUC', 'HDL', 'PAD', 'BMI']
# features = ['GLUC', 'HDL', 'LDL', 'CHOL', 'TRYG', 'PAD', 'PAS', 'BMI']
static_features, target_col, id_col = ["DIPNIV_1.0","DIPNIV_2.0","DIPNIV_3.0", "SEX_1.0", "SEX_2.0", "AGE0"], "ISA15", "NUM_ID"

df = pd.read_csv("3C_dataset/train_3C_data_1.csv", na_values=["NA", ""])
df["SUIVI"] = pd.to_datetime(df["SUIVI"])

model = CDEModel(len(features)*2+1, len(static_features), LATENT_DIM, device).to(device)
model.load_state_dict(torch.load('EXPs/model_latent_4_CDE_diagoG.pth', map_location='cpu')['model_state_dict'])

permuted_df, _, _ = permute_bmi_keep_length_truncate_or_keep(df)
permuted_features = ['GLUC', 'HDL', 'PAD', 'BMI_perm']
data = process_data(permuted_df, id_col, permuted_features, static_features, target_col, 
                        with_only_static_features=with_only_static_features, scaler=None,metabolic_baseline=metabolic_features_baselines)
dataset = PatientDataset(data)
N_subject = len(dataset)
loader = DataLoader(dataset, batch_size=256, shuffle=False, collate_fn=collate_fn)

permuted_log_likelihood = lme_log_likelihood(model, loader, device)
print("BMI permuted train dataset likelihood", permuted_log_likelihood)

permuted_prediction_mse = calculate_prediction_mse_with_blup(model, loader, device)
print(f"BMI permuted train dataset pred MSE: {permuted_prediction_mse:.4f}")

permuted_df, _, _ = permute_bmi_keep_length_truncate_or_keep(df, perm_col='GLUC')
permuted_features = ['GLUC_perm', 'HDL', 'PAD', 'BMI']
data = process_data(permuted_df, id_col, permuted_features, static_features, target_col, 
                        with_only_static_features=with_only_static_features, scaler=None,metabolic_baseline=metabolic_features_baselines)
dataset = PatientDataset(data)
N_subject = len(dataset)
loader = DataLoader(dataset, batch_size=256, shuffle=False, collate_fn=collate_fn)

permuted_log_likelihood = lme_log_likelihood(model, loader, device)
print("GLUC permuted train dataset likelihood", permuted_log_likelihood)

permuted_prediction_mse = calculate_prediction_mse_with_blup(model, loader, device)
print(f"GLUC permuted train dataset pred MSE: {permuted_prediction_mse:.4f}")

permuted_df, _, _ = permute_bmi_keep_length_truncate_or_keep(df, perm_col='HDL')
permuted_features = ['GLUC', 'HDL_perm', 'PAD', 'BMI']
data = process_data(permuted_df, id_col, permuted_features, static_features, target_col, 
                        with_only_static_features=with_only_static_features, scaler=None,metabolic_baseline=metabolic_features_baselines)
dataset = PatientDataset(data)
N_subject = len(dataset)
loader = DataLoader(dataset, batch_size=256, shuffle=False, collate_fn=collate_fn)

permuted_log_likelihood = lme_log_likelihood(model, loader, device)
print("HDL permuted train dataset likelihood", permuted_log_likelihood)

permuted_prediction_mse = calculate_prediction_mse_with_blup(model, loader, device)
print(f"HDL permuted train dataset pred MSE: {permuted_prediction_mse:.4f}")

permuted_df, _, _ = permute_bmi_keep_length_truncate_or_keep(df, perm_col='PAD')
permuted_features = ['GLUC', 'HDL', 'PAD_perm', 'BMI']
data = process_data(permuted_df, id_col, permuted_features, static_features, target_col, 
                        with_only_static_features=with_only_static_features, scaler=None,metabolic_baseline=metabolic_features_baselines)
dataset = PatientDataset(data)
N_subject = len(dataset)
loader = DataLoader(dataset, batch_size=256, shuffle=False, collate_fn=collate_fn)

permuted_log_likelihood = lme_log_likelihood(model, loader, device)
print("PAD permuted train dataset likelihood", permuted_log_likelihood)

permuted_prediction_mse = calculate_prediction_mse_with_blup(model, loader, device)
print(f"PAD permuted train dataset pred MSE: {permuted_prediction_mse:.4f}")

data = process_data(df, id_col, features, static_features, target_col, 
                        with_only_static_features=with_only_static_features, scaler=None,metabolic_baseline=metabolic_features_baselines)
dataset = PatientDataset(data)
N_subject = len(dataset)

loader = DataLoader(dataset, batch_size=256, shuffle=False, collate_fn=collate_fn)

train_log_likelihood = lme_log_likelihood(model, loader, device)
print("nopermuted train dataset likelihood", train_log_likelihood)
prediction_mse = calculate_prediction_mse_with_blup(model, loader, device)
print(f"nopermuted train dataset pred MSE: {prediction_mse:.4f}")


