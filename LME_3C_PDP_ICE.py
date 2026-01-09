# ###PDP/ICE####
from torch.utils.data import DataLoader
from utils import *
from torch.utils.data import Subset

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

## data plot
df["time"] = (
    (df["SUIVI"] - df.groupby("NUM_ID")["SUIVI"].transform("min"))
      .dt.total_seconds() / (60 * 60 * 24 * 365)
)

# Make sure ISA15 is numeric (in case it was read as strings)
df["ISA15"] = pd.to_numeric(df["ISA15"], errors="coerce")

# pick 50 subjects that have at least 1 non-missing ISA15
eligible_ids = (
    df.loc[df["ISA15"].notna(), "NUM_ID"]
      .dropna()
      .unique()
)

np.random.seed(0)
selected_ids = np.random.choice(eligible_ids, size=1000, replace=False)

d50 = (
    df.loc[df["NUM_ID"].isin(selected_ids), ["NUM_ID", "time", "ISA15", 'BMI', 'GLUC', 'HDL', 'PAD']]
      .dropna(subset=["NUM_ID", "time", "ISA15", 'BMI', 'GLUC', 'HDL', 'PAD'])
      .sort_values(["NUM_ID", "time"])
)
plt.figure(figsize=(8, 6))
plt.rcParams["font.size"] = 16

for sid, g in d50.groupby("NUM_ID"):
    plt.plot(g["time"], g["ISA15"], color="black", alpha=0.6)

plt.xlabel("Follow-up time (years since first visit)")
plt.ylabel("ISA15")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("figures/ISA15_data.pdf")
plt.close()

plt.figure(figsize=(8, 6))
for sid, g in d50.groupby("NUM_ID"):
    plt.plot(g["time"], g["BMI"], color="green", alpha=0.6)

plt.xlabel("Follow-up time (years since first visit)")
plt.ylabel("Body Mass Index (KG/CM²)")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("figures/BMI_data.pdf")
plt.close()

plt.figure(figsize=(8, 6))
for sid, g in d50.groupby("NUM_ID"):
    plt.plot(g["time"], g["GLUC"], color="blue", alpha=0.6)

plt.xlabel("Follow-up time (years since first visit)")
plt.ylabel("Glucose (mmol/L)")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("figures/GLUC_data.pdf")
plt.close()

plt.figure(figsize=(8, 6))
for sid, g in d50.groupby("NUM_ID"):
    plt.plot(g["time"], g["HDL"], color="magenta", alpha=0.6)

plt.xlabel("Follow-up time (years since first visit)")
plt.ylabel("HDL cholesterol")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("figures/HDL_data.pdf")
plt.close()

plt.figure(figsize=(8, 6))
for sid, g in d50.groupby("NUM_ID"):
    plt.plot(g["time"], g["PAD"], color="orange", alpha=0.6)

plt.xlabel("Follow-up time (years since first visit)")
plt.ylabel("Diastolic blood pressure")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("figures/PAD_data.pdf")
plt.close()

# # Define bins around your visit schedule
# bins = [-0.5, 1, 3, 5.5, 8.5, 11, 13]  # bins centered on 0,2,4,7,10,12

# # Select low/high BMI groups by baseline BMI
# low_ids  = select_ids_by_bmi(df, 18.5, 22.0, how="baseline", baseline_time=0.0, min_visits=2)
# high_ids = select_ids_by_bmi(df, 28.0, 35.0, how="baseline", baseline_time=0.0, min_visits=2)

# # Plot binned means
# plot_binned_groups(
#     df,
#     groups={"Baseline low BMI": low_ids, "Baseline high BMI": high_ids},
#     bins=bins,
#     min_n=20,
#     title="ISA15 vs time (binned), stratified by baseline BMI",
# )


### PDP ####

# model = CDEModel(len(features)*2+1, len(static_features), LATENT_DIM, device).to(device)
# model.load_state_dict(torch.load('EXPs/model_latent_4_CDE_diagoG.pth', map_location='cpu')['model_state_dict'])

# data = process_data(df, id_col, features, static_features, target_col, 
#                         with_only_static_features=with_only_static_features, scaler=None,metabolic_baseline=metabolic_features_baselines)
# dataset = PatientDataset(data)
# N_subject = len(dataset)

# loader = DataLoader(dataset, batch_size=256, shuffle=False, collate_fn=collate_fn)

# grids = {}
# name = 'BMI'
# grids[name] = get_medical_grid(df, name, num_points=6)
# print(f"{name} Grid: {grids[name]} for {N_subject}")
# profils = create_fictive_profiles(grids[name])
# t_query = torch.tensor([0., 2., 4., 7., 10., 12.], device=device)
# pdp = compute_pdp_delta_ice(
#     model=model,
#     dataloader=loader,
#     features=features,
#     feature_idx=features.index(name),
#     grid_values=profils,
#     time_ch=0,
#     t_query=t_query,
#     use_delta=False,   # delta-ICE PDP (recommended)
# )

