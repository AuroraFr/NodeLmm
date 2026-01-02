import torch
import torch.optim as optim
import pandas as pd
from torch.utils.data import DataLoader
# You will need to have this installed in your environment
# pip install torchcde
import torchcde
from sklearn.preprocessing import StandardScaler
from LME_3C_model import process_data, PatientDataset, collate_fn, ODEFunc, CDEModel, ODENet, masked_NLL
from torch.optim.lr_scheduler import ReduceLROnPlateau
from LME_3C_evaluation import calculate_fit_mse_with_blup
# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import warnings
# Add this after your imports to ignore this specific warning
warnings.filterwarnings("ignore", category=UserWarning)


# --- MAIN SCRIPT ---
if __name__ == "__main__":

    LATENT_DIM, LEARNING_RATE, WEIGHT_DECAY, EPOCHS, BATCH_SIZE = 4, 0.001, 1e-4, 10000, 128
    features = ['HDL', 'PAD', 'BMI']
    # features = ['GLUC', 'HDL', 'LDL', 'CHOL', 'TRYG', 'PAD', 'PAS', 'BMI']
    static_features, target_col, id_col = ["DIPNIV_1.0","DIPNIV_2.0","DIPNIV_3.0", "SEX_1.0", "SEX_2.0", "AGE0"], "ISA15", "NUM_ID"
    # static_features, target_col, id_col = ["AGE0", "SEX_1.0", "SEX_2.0"], "ISA15", "NUM_ID"
    # static_features, target_col, id_col = ["DIPNIV_1","DIPNIV_2","DIPNIV_3", "SEX_1", "SEX_2", "AGE0"], "ISA15", "NUM_ID"
    with_only_static_features = False
    scale_dynamic_features = False
    metabolic_features_baselines = False

    # full_df = pd.read_csv("data_3C.csv", na_values=["NA", ""])
    train_df = pd.read_csv("3C_dataset/train_3C_data_1.csv", na_values=["NA", ""])
    train_df["SUIVI"] = pd.to_datetime(train_df["SUIVI"])

    val_df = pd.read_csv("3C_dataset/val_3C_data_1.csv", na_values=["NA", ""])
    val_df["SUIVI"] = pd.to_datetime(val_df["SUIVI"])    

    if scale_dynamic_features:

        scaler = StandardScaler()
        scaler.fit(train_df[features].dropna())

        # Scaler for the target variable
        y_scaler = StandardScaler()
        # Fit ONLY on the training data's target column
        y_scaler.fit(train_df[[target_col]])
        train_df[target_col] = y_scaler.transform(train_df[[target_col]])
        val_df[target_col]   = y_scaler.transform(val_df[[target_col]])

    train_data = process_data(train_df, id_col, features, static_features, target_col, with_only_static_features=with_only_static_features, scaler=None, metabolic_baseline=metabolic_features_baselines)
    val_data = process_data(val_df, id_col, features, static_features, target_col, with_only_static_features=with_only_static_features, scaler=None, metabolic_baseline=metabolic_features_baselines)
   
    train_dataset = PatientDataset(train_data)
    val_dataset = PatientDataset(val_data)

    # Use the new collate_fn
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    print(f"Data split: {len(train_dataset)} patients for training, {len(val_dataset)} for validation.")
    
    static_features_dim = 6
    metabolic_baseline_features_dim = 4

    # 1. Create an instance of your original ODEFunc
    if metabolic_features_baselines:
        ode_dynamics = ODEFunc(hidden_dim=LATENT_DIM, static_dim=static_features_dim + len(features))
    else:
        ode_dynamics = ODEFunc(hidden_dim=LATENT_DIM, static_dim=static_features_dim)

    # 2. Apply torch.jit.script to it
    if with_only_static_features:
        print("Scripting the ODE dynamics function...")
        scripted_ode_func = torch.jit.script(ode_dynamics)
        print("Scripting complete.")

    if not with_only_static_features:
        model = CDEModel(len(features) * 2 + 1, static_features_dim, LATENT_DIM, device, fullG=False).to(device)
    elif (not metabolic_features_baselines) and with_only_static_features:
        model = ODENet(scripted_ode_func, static_features_dim, LATENT_DIM, device, fullG=False).to(device)
    else:
        model = ODENet(scripted_ode_func, static_features_dim+len(features), LATENT_DIM, device, fullG=False).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    # It will reduce LR if val_loss doesn't improve for 'patience' epochs.
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=100, verbose=True)

    # --- Initialization for saving best model and early stopping ---
    best_val_loss = float('inf')
    early_stop_patience = 300
    patience_counter = 0
    model_save_path = 'EXPs/LME_exp/latent_4_CDE_diagoG_noGLUC.pth'
    # ----------------------------------------------------------------

    # Check if a checkpoint exists to load from
    import os
    if os.path.exists(model_save_path):
        print('continue training')
        checkpoint = torch.load(model_save_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint['best_val_loss']
        patience_counter = checkpoint['patience_counter']
        print(f"Checkpoint loaded. Resuming training from epoch {start_epoch}")
    # ---------------------------
    
    print(f"Using device: {device}")
    print("Starting training...")
    
    # --- MODIFIED Training Loop ---
    for epoch in range(EPOCHS):
        model.train()
        total_train_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            
            if not with_only_static_features:
                # Move all batch data to the device
                t, y, s_i, mask, id, t_i, x_aug = [d.to(device) for d in batch.values()]
                # coeffs = torchcde.natural_cubic_spline_coeffs(x_aug, t=t)
                coeffs = torchcde.hermite_cubic_coefficients_with_backward_differences(x_aug)
                X = torchcde.CubicSpline(coeffs)
            
                predicted_mean, V = model(s_i, X, len(t))
            
            elif with_only_static_features and metabolic_features_baselines:

                t, y, s_i, mask, subject_id, metabolic_baseline = [d.to(device) for d in batch.values()]
                predicted_mean, V = model(s_i, t, metabolic_baseline=metabolic_baseline)
            
            else:
                t, y, s_i, mask, _ = [d.to(device) for d in batch.values()]
                predicted_mean, V = model(s_i, t)

            # Use the mask in the loss function
            loss = masked_NLL(predicted_mean, y, V, mask)
            
            loss.backward()
            # for n, p in model.encoder.named_parameters():
            #     print(n,
            #         " | w_meanabs:", p.data.abs().mean().item(),
            #         " | grad_meanabs:", (p.grad.abs().mean().item() if p.grad is not None else None),
            #         " | requires_grad:", p.requires_grad)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_train_loss += loss.item()

        if with_only_static_features:
            fitted_train_mse = calculate_fit_mse_with_blup(model, train_loader, device)
        
        avg_train_loss = total_train_loss / len(train_loader)

        # --- MODIFIED Validation Phase ---
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch in val_loader:

                if not with_only_static_features:
                    # Move all batch data to the device
                    t, y, s_i, mask, id, t_i, x_aug = [d.to(device) for d in batch.values()]
                    coeffs = torchcde.hermite_cubic_coefficients_with_backward_differences(x_aug)
                    X = torchcde.CubicSpline(coeffs)
                
                    predicted_mean, V = model(s_i, X, len(t))
                    
                elif with_only_static_features and metabolic_features_baselines:

                    t, y, s_i, mask, patient_id, metabolic_baseline = [d.to(device) for d in batch.values()]
                    predicted_mean, V = model(s_i, t, CDE=False, metabolic_baseline=metabolic_baseline)
            
                else:
                    t, y, s_i, mask, _ = [d.to(device) for d in batch.values()]
                    predicted_mean, V = model(s_i, t, CDE=False)
                
                loss = masked_NLL(predicted_mean, y, V, mask)
                total_val_loss += loss.item()


        avg_val_loss = total_val_loss / len(val_loader) # Divide by number of batches
        scheduler.step(avg_val_loss)

        if with_only_static_features:
            fitted_val_mse = calculate_fit_mse_with_blup(model, val_loader, device)

            print(f"Epoch {epoch+1}/{EPOCHS}: Train Loss = {avg_train_loss:.4f}, Validation Loss = {avg_val_loss:.4f}, train_fitted MSE = {fitted_train_mse:4f}, val_fitted_MSE:{fitted_val_mse:4f}")
        else:
            print(f"Epoch {epoch+1}/{EPOCHS}: Train Loss = {avg_train_loss:.4f}, Validation Loss = {avg_val_loss:.4f}")

        # --- Logic for saving the best model and early stopping ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), model_save_path)
            patience_counter = 0
            print(f"Validation loss improved. Saving model to {model_save_path}")

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
                'patience_counter': patience_counter
            }, model_save_path)
        else:
            patience_counter += 1
            print(f"Validation loss did not improve. Patience: {patience_counter}/{early_stop_patience}")

        if patience_counter >= early_stop_patience:
            print("Early stopping triggered.")
            break
        # -----------------------------------------------------------


    print("Training finished.")
