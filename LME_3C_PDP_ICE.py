# ###PDP/ICE####

# grids = {}
# name = 'BMI'
# grids[name] = get_medical_grid(df, name, num_points=6)
# print(f"{name} Grid: {grids[name]} for {N_subject}")
# profils = create_fictive_profiles(grids[name])
# compute_pdp(model, loader, features.index(name), profils)

# representative_indices = get_representative_indices_latent(model, loader, num_subjects=20, device='cuda')
# # Create a subset of your original dataset
# ice_dataset = Subset(dataset, representative_indices)
# # Create a generic dataloader for these specific people (batch_size=1 is easiest for ICE)
# ice_loader = DataLoader(ice_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)
# # Run calculation
# results = compute_cde_ice_representatives(model, ice_loader, feature_idx=3, test_values=profils)

# # Plot
# plot_representative_ice(results, name)