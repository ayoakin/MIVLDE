from odeformer.model import SymbolicTransformerRegressor
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import torch
import numpy as np

np.random.seed(2)

times = np.linspace(1, 10, 50)
val_range = .2          # range of y values to modify [-A, A]
number_of_samples = 31  # Number of points to evaluate in. (uneven to capture 0)

base_path = f"results/MAT_change_y/"    # Make sure path exists!
savefile = f"{base_path}averaged_MAT_{val_range}_{number_of_samples}.pdf"
create_data = True     # create new data or load existing data

# 1D system
c = 2
a = 1.2
x1 = c * np.exp(-a * times)
trajectory_base = np.stack([x1], axis=1)

# create linearly spaced values to modify
z_modify_range = [-val_range, val_range]
z_arr = np.linspace(z_modify_range[0], z_modify_range[1], number_of_samples)
# MAT id MAT=Most-Attended Token
mat_id = 29

if create_data:
    # Averaged attention array
    averaged_attention_mat = np.zeros((number_of_samples, 12))

    for i, z_val in enumerate(z_arr):
        # Re-initialize the model. Not doing this every loop causes inconsistent & wrong results.
        dstr = SymbolicTransformerRegressor(from_pretrained=True, plot_token_charts=False)
        model_args = {'beam_size': 2, 'beam_temperature': .1}
        dstr.set_model_args(model_args)

        print(f"Starting iteration {i} with z_val: {z_val}")
        # Overwrite trajectory data, and subtract and add the Most Attended Token from [-z, z]
        new_trajectory = trajectory_base.copy()
        new_trajectory[mat_id] -= z_val

        # Run model and get attention data
        dstr.fit(times, new_trajectory)
        all_attentions = dstr.get_stored_attentions()
        intermediate_tokens = list(dstr.get_intermediate_tokens().values())

        # Process attention data
        num_tokens = len(intermediate_tokens)
        token_ids = range(num_tokens)
        # Process each decoder block
        for block_id_param in range(12):
            block_idx, beam_idx = block_id_param, 0
            cross_attns = []

            for token_id in token_ids:
                attn = torch.tensor(all_attentions[f"token_{token_id}"]["cross_attention"])
                cross_attns.append(attn)

            # Stack and process attention data
            stacked = torch.cat(cross_attns, dim=3)  # [layers, beam, heads, tokens, seq]
            block_data = stacked[block_idx, beam_idx]  # [heads, tokens, seq]

            # Sum of all heads
            sum_attn = block_data.sum(dim=0).numpy()
            if sum_attn.max() > 0:
                sum_attn /= sum_attn.max()

            # Average attention over all tokens for MAT for this decoder layer
            mat_id_attn = sum_attn[:,mat_id].mean()
            # Store for plotting
            averaged_attention_mat[i, block_id_param] = mat_id_attn

    # Store averaged attention data
    np.savez(f"{base_path}averaged_attention_data_{val_range}.npz", averaged_attention_mat=averaged_attention_mat)
else:
    # Load the saved data
    data = np.load(f"{base_path}averaged_attention_data_{val_range}.npz")
    # Access the arrays by name
    averaged_attention_mat = data['averaged_attention_mat']


# Plot, with x-axis [-A, A], y-axis the cross attention
colors = cm.viridis(np.linspace(0, 1, 12))  # Define some colormap
fig, ax = plt.subplots(figsize=(7, 3))
for i in range(12):
    ax.plot(z_arr, averaged_attention_mat[:, i], color=colors[i], label=f"Layer {i+1}") # , marker='o', alpha=.5
    ax.text(z_arr[-1] + val_range * 0.02, averaged_attention_mat[-1, i], str(i+1), color=colors[i], fontsize=7)

# Add color bar to show the layer progression
sm = plt.cm.ScalarMappable(cmap=cm.viridis, norm=plt.Normalize(1, 12))
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax, label='Layer')
cbar.set_ticks([1, 6, 12])
cbar.set_ticklabels(['1', '6', '12'])

# Set axis labels
ax.set_xlabel(f'Change in MAT ({mat_id}) y value')
ax.set_ylabel(f'Averaged Cross Attention MAT ({mat_id})')

# plt.legend()
plt.savefig(savefile, bbox_inches='tight', pad_inches=0.01, dpi=300, format='pdf')
# plt.savefig(savefile[:-4] + '.png', bbox_inches='tight', pad_inches=0.01, dpi=300, format='png')

