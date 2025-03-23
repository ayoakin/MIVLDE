from odeformer.model import SymbolicTransformerRegressor
import matplotlib.pyplot as plt
import torch
import numpy as np

np.random.seed(2)

times = np.linspace(1, 10, 50)
base_path = f"results/overlap/2dsincos/"

block_idx, head_idx, beam_idx = 6, 4, 0  # decoder layer 6, head 4, beam 0

extra_name = f"_dec{block_idx}_h{head_idx}"
save_attention = f"{base_path}attentions{extra_name}"
savefile = f"{base_path}func_comparison_head{extra_name}"

a = [1, 2, 1, 1]
b = [1, 1, .8, 1]
c = [1, 1, 1, 1]
d = [1, 1, 1, .8]

all_head_attentions = []
all_trajectories = []
all_token_labels = []

for i in range(len(a)):
# for i in range(2):
    print(f"running model: a={a[i]}, b={b[i]}, c={c[i]}, d={d[i]} ")
    # Re-initialize the model. Not doing this every loop causes inconsistent & wrong results.
    dstr = SymbolicTransformerRegressor(from_pretrained=True, plot_token_charts=False)
    model_args = {'beam_size': 2, 'beam_temperature': .1}
    dstr.set_model_args(model_args)

    x1 = a[i] * np.cos(b[i] * times)
    y1 = c[i] * np.sin(d[i] * times)

    trajectory = np.stack([x1, y1], axis=1)
    all_trajectories.append(trajectory.copy())

    # Run model and get attention data
    dstr.fit(times, trajectory)
    all_attentions = dstr.get_stored_attentions()
    intermediate_tokens = list(dstr.get_intermediate_tokens().values())
    num_tokens = len(intermediate_tokens)
    token_ids = range(num_tokens)
    ytick_labels = [intermediate_tokens[i][-1][1][-1] for i in range(num_tokens)]
    all_token_labels.append(ytick_labels.copy())

    # Store attentions
    cross_attns = []

    for token_id in token_ids:
        attn = torch.tensor(all_attentions[f"token_{token_id}"]["cross_attention"])
        cross_attns.append(attn)

    # Stack and process attention data
    stacked = torch.cat(cross_attns, dim=3)  # [layers, beam, heads, tokens, seq]
    head_data = stacked[block_idx, beam_idx, head_idx].numpy()  # [tokens, seq]
    all_head_attentions.append(head_data.copy())

## Create figure:
print(f"starting plotting figure")
fig = plt.figure(figsize=(11, 10))
gs = fig.add_gridspec(2, 2, height_ratios=[1, 1])

# Plots in 2x2 grid
for i in range(len(a)):
    ytick_labels = all_token_labels[i]
    head_attn = all_head_attentions[i]
    trajectory = all_trajectories[i]

    # Plot settings
    xtick_labels = [f"t={t:.1f}" for t in times]
    xticks = np.arange(len(times))
    yticks = np.arange(len(ytick_labels))

    row = (i // 2)
    col = i % 2
    print(f"row={row}, col={col}")
    ax = fig.add_subplot(gs[row, col])

    if head_attn.max() > 0:
        head_attn /= head_attn.max()

    ax.imshow(head_attn, cmap='viridis', aspect='auto')
    ax.set_title(f"a={a[i]}, b={b[i]}, c={c[i]}, d={d[i]}")
    ax.set_xticks(xticks[::4])
    ax.set_xticklabels(xtick_labels[::4], rotation=45, ha='right')
    ax.set_yticks(yticks[::2])
    ax.set_yticklabels(ytick_labels[::2])

    # Plot the original trajectory in the same plot overlapping the imshow
    move_along_y_axis = len(ytick_labels)/4 - 1/2
    move_along_y_axis_2 = len(ytick_labels)*3/4 - 1/2
    # scale x-axis to fit the number of xticks
    scaled_times = (times - times[0]) / (times[-1] - times[0]) * (len(xtick_labels) - 1)
    # scale y-axis to better fit in ytick_labels
    trajectory_factor = len(ytick_labels) / (2 * 4)

    # horizontal lines representing -1, 0, and 1
    ax.axhline(move_along_y_axis_2, color='white', linestyle='--', alpha=.5)
    ax.axhline(move_along_y_axis_2 + trajectory_factor, color='white', linestyle='-.', alpha=.5)
    ax.axhline(move_along_y_axis_2 - trajectory_factor, color='white', linestyle='-.', alpha=.5)
    ax.axhline(move_along_y_axis, color='red', linestyle='--', alpha=.5)
    ax.axhline(move_along_y_axis + trajectory_factor, color='red', linestyle='-.', alpha=.5)
    ax.axhline(move_along_y_axis - trajectory_factor, color='red', linestyle='-.', alpha=.5)

    ax.scatter(scaled_times, trajectory[:, 0] * trajectory_factor + move_along_y_axis_2, color='white', label=f'$x_{0}$', marker='o', edgecolor='0', alpha=.9)
    ax.scatter(scaled_times, trajectory[:, 1] * trajectory_factor + move_along_y_axis, color='red', label=f'$x_{1}$', marker='o',  edgecolor='0', alpha=.9)
    if i == 0:
        plt.legend(loc='upper right', fontsize=15, frameon=True)

plt.tight_layout()
plt.savefig(savefile + '.pdf', bbox_inches='tight', pad_inches=0.01, dpi=300, format='pdf')
plt.savefig(savefile + '.png', bbox_inches='tight', pad_inches=0.01, dpi=300, format='png')
