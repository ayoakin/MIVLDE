# import odeformer
from odeformer.model import SymbolicTransformerRegressor
import matplotlib.pyplot as plt
# import plotly.graph_objects as go
# from plotly.subplots import make_subplots
# import seaborn as sns
import torch, json
import numpy as np
from odeformer.metrics import r2_score
import io
from PyPDF2 import PdfMerger

np.random.seed(2)



dstr = SymbolicTransformerRegressor(from_pretrained=True, plot_token_charts=False)
model_args = {'beam_size':2, 'beam_temperature':.1}
dstr.set_model_args(model_args)

times = np.linspace(1, 10, 50)
# 1D
# base_path = f"results/overwriting_t_vals/"
# c = 2
# # a = 1.2
# # x1 = c * np.exp(-a * times)
#
# t0 = 20
# x1 = c / (t0 - times)
# trajectory = np.stack([x1], axis=1)
# c1_val = 0.1

# 2D sincos
# base_path = f"results/2d_asinx_bsinc/"
# x1 = 2.3 * np.cos(times + 0.5)
# y1 = 1.2 * np.sin(times + 0.1)

# 2D
base_path = f"results/2d_sin_cos/"
x1 = np.cos(times)
y1 = np.sin(times)
# solution: x' = A * y
        #   y' = B * x

trajectory = np.stack([x1, y1], axis=1)


save_attention = f"{base_path}cross_attention_plot_cur"
savefile_trajectory = f"{base_path}trajectory_cur.pdf"

# Overwrite times data to study attention influence
times_overwrite = times.copy()
# Test 1: Overwrite the most attended token
# times_overwrite[29] = 4.0

# Test 2: Overwrite neighboring tokens to have same time stamp
# times_overwrite[26:31] = times_overwrite[29]


# Run model and get attention data
dstr.fit(times_overwrite, trajectory)
all_attentions = dstr.get_stored_attentions()
intermediate_tokens = list(dstr.get_intermediate_tokens().values())
# print(f"intermediate_tokens: {intermediate_tokens}")

# Dynamic token range based on intermediate tokens
num_tokens = len(intermediate_tokens)
token_ids = range(num_tokens)
ytick_labels = [intermediate_tokens[i][-1][1][-1] for i in range(num_tokens)]

# Get cross-attention data for all blocks, beam 0
pdf_merger = PdfMerger()

for block_id_param in range(12):
    block_idx, beam_idx = block_id_param, 0
    cross_attns = []

    for token_id in token_ids:
        attn = torch.tensor(all_attentions[f"token_{token_id}"]["cross_attention"])
        cross_attns.append(attn)

    # Stack and process attention data
    stacked = torch.cat(cross_attns, dim=3)  # [layers, beam, heads, tokens, seq]
    block_data = stacked[block_idx, beam_idx]  # [heads, tokens, seq]

    # Create figure layout
    num_heads = block_data.shape[0]
    fig = plt.figure(figsize=(24, 20))
    gs = fig.add_gridspec(5, 4, height_ratios=[1, 1, 1, 1, 1])

    # Plot settings
    xtick_labels = [f"t={t:.1f}\nx={x:.2f}" for t, x in zip(times_overwrite, x1)]
    xticks = np.arange(len(times_overwrite))
    yticks = np.arange(len(ytick_labels))

    # Plot 1: Sum of all heads
    ax1 = fig.add_subplot(gs[0, :])
    sum_attn = block_data.sum(dim=0).numpy()
    if sum_attn.max() > 0:
        sum_attn /= sum_attn.max()
    # print maximum id of sum_attn[-1]
    most_attended_id = sum_attn[-1].argmax()    # get highest val from the last token
    print(f"most_attended_id: {most_attended_id}")
    print(f"highest value : {sum_attn[-1, most_attended_id]}")

    im = ax1.imshow(sum_attn, cmap='viridis', aspect='auto')
    ax1.set_title(f"Decoder layer {block_id_param} Beam 0 - Sum of All Heads", pad=20)
    ax1.set_xticks(xticks[::2])
    ax1.set_xticklabels(xtick_labels[::2], rotation=45, ha='right')
    ax1.set_yticks(yticks)
    ax1.set_yticklabels(ytick_labels)
    plt.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)

    # Plot 2-17: Individual heads in 4x4 grid
    for head_idx in range(num_heads):
        row = (head_idx // 4) + 1
        col = head_idx % 4
        ax = fig.add_subplot(gs[row, col])

        head_attn = block_data[head_idx].numpy()
        if head_attn.max() > 0:
            head_attn /= head_attn.max()

        ax.imshow(head_attn, cmap='viridis', aspect='auto')
        ax.set_title(f"Head {head_idx}")
        ax.set_xticks(xticks[::4])
        ax.set_xticklabels(xtick_labels[::4], rotation=45, ha='right')
        ax.set_yticks(yticks[::2])
        ax.set_yticklabels(ytick_labels[::2])

    plt.tight_layout()

    # Save the current figure to a PDF buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='pdf', bbox_inches='tight')
    buf.seek(0)

    # Add this page to the PDF merger
    pdf_merger.append(buf)

# Write the merged PDF file
with open(f"{save_attention}_combined.pdf", "wb") as output_file:
    pdf_merger.write(output_file)

pdf_merger.close()

print(f"times: {times}")
print(f"trajectory: {trajectory.T}")
pred_trajectory = dstr.predict(times, trajectory[0])
print("R2 Score:", r2_score(trajectory, pred_trajectory))

dimension = len(trajectory[0])
ax, fig = plt.subplots(figsize=(3,2))
for dim in range(dimension):
    plt.scatter(times_overwrite, trajectory[:, dim], color = f'C{0}', label=f'$x_{dim}$', marker='o', alpha=.3)
    plt.plot(times, pred_trajectory[:, dim],  color = f'C{0}', label=f'$x_{dim}$ predicted')

    # plt.scatter(times, trajectory_2[:, dim], color = f'C{1}', label=f'$x_{dim}$', marker='o', alpha=.3)
    # plt.plot(times, pred_trajectory_2[:, dim],  color = f'C{1}', label=f'$x_{dim}$ predicted')

plt.legend()
plt.savefig(savefile_trajectory, bbox_inches='tight', pad_inches=0.01, dpi=300, format='pdf')



