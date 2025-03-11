import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def plot_activation_along_trajectory(trajectory, activations, feature_idx, time_points=None, 
                                     equation=None, figsize=(12, 8), cmap='viridis', 
                                     save_path=None, linewidth=2.5):
    fig, axes = plt.subplots(2, 1, figsize=figsize, gridspec_kw={'height_ratios': [3, 1]})
    
    feature_activations = activations[:, feature_idx]
    
    if time_points is not None and len(time_points) == len(trajectory):
        x_values = np.array(time_points)
    else:
        x_values = np.arange(len(trajectory))

    cmap_obj = plt.get_cmap(cmap)
    
    ax1 = axes[0]
    
    for j in range(trajectory.shape[1]):
        var_points = np.array([x_values, trajectory[:, j]]).T.reshape(-1, 1, 2)
        var_segments = np.concatenate([var_points[:-1], var_points[1:]], axis=1)
        
        lc = plt.matplotlib.collections.LineCollection(
            var_segments, cmap=cmap_obj, linewidth=linewidth
        )
        lc.set_array(feature_activations[:-1])
        line = ax1.add_collection(lc)

        ax1.plot(x_values, trajectory[:, j], alpha=0.2, label=f'Var {j+1}')
    
    # Set plot limits
    ax1.set_xlim(x_values.min(), x_values.max())
    y_min = np.min(trajectory)
    y_max = np.max(trajectory)
    padding = (y_max - y_min) * 0.05
    ax1.set_ylim(y_min - padding, y_max + padding)
    
    ax1.legend()
    
    # Add colorbar
    cbar = fig.colorbar(line, ax=ax1)
    cbar.set_label(f'Feature {feature_idx} Activation')
    
    # Set labels
    ax1.set_title(f'Trajectory with Feature {feature_idx} Activation')
    
    if equation is not None:
        ax1.text(0.5, 0.01, f"Equation: {equation}", transform=ax1.transAxes, 
                horizontalalignment='center', fontsize=9)
    
    ax1.set_xlabel('Time Points')
    ax1.set_ylabel('Value')
    ax1.grid(True, alpha=0.3)
    
    ax2 = axes[1]
    ax2.plot(x_values, feature_activations, 'o-', markersize=3, color='blue', linewidth=1.5)
    ax2.axhline(y=np.mean(feature_activations), color='r', linestyle='--', 
                label=f'Mean: {np.mean(feature_activations):.4f}')
    
    ax2.set_title(f'Feature {feature_idx} Activation Sequence')
    ax2.set_xlabel('Sequence Position')
    ax2.set_ylabel('Activation')
    ax2.set_xlim(x_values.min(), x_values.max())
    
    min_val = np.min(feature_activations)
    max_val = np.max(feature_activations)
    range_val = max_val - min_val
    if range_val < 1e-5:
        ax2.set_ylim(min_val - 0.01, max_val + 0.01)
    else:
        padding = range_val * 0.1
        ax2.set_ylim(min_val - padding, max_val + padding)
    
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig, axes