import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src.probes.utils import verbose_eval_regression_probe, load_probe_from_path
from scipy.stats import spearmanr, pearsonr
from src.datasets.activations_dataset import ActivationsDataset
from src.datasets.utils import split_dataset, get_d_in
from torch.utils.data import DataLoader
sns.set_theme()


def summarise_experiment(experiment_data, incl_acc=False, incl_extras=False):
    """
    Return a summary dataframe of experiment_data generated via the experiment runners.
    Depending on what type of experiment is run, the accuracy and extra measures (r2, spearman, pearson) can be chosen to be included.

    Args:
    - experiment_data (pd.DataFrame): A DataFrame containing experimental results across different layers and runs
    - incl_acc (bool, optional): Whether to include the accuracy in the summary DataFrame. Defaul is False.
    - incl_extras (bool, optional): Whether to include the r2, spearman and pearson coefficients in the summary Dataframe. Default is False
    """
    if incl_acc:
        experiment_summary = experiment_data.groupby("layer").agg(
            accuracy_mean=("test_accuracy", "mean"),
            accuracy_std=("test_accuracy", "std"),
            loss_mean=("test_loss", "mean"),
            loss_std=("test_loss", "std")
        ).reset_index()
    elif incl_extras:
        experiment_summary = experiment_data.groupby("layer").agg(
            r2_mean=("test_r2", "mean"),
            r2_std=("test_r2", "std"),
            spearman_mean=("test_spearman", "mean"),
            spearman_std=("test_spearman", "std"),
            pearson_mean=("test_pearson", "mean"),
            pearson_std=("test_pearson", "std"),
            loss_mean=("test_loss", "mean"),
            loss_std=("test_loss", "std")
        ).reset_index()
    else:
        experiment_summary = experiment_data.groupby("layer").agg(
            loss_mean=("test_loss", "mean"),
            loss_std=("test_loss", "std")
        ).reset_index()

    return experiment_summary

def plot_from_summary(experiment_summary,  incl_acc=False, incl_extras=False,descriptor='placeholder', in_notebook=True, fig_dir='plots/', layers = None):
    """
    Plots the mean and standard deviation of test loss (and optionally accuracy or extras (r2, spearman, pearson)) across transformer layers
    from an experiment summary.

    Args:
    - experiment_summary (pd.DataFrame): A DataFrame containing layer-wise summary statistics with the following columns:
        - "layer" (int): Layer index.
        - "loss_mean" (float): Mean test loss across multiple runs.
        - "loss_std" (float): Standard deviation of test loss.
        - "accuracy_mean" (float, optional): Mean test accuracy across multiple runs (required if `incl_acc=True`).
        - "accuracy_std" (float, optional): Standard deviation of test accuracy (required if `incl_acc=True`).
    - incl_acc (bool, optional): If True, also plots test accuracy in addition to loss. Default is False.
    - incl_extras (bool, optional): Whether to plot the test r2, spearman and pearson coefficients in addition to loss. Default is False
    - descriptor (str, optional): A label used for saving the plot file. Default is 'placeholder'.
    - in_notebook (bool, optional): If True, displays the plot in a Jupyter Notebook; otherwise, saves it to a file. Default is True.
    - fig_dir (str, optional): Directory path to save plots if `in_notebook=False`. Default is 'plots/'.

    Returns:
    - None: The function either displays the plot or saves it to `fig_dir/expt_plot_{descriptor}.png`.

    Notes:
    - If `incl_acc=True`, two subplots are generated: one for loss and one for accuracy.
    - If `incl_extras=True`, four subplots are generated: one each for loss, r2, spearman, pearson
    - If `in_notebook=True`, the plot is displayed inline using `plt.show()`.
    - If `in_notebook=False`, the plot is saved as a high-resolution PNG file.
    - Ensure that `experiment_summary` contains the required statistics before calling this function.
    """
    if layers:
      experiment_summary = experiment_summary[experiment_summary['layer'].isin(layers)]
    # Include both loss and accuracy in plots or just include loss
    if incl_acc:
        fig, ax = plt.subplots(2, figsize=(8,8))
        ax[0].errorbar(experiment_summary['layer'], experiment_summary['loss_mean'], yerr=experiment_summary['loss_std'], marker='x')
        ax[0].set(title='Probe test loss over layers', xlabel='Layer index', ylabel='Mean loss on test set')
        ax[1].errorbar(experiment_summary['layer'], experiment_summary['accuracy_mean'], yerr=experiment_summary['accuracy_std'], marker='x')
        ax[1].set(title='Probe test accuracy over layers', xlabel='Layer index', ylabel='Mean accuracy on test set')
        plt.tight_layout()
    elif incl_extras:
        fig, ax = plt.subplots(4, figsize=(6,12)) # TODO: determine best figsize
        ax[0].errorbar(experiment_summary['layer'], experiment_summary['loss_mean'], yerr=experiment_summary['loss_std'], marker='x')
        ax[0].set(title='Probe test loss over layers', xlabel='Layer index', ylabel='Mean loss on test set')
        ax[1].errorbar(experiment_summary['layer'], experiment_summary['r2_mean'], yerr=experiment_summary['r2_std'], marker='x')
        ax[1].set(title='Probe test r2 over layers', xlabel='Layer index', ylabel='Mean r2 on test set')
        ax[2].errorbar(experiment_summary['layer'], experiment_summary['spearman_mean'], yerr=experiment_summary['spearman_std'], marker='x')
        ax[2].set(title='Probe test spearman over layers', xlabel='Layer index', ylabel='Mean spearman on test set')
        ax[3].errorbar(experiment_summary['layer'], experiment_summary['pearson_mean'], yerr=experiment_summary['pearson_std'], marker='x')
        ax[3].set(title='Probe test pearson over layers', xlabel='Layer index', ylabel='Mean pearson on test set')
        plt.tight_layout()
    else:
        fig, ax = plt.subplots(figsize=(8,4))
        ax.errorbar(experiment_summary['layer'], experiment_summary['loss_mean'], yerr=experiment_summary['loss_std'], marker='x')
        ax.set(title='Probe test loss over layers', xlabel='Layer index', ylabel='Mean loss on test set')
    
    # Display plot in notebook or save to path
    if in_notebook:
        plt.show()
    else:
        fig_path = f'{fig_dir}/expt_plot_{descriptor}.png'
        plt.savefig(fig_path, dpi=300)
    return

def plot_scalar_predictions(probe, test_dataloader, ax, color, feature_descriptor, descriptor='placeholder', in_notebook=True, fig_dir='plots/'):
    '''
    Plots a probe's predictions agains ground truth, and computes the spearman coefficient
    '''
    test_results, avg_test_loss = verbose_eval_regression_probe(probe, test_dataloader)
    probe_outputs = np.array([test_results[i][0] for i in range(len(test_results))], dtype=float)
    ground_truths = np.array([test_results[i][1] for i in range(len(test_results))], dtype=float)

    ax.scatter(ground_truths, probe_outputs, alpha=0.3, color=color)
    gt_minimal_set = [min(ground_truths), max(ground_truths)]
    ax.plot(gt_minimal_set, gt_minimal_set, color='tab:red')
    ax.set(xlabel='Ground Truth', ylabel='Probe Prediction', title=f'{feature_descriptor} prediction')

    spearman_corr, _ = spearmanr(ground_truths, probe_outputs)
    pearson_corr, _ = pearsonr(ground_truths, probe_outputs)
    
    # plt.annotate(f"Spearman r={spearman_corr:.3f}",
    #             xy=(0.95, 0.05),  # bottom-right in axes coords
    #             xycoords='axes fraction',
    #             ha='right', fontsize=10)

    # plt.tight_layout()

    # if in_notebook:
    #     plt.show()
    # else:
    #     fig_path = f'{fig_dir}/regression_expt_plot_{descriptor}.png'
    #     plt.savefig(fig_path, dpi=300)
    return spearman_corr, pearson_corr[0]


def plot_all_scalar_predictions(target_feature, feature_descriptor, descriptor, activations_path, probes_path, layers, num_repeats=1, r2_threshold=None, shuffle_datasets=True, data_split=[0.8, 0.1, 0.1]):

  '''
  Plot probe predictions against ground truth - creates one plot for each layer, and plots each repeat in a different colour. 
  '''  
  # Iterate over the specified layers
  for layer_idx in layers:
    full_dataset = ActivationsDataset(activations_path=activations_path, feature_label=target_feature, layer_idx=layer_idx, r2_threshold=r2_threshold)

    train_dataset, val_dataset, test_dataset = split_dataset(full_dataset, lengths=data_split)
    test_dataloader = DataLoader(test_dataset, shuffle=shuffle_datasets)
    # Load the specified number of repeats
    fig, ax = plt.subplots(figsize=(4,4))
    for run in range(num_repeats):
      print(f'Repeat {run} of layer {layer_idx}')
      color = plt.rcParams['axes.prop_cycle'].by_key()['color'][run % 10]

      d_in = get_d_in(full_dataset)

      # Load probe based on expected naming format
      probe_name = f'probe_{target_feature}_{layer_idx}_{run}.pt'
      probe = load_probe_from_path(f'{probes_path}/{probe_name}', d_in=d_in)

      spearman_corr, pearson_corr = plot_scalar_predictions(probe, test_dataloader, ax, color, feature_descriptor, descriptor)

      y_base = 0.05 + run * 0.12  # Slightly increase spacing for both annotations

      ax.annotate(f"Spearman r={spearman_corr:.3f}", xy=(0.95, y_base),
                  xycoords='axes fraction', ha='right', fontsize=10,
                  color=color)

      ax.annotate(f"Pearson r={pearson_corr:.3f}", xy=(0.95, y_base + 0.05),
                  xycoords='axes fraction', ha='right', fontsize=10,
                  color=color)

    plt.tight_layout()
    plt.show()
  return