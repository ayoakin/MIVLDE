import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
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

def plot_from_summary(experiment_summary,  incl_acc=False, incl_extras=False,descriptor='placeholder', in_notebook=True, fig_dir='plots/'):
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
    
    # Include both loss and accuracy in plots or just include loss
    if incl_acc:
        fig, ax = plt.subplots(2, figsize=(8,8))
        ax[0].errorbar(experiment_summary['layer'], experiment_summary['loss_mean'], yerr=experiment_summary['loss_std'], marker='x')
        ax[0].set(title='Probe test loss over layers', xlabel='Layer index', ylabel='Mean loss on test set')
        ax[1].errorbar(experiment_summary['layer'], experiment_summary['accuracy_mean'], yerr=experiment_summary['accuracy_std'], marker='x')
        ax[1].set(title='Probe test accuracy over layers', xlabel='Layer index', ylabel='Mean accuracy on test set')
        plt.tight_layout()
    elif incl_extras:
        fig, ax = plt.subplots(4, figsize=(8,8))
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