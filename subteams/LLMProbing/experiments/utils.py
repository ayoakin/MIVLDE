import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()

def summarise_experiment(experiment_data):
    experiment_summary = experiment_data.groupby("layer").agg(
        accuracy_mean=("test_accuracy", "mean"),
        accuracy_std=("test_accuracy", "std"),
        loss_mean=("test_loss", "mean"),
        loss_std=("test_loss", "std")
    ).reset_index()

    return experiment_summary

def plot_from_summary(experiment_summary, descriptor, in_notebook=True, fig_dir='plots/'):
    fig, ax = plt.subplots(figsize=(8,4))
    ax.errorbar(experiment_summary['layer'], experiment_summary['accuracy_mean'], yerr=experiment_summary['accuracy_std'], marker='x')
    ax.set(title='Probe test accuracy over layers', xlabel='Layer index', ylabel='Mean accuracy on test set')
    if in_notebook:
        plt.show()
    else:
        fig_path = f'{fig_dir}/acc_plot_{descriptor}.png'
        plt.savefig(fig_path, dpi=300)
    pass