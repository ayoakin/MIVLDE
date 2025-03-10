import pandas as pd

def summarise_experiment(experiment_data):
    experiment_summary = experiment_data.groupby("layer").agg(
        accuracy_mean=("test_accuracy", "mean"),
        accuracy_std=("test_accuracy", "std"),
        loss_mean=("test_loss", "mean"),
        loss_std=("test_loss", "std")
    ).reset_index()

    return experiment_summary