import pandas as pd

def summarise_experiment(experiment_data):
    experiment_summary = experiment_data.groupby("layer").agg(
        accuracy_mean=("accuracy", "mean"),
        accuracy_std=("accuracy", "std"),
        loss_mean=("loss", "mean"),
        loss_std=("loss", "std")
    ).reset_index()

    return experiment_summary