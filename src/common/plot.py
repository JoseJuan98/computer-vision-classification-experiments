# -*- coding: utf-8 -*-
"""Plot utils.

This module contains the plot utilities to be used in the project.
"""
import logging
import os
import pathlib

import pandas
import seaborn
from matplotlib import pyplot


def plot_csv_logger_metrics(
    plots_path: pathlib.Path, csv_dir: str, experiment: str, logger: logging.Logger = None
) -> None:
    """Plot the metrics."""
    metrics = pandas.read_csv(filepath_or_buffer=os.path.join(csv_dir, "metrics.csv"))

    metrics.drop(columns=["step", "n_samples"], axis=1, inplace=True)
    metrics.set_index("epoch", inplace=True)

    test_loss = metrics["test_loss"].dropna(how="all").mean()
    test_acc = metrics["test_acc"].dropna(how="all").mean()

    if logger is None:
        print(f"\nExperiment {experiment}\n\tTest loss: {test_loss}.\n\tTest accuracy: {test_acc}.\n\n")
    else:
        logger.info(f"\nExperiment {experiment}\n\tTest loss: {test_loss}.\n\tTest accuracy: {test_acc}.\n\n")

    os.makedirs(name=plots_path, exist_ok=True)

    metrics.drop(columns=["test_loss", "test_acc"], axis=1, inplace=True)
    seaborn.relplot(data=metrics, kind="line")
    pyplot.savefig(fname=plots_path / f"metrics_{experiment}.png")
    pyplot.show()
