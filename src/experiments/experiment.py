# -*- coding: utf-8 -*-
"""Experiment utils for CNN."""
import os
import pathlib

from typing import Type, Union

import lightning
import pandas
import torch
import torchvision
from lightning.pytorch.callbacks import ModelSummary
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
from torch.utils.data import DataLoader

from cnn import CustomCNN
from common.log import get_logger
from common.plot import plot_csv_logger_metrics

# to show all columns without cuts
pandas.set_option("display.max_columns", None)

logger = get_logger()


def run_experiment(
    cnn: Union[CustomCNN, torch.nn.Module],
    experiment_name: str,
    train_val_dataset: Union[Type[torchvision.datasets.VisionDataset], torchvision.datasets.VisionDataset],
    test_dataset: Union[Type[torchvision.datasets.VisionDataset], torchvision.datasets.VisionDataset],
    validation_size: float = 0.2,
    n_epochs: int = 50,
    batch_size: int = 256,
    num_workers: int = None,
    precision: str = None,
    fast_dev_run: Union[bool, int] = None,
) -> Union[CustomCNN, torch.nn.Module]:
    """Run the CNN experiment with the Cifar10 dataset  given a CustomCNN object.

    Args:
        cnn (CustomCNN): CustomCNN object.
        experiment_name (str): Experiment name.
        n_epochs (int): Number of epochs.
        batch_size (int): Batch size.
        num_workers (int): Number of workers to use in the DataLoader. If not provided, half of
            the available CPUs are.
        precision (str): Precision to use. Default is "16-mixed".
        train_val_dataset (torch.utils.data.Dataset): Train and validation dataset.
        test_dataset (torch.utils.data.Dataset): Test dataset.
        validation_size (float): Validation size taken from the train dataset. Default is 0.2.
        fast_dev_run (Union[bool, int]): If True, runs 1 batch of train, test and val to find any bugs. Also, it can be
            specified the number of batches to run as an integer
    """
    artifacts_path = pathlib.Path(__file__).parent.parent.parent / "artifacts"
    models_path = artifacts_path / "models"
    logs_path = artifacts_path / "logs"
    plots_path = pathlib.Path(__file__).parent.parent.parent / "docs" / "plots"

    if num_workers is None:
        num_workers = int(os.cpu_count() / 2)

    os.makedirs(name=models_path.absolute(), exist_ok=True)
    os.makedirs(name=logs_path.absolute(), exist_ok=True)

    # use 20% of training data for validation
    train_set_size = int(len(train_val_dataset) * (1 - validation_size))
    val_set_size = len(train_val_dataset) - train_set_size
    train_set, val_set = torch.utils.data.random_split(train_val_dataset, [train_set_size, val_set_size])

    train_loader = DataLoader(dataset=train_set, shuffle=True, batch_size=batch_size, num_workers=num_workers)
    val_loader = DataLoader(dataset=val_set, batch_size=batch_size, num_workers=num_workers)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, num_workers=num_workers)

    random_image, labels = next(iter(train_loader))

    logger.info(f"\nRandom image shape: {random_image[0].shape}")
    logger.info(f"Number of samples in the training set: {len(train_set)}\n\n")

    trainer = lightning.Trainer(
        default_root_dir=artifacts_path,
        max_epochs=n_epochs,
        devices=1,
        accelerator="auto",
        precision=precision,
        logger=[
            CSVLogger(save_dir=logs_path / "csv", name=experiment_name),
            TensorBoardLogger(save_dir=logs_path / "board", name=experiment_name),
        ],
        callbacks=[ModelSummary(max_depth=-1)],
        # measures all the key methods across Callbacks, DataModules and the LightningModule in the training loop.
        # profiler="simple",
        fast_dev_run=fast_dev_run,
        log_every_n_steps=1,
    )

    if n_epochs > 0:
        trainer.fit(model=cnn, train_dataloaders=train_loader, val_dataloaders=val_loader)

    trainer.test(model=cnn, dataloaders=test_loader)

    # plot metrics if there was training
    if n_epochs > 0 and fast_dev_run is None:
        plot_csv_logger_metrics(
            csv_dir=trainer.logger.log_dir, experiment=experiment_name, logger=logger, plots_path=plots_path
        )

    return cnn
