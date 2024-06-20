# -*- coding: utf-8 -*-
"""Experiment utils for CNN."""
import os
import pathlib
from typing import Union

import lightning
import pandas
import torch
import torchvision
from lightning.pytorch.callbacks import ModelSummary
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
from torch.utils.data import DataLoader

from common.log import get_logger
from cnn import CustomCNN
from common.plot import plot_csv_logger_metrics

# to show all columns without cuts
pandas.set_option("display.max_columns", None)

logger = get_logger()


def run_cifar10_experiment(
    cnn: Union[CustomCNN, torch.nn.Module],
    experiment_name: str,
    n_epochs: int = 50,
    batch_size: int = 256,
    transform: torchvision.transforms.Compose = None,
    num_workers: int = None,
    precision: str = None,
) -> None:
    """Run the CNN experiment with the Cifar10 dataset  given a CustomCNN object.

    Args:
        cnn (CustomCNN): CustomCNN object.
        experiment_name (str): Experiment name.
        n_epochs (int): Number of epochs.
        batch_size (int): Batch size.
        transform (torchvision.transforms.Compose): [Optional] Transform to apply to the data.
            If not provided, a default transform is used.
        num_workers (int): Number of workers to use in the DataLoader. If not provided, half of
            the available CPUs are.
        precision (str): Precision to use. Default is "16-mixed".
    """
    artifacts_path = pathlib.Path(__file__).parent.parent.parent / "artifacts"
    data_path = artifacts_path / "data"
    models_path = artifacts_path / "models"
    logs_path = artifacts_path / "logs"
    plots_path = pathlib.Path(__file__).parent.parent.parent / "docs" / "plots"

    if num_workers is None:
        num_workers = int(os.cpu_count() / 2)

    os.makedirs(name=data_path.absolute(), exist_ok=True)
    os.makedirs(name=models_path.absolute(), exist_ok=True)
    os.makedirs(name=logs_path.absolute(), exist_ok=True)

    if transform is None:
        transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.RandomCrop(size=32, padding=4),
                torchvision.transforms.RandomRotation(10),
                torchvision.transforms.RandomHorizontalFlip(p=0.5),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ],
        )

    train_val_cifar_dataset = torchvision.datasets.CIFAR10(
        root=str(data_path), transform=transform, download=True, train=True
    )

    test_cifar_dataset = torchvision.datasets.CIFAR10(
        root=str(data_path), transform=transform, download=True, train=False
    )

    # use 20% of training data for validation
    train_set_size = int(len(train_val_cifar_dataset) * 0.8)
    val_set_size = len(train_val_cifar_dataset) - train_set_size
    train_set, val_set = torch.utils.data.random_split(train_val_cifar_dataset, [train_set_size, val_set_size])

    train_loader = DataLoader(dataset=train_set, shuffle=True, batch_size=batch_size, num_workers=num_workers)
    val_loader = DataLoader(dataset=val_set, batch_size=batch_size, num_workers=num_workers)
    test_loader = DataLoader(dataset=test_cifar_dataset, batch_size=batch_size, num_workers=num_workers)

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
        # he trainer runs 2 batch of training, validation, test and prediction data through to see if there are any bugs
        # fast_dev_run=2,
        log_every_n_steps=1,
    )

    if n_epochs > 0:
        trainer.fit(model=cnn, train_dataloaders=train_loader, val_dataloaders=val_loader)

    trainer.test(model=cnn, dataloaders=test_loader)

    # plot metrics if there was training
    if n_epochs > 0:
        plot_csv_logger_metrics(
            csv_dir=trainer.logger.log_dir, experiment=experiment_name, logger=logger, plots_path=plots_path
        )
