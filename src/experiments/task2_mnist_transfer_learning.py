# -*- coding: utf-8 -*-
"""MNIST Experiment 2. Transfer Learning."""
import os
import pathlib

import torch
import torchvision

from cnn import CustomCNN

from common.log import get_logger
from experiment import run_experiment

logger = get_logger()


def task2_mnist_cnn_transfer_learning(batch_size: int = 4000, n_epochs: int = 10) -> None:

    data_path = pathlib.Path(__file__).parent.parent.parent / "artifacts" / "data"
    os.makedirs(name=data_path.absolute(), exist_ok=True)

    logger.info("\t=> 0.2.2 Transfer Learning from MNIST Experiment 1 - Train CNN.\n")

    model = CustomCNN(
        net=torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1),
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1),
            torch.nn.LeakyReLU(),
            torch.nn.MaxPool2d(2, 2),
            torch.nn.Flatten(),
            torch.nn.Linear(3200, 64),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(in_features=64, out_features=10),
        ),
        optimizer=torch.optim.Adam,
        lr=1e-3,
        cost_function=torch.nn.CrossEntropyLoss,
        optimizer_kwargs=dict(
            weight_decay=0.01,  # L2 regularization
        ),
    )

    transform_mnist = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize((28, 28)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.5], std=[0.5]),
        ]
    )

    model = run_experiment(
        experiment_name="mnist_cnn_train",
        cnn=model,
        batch_size=batch_size,
        n_epochs=n_epochs,
        num_workers=4,
        train_val_dataset=torchvision.datasets.MNIST(
            root=str(data_path), transform=transform_mnist, download=True, train=True
        ),
        test_dataset=torchvision.datasets.MNIST(
            root=str(data_path), transform=transform_mnist, download=True, train=False
        ),
    )

    transform_svhn = torchvision.transforms.Compose(
        [
            # MNIST is in grey scale, so we need to convert the images of SVHN to grey scale
            torchvision.transforms.Grayscale(num_output_channels=1),
            # same image size as MNIST
            torchvision.transforms.Resize((28, 28)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.5], std=[0.5]),
        ]
    )

    logger.info("\t=> 0.2.2 Transfer Learning from MNIST Experiment 2 - Transfer Learning from MNIST CNN to SVHN.\n")
    run_experiment(
        experiment_name="svhn_transfer_learning",
        cnn=model,
        batch_size=batch_size,
        # just for testing
        n_epochs=0,
        num_workers=4,
        train_val_dataset=torchvision.datasets.SVHN(
            root=str(data_path), split="train", transform=transform_svhn, download=True
        ),
        test_dataset=torchvision.datasets.SVHN(
            root=str(data_path), split="test", transform=transform_svhn, download=True
        ),
    )

    logger.info("\t=> 0.2.2 Transfer Learning from MNIST Experiment 3 - Fine tuning MNIST CNN to SVHN.\n")
    _ = run_experiment(
        experiment_name="svhn_fine_tuning",
        cnn=model,
        batch_size=batch_size,
        n_epochs=n_epochs,
        num_workers=4,
        train_val_dataset=torchvision.datasets.SVHN(
            root=str(data_path), split="train", transform=transform_svhn, download=True
        ),
        test_dataset=torchvision.datasets.SVHN(
            root=str(data_path), split="test", transform=transform_svhn, download=True
        ),
    )


if __name__ == "__main__":
    task2_mnist_cnn_transfer_learning(batch_size=1000, n_epochs=10)
