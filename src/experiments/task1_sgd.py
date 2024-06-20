# -*- coding: utf-8 -*-
"""Custom CNN Experiment 1. CNN with LeakyReLU activation function, SGD optimizer with lr=0.0001."""


import torch
import torchvision

from cnn import CustomCNN

from common.log import get_logger
from experiment_cifar10 import run_cifar10_experiment


logger = get_logger()


def task1_sgd_experiment(batch_size: int = 256, n_epochs: int = 50) -> None:
    logger.info("\t=> Experiment 1 - CNN with LeakyReLU activation function, SGD optimizer with lr=0.0001.\n")
    cnn1 = CustomCNN(
        net=torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5, stride=1),
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1),
            torch.nn.LeakyReLU(),
            torch.nn.MaxPool2d(2, 2),
            torch.nn.Flatten(),
            torch.nn.Linear(4608, 64),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(in_features=64, out_features=10),
        ),
        optimizer=torch.optim.SGD,
        lr=1e-4,
        cost_function=torch.nn.CrossEntropyLoss,
    )
    run_cifar10_experiment(
        experiment_name="custom_cnn1_lr0001",
        cnn=cnn1,
        batch_size=batch_size,
        n_epochs=n_epochs,
        precision="16-mixed",
        transform=torchvision.transforms.Compose(
            [
                torchvision.transforms.RandomCrop(size=32, padding=4),
                torchvision.transforms.RandomRotation(10),
                torchvision.transforms.RandomHorizontalFlip(p=0.5),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ],
        ),
    )


if __name__ == "__main__":
    task1_sgd_experiment(batch_size=1000, n_epochs=50)
