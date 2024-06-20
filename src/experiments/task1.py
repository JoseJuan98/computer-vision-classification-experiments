# -*- coding: utf-8 -*-
"""Task 1 - Custom CNN experiments execution.

Script to execute all the experiments together.
"""

from common.log import msg_task

from task1_sgd import task1_sgd_experiment
from task1_adam import task1_adam_experiment
from task1_tanh import task1_tanh_experiment


def task1() -> None:
    """Task 1 - Custom CNN.

    Main method to train all the CNN models using the CIFAR10 dataset.
    """
    # check if the images fit in your GPU memory, if not reduce the batch size. The current one is about ~3 Gib
    batch_size = 10000
    n_epochs = 50

    msg_task(msg="Task 1 - Custom CNN")

    task1_sgd_experiment(batch_size=batch_size, n_epochs=n_epochs)

    task1_adam_experiment(batch_size=batch_size, n_epochs=n_epochs)

    task1_tanh_experiment(batch_size=batch_size, n_epochs=n_epochs)


if __name__ == "__main__":
    task1()
