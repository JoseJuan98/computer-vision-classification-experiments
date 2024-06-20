# -*- coding: utf-8 -*-
"""Task 2 - Transfer Learning.

Script to execute all the experiments together.
"""

from common.log import msg_task

from task2_alexnet_fine_tuning import task2_alexnet_fine_tuning
from task2_alexnet_feature_extraction import task2_alexnet_feature_extraction
from task2_mnist_transfer_learning import task2_mnist_cnn_transfer_learning


def task2() -> None:
    """Task 2 - Transfer Learning."""

    msg_task(msg="Task 2 - Transfer Learning")

    # ~3.5 Gib of GPU memory
    task2_alexnet_fine_tuning(batch_size=500, n_epochs=10)

    # check if the images fit in your GPU memory or RAM, if not reduce the batch size. The current one peaks at
    #  ~30 Gib of RAM
    task2_alexnet_feature_extraction(batch_size=1500)

    # ~2.9 Gib of GPU memory
    task2_mnist_cnn_transfer_learning(batch_size=10000, n_epochs=10)


if __name__ == "__main__":
    task2()
