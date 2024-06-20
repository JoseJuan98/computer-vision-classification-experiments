# -*- coding: utf-8 -*-
"""AlexNet Experiment 1. Fine Tuning."""

import torch
import torchvision

from cnn import CustomCNN

from common.log import get_logger
from experiment_cifar10 import run_cifar10_experiment

logger = get_logger()


def task2_alexnet_fine_tuning(batch_size: int = 256, n_epochs: int = 50) -> None:
    logger.info("\t=> 0.2.1 Transfer Learning from ImageNet Experiment 1 - Fine tuning.\n")

    # Load AlexNet with pretrained weights for feature extraction
    alexnet_weights = torchvision.models.AlexNet_Weights.DEFAULT
    alexnet = torchvision.models.alexnet(weights=alexnet_weights)

    # modifying last fully connected layer to output 10 classes
    in_features_last_layer = alexnet.classifier[-1].in_features
    alexnet.classifier[-1] = torch.nn.Linear(in_features=in_features_last_layer, out_features=10)

    model = CustomCNN(
        net=alexnet,
        optimizer=torch.optim.Adam,
        lr=1e-3,
        cost_function=torch.nn.CrossEntropyLoss,
        optimizer_kwargs=dict(
            weight_decay=0.01,  # L2 regularization
        ),
    )

    run_cifar10_experiment(
        experiment_name="alexnet_fine_tuning",
        cnn=model,
        batch_size=batch_size,
        n_epochs=n_epochs,
        transform=torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize(256),
                torchvision.transforms.CenterCrop(224),  # Resize images to 224x224, as required by AlexNet
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        ),
        num_workers=4,
    )


if __name__ == "__main__":
    task2_alexnet_fine_tuning(batch_size=1000, n_epochs=10)
