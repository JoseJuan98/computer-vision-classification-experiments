# -*- coding: utf-8 -*-
"""Custom CNN."""
from typing import Any, Type

import lightning
import torch
import torchmetrics
from lightning.pytorch.utilities.types import STEP_OUTPUT

from common.log import get_logger

logger = get_logger()


class CustomCNN(lightning.LightningModule):
    """Custom CNN using pytorch lightning API to reuse the network and make it reusable for different experiments."""

    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: Type[torch.optim.Optimizer],
        cost_function: Type[torch.nn.Module],
        lr: float = 0.001,
        optimizer_kwargs = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        if optimizer_kwargs is None:
            optimizer_kwargs = {}

        self.net = net
        self.optimizer = optimizer
        self.cost_function = cost_function()
        self.lr = lr
        self.optimizer_kwargs = optimizer_kwargs

        self.save_hyperparameters(ignore=["net"])

        # Check if the optimizer accepts the parameters
        # No need to check for parameters as the base class defines them
        if "lr" not in self.optimizer.__init__.__code__.co_varnames:
            # Warn that this object does not accept the learning rate parameter
            logger.warning(f"Optimizer initialized for '{self.__class__.__name__}' may not accept the learning rate parameter, this can lead to errors later.")

    def forward(self, x):
        out = self.net(x)
        return torch.nn.functional.log_softmax(out, dim=1)

    def training_step(self, batch, *args: Any, **kwargs: Any) -> STEP_OUTPUT:
        """compute and return the training loss and some additional metrics for e.g. the progress bar or
        logger.

        Args:
            batch: The output of your data iterable, normally a :class:`~torch.utils.data.DataLoader`.
            batch_idx: The index of this batch.
            dataloader_idx: The index of the dataloader that produced this batch.
                (only if multiple dataloaders used)

        Return:
            - :class:`~torch.Tensor` - The loss tensor
            - ``dict`` - A dictionary which can include any keys, but must include the key ``'loss'`` in the case of
              automatic optimization.
            - ``None`` - In automatic optimization, this will skip to the next batch (but is not supported for
              multi-GPU, TPU, or DeepSpeed). For manual optimization, this has no special meaning, as returning
              the loss is not required.

        In this step you'd normally do the forward pass and calculate the loss for a batch.
        You can also do fancier things like multiple forward passes or something model specific.
        """
        return self.evaluate(batch, stage="train")

    def evaluate(self, batch, stage: str, progress_bar: bool = True):
        x, y = batch

        outputs = self(x)

        loss = self.cost_function(outputs, y)

        preds = torch.argmax(outputs, dim=1)

        acc = torchmetrics.functional.accuracy(preds=preds, target=y, task="multiclass", num_classes=10)

        metrics = {f"{stage}_loss": loss, f"{stage}_acc": acc, "n_samples": len(y)}
        self.log_dict(dictionary=metrics, prog_bar=progress_bar, logger=True, reduce_fx="mean")

        # it's needed for the train step
        if stage == "train":
            metrics["loss"] = loss

        return metrics

    def validation_step(self, batch, batch_idx):
        return self.evaluate(batch, stage="val")

    def test_step(self, batch, batch_idx):
        return self.evaluate(batch, stage="test")

    def configure_optimizers(self):
        kwargs = {
            "lr": self.lr
        }
        kwargs.update(self.optimizer_kwargs)

        return self.optimizer(self.parameters(), **kwargs)
