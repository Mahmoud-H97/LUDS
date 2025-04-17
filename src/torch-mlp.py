#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np
from pathlib import Path as P

# Vanilla PyTorch
import torch
from torch.nn import functional as F
from torch import nn
from torch.utils.data import DataLoader, random_split

# Torchvision for CV
from torchvision.datasets import MNIST
from torchvision import transforms

# PyTorch Lightning
import lightning.pytorch as pl
import torchmetrics

# Logging
from lightning.pytorch.loggers import CSVLogger

# remove slow mirror from list of MNIST mirrors
MNIST.mirrors = [
    mirror for mirror in MNIST.mirrors if not mirror.startswith("http://yann.lecun.com")
]



class LitMLP(pl.LightningModule):

    def __init__(self, in_dims, n_classes=10, n_layer_1=128, n_layer_2=256, lr=1e-3):
        super().__init__()

        # we flatten the input Tensors and pass them through an MLP
        self.layer_1 = nn.Linear(np.prod(in_dims), n_layer_1)
        self.layer_2 = nn.Linear(n_layer_1, n_layer_2)
        self.layer_3 = nn.Linear(n_layer_2, n_classes)

        # log hyperparameters
        self.save_hyperparameters()

        # compute the accuracy -- no need to roll your own!
        self.train_acc = torchmetrics.Accuracy(task="multiclass", num_classes=n_classes)
        self.valid_acc = torchmetrics.Accuracy(task="multiclass", num_classes=n_classes)
        self.test_acc = torchmetrics.Accuracy(task="multiclass", num_classes=n_classes)

    def forward(self, x):
        """
        Defines a forward pass using the Stem-Learner-Task
        design pattern from Deep Learning Design Patterns:
        https://www.manning.com/books/deep-learning-design-patterns
        """
        batch_size, *dims = x.size()

        # stem: flatten
        x = x.view(batch_size, -1)


        # task: compute class logits
        x = self.layer_3(x)
        x = F.log_softmax(x, dim=1)

        return x

    # convenient method to get the loss on a batch
    def loss(self, xs, ys):
        logits = self(xs)  # this calls self.forward
        loss = F.nll_loss(logits, ys)
        return logits, loss

    def training_step(self, batch, batch_idx):
        xs, ys = batch
        logits, loss = self.loss(xs, ys)
        preds = torch.argmax(logits, 1)

        # logging metrics we calculated by hand
        self.log("train/loss", loss, on_epoch=True)
        # logging a pl.Metric
        self.train_acc(preds, ys)
        self.log("train/acc", self.train_acc, on_epoch=True)

        return loss

        # task: compute class logits
        x = self.layer_3(x)
        x = F.log_softmax(x, dim=1)

        return x

    # convenient method to get the loss on a batch
    def loss(self, xs, ys):
        logits = self(xs)  # this calls self.forward
        loss = F.nll_loss(logits, ys)
        return logits, loss

    def training_step(self, batch, batch_idx):
        xs, ys = batch
        logits, loss = self.loss(xs, ys)
        preds = torch.argmax(logits, 1)

        # logging metrics we calculated by hand
        self.log("train/loss", loss, on_epoch=True)
        # logging a pl.Metric
        self.train_acc(preds, ys)
        self.log("train/acc", self.train_acc, on_epoch=True)

        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams["lr"])

    def test_step(self, batch, batch_idx):
        xs, ys = batch
        logits, loss = self.loss(xs, ys)
        preds = torch.argmax(logits, 1)

        self.test_acc(preds, ys)
        self.log("test/loss_epoch", loss, on_step=False, on_epoch=True)
        self.log("test/acc_epoch", self.test_acc, on_step=False, on_epoch=True)

    def on_test_epoch_end(self):
        dummy_input = torch.zeros(self.hparams["in_dims"], device=self.device)
        model_filename = "output/model_final.onnx"
        self.to_onnx(model_filename, dummy_input, export_params=True)

    def on_validation_epoch_start(self):
        self.validation_step_outputs = []

    def validation_step(self, batch, batch_idx):
        xs, ys = batch
        logits, loss = self.loss(xs, ys)
        preds = torch.argmax(logits, 1)
        self.valid_acc(preds, ys)

        self.log("valid/loss_epoch", loss)  # default on val/test is on_epoch only
        self.log("valid/acc_epoch", self.valid_acc)

        self.validation_step_outputs.append(logits)

        return logits

    def on_validation_epoch_end(self):
        validation_step_outputs = self.validation_step_outputs
        flattened_logits = torch.flatten(torch.cat(validation_step_outputs))
        # Log metrics to CSV file
        self.logger.log_metrics(
            {
                "valid/logits": flattened_logits.mean().item(),
                "global_step": self.global_step,
            }
        )


    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams["lr"])

    def test_step(self, batch, batch_idx):
        xs, ys = batch
        logits, loss = self.loss(xs, ys)
        preds = torch.argmax(logits, 1)lf.log("test/acc_epoch", self.test_acc, on_step=False, on_epoch=True)

    def on_test_epoch_end(self):
        dummy_input = torch.zeros(self.hparams["in_dims"], device=self.device)
        model_filename = "output/model_final.onnx"
        self.to_onnx(model_filename, dummy_input, export_params=True)

    def on_validation_epoch_start(self):
        self.validation_step_outputs = []

    def validation_step(self, batch, batch_idx):
        xs, ys = batch
        logits, loss = self.loss(xs, ys)
        preds = torch.argmax(logits, 1)
        self.valid_acc(preds, ys)

        self.log("valid/loss_epoch", loss)  # default on val/test is on_epoch only
        self.log("valid/acc_epoch", self.valid_acc)

        self.validation_step_outputs.append(logits)

        return logits

    def on_validation_epoch_end(self):
        validation_step_outputs = self.validation_step_outputs
        flattened_logits = torch.flatten(torch.cat(validation_step_outputs))
        # Log metrics to CSV file
        self.logger.log_metrics(
            {
                "valid/logits": flattened_logits.mean().item(),
                "global_step": self.global_step,
            }
        )

        self.test_acc(preds, ys)
        self.log("test/loss_epoch", loss, on_step=False, on_epoch=True)
        self.log("test/acc_epoch", self.test_acc, on_step=False, on_epoch=True)

    def on_test_epoch_end(self):
        dummy_input = torch.zeros(self.hparams["in_dims"], device=self.device)
        model_filename = "output/model_final.onnx"
        self.to_onnx(model_filename, dummy_input, export_params=True)

    def on_validation_epoch_start(self):
        self.validation_step_outputs = []

    def validation_step(self, batch, batch_idx):
        xs, ys = batch
        logits, loss = self.loss(xs, ys)
        preds = torch.argmax(logits, 1)
        self.valid_acc(preds, ys)

        self.log("valid/loss_epoch", loss)  # default on val/test is on_epoch only
        self.log("valid/acc_epoch", self.valid_acc)

        self.validation_step_outputs.append(logits)

        return logits

    def on_validation_epoch_end(self):
        validation_step_outputs = self.validation_step_outputs
        flattened_logits = torch.flatten(torch.cat(validation_step_outputs))
        # Log metrics to CSV file
        self.logger.log_metrics(
            {
                "valid/logits": flattened_logits.mean().item(),
                "global_step": self.global_step,
            }
        )

   # learner: two fully-connected layers
        x = F.relu(self.layer_1(x))
        x = F.relu(self.layer_2(x))

        # task: compute class logits
        x = self.layer_3(x)
        x = F.log_softmax(x, dim=1)

        return x
lf.log("test/acc_epoch", self.test_acc, on_step=False, on_epoch=True)

    def on_test_epoch_end(self):
        dummy_input = torch.zeros(self.hparams["in_dims"], device=self.device)
        model_filename = "output/model_final.onnx"
        self.to_onnx(model_filename, dummy_input, export_params=True)

    def on_validation_epoch_start(self):
        self.validation_step_outputs = []

    def validation_step(self, batch, batch_idx):
        xs, ys = batch
        logits, loss = self.loss(xs, ys)
        preds = torch.argmax(logits, 1)
        self.valid_acc(preds, ys)

        self.log("valid/loss_epoch", loss)  # default on val/test is on_epoch only
        self.log("valid/acc_epoch", self.valid_acc)

        self.validation_step_outputs.append(logits)

        return logits

    def on_validation_epoch_end(self):
        validation_step_outputs = self.validation_step_outputs
        flattened_logits = torch.flatten(torch.cat(validation_step_outputs))
        # Log metrics to CSV file
        self.logger.log_metrics(
            {
                "valid/logits": flattened_logits.mean().item(),
                "global_step": self.global_step,
            }
        )
    # convenient method to get the loss on a batch
    def loss(self, xs, ys):
        logits = self(xs)  # this calls self.forward
        loss = F.nll_loss(logits, ys)
        return logits, loss

    def training_step(self, batch, batch_idx):
        xs, ys = batch
        logits, loss = self.loss(xs, ys)
        preds = torch.argmax(logits, 1)

        # logging metrics we calculated by hand
        self.log("train/loss", loss, on_epoch=True)
        # logging a pl.Metric
        self.train_acc(preds, ys)
        self.log("train/acc", self.train_acc, on_epoch=True)

        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams["lr"])

    def test_step(self, batch, batch_idx):
        xs, ys = batch
        logits, loss = self.loss(xs, ys)
        preds = torch.argmax(logits, 1)

        self.test_acc(preds, ys)
        self.log("test/loss_epoch", loss, on_step=False, on_epoch=True)
        self.log("test/acc_epoch", self.test_acc, on_step=False, on_epoch=True)

    def on_test_epoch_end(self):
        dummy_input = torch.zeros(self.hparams["in_dims"], device=self.device)
        model_filename = "output/model_final.onnx"
        self.to_onnx(model_filename, dummy_input, export_params=True)

    def on_validation_epoch_start(self):
        self.validation_step_outputs = []

    def validation_step(self, batch, batch_idx):
        xs, ys = batch
        logits, loss = self.loss(xs, ys)
        preds = torch.argmax(logits, 1)
        self.valid_acc(preds, ys)

        self.log("valid/loss_epoch", loss)  # default on val/test is on_epoch only
        self.log("valid/acc_epoch", self.valid_acc)

        self.validation_step_outputs.append(logits)

        return logits

    def on_validation_epoch_end(self):
        validation_step_outputs = self.validation_step_outputs
        flattened_logits = torch.flatten(torch.cat(validation_step_outputs))
        # Log metrics to CSV file
        self.logger.log_metrics(
            {
                "valid/logits": flattened_logits.mean().item(),
                "global_step": self.global_step,
            }
        )
