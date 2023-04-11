#!/usr/bin/env python3

import math
import time
from typing import Dict, List, Optional, Tuple
import torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim
from torch.utils.data import Dataset, DataLoader
# import torchtext
import random

import ntcnetwork
import dataset
from utils import device

import ignite.metrics as metrics
import ignite.engine
from ignite.engine import Events
import ignite.contrib.handlers


def train(train_set_dir, val_set_dir, epochs, batch_size, learning_rate, logdir):

    train_loader = DataLoader(dataset.StructuresDataset(train_set_dir), batch_size=4, shuffle=True, collate_fn=dataset.StructuresDataset.collate_fn)
    val_loader = DataLoader(dataset.StructuresDataset(val_set_dir), batch_size=4, shuffle=False, collate_fn=dataset.StructuresDataset.collate_fn)

    model = ntcnetwork.Network(
        embedding_size=32,
        hidden_size=32
    )

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    trainer = ignite.engine.create_supervised_trainer(model, optimizer, model.loss, device)

    recall_metric = metrics.Recall(output_transform=lambda x: (x[0]["NtC"], x[1]["NtC"]), average=False)
    precision_metric = metrics.Precision(output_transform=lambda x: (x[0]["NtC"], x[1]["NtC"]), average=False)
    val_metrics: Dict[str, metrics.Metric] = {
        "accuracy": metrics.Accuracy(output_transform=lambda x: (x[0]["NtC"], x[1]["NtC"])),
        "f1": (precision_metric * recall_metric * 2 / (precision_metric + recall_metric)).nanmean(),
        "miou": metrics.mIoU(metrics.ConfusionMatrix(len(ntcnetwork.Network.NTC_LABELS), output_transform=lambda x: (x[0]["NtC"], x[1]["NtC"]))),
        "loss": metrics.Loss(model.loss)
    }
    train_evaluator = ignite.engine.create_supervised_evaluator(model, metrics=val_metrics, device=device)
    val_evaluator = ignite.engine.create_supervised_evaluator(model, metrics=val_metrics, device=device)

    @trainer.on(Events.ITERATION_COMPLETED(every=100))
    def log_training_loss(engine):
        print(f"train - Epoch[{engine.state.epoch}], Iter[{engine.state.iteration}] Loss: {engine.state.output:.2f}")

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training_results(trainer):
        train_evaluator.run(train_loader)
        metrics = train_evaluator.state.metrics
        print(f"evalT - Epoch[{trainer.state.epoch}] accuracy: {metrics['accuracy']:.2f} F1: {metrics['f1']:.2f} loss: {metrics['loss']:.2f}")


    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(trainer):
        val_evaluator.run(val_loader)
        metrics = val_evaluator.state.metrics
        print(f"evalV - Epoch[{trainer.state.epoch}] accuracy: {metrics['accuracy']:.2f} F1: {metrics['f1']:.2f} loss: {metrics['loss']:.2f}")

    setup_tensorboard_logger(trainer, train_evaluator, val_evaluator, logdir)

    trainer.run(train_loader, max_epochs=epochs)


class Clock:
    def __init__(self):
        self.start_time = time.time()
        self.last_time = self.start_time

    def measure(self):
        now = time.time()
        elapsed = now - self.last_time
        self.last_time = now
        return elapsed


def setup_tensorboard_logger(trainer, train_evaluator, val_evaluator, logdir):
    # Define a Tensorboard logger
    tb_logger = ignite.contrib.handlers.TensorboardLogger(log_dir=logdir)
    
    clock_step = Clock()
    clock_epoch = Clock()

    # Attach handler to plot trainer's loss every 100 iterations
    tb_logger.attach_output_handler(
        trainer,
        event_name=Events.ITERATION_COMPLETED(every=100),
        tag="training",
        output_transform=lambda loss: {"batch_loss": loss, "step_time": clock_step.measure()},
    )

    # Attach handler for plotting both evaluators' metrics after every epoch completes
    for tag, evaluator in [("training", train_evaluator), ("validation", val_evaluator)]:
        tb_logger.attach_output_handler(
            evaluator,
            event_name=Events.EPOCH_COMPLETED,
            tag=tag,
            metric_names="all",
            global_step_transform=ignite.contrib.handlers.global_step_from_engine(trainer),
        )

    tb_logger.attach_output_handler(
        trainer,
        event_name=Events.EPOCH_COMPLETED,
        tag="system",
        output_transform=lambda loss: {
            #"learning_rate": trainer.state.optimizer.param_groups[0]["lr"]
            "epoch_time": clock_epoch.measure()
        },
    )

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Train NtC network using PyTorch ignite')
    parser.add_argument('--train_set', type=str, help='Path to directory training data CSVs')
    parser.add_argument('--val_set', type=str, help='Path to directory validation data CSVs')
    parser.add_argument('--logdir', type=str, default="tb-logs", help='Path for saving Tensorboard logs and other outputs')
    parser.add_argument('--epochs', type=int, default=500, help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=0.004, help='Learning rate')

    args = parser.parse_args()

    train(args.train_set, args.val_set, args.epochs, args.batch_size, args.learning_rate, args.logdir)
