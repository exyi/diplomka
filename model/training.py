#!/usr/bin/env python3

import math
import time
from typing import Dict, List, Optional, Tuple
import torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import dataclasses
from dataclasses import dataclass, field
# import torchtext
import random
from hparams import Hyperparams

import ntcnetwork
import dataset
from utils import count_parameters, device

import ignite.metrics as metrics
import ignite.engine
from ignite.engine import Events
import ignite.contrib.handlers

def print_model(model: nn.Module):
    for name, module in model.named_modules():
        if name == "":
            name = "total"
        if len(name) > 30:
            name = "…" + name[len(name)-29:]
        parameters = count_parameters(module)
        if parameters > 0:
            print(f"#Parameters {name:30s}: {parameters}")
        
    print(model)

def train(train_set_dir, val_set_dir, p: Hyperparams, logdir):
    train_loader = DataLoader(
        dataset.StructuresDataset(train_set_dir), batch_size=p.batch_size, shuffle=True, collate_fn=dataset.StructuresDataset.collate_fn, num_workers=4)
    batch_count = len(train_loader)
    val_loader = DataLoader(dataset.StructuresDataset(val_set_dir), batch_size=64, shuffle=False, collate_fn=dataset.StructuresDataset.collate_fn, num_workers=4)

    model = ntcnetwork.Network(p).to(device)
    print_model(model)

    optimizer = optim.Adam(model.parameters(), lr=p.learning_rate)
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
        print(f"train - Epoch[{engine.state.epoch}], Iter[{engine.state.iteration:7d}|{engine.state.iteration / batch_count:6.2f}] Loss: {engine.state.output:.2f}")

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

    trainer.run(train_loader, max_epochs=p.epochs)


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

    for k, w in Hyperparams.__dataclass_fields__.items():
        p_config = {}
        if w.metadata.get("list", False):
            p_config["nargs"] = "+"
            p_config["type"] = w.type.__args__[0]
        elif dataclasses.MISSING != w.type:
            p_config["type"] = w.type
        else: 
            p_config["type"] = type(w.default)
        if dataclasses.MISSING != w.default:
            p_config["default"] = w.default
        else:
            p_config["required"] = True
        
        if "help" in w.metadata:
            p_config["help"] = w.metadata["help"]

        if "choices" in w.metadata:
            p_config["choices"] = w.metadata["choices"]
        
        parser.add_argument(f'--{k}', **p_config)

    args = parser.parse_args()

    hyperparameters = Hyperparams(**{ k: v for k, v in vars(args).items() if k in Hyperparams.__dataclass_fields__ })
    
    print(f"device: {device}    logdir: {args.logdir}")
    print(hyperparameters)
    train(args.train_set, args.val_set, hyperparameters, args.logdir)
