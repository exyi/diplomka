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
from model import hyperparameters, epochschedule
from model.hyperparameters import Hyperparams

from . import ntcnetwork, torchutils, dataset_wrapper
from .torchutils import count_parameters, device

import ignite.metrics as metrics
import ignite.engine
import ignite.handlers
import ignite.handlers.param_scheduler
from ignite.engine import Events
import ignite.handlers.early_stopping
import ignite.contrib.handlers.tensorboard_logger as tensorboard_logger
import itertools

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

def _output_field_transform(field):
    def transform(x):
        return (x[0][field], x[1][field])
    
    return transform

def train(train_set_dir, val_set_dir, p: Hyperparams, logdir):
    ds = dataset_wrapper.make_test_val(train_set_dir, val_set_dir, p)
    seq_len_schedule = epochschedule.parse_epoch_schedule(p.seq_length_schedule, p.epochs, tt=int)
    batch_size_schedule = epochschedule.get_batch_size_from_maxlen(seq_len_schedule, p.batch_size, p.max_batch_size)
    batch_count = epochschedule.get_step_count(batch_size_schedule, ds.train_size)

    model = ntcnetwork.Network(p).to(device)
    print_model(model)

    optimizer = optim.Adam(model.parameters(), lr=p.learning_rate)
    trainer = ignite.engine.create_supervised_trainer(model, optimizer, model.loss, device)

    if p.lr_decay == "cosine":
        torch_lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=batch_count * p.epochs)
        # scheduler = ignite.handlers.param_scheduler.LRScheduler(torch_lr_scheduler)
        # trainer.add_event_handler(Events.ITERATION_STARTED, scheduler)
        @trainer.on(Events.ITERATION_COMPLETED)
        def lr_decay():
            torch_lr_scheduler.step()
    elif p.lr_decay != "none" and p.lr_decay is not None:
        raise ValueError(f"Unknown lr_decay method {p.lr_decay}")

    output_ntc = _output_field_transform("NtC")
    recall_metric = metrics.Recall(output_transform=output_ntc, average=False)
    precision_metric = metrics.Precision(output_transform=output_ntc, average=False)
    val_metrics: Dict[str, metrics.Metric] = {
        "accuracy": metrics.Accuracy(output_transform=output_ntc),
        "f1": metrics.Fbeta(1, output_transform=output_ntc),
        # "miou": metrics.mIoU(metrics.ConfusionMatrix(len(ntcnetwork.Network.NTC_LABELS), output_transform=output_ntc)),
        "loss": metrics.Loss(model.loss)
    }
    train_evaluator = ignite.engine.create_supervised_evaluator(model, metrics=val_metrics, device=device)
    val_evaluator = ignite.engine.create_supervised_evaluator(model, metrics=val_metrics, device=device)

    metrics.Accuracy(output_transform=output_ntc).attach(trainer, "accuracy")
    metrics.Fbeta(beta=1.0, output_transform=output_ntc).attach(trainer, "f1")

    logger_clock = Clock()
    epoch_clock = Clock()
    @trainer.on(Events.ITERATION_COMPLETED(every=10))
    def log_training_loss(engine):
        print(f"train - Epoch[{engine.state.epoch}], Iter[{engine.state.iteration:7d}|{engine.state.iteration / batch_count:6.2f}] Loss: {engine.state.output:.2f} Time: {logger_clock.measure():3.1f}s      ")

    @trainer.on(Events.EPOCH_STARTED)
    def reset_timer(trainer):
        res_step = logger_clock.measure()
        res_epoch = epoch_clock.measure()
        print(f"Epoch[{trainer.state.epoch}] Residual time: {res_epoch/60:3.1f}m residual step time: {res_step/60:3.1f}m    ")

    # @trainer.on(Events.EPOCH_COMPLETED)
    # def log_training_results(trainer):
    #     epoch_time = epoch_clock.measure()
    #     eval_clock = Clock()
    #     train_evaluator.run(itertools.islice(train_loader, 6)) # limit to few batches, otherwise it takes longer than training (??)
    #     eval_time = eval_clock.measure()
    #     metrics = train_evaluator.state.metrics
    #     print(f"evalT - Epoch[{trainer.state.epoch}] accuracy: {metrics['accuracy']:.2f} F1: {metrics['f1']:.2f} loss: {metrics['loss']:.2f} Train Time: {epoch_time/60:3.1f}m Eval Time: {eval_time/60:3.2f}m    ")


    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(trainer):
        eval_clock = Clock()
        val_evaluator.run(ds.get_validation_ds())
        eval_time = eval_clock.measure()
        metrics = val_evaluator.state.metrics
        print(f"evalV - Epoch[{trainer.state.epoch}] accuracy: {metrics['accuracy']:.2f} F1: {metrics['f1']:.2f} loss: {metrics['loss']:.2f} Eval Time: {eval_time/60:3.2f}m")

    early_stopper = ignite.handlers.early_stopping.EarlyStopping(patience=8, score_function=lambda engine: -engine.state.metrics["loss"], trainer=trainer)
    val_evaluator.add_event_handler(Events.COMPLETED, early_stopper)

    tb_log = setup_tensorboard_logger(trainer, optimizer, train_evaluator, val_evaluator, logdir)
    if tb_log is not None:
        tblog_architecture(tb_log, model, ds, p)

    for seq_len, batch_size in zip(epochschedule.schedule_to_list(seq_len_schedule), epochschedule.schedule_to_list(batch_size_schedule)):
        print(f"Epoch[{trainer.state.epoch}] seq_len: {seq_len} batch_size: {batch_size}")
        trainer.run(data=ds.get_train_ds(seq_len, batch_size))

def tblog_architecture(tb_log: tensorboard_logger.TensorboardLogger, model: nn.Module, ds: dataset_wrapper.Datasets, p: Hyperparams):
    from torch.utils.tensorboard._pytorch_graph import graph
    tb_log.writer._get_file_writer().add_graph(graph(model, next(iter(ds.get_validation_ds())), use_strict_trace=False))

    hparams = dict(vars(p))
    for k, v in hparams.items():
        if not (isinstance(v, str) or isinstance(v, int) or isinstance(v, float) or isinstance(v, bool)):
            hparams[k] = str(v)

    import tensorboardX.summary
    hmetrics = {
        "model/total_params": 0,
        "training/loss": 0,
        "training/step_time": 0,
        "validation/miou": 0,
        "validation/f1": 0
    }
    a, b, c = tensorboardX.summary.hparams(hparams, hmetrics)
    tb_log.writer._get_file_writer().add_summary(a)
    tb_log.writer._get_file_writer().add_summary(b)
    tb_log.writer._get_file_writer().add_summary(c)
    tb_log.writer.add_scalar("model/total_params", count_parameters(model))
    tb_log.writer.add_text("model/structure", str(model))

class Clock:
    def __init__(self):
        self.start_time = time.time()
        self.last_time = self.start_time

    def measure(self):
        now = time.time()
        elapsed = now - self.last_time
        self.last_time = now
        return elapsed


def setup_tensorboard_logger(trainer: ignite.engine.Engine, optimizer: optim.Optimizer, train_evaluator, val_evaluator, logdir) -> tensorboard_logger.TensorboardLogger:
    # Define a Tensorboard logger
    tb_logger = tensorboard_logger.TensorboardLogger(log_dir=logdir)

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
            global_step_transform=tensorboard_logger.global_step_from_engine(trainer),
        )

    tb_logger.attach_output_handler(
        trainer,
        event_name=Events.EPOCH_COMPLETED,
        tag="training",
        output_transform=lambda loss: {
            "learning_rate": optimizer.param_groups[0]["lr"],
            "epoch_time": clock_epoch.measure()
        },
    )
    return tb_logger

def init_argparser(parser):
    parser.add_argument('--train_set', type=str, help='Path to tfrecord file with training data. The *.tfrecord.meta.json must be in the same directory')
    parser.add_argument('--val_set', type=str, help='Path to tfrecord file with validation data. The *.tfrecord.meta.json must be in the same directory')
    parser.add_argument('--logdir', type=str, default="tb-logs", help='Path for saving Tensorboard logs and other outputs')
    hyperparameters.add_parser_args(parser, Hyperparams)

def main(argv: list[str]):
    import argparse
    parser = argparse.ArgumentParser(description='Train NtC network using PyTorch ignite')
    init_argparser(parser)
    args = parser.parse_args(argv)

    p = Hyperparams.from_args(args)

    print(f"device: {device}    logdir: {args.logdir}")
    print(p)
    train(args.train_set, args.val_set, p, args.logdir)

if __name__ == '__main__':
    print("Use the launch.py script using `torch-train` command.")
    exit(1)
