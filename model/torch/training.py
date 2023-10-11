#!/usr/bin/env python3

import math, random, time, os, sys

if os.environ.get("NTCNET_INTERNAL_NO_HEAVY_IMPORTS", "") == "1":
    raise Exception("This module is not supposed to be imported before args are parsed")

from typing import Callable, Dict, List, Optional, Tuple
import torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import dataclasses
from dataclasses import dataclass, field
# import torchtext
from model import hyperparameters, epochschedule
from model.hyperparameters import Hyperparams

from . import ntcnetwork, torchutils, dataset_wrapper
from .torchutils import TensorDict, count_parameters, device

import ignite.metrics as metrics
import ignite.engine
import ignite.handlers
import ignite.handlers.param_scheduler
from ignite.engine import Events
import ignite.handlers.early_stopping
import ignite.contrib.handlers.tqdm_logger
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

def lengths_mask(lengths: torch.LongTensor, max_len: int) -> torch.BoolTensor:
    mask = torch.arange(max_len, device=lengths.device, dtype=lengths.dtype).expand(lengths.shape[0], -1) < lengths.unsqueeze(-1)
    return mask # type: ignore

def maskout_for_metrics(values: torch.Tensor, mask: torch.Tensor, broadcast_dims: Optional[list[bool]] = None) -> torch.Tensor:
    if broadcast_dims is None:
        assert mask.shape == values.shape, f"mask{list(mask.shape)} != values{list(values.shape)}"
        values = values.reshape([ -1 ])
        mask = mask.reshape([ -1 ])
    else:
        batch_dimensions = [ i for i, br in enumerate(broadcast_dims) if not br ]
        assert mask.shape == tuple([ values.shape[i] for i in batch_dimensions ]), f"mask{list(mask.shape)} != values{list(values.shape)}"
        broadcast_dimensions = [ i for i, br in enumerate(broadcast_dims) if br ]
        values = values.permute([ *batch_dimensions, *broadcast_dimensions ])
        values = values.reshape([ -1, *values.shape[len(batch_dimensions):] ])
        mask = mask.reshape([ -1 ])
        # new_dims = list(reversed(mask.shape))
        # new_dims = [ new_dims.pop() if br else 1 for br in broadcast_dims ]
        # mask = mask.reshape(new_dims).expand(values.shape) # type: ignore
    assert mask.dtype == torch.bool, f"mask.dtype={mask.dtype}"
    return values[mask]

def _NANT_filter(x: TensorDict, y: TensorDict) -> torch.Tensor:
    return y["NtC"] != 0

def _output_field_transform(field, len_offset: int, filter: Optional[Callable[[TensorDict, TensorDict], torch.Tensor]] = None):
    def transform(output):
        assert isinstance(output, dict), f"Expected dict, got {type(output)}: {output}"
        assert "y" in output and "y_pred" in output, f"Expected dict with keys 'y' and 'y_pred', got {output.keys()}, {output}"
        y, y_pred = output["y"], output["y_pred"]
        lengths = y["lengths"]
        yf = y[field]
        yf_pred = y_pred[field]
        mask = lengths_mask(lengths - len_offset, yf.shape[1]).to(yf_pred.device)
        if filter is not None:
            mask = mask & filter(output.get("x", {}), output.get("y", {})).to(yf_pred.device)
        yf = maskout_for_metrics(yf, mask)
        yf_pred = maskout_for_metrics(yf_pred, mask, broadcast_dims=[False, True, False])

        return (yf_pred, yf)
    
    return transform

global_epoch = -1

def testtheshit(ds: dataset_wrapper.Datasets, optimizer: torch.optim.Optimizer, model: ntcnetwork.Network):
    accuracy = metrics.Accuracy()
    data = ds.get_train_ds(512, 12)
    batch1X, batch1Y = next(iter(data))
    x, y = torchutils.to_device(batch1X), torchutils.to_device(batch1Y)
    for name, param in model.named_parameters():
        if "weight" in name and param.dim() > 1:
            nn.init.xavier_uniform_(param)
    for i in range(1000000):
        optimizer.zero_grad()
        model.train()
        y_pred = model(x)
        loss = model.loss(y_pred, y)
        loss.backward()

        max_grads = [
            (n, p.grad.abs().mean().item())
            for n, p in model.named_parameters()
            if p.grad is not None
        ]
        no_grad = [ name + ": " + str(p.shape) for name, p in model.named_parameters() if p.grad is None ]
        torch.nn.utils.clip_grad.clip_grad_value_(model.parameters(), 1.0)

        accuracy.update((y_pred["NtC"], y["NtC"]))
        
        print(f"i={i}, loss = {loss.item()}, lr = {optimizer.param_groups[0]['lr']}, acc = {accuracy.compute()}")
        if i % 100 == 0:
            print(max_grads)
        # print("no grad: ", *no_grad)

        optimizer.step()

    exit(1)



def train(train_set_dir, val_set_dir, p: Hyperparams, logdir):
    ds = dataset_wrapper.make_test_val(train_set_dir, val_set_dir, p)
    seq_len_schedule = epochschedule.parse_epoch_schedule(p.seq_length_schedule, p.epochs, tt=int)
    batch_size_schedule = epochschedule.get_batch_size_from_maxlen(seq_len_schedule, p.batch_size, p.max_batch_size)
    batch_count = epochschedule.get_step_count(batch_size_schedule, ds.train_size)

    model = ntcnetwork.Network(p).to(device)
    print_model(model)

    optimizer = optim.Adam(model.parameters(), lr=p.learning_rate)
    optimizer.step()

    testtheshit(ds, optimizer, model)
    trainer = ignite.engine.create_supervised_trainer(
        model, optimizer, model.loss, device,
        output_transform=lambda x, y, y_pred, loss: { "loss": loss.item(), "y": y, "y_pred": y_pred, "x": x }
    )

    print("expected batch count", batch_count)
    if p.lr_decay == "cosine":
        torch_lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=batch_count * p.epochs)
        # scheduler = ignite.handlers.param_scheduler.LRScheduler(torch_lr_scheduler)
        # trainer.add_event_handler(Events.ITERATION_STARTED, scheduler)
        @trainer.on(Events.ITERATION_COMPLETED)
        def lr_decay():
            torch_lr_scheduler.step()
    elif p.lr_decay != "none" and p.lr_decay is not None:
        raise ValueError(f"Unknown lr_decay method {p.lr_decay}")

    output_ntc = _output_field_transform("NtC", len_offset=1, filter=_NANT_filter)
    recall_metric = metrics.Recall(output_transform=output_ntc, average=False)
    precision_metric = metrics.Precision(output_transform=output_ntc, average=False)
    val_metrics: Dict[str, metrics.Metric] = {
        "accuracy": metrics.Accuracy(output_transform=output_ntc),
        "f1": metrics.Fbeta(1, output_transform=output_ntc),
        "f1_unfiltered": metrics.Fbeta(1, output_transform=_output_field_transform("NtC", len_offset=1)),
        # "miou": metrics.mIoU(metrics.ConfusionMatrix(len(ntcnetwork.Network.NTC_LABELS), output_transform=output_ntc)),
        "loss": metrics.Loss(model.loss)
    }
    train_evaluator = ignite.engine.create_supervised_evaluator(model, metrics=val_metrics, device=device, output_transform=lambda x, y, y_pred: { "y": y, "y_pred": y_pred, "x": x, "criterion_kwargs":{} })
    val_evaluator = ignite.engine.create_supervised_evaluator(model, metrics=val_metrics, device=device, output_transform=lambda x, y, y_pred: { "y": y, "y_pred": y_pred, "x": x, "criterion_kwargs":{} })

    metrics.Accuracy(output_transform=output_ntc).attach(trainer, "accuracy")
    metrics.Fbeta(beta=1.0, output_transform=output_ntc).attach(trainer, "f1")
    # metrics.Loss(model.loss).attach(trainer, "loss")

    tqdm_progbar = ignite.contrib.handlers.tqdm_logger.ProgressBar()
    tqdm_progbar.attach(trainer)

    logger_clock = Clock()
    epoch_clock = Clock()
    # @trainer.on(Events.ITERATION_COMPLETED(every=1))
    # def log_training_loss(engine):
    #     print(f"train - Epoch[{engine.state.epoch}], Iter[{engine.state.iteration:7d}|{engine.state.iteration / batch_count:6.2f}] Loss: {engine.state.output['loss']:.2f} Time: {logger_clock.measure():3.1f}s      ")

    @trainer.on(Events.EPOCH_STARTED)
    def reset_timer(trainer):
        res_step = logger_clock.measure()
        res_epoch = epoch_clock.measure()
        print(f"Epoch[{global_epoch}] Residual time: {res_epoch/60:3.1f}m residual step time: {res_step/60:3.1f}m    ")

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training_results(trainer):
        epoch_time = epoch_clock.measure()
        eval_clock = Clock()
        # train_evaluator.run(itertools.islice(train_loader, 6)) # limit to few batches, otherwise it takes longer than training (??)
        eval_time = eval_clock.measure()
        metrics = trainer.state.metrics
        print(f"evalT - Epoch[{global_epoch}] accuracy: {metrics['accuracy']:.2f} F1: {metrics['f1']:.2f} training loss: {trainer.state.output['loss']:.2f} Train Time: {epoch_time/60:3.1f}m Eval Time: {eval_time/60:3.2f}m    ")


    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(trainer):
        eval_clock = Clock()
        val_evaluator.run(ds.get_validation_ds())
        eval_time = eval_clock.measure()
        metrics = val_evaluator.state.metrics
        print(f"evalV - Epoch[{global_epoch}] accuracy: {metrics['accuracy']:.2f} F1: {metrics['f1']:.2f} loss: {metrics['loss']:.2f} Eval Time: {eval_time/60:3.2f}m")

    early_stopper = ignite.handlers.early_stopping.EarlyStopping(patience=8, score_function=lambda engine: -engine.state.metrics["loss"], trainer=trainer)
    val_evaluator.add_event_handler(Events.COMPLETED, early_stopper)

    tb_log = setup_tensorboard_logger(trainer, optimizer, train_evaluator, val_evaluator, logdir)
    if tb_log is not None:
        tblog_architecture(tb_log, model, ds, p)

    param_schedule = list(zip(epochschedule.schedule_to_list(seq_len_schedule), epochschedule.schedule_to_list(batch_size_schedule)))
    # def replace_dataloader():
    #     current_epoch = trainer.state.epoch
    #     seq_len, batch_size = param_schedule[min(current_epoch, len(param_schedule)-1)]
    #     print(f"Setting new data loader with seq_len: {seq_len} batch_size: {batch_size}")
    #     trainer.set_data(ds.get_train_ds(seq_len, batch_size))
    # trainer.add_event_handler(Events.EPOCH_STARTED, lambda _: replace_dataloader())
    # replace_dataloader()
    for epoch, (seq_len, batch_size) in enumerate(param_schedule):
        global global_epoch
        global_epoch = epoch
        result = trainer.run(max_epochs=1, data=ds.get_train_ds(seq_len, batch_size))
        if trainer.should_terminate:
            print("Should terminate -> exiting")
            break
        # print("training result", result)

def tblog_architecture(tb_log: tensorboard_logger.TensorboardLogger, model: nn.Module, ds: dataset_wrapper.Datasets, p: Hyperparams):
    # from torch.utils.tensorboard._pytorch_graph import graph
    # tb_log.writer._get_file_writer().add_graph(graph(model, next(iter(ds.get_validation_ds())), use_strict_trace=False))

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

if __name__ == '__main__':
    print("Use the launch.py script using `torch-train` command.")
    exit(1)
