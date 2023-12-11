#!/usr/bin/env python3

import math, random, time, os, sys
import numbers

import numpy as np

if os.environ.get("NTCNET_INTERNAL_NO_HEAVY_IMPORTS", "") == "1":
    raise Exception("This module is not supposed to be imported before args are parsed")

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
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
import ignite.utils
import ignite.handlers
import ignite.handlers.param_scheduler
from ignite.engine import Events
import ignite.handlers.early_stopping
import ignite.contrib.handlers.tqdm_logger
import ignite.contrib.handlers.tensorboard_logger as tensorboard_logger
import itertools
import matplotlib.pyplot as plt
import matplotlib.colors

import traceback
import warnings
import sys

def warn_with_traceback(message: warnings.WarningMessage, category, filename, lineno, file=None, line=None):
    log = file if hasattr(file,'write') else sys.stderr
    if "can not log metrics value type" in str(message):
        traceback.print_stack(file=log)
    log.write(warnings.formatwarning(message, category, filename, lineno, line))

warnings.showwarning = warn_with_traceback

def print_model(model: nn.Module):
    for name, module in model.named_modules():
        if name == "":
            name = "total"
        if len(name) > 30:
            name = "…" + name[len(name)-29:]
        parameters = count_parameters(module)
        if parameters > 0:
            print(f"#Parameters {name:30s}: {parameters:,}")
        
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
        yf = (yf_unfiltered := y[field])
        yf_pred = (yf_pred_unfiltered := y_pred[field])
        mask = lengths_mask(lengths - len_offset, yf.shape[1]).to(yf_pred.device)
        if filter is not None:
            mask = mask & filter(output.get("x", {}), output.get("y", {})).to(yf_pred.device)
        yf = maskout_for_metrics(yf, mask)
        yf_pred = maskout_for_metrics(yf_pred, mask, broadcast_dims=[False, False, True])

        # print(yf_pred.shape, yf.shape, yf_unfiltered.shape, output["x"]["lengths"])

        if yf.shape[0] == 0:
            # this crashes some metrics, keep first element filtered-out element instead of returning an empty tensor
            return (yf_pred_unfiltered.view(-1, yf_pred.shape[1])[:2, :], yf_unfiltered.view(-1)[:2]) 

        return (yf_pred, yf)
    
    return transform

global_epoch = -1

def testtheshit(ds: dataset_wrapper.Datasets, optimizer: torch.optim.Optimizer, model: ntcnetwork.Network):
    accuracy = metrics.Accuracy()
    data = ds.get_train_ds(512, 12)
    dataiter = iter(data)
    batch1X, batch1Y = next(dataiter)
    batch2X, batch2Y = next(dataiter)
    print(batch1Y["NtC"])
    most_frequent = torch.mode(torch.concat([batch1Y["NtC"].view(-1), batch2Y["NtC"].view(-1)])).values.item()
    optimal_acc = ((batch1Y["NtC"] == most_frequent).float().mean().item() + (batch2Y["NtC"] == most_frequent).float().mean().item()) / 2
    print("most frequent", most_frequent)
    batch1X, batch1Y = torchutils.to_device(batch1X), torchutils.to_device(batch1Y)
    batch2X, batch2Y = torchutils.to_device(batch2X), torchutils.to_device(batch2Y)
    # x, y = torchutils.to_device(batch1X), torchutils.to_device(batch1Y)
    for name, param in model.named_parameters():
        if "weight" in name and param.dim() > 1:
            nn.init.xavier_uniform_(param)
    model.train()

    # closs = torch.nn.CrossEntropyLoss()
    linmodel = nn.Linear(10, 97).to(device)
    linmodel.train()
    optimizer = optim.Adam([ p for p in model.parameters() if p.requires_grad ], lr=0.001)
    for i in range(1000000):
        if random.randint(0, 1) == 0:
            x, y = batch1X, batch1Y
        else:
            x, y = batch2X, batch2Y
        optimizer.zero_grad()
        y_pred = model(x)
        # y_pred = {"NtC": linmodel(F.dropout(F.one_hot(x["sequence"], num_classes=10).type(torch.float32), p=0.5))[:, :-1, :]}
        # print(y_pred["NtC"].shape, y["NtC"].shape)
        loss = model.loss(y_pred, y)
        # loss = closs(y_pred["NtC"].contiguous().view(-1, 97), y["NtC"].view(-1))
        loss.backward()
        torch.nn.utils.clip_grad.clip_grad_value_(model.parameters(), 1.0)
        optimizer.step()

        max_grads = [
            (n, p.grad.abs().mean().item())
            for n, p in model.named_parameters()
            if p.grad is not None
        ]
        no_grad = [ name + ": " + str(p.shape) for name, p in model.named_parameters() if p.grad is None ]

        accuracy.update((y_pred["NtC"].contiguous().view(-1, 97), y["NtC"].view(-1)))
        pred_ntclist = torch.argmax(y_pred["NtC"], dim=1)
        print(pred_ntclist)
        most_frequent_pred = torch.mode(pred_ntclist.view(-1)).values.item()
        most_frequent_frequence = (pred_ntclist == most_frequent_pred).float().mean().item()
        most_frequence_data_frequence = (y["NtC"] == most_frequent_pred).float().mean().item()
        
        print(f"i={i}, loss = {loss.item()}, lr = {optimizer.param_groups[0]['lr']}, acc = {accuracy.compute()} / {optimal_acc:.2f}  (pred={most_frequent_pred} in {most_frequent_frequence:.5f} -> {most_frequence_data_frequence:.5f})")
        if i % 100 == 0:
            print(max_grads)
        # print("no grad: ", *no_grad)


    exit(1)

def create_trainer(p: Hyperparams, model: ntcnetwork.Network, optimizer: torch.optim.Optimizer, loss_fn):
    # trainer = ignite.engine.create_supervised_trainer(
    #     model, optimizer, model.loss, device,
    #     output_transform=lambda x, y, y_pred, loss: { "loss": loss.item(), "y": y, "y_pred": y_pred, "x": x }
    # )
    def update(engine: ignite.engine.Engine, batch: Sequence[torch.Tensor]) -> Union[Any, Tuple[torch.Tensor]]:
        x, y = batch
        x, y = ignite.utils.convert_tensor(x, device=device, non_blocking=True), ignite.utils.convert_tensor(y, device=device, non_blocking=True)
        optimizer.zero_grad()
        model.train()
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        loss.backward()
        if p.clip_grad is not None:
            torch.nn.utils.clip_grad.clip_grad_norm_(model.parameters(), p.clip_grad)
        if p.clip_grad_value is not None:
            torch.nn.utils.clip_grad.clip_grad_value_(model.parameters(), p.clip_grad_value)
        optimizer.step()

        nz_grad = sum([
            p.grad.not_equal(0).sum()
            for n, p in model.named_parameters()
            if p.grad is not None
        ])
        print(f"Non-zero gradients: {nz_grad:,}")
        return { "loss": loss.item(), "y": y, "y_pred": y_pred, "x": x }

    return ignite.engine.Engine(update)


def train(train_set_dir, val_set_dir, p: Hyperparams, logdir):
    if p.finetune and not p.external_embedding:
        raise ValueError("Cannot finetune without external_embedding")
    ds = dataset_wrapper.make_test_val(train_set_dir, val_set_dir, p)
    seq_len_schedule = epochschedule.parse_epoch_schedule(p.seq_length_schedule, p.epochs, tt=int)
    batch_size_schedule = epochschedule.get_batch_size_from_maxlen(seq_len_schedule, p.batch_size, p.max_batch_size)
    batch_count = epochschedule.get_step_count(batch_size_schedule, ds.train_size)

    model = ntcnetwork.Network(p).to(device)
    print_model(model)

    optimizer = optim.Adam(model.parameters(), lr=p.learning_rate)
    # optimizer.step()

    # testtheshit(ds, optimizer, model)
    trainer = create_trainer(p, model, optimizer, model.loss)

    print("expected batch count", batch_count)
    if p.lr_decay == "cosine":
        torch_lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=batch_count)
        # scheduler = ignite.handlers.param_scheduler.LRScheduler(torch_lr_scheduler)
        # trainer.add_event_handler(Events.ITERATION_STARTED, scheduler)
        @trainer.on(Events.ITERATION_COMPLETED)
        def lr_decay():
            torch_lr_scheduler.step()
    elif p.lr_decay != "none" and p.lr_decay is not None:
        raise ValueError(f"Unknown lr_decay method {p.lr_decay}")

    output_ntc = _output_field_transform("NtC", len_offset=1, filter=_NANT_filter)
    # recall_metric = metrics.Recall(output_transform=output_ntc, average=False)
    # precision_metric = metrics.Precision(output_transform=output_ntc, average=False)
    val_metrics: Dict[str, metrics.Metric] = {
        "accuracy": metrics.Accuracy(output_transform=output_ntc),
        "f1": metrics.Fbeta(1, output_transform=output_ntc),
        "f1_unfiltered": metrics.Fbeta(1, output_transform=_output_field_transform("NtC", len_offset=1)),
        "confusion": metrics.ConfusionMatrix(len(ntcnetwork.Network.NTC_LABELS), output_transform=_output_field_transform("NtC", len_offset=1)),
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
        print(metrics)
        print(f"evalT - Epoch[{global_epoch}] accuracy: {metrics['accuracy']:.2f} F1: {metrics['f1']:.2f} training loss: {trainer.state.output['loss']:.2f} Train Time: {epoch_time/60:3.1f}m Eval Time: {eval_time/60:3.2f}m    ")

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(trainer):
        eval_clock = Clock()
        val_evaluator.run(ds.get_validation_ds())
        eval_time = eval_clock.measure()
        metrics = val_evaluator.state.metrics
        print(f"evalV - Epoch[{global_epoch}] accuracy: {metrics['accuracy']:.2f} F1: {metrics['f1']:.2f} loss: {metrics['loss']:.2f} Eval Time: {eval_time/60:3.2f}m")

    # early_stopper = ignite.handlers.early_stopping.EarlyStopping(patience=8, score_function=lambda engine: -engine.state.metrics["loss"], trainer=trainer)
    # val_evaluator.add_event_handler(Events.COMPLETED, early_stopper)

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
    global global_epoch
    for epoch, (seq_len, batch_size) in enumerate(param_schedule):
        global_epoch = epoch
        result = trainer.run(max_epochs=1, data=ds.get_train_ds(seq_len, batch_size))
        if trainer.should_terminate:
            print("Should terminate -> exiting")
            break
        # print("training result", result)

    if p.finetune:
        import gc
        gc.collect()
        torch.cuda.empty_cache()

        print(f"Starting finetuning: {p.finetune}")
        assert model.external_embedding is not None
        model.external_embedding.enable_training_()
        seq_len, batch_size = param_schedule[-1]
        args = p.finetune.split(",")
        ft_epochs = int(args[1])
        ft_lr = p.learning_rate / float(args[0])
        ft_batch_size = batch_size if len(args) < 3 else int(args[2])
        ft_batch_count = math.ceil(ds.train_size // ft_batch_size) * ft_epochs
        normal_epochs = p.epochs
        assert normal_epochs == global_epoch + 1
        # torch_lr_scheduler.
        torch_lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=ft_batch_count + 1)
        torch_lr_scheduler.step()
        for ft_epoch in range(ft_epochs):
            global_epoch = normal_epochs + ft_epoch
            result = trainer.run(max_epochs=1, data=ds.get_train_ds(seq_len, ft_batch_size))
            if trainer.should_terminate:
                print("Should terminate -> exiting")
                break


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
        "training/f1": 0,
        "valset/accuracy": 0,
        "valset/miou": 0,
        "valset/f1": 0
    }
    a, b, c = tensorboardX.summary.hparams(hparams, hmetrics)
    tb_log.writer._get_file_writer().add_summary(a)
    tb_log.writer._get_file_writer().add_summary(b)
    tb_log.writer._get_file_writer().add_summary(c)
    tb_log.writer.add_scalar("model/total_params", count_parameters(model))
    tb_log.writer.add_text("model/structure", "```\n" + str(model) + "\n```\n")

class Clock:
    def __init__(self):
        self.start_time = time.time()
        self.last_time = self.start_time

    def measure(self):
        now = time.time()
        elapsed = now - self.last_time
        self.last_time = now
        return elapsed
    
    def elapsed(self):
        return time.time() - self.start_time

def setup_tensorboard_logger(trainer: ignite.engine.Engine, optimizer: optim.Optimizer, train_evaluator, val_evaluator, logdir) -> tensorboard_logger.TensorboardLogger:
    # Define a Tensorboard logger
    tb_logger = tensorboard_logger.TensorboardLogger(log_dir=logdir)

    clock_start = Clock()
    count_step = 0
    clock_step = Clock()
    count_epoch = 0
    clock_epoch = Clock()

    batch_loss = []

    def write_scalar(category, name, value, step = None):
        if step is None:
            step = count_epoch
        tb_logger.writer.add_scalar(category + "/" + name, float(value), step, walltime=clock_start.elapsed())

    def write_metrics(category, metrics, step = None):
        for name, value in metrics.items():
            if isinstance(value, numbers.Number):
                write_scalar(category, name, value, step)
            else:
                if name not in ["confusion", "y", "y_pred"]:
                    print(f"WARNING: metric {name} at {step} is not a number: {value}")

    def after_batch(_):
        nonlocal count_step
        count_step += 1

        if trainer.state.output is not None:
            batch_loss.append(trainer.state.output["loss"]) #type:ignore

    trainer.add_event_handler(Events.ITERATION_COMPLETED, after_batch)

    def after_epoch(_):
        nonlocal count_epoch
        count_epoch += 1
        write_scalar("training", "loss", np.mean(batch_loss))
        batch_loss.clear()
        write_scalar("training", "runtime", clock_epoch.measure())
        write_scalar("training", "learning_rate", optimizer.param_groups[0]["lr"])
        write_metrics("training", trainer.state.metrics)

    trainer.add_event_handler(Events.EPOCH_COMPLETED, after_epoch)

    def after_trainset(_):
        write_metrics("trainset", train_evaluator.state.metrics)
    trainer.add_event_handler(Events.EPOCH_COMPLETED, after_trainset)
    def after_valset(_):
        write_metrics("valset", val_evaluator.state.metrics)

        confmatrix: torch.Tensor = val_evaluator.state.metrics["confusion"]
        figure, ax = plt.subplots()
        im = ax.imshow(confmatrix.numpy(), norm=matplotlib.colors.LogNorm())
        plt.colorbar(im, ax=ax)
        tb_logger.writer.add_figure("valset/confusion", figure, count_epoch, walltime=clock_start.elapsed())
        plt.close(figure)
        tb_logger.writer.add_histogram("valset/log_predictions", np.log10(1 + np.sum(confmatrix.numpy(), axis=0)), count_epoch, walltime=clock_start.elapsed())
        tb_logger.writer.add_histogram("valset/log_labels", np.log10(1 + np.sum(confmatrix.numpy(), axis=1)), count_epoch, walltime=clock_start.elapsed())
    trainer.add_event_handler(Events.EPOCH_COMPLETED, after_valset)

    return tb_logger

if __name__ == '__main__':
    print("Use the launch.py script using `torch-train` command.")
    exit(1)
