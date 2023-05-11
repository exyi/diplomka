#!/usr/bin/env python3

import math, time, os, sys
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
from typing import Any, Dict, List, Optional, Tuple
import tensorflow as tf
import dataclasses
# import torchtext
import random
from hparams import Hyperparams

import ntcnetwork_tf as ntcnetwork
import dataset_tf

import itertools

def filter_dict(d: Dict[str, Any], keys: List[str]) -> Dict[str, Any]:
    return { k: v for k, v in d.items() if k in keys }

def create_dataset(file, batch_size: int, shuffle=False):
    loader = dataset_tf.NtcDatasetLoader(file)
    dataset = loader.dataset
    if shuffle:
        dataset = dataset.shuffle(3000)
    dataset = dataset.map(lambda x: (
        filter_dict(x, ["is_dna", "sequence"]),
        filter_dict(x, ["NtC"]),
    ))
    dataset = dataset.ragged_batch(batch_size)
    return dataset

def create_model(p: Hyperparams, step_count, logdir, eager=False):
    model = ntcnetwork.Network(p)
    if p.lr_decay == "cosine":
        learning_rate: Any = tf.optimizers.schedules.CosineDecay(
            p.learning_rate,
            decay_steps=step_count,
            alpha=p.learning_rate/50,
        )
    else:
        learning_rate = p.learning_rate
    optimizer = tf.optimizers.Adam(learning_rate)

    # recall_metric = metrics.Recall(output_transform=lambda x: (x[0]["NtC"], x[1]["NtC"]), average=False)
    # precision_metric = metrics.Precision(output_transform=lambda x: (x[0]["NtC"], x[1]["NtC"]), average=False)
    # val_metrics: Dict[str, metrics.Metric] = {
    #     "accuracy": metrics.Accuracy(output_transform=lambda x: (x[0]["NtC"], x[1]["NtC"])),
    #     "f1": (precision_metric * recall_metric * 2 / (precision_metric + recall_metric)).nanmean(),
    #     "miou": metrics.mIoU(metrics.ConfusionMatrix(len(ntcnetwork.Network.NTC_LABELS), output_transform=lambda x: (x[0]["NtC"], x[1]["NtC"]))),
    #     "loss": metrics.Loss(model.loss)
    # }

    model.compile(optimizer=optimizer, loss={
        "NtC": model.ntcloss,
    }, metrics={
        "NtC": [
            # tf.keras.metrics.Accuracy(),
            # tf.keras.metrics.
        ]
    })
    model.run_eagerly = eager
    model.tb_callback = tf.keras.callbacks.TensorBoard(args.logdir)
    return model

def train(train_set_dir, val_set_dir, p: Hyperparams, logdir, eager=False):
    train_ds = create_dataset(train_set_dir, p.batch_size, shuffle=True)
    batch_count = int(train_ds.cardinality())
    assert batch_count > 0, f"batch_count = {batch_count}"
    val_ds = create_dataset(val_set_dir, p.batch_size)
    
    model = create_model(p, batch_count, logdir, eager=eager)
    # build the model, otherwise the .summary() call is unhappy
    predictions = model.predict(next(iter(train_ds.map(lambda x, y: x))))
    # print(predictions)

    logger_clock = Clock()
    epoch_clock = Clock()
    model.summary()
    summary_text = []
    model.summary(print_fn=lambda x: summary_text.append(x))
    tf.summary.text("model/structure", "\n".join(summary_text), step=0)
    tf.summary.scalar("model/total_params", model.count_params(), step=0)

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=p.epochs,
        callbacks=[ model.tb_callback ],
    )


class Clock:
    def __init__(self):
        self.start_time = time.time()
        self.last_time = self.start_time

    def measure(self):
        now = time.time()
        elapsed = now - self.last_time
        self.last_time = now
        return elapsed


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Train NtC network using PyTorch ignite')
    parser.add_argument('--train_set', type=str, help='Path to directory training data CSVs')
    parser.add_argument('--val_set', type=str, help='Path to directory validation data CSVs')
    parser.add_argument('--logdir', type=str, default="tb-logs", help='Path for saving Tensorboard logs and other outputs')
    parser.add_argument('--eager', action="store_true", help='Run in eager mode', default=False)

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

    tb_writer = tf.summary.create_file_writer(args.logdir)
    with tb_writer.as_default():
        

        print(f"logdir: {args.logdir}")
        print(f"devices: {[ x.name for x in tf.config.list_physical_devices() ]}")
        print(hyperparameters)
        tf.summary.text("model/hyperparams", str(hyperparameters), step=0)

        try:
            train(args.train_set, args.val_set, hyperparameters, args.logdir, eager=args.eager)
        except KeyboardInterrupt:
            print("Keyboard interrupt")
        except Exception as e:
            tf.summary.text("crash", str(e), step=(tf.summary.experimental.get_step() or 0))
            raise e
