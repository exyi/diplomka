#!/usr/bin/env python3

import math, time, os, sys
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
from typing import Any, Dict, List, Optional, Tuple
import tensorflow as tf, tensorflow_addons as tfa
import dataclasses
import random
from hparams import Hyperparams

import ntcnetwork_tf as ntcnetwork
import dataset_tf
import csv_loader

class NtcMetricWrapper(tf.keras.metrics.Metric):
    def __init__(self, metric: tf.keras.metrics.Metric, argmax_output=False):
        super().__init__(metric.name, metric.dtype)
        self.inner_metric = metric
        self.argmax_output = argmax_output

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = y_pred.values
        y_true = y_true.values
        y_true = tf.one_hot(y_true, ntcnetwork.Network.OUTPUT_NTC_SIZE)

        if self.argmax_output:
            y_pred = tf.argmax(y_pred, axis=-1)
            y_pred = tf.one_hot(y_pred, ntcnetwork.Network.OUTPUT_NTC_SIZE)

        self.inner_metric.update_state(y_true, y_pred, sample_weight)

        if False:
            y_pred_argmax = tf.argmax(y_pred, axis=-1)
            y_true_argmax = tf.argmax(y_true, axis=-1)
            tf.print("Prediction vs true NTC: ", tf.stack([
                tf.gather(csv_loader.ntcs, y_pred_argmax),
                tf.gather(csv_loader.ntcs, y_true_argmax)
            ], axis=1))
    def result(self):
        return self.inner_metric.result()
    def reset_state(self):
        self.inner_metric.reset_state()
    def get_config(self):
        return self.inner_metric.get_config()

def create_model(p: Hyperparams, step_count, logdir, eager=False, profile=False):
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

    model.compile(optimizer=optimizer, loss={
        "NtC": model.ntcloss,
    }, metrics={
        "NtC": [
            tf.keras.metrics.SparseCategoricalAccuracy(name="acc"),
            tf.keras.metrics.SparseTopKCategoricalAccuracy(k=2, name="acc2"),
            tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5, name="acc5"),
            NtcMetricWrapper(tfa.metrics.F1Score(name="f1", num_classes=ntcnetwork.Network.OUTPUT_NTC_SIZE, average="macro")),
        ]
    },
    )
    model.run_eagerly = eager
    model.tb_callback = tf.keras.callbacks.TensorBoard(args.logdir)
    return model

def train(train_set_dir, val_set_dir, p: Hyperparams, logdir, eager=False):
    train_ds = dataset_tf.NtcDatasetLoader(train_set_dir).get_data(max_len=p.seq_length_limit, batch=p.batch_size, shuffle=3000)
    batch_count = int(train_ds.cardinality())
    assert batch_count > 0, f"batch_count = {batch_count}"
    val_ds = dataset_tf.NtcDatasetLoader(val_set_dir).get_data(batch=p.batch_size)
    
    model = create_model(p, batch_count, logdir, eager=eager, profile=profile)
    # build the model, otherwise the .summary() call is unhappy
    predictions = model.predict(next(iter(train_ds.map(lambda x, y: x))))
    # print(predictions)

    logger_clock = Clock()
    epoch_clock = Clock()
    model.summary(expand_nested=True)
    summary_text = []
    model.summary(expand_nested=True, print_fn=lambda x: summary_text.append(x))
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
