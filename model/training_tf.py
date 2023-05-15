#!/usr/bin/env python3

import math, time, os, sys
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
from typing import Any, Dict, List, Optional, Tuple
import tensorflow as tf, tensorflow_addons as tfa
import tensorboard.plugins.hparams.api as hparams
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
                tf.gather(dataset_tf.NtcDatasetLoader.ntc_mapping.get_vocabulary(), y_pred_argmax),
                tf.gather(dataset_tf.NtcDatasetLoader.ntc_mapping.get_vocabulary(), y_true_argmax)
            ], axis=1))
    def result(self):
        return self.inner_metric.result()
    def reset_state(self):
        self.inner_metric.reset_state()
    def get_config(self):
        return self.inner_metric.get_config()
    
def parse_len_schedule(input_str: str, num_epochs: int) -> List[Tuple[int, int]]:
    splits = [ s.split('*') for s in input_str.split(";") ]
    non_x_sum = sum([ int(epochs) for epochs, _ in splits if epochs != "x" ])
    x_count = sum([ 1 for epochs, _ in splits if epochs == "x" ])
    x_min = (num_epochs - non_x_sum) // x_count
    x_mod = (num_epochs - non_x_sum) % x_count
    x = [ x_min + (1 if i < x_mod else 0) for i in range(x_count) ]
    x.reverse()
    return [ ((int(epochs) if epochs != "x" else x.pop()), int(max_len)) for epochs, max_len in splits ]

def get_step_count(seq_len_schedule: List[Tuple[int, int]], ds_cardinality, base_batch_size: int) -> int:
    base_seq_len = min([ seq_len for _, seq_len in seq_len_schedule ])
    return sum([ epochs * math.ceil(ds_cardinality / math.ceil(base_seq_len * base_batch_size / seq_len)) for epochs, seq_len in seq_len_schedule ])

class HackedTB(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if isinstance(logs, dict):
            for k, v in logs.items():
                tf.summary.scalar(f"epochmetrics/{k}", v, step=epoch)

class FilteredProgbar(tf.keras.callbacks.ProgbarLogger):
    def __init__(self, count_mode: str = "samples", stateful_metrics=None, include: Optional[List[str]] = None, exclude: Optional[List[str]] = None):
        super().__init__(count_mode, stateful_metrics)
        self.include = set(include) if include is not None else None
        self.exclude = set(exclude) if exclude is not None else None
        # print("FilteredProgbar: include=", self.include, " exclude=", self.exclude)

    def _filter_logs(self, logs):
        if logs is None or not isinstance(logs, dict):
            return logs

        if self.include is not None:
            logs = { k: v for k, v in logs.items() if k in self.include }
        if self.exclude is not None:
            logs = { k: v for k, v in logs.items() if k not in self.exclude }
        
        # print("FilteredProgbar: logs=", logs)
        return logs
    
    def on_train_batch_end(self, batch, logs=None):
        return super().on_train_batch_end(batch, self._filter_logs(logs))
    def on_test_batch_end(self, batch, logs=None):
        return super().on_test_batch_end(batch, self._filter_logs(logs))
    def on_predict_batch_end(self, batch, logs=None):
        return super().on_predict_batch_end(batch, self._filter_logs(logs))

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
    optimizer = tf.optimizers.Adam(learning_rate, global_clipnorm=p.clip_grad)

    all_submodules: List[tf.Module] = model.submodules
    config_json = {
        "hyperparams": dataclasses.asdict(p),
        "optimizer": optimizer.get_config(),
        "submodules": [m.get_config() for m in all_submodules],
    }
    with open(os.path.join(logdir, "tf_modules.json"), "w") as f:
        import json
        json.dump(config_json, f, indent=4)

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
    if profile == False:
        profile_batch: Any = 0
    elif profile == True or profile == None:
        profile_batch = (2, 100)
    else:
        profile_batch = tuple(map(int, profile.split(",")))
    model.tb_callbacks = [
        tf.keras.callbacks.TensorBoard(logdir, profile_batch=profile_batch, write_steps_per_second=True),
        HackedTB(),
        FilteredProgbar(count_mode="steps", include=["loss", "acc", "f1", "val_acc", "val_acc5", "val_f1"])
    ]
    return model

def model_fit(model: tf.keras.Model, train_loader: dataset_tf.NtcDatasetLoader, val_ds, seq_len_schedule: List[Tuple[int, int]], base_batch_size: int):
    base_seq_len = min([ seq_len for _, seq_len in seq_len_schedule ])
    e = 0
    for epochs, seq_len in seq_len_schedule:
        batch_size = math.ceil(base_batch_size * base_seq_len / seq_len)
        model.fit(
            train_loader.get_data(batch=batch_size, max_len=seq_len, shuffle=15000),
            validation_data=val_ds,
            initial_epoch=e,
            epochs=e + epochs,
            callbacks=[ *model.tb_callbacks ],
        )
        e += epochs

def train(train_set_dir, val_set_dir, p: Hyperparams, logdir, eager=False, profile=False):
    seq_len_schedule = parse_len_schedule(p.seq_length_schedule, p.epochs)
    train_loader = dataset_tf.NtcDatasetLoader(train_set_dir)
    step_count = get_step_count(seq_len_schedule, train_loader.cardinality, p.batch_size)
    assert step_count > 0, f"{step_count=}"
    val_ds = dataset_tf.NtcDatasetLoader(val_set_dir).get_data(batch=p.batch_size)
    
    model = create_model(p, step_count, logdir, eager=eager, profile=profile)
    # tf.summary.trace_on(graph=True, profiler=False)
    # build the model, otherwise the .summary() call is unhappy
    predictions = model(next(iter(train_loader.get_data(max_len=128, batch=2).map(lambda x, y: x))))
    # tf.summary.trace_export(name="model_trace", step=0, profiler_outdir=logdir)
    # tf.summary.trace_off()

    summary_text = []
    model.summary(expand_nested=True, print_fn=lambda x: summary_text.append(x))
    print("\n".join(summary_text))

    tf.summary.text("model/structure", "```\n" + "\n".join(summary_text) + "\n```\n", step=0)
    tf.summary.scalar("model/total_params", model.count_params(), step=0)
    # tf.keras.utils.plot_model(model, to_file=os.path.join(logdir, "model.png"), expand_nested=True, show_shapes=True, show_layer_activations=True, dpi=300, show_trainable=True)

    # with tf.profiler.experimental.Profile(logdir):

    model_fit(model, train_loader, val_ds, seq_len_schedule, p.batch_size)

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
    parser.add_argument('--profile', type=str, help='Run tensorflow profiler. The value specified for which batches the profiler should be run (for example 10,20 for 10..20)', default=False)
    parser.add_argument('--fp16', action="store_true", help='Run in (mixed) Float16 mode. By default, float32 is used', default=False)
    parser.add_argument('--bfp16', action="store_true", help='Run in (mixed) BFloat16 mode', default=False)

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

    if args.fp16:
        tf.keras.mixed_precision.set_global_policy(tf.keras.mixed_precision.Policy("mixed_float16"))
    elif args.bfp16:
        tf.keras.mixed_precision.set_global_policy(tf.keras.mixed_precision.Policy("mixed_bfloat16"))

    hyperparameters = Hyperparams(**{ k: v for k, v in vars(args).items() if k in Hyperparams.__dataclass_fields__ })

    tb_writer = tf.summary.create_file_writer(args.logdir)
    with tb_writer.as_default():
        print(f"logdir: {args.logdir}")
        print(f"devices: {[ x.name for x in tf.config.list_physical_devices() ]}")
        print(hyperparameters)
        print(hyperparameters.get_nondefault())
        tf.summary.trace_off()
        tf.summary.text("model/hyperparams", "```python\n" + str(hyperparameters) + "\n\n" + str(hyperparameters.get_nondefault()) + "\n```\n", step=0)

        ## log hparams
        # tensorboard.plugins.hparams.api.hparams(
        hparams_keys = {f.name: hparams.HParam(name=f.name, description=f.metadata.get("help", None)) for f in dataclasses.fields(Hyperparams) if f.metadata.get("hyperparameter", False)}
        hparams.hparams_config(
            hparams=list(hparams_keys.values()),
            metrics=[
                hparams.Metric("epochmetrics/val_f1", display_name="Validation F1 score"),
                hparams.Metric("epochmetrics/val_acc", display_name="Validation accuracy"),
                hparams.Metric("epochmetrics/val_loss", display_name="Validation loss"),
                hparams.Metric("epochmetrics/f1", display_name="Training F1 score"),
                hparams.Metric("epochmetrics/acc", display_name="Training accuracy"),
            ]
        )
        hparams.hparams(
            {k: (v if isinstance(v, (int, float, bool, str)) else str(v)) for k, v in dataclasses.asdict(hyperparameters).items() if k in hparams_keys},
            trial_id=os.path.basename(args.logdir)
        )

        try:
            train(args.train_set, args.val_set, hyperparameters, args.logdir, eager=args.eager, profile=args.profile)
        except KeyboardInterrupt:
            print("Keyboard interrupt")
        except Exception as e:
            tf.summary.text("crash", "```\n" + str(e) + "\n```\n", step=(tf.summary.experimental.get_step() or 0))
            raise e
