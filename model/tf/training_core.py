import math, time, os, sys

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import tensorflow as tf
from packaging.version import Version
tensorflow_2_13 = Version(tf.__version__).release >= Version("2.13").release
if not tensorflow_2_13:
    import tensorflow_addons as tfa
else:
    tfa=None
import dataclasses
import random

from . import ntcnetwork

from model.utils import filter_dict
from model import hyperparameters, utils, sample_weight, dataset_tf, csv_loader, epochschedule
from model.hyperparameters import Hyperparams
from .metrics import FilteredSparseCategoricalAccuracy, NtcMetricWrapper

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

def utf8decode(x):
    if isinstance(x, tf.Tensor):
        return x.numpy().decode('utf-8')
    if isinstance(x, bytes):
        return x.decode('utf-8')
    return str(x)

def print_results(file, model: ntcnetwork.Network, val_dataset):
    vocab_letter = dataset_tf.NtcDatasetLoader.letters_mapping.get_vocabulary()
    vocab_ntc = dataset_tf.NtcDatasetLoader.ntc_mapping.get_vocabulary()
    vocab_cana = dataset_tf.NtcDatasetLoader.cana_mapping.get_vocabulary()
    for i, (x, y, *_) in enumerate(val_dataset):
        pred_batch = model(x)
        pred_ntc_decoded = model.crf_ntc_decode(pred_batch['NtC'])
        batch_size = x["sequence"].shape[0]
        for i in range(batch_size):
            sequence = x['sequence'][i].numpy()
            ntcs_true = y['NtC'][i].numpy()
            nearest_ntcs_true = x['nearest_NtC'][i].numpy()
            canas_true = y['CANA'][i].numpy() if 'CANA' in y else None
            is_dna = x['is_dna'][i].numpy()
            struct_len = sequence.shape[0]
            print(f"### Structure X ({x['pdbid'][i].numpy()}  {np.sum(x['sequence'][i].numpy() != dataset_tf.NtcDatasetLoader.letters_mapping(' '))}nt {1+np.sum(x['sequence'][i].numpy() == dataset_tf.NtcDatasetLoader.letters_mapping(' '))}ch ===========", file=file)
            ntcs_pred_dist = tf.nn.softmax(pred_batch['NtC'][i]).numpy()
            canas_pred_dist = pred_batch['CANA'][i].numpy() if 'CANA' in pred_batch else None
            for j in range(struct_len - 1):
                letter = ('d' if is_dna[j] else ' ') + vocab_letter[sequence[j]]
                ntc_true = vocab_ntc[ntcs_true[j]]
                ntc_nearest_true = vocab_ntc[nearest_ntcs_true[j]]
                ntc_pred = vocab_ntc[pred_ntc_decoded[i][j]] # vocab_ntc[np.argmax(ntcs_pred_dist[j], axis=-1)]
                ntc_sureness = ntcs_pred_dist[j][np.argmax(ntcs_pred_dist[j], axis=-1)]
                ntc_true_sureness = ntcs_pred_dist[j][ntcs_true[j]]
                ntc_true_order = np.sum(ntcs_pred_dist[j] > ntc_true_sureness)

                if canas_pred_dist is not None:
                    cana_pred = vocab_cana[np.argmax(canas_pred_dist[j], axis=-1)]
                    cana_correct = cana_pred == vocab_cana[canas_true[j]]
                    cana_pred = cana_pred + "-"
                else:
                    cana_correct = ntc_true[0:2] == ntc_pred[0:2] or ntc_true[0:2] == ntc_nearest_true[0:2]
                    cana_pred = ""

                if ntc_true == ntc_pred or ntc_nearest_true == ntc_pred:
                    warning = "   "
                elif ntc_true == "NANT":
                    warning = " . "
                elif ntc_true_order < 3:
                    warning = " * "
                elif cana_correct:
                    warning = "!!"
                else:
                    warning = "!!!"

                if ntc_true == "NANT":
                    ntc_true_l = "N(" + ntc_nearest_true + ")"
                else:
                    ntc_true_l = "  " + ntc_true + " "

                print(f"{j: 5} {letter} {ntc_true_l} {cana_pred}{ntc_pred}  {int(round(ntc_sureness*100)): >2}%  {int(round(ntc_true_sureness*100)): >2}%  {ntc_true_order: >2}        {warning}", file=file)
            print(f"{struct_len-1: 5} {('d' if is_dna[-1] else ' ')}{vocab_letter[sequence[-1]]}  ...", file=file)
            print('', file=file)

def create_model(p: Hyperparams, step_count, logdir, profile=False):
    model = ntcnetwork.Network(p)
    if p.lr_decay == "cosine":
        learning_rate: Any = tf.optimizers.schedules.CosineDecay(
            p.learning_rate,
            decay_steps=step_count,
            alpha=p.learning_rate/50,
        )
    else:
        learning_rate = p.learning_rate
    optimizer = tf.optimizers.Adam(learning_rate, global_clipnorm=p.clip_grad, clipvalue=p.clip_grad_value)

    all_submodules: List[tf.Module] = model.submodules
    module_names = set()
    # for x in all_submodules:
    #     print(x.name)
    #     if x.name in module_names:
    #         raise ValueError(f"Duplicate module name {x.name} of type {x.__class__}")
    #     module_names.add(x.name)
    config_json = {
        "hyperparams": dataclasses.asdict(p),
        "optimizer": optimizer.get_config(),
        "submodules": [m.get_config() for m in all_submodules if hasattr(m, "get_config")],
    }
    with open(os.path.join(logdir, "tf_modules.json"), "w") as f:
        import json
        def serialize_weird_object(x):
            if isinstance(x, tf.TensorShape):
                return list(x)
            else:
                raise TypeError(f"Can't serialize {x}: {type(x)}")
        json.dump(config_json, f, indent=4, default=serialize_weird_object)

    model.compile(optimizer=optimizer, loss=filter_dict({
        "NtC": model.ntcloss,
        # "NtC": model.unweighted_ntcloss, 
        # "NtC": tf.keras.losses.SparseCategoricalCrossentropy(reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE),
        # "CANA": tf.keras.losses.SparseCategoricalCrossentropy()
        "CANA": model.canaloss
    }, p.outputs), metrics=filter_dict({
        "NtC": [
            NtcMetricWrapper(tf.keras.metrics.CategoricalAccuracy(name="acc")),
            NtcMetricWrapper(tf.keras.metrics.TopKCategoricalAccuracy(k=2, name="acc2")),
            NtcMetricWrapper(tf.keras.metrics.TopKCategoricalAccuracy(k=5, name="acc5")),
            NtcMetricWrapper(
                tf.metrics.F1Score(name="f1", average="macro")
                if tensorflow_2_13 else
                tfa.metrics.F1Score(name="f1", num_classes=ntcnetwork.Network.OUTPUT_NTC_SIZE, average="macro"),
            ),
            # NtcMetricWrapper(
            #     tf.metrics.F1Score(name="CRFf1", average="macro")
            #     if tensorflow_2_13 else
            #     tfa.metrics.F1Score(name="CRFf1", num_classes=ntcnetwork.Network.OUTPUT_NTC_SIZE, average="macro"),
            # decoder=model.crf_ntc_decode),
            FilteredSparseCategoricalAccuracy(
                ignored_labels=dataset_tf.NtcDatasetLoader.ntc_mapping(["<UNK>", "NANT", "AA00", "AA08"]),
                name="accF"
            )
        ],
        "CANA": [
            tf.keras.metrics.SparseCategoricalAccuracy(name="acCANA")
        ]
    }, p.outputs),
     weighted_metrics=filter_dict({
        "NtC": [
            tf.keras.metrics.SparseCategoricalAccuracy(name="accW"),
        ]
     }, p.outputs),
     from_serialized=True # don't rename my metrics when other output is added
    )
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

def model_fit(model: tf.keras.Model, train_loader: dataset_tf.NtcDatasetLoader, val_ds, seq_len_schedule: List[Tuple[int, int]], batch_size_schedule: List[Tuple[int, int]]):
    e = 0
    for (epochs, seq_len), (epochs_, batch_size) in zip(seq_len_schedule, batch_size_schedule):
        assert epochs == epochs_
        model.fit(
            train_loader.get_data(batch=batch_size, max_len=seq_len, shuffle=15000),
            validation_data=val_ds,
            initial_epoch=e,
            epochs=e + epochs,
            callbacks=[ *model.tb_callbacks ],
        )
        e += epochs

def train(train_set_dir, val_set_dir, load_model, p: Hyperparams, logdir, eager=False, profile=False):
    seq_len_schedule = epochschedule.parse_epoch_schedule(p.seq_length_schedule, p.epochs, tt=int)
    batch_size_schedule = epochschedule.get_batch_size_from_maxlen(seq_len_schedule, p.batch_size, p.max_batch_size)
    print("Epoch/sequence length schedule: ", seq_len_schedule)

    sample_weighter = None # sample_weight.get_weighter(p.sample_weight, tf, dataset_tf.NtcDatasetLoader.ntc_mapping.get_vocabulary())
    train_loader = dataset_tf.NtcDatasetLoader(train_set_dir, features=p.outputs, ntc_rmsd_threshold=p.nearest_ntc_threshold).set_sample_weighter(sample_weighter)
    step_count = epochschedule.get_step_count(batch_size_schedule, train_loader.cardinality)
    assert step_count > 0, f"{step_count=}"
    val_loader = dataset_tf.NtcDatasetLoader(val_set_dir, features=p.outputs, ntc_rmsd_threshold=0).set_sample_weighter(sample_weighter)
    val_ds = val_loader.get_data(batch=p.batch_size)
    
    if load_model:
        model = tf.keras.models.load_model(load_model, custom_objects={
            "Network": ntcnetwork.Network,
            "unweighted_ntcloss": ntcnetwork.Network.ntcloss,
            "NtcMetricWrapper": NtcMetricWrapper,
        })
    else:
        model = create_model(p, step_count, logdir, profile=profile)
    model.run_eagerly = eager
    # tf.summary.trace_on(graph=True, profiler=False)
    # build the model, otherwise the .summary() call is unhappy
    predictions = model(next(iter(train_loader.get_data(max_len=128, batch=2).map(lambda x, y, sw: x))))
    # tf.summary.trace_export(name="model_trace", step=0, profiler_outdir=logdir)
    # tf.summary.trace_off()

    summary_text = []
    model.summary(expand_nested=True, print_fn=lambda x: summary_text.append(x))
    print("\n".join(summary_text))

    tf.summary.text("model/structure", "```\n" + "\n".join(summary_text) + "\n```\n", step=0)
    tf.summary.scalar("model/total_params", model.count_params(), step=0)
    # tf.keras.utils.plot_model(model, to_file=os.path.join(logdir, "model.png"), expand_nested=True, show_shapes=True, show_layer_activations=True, dpi=300, show_trainable=True)

    # with tf.profiler.experimental.Profile(logdir):

    model_fit(model, train_loader, val_ds, seq_len_schedule, batch_size_schedule)

    with open(os.path.join(logdir, "predictions.txt"), "w") as file:
        print_results(file, model, val_ds)

    model.save(os.path.join(logdir, "final_model.h5"))

    return model

class Clock:
    def __init__(self):
        self.start_time = time.time()
        self.last_time = self.start_time

    def measure(self):
        now = time.time()
        elapsed = now - self.last_time
        self.last_time = now
        return elapsed

def main_args(args):
    import tensorboard.plugins.hparams.api as hparams

    if args.fp16:
        tf.keras.mixed_precision.set_global_policy(tf.keras.mixed_precision.Policy("mixed_float16"))
    elif args.bfp16:
        tf.keras.mixed_precision.set_global_policy(tf.keras.mixed_precision.Policy("mixed_bfloat16"))

    p = Hyperparams.from_args(args)

    assert utils.get_logdir() == args.logdir
    tb_writer = tf.summary.create_file_writer(args.logdir)
    with tb_writer.as_default():
        print(f"logdir: {args.logdir}")
        print(f"devices: {[ x.name for x in tf.config.list_physical_devices() ]}")
        print(p)
        print(p.get_nondefault())
        tf.summary.trace_off()
        tf.summary.text("model/hyperparams", "```python\n" + str(p) + "\n\n" + str(p.get_nondefault()) + "\n```\n", step=0)

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
            {k: (v if isinstance(v, (int, float, bool, str)) else str(v)) for k, v in dataclasses.asdict(p).items() if k in hparams_keys},
            trial_id=os.path.basename(args.logdir)
        )

        tf.summary.experimental.set_step(-1)

        try:
            model = train(args.train_set, args.val_set, args.load_model, p, args.logdir, eager=args.eager, profile=args.profile)
        except KeyboardInterrupt:
            print("Keyboard interrupt")
        except Exception as e:
            tf.summary.text("crash", "```\n" + str(e) + "\n```\n", step=(tf.summary.experimental.get_step() or 0))
            raise e


