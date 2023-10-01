import tensorflow as tf
import model.tf.ntcnetwork as ntcnetwork

class NtcMetricWrapper(tf.keras.metrics.Metric):
    def __init__(self, metric: tf.keras.metrics.Metric, argmax_output=False, decoder=None, ignore_nants=True, name=None, dtype=None):
        super().__init__(name or metric.name, dtype or metric.dtype)
        self.inner_metric = metric
        self.argmax_output = argmax_output
        self.decoder = decoder
        self.ignore_nants = ignore_nants

    def update_state(self, y_true, y_pred, sample_weight=None):
        if self.decoder is not None:
            y_pred = self.decoder(y_pred)
            y_pred = tf.one_hot(y_pred, ntcnetwork.Network.OUTPUT_NTC_SIZE)
        elif self.argmax_output:
            y_pred = tf.argmax(y_pred, axis=-1)
            y_pred = tf.one_hot(y_pred, ntcnetwork.Network.OUTPUT_NTC_SIZE)

        y_pred = y_pred.values
        y_true = y_true.values
        is_not_nant = (y_true != ntcnetwork.dataset.NtcDatasetLoader.ntc_mapping("NANT")) & (y_true != ntcnetwork.dataset.NtcDatasetLoader.ntc_mapping("<UNK>"))
        y_true = tf.one_hot(y_true, ntcnetwork.Network.OUTPUT_NTC_SIZE)

        if self.ignore_nants:
            # y_true = tf.boolean_mask(tf.broadcast_to(tf.expand_dims(is_not_nant, axis=-1), tf.shape(y_true)), y_true)
            y_true = tf.boolean_mask(y_true, is_not_nant, axis=0)
            y_pred = tf.boolean_mask(y_pred, is_not_nant, axis=0)

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
        return {
            **super().get_config(),
            "metric": tf.keras.metrics.serialize(self.inner_metric),
        }

class FilteredSparseCategoricalAccuracy(tf.keras.metrics.MeanMetricWrapper):
    def __init__(self, ignored_labels, name=None, dtype=None, **kwargs):
        super().__init__(self.accuracy, name, dtype, **kwargs)
        self.ignored_labels = ignored_labels

    # @tf.function
    def accuracy(self, y_true, y_pred):
        # print(y_true, self.ignored_labels)
        is_ignored = tf.reduce_any(tf.equal(tf.cast(y_true, dtype=tf.int64), self.ignored_labels), axis=-1)
        not_ignored = tf.logical_not(is_ignored)
        return tf.keras.metrics.sparse_categorical_accuracy(tf.boolean_mask(y_true, not_ignored, axis=0), tf.boolean_mask(y_pred, not_ignored))


