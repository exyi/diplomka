import tensorflow as tf
import ntcnetwork_tf as ntcnetwork

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


