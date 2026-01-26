import tensorflow as tf


class R2(tf.keras.metrics.Metric):
    def __init__(self, name="r2", **kwargs):
        super().__init__(name=name, **kwargs)
        self.ssr = self.add_weight(name="ssr", initializer="zeros")
        self.sum_y = self.add_weight(name="sum_y", initializer="zeros")
        self.sum_y2 = self.add_weight(name="sum_y2", initializer="zeros")
        self.count = self.add_weight(name="count", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        residuals = y_true - y_pred
        self.ssr.assign_add(tf.reduce_sum(tf.square(residuals)))

        self.sum_y.assign_add(tf.reduce_sum(y_true))
        self.sum_y2.assign_add(tf.reduce_sum(tf.square(y_true)))
        self.count.assign_add(tf.cast(tf.size(y_true), tf.float32))

    def result(self):
        sst = self.sum_y2 - tf.square(self.sum_y) / self.count
        return 1.0 - self.ssr / (sst + tf.keras.backend.epsilon())

    def reset_state(self):
        for v in self.variables:
            v.assign(0.0)