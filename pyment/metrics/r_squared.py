import tensorflow as tf


class R2(tf.keras.metrics.Metric):
    def __init__(self, name: str = 'r2', **kwargs):
        super().__init__(name=name, **kwargs)

        self.ssr = self.add_weight(name='ssr', initializer='zeros')
        self.sum_y = self.add_weight(name='sum_y', initializer='zeros')
        self.sum_y2 = self.add_weight(name='sum_y2', initializer='zeros')
        self.count = self.add_weight(name='count', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None) -> None:
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        residuals = y_true - y_pred

        self.ssr.assign_add(tf.reduce_sum(tf.square(residuals)))  # type: ignore
        self.sum_y.assign_add(tf.reduce_sum(y_true))  # type: ignore
        self.sum_y2.assign_add(tf.reduce_sum(tf.square(y_true)))  # type: ignore
        self.count.assign_add(tf.cast(tf.size(y_true), tf.float32))  # type: ignore

    def result(self) -> tf.Tensor:
        sst = self.sum_y2 - tf.square(self.sum_y) / self.count  # type: ignore

        return 1.0 - self.ssr / (sst + tf.keras.backend.epsilon())

    def reset_state(self) -> None:
        for v in self.variables:
            v.assign(0.0)
