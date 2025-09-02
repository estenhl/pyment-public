import tensorflow as tf


def optimizer_factory(name: str) -> tf.optimizers.Optimizer:
    if name.lower() == 'adam':
        return tf.optimizers.Adam

    raise KeyError(f'Unknown optimizer {name}')