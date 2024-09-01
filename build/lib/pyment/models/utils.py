import tensorflow as tf

from tensorflow.keras.layers import Add, ReLU, GlobalAveragePooling3D, \
                                    GlobalMaxPooling3D


def get_global_pooling_layer(name: str, dimensions: int):
    if dimensions == 3:
        if name in ['avg', 'average']:
            return GlobalAveragePooling3D
        elif name in ['max']:
            return GlobalMaxPooling3D

    raise NotImplementedError(f'Invalid global pooling layer name {name} '
                              f'for {dimensions} dimensions')

def restrict_range(x: tf.Tensor, lower: int, upper: int,
                   name: str = 'restrict_range') -> tf.Tensor:
    assert upper > lower, 'upper must be greater than lower'

    x = ReLU(max_value=upper - lower, name=f'{name}/relu')(x)
    x = Add(name=f'{name}/add')([x, tf.constant([[lower]], dtype=tf.float32)])

    return x
