import numpy as np
import tensorflow as tf

from tensorflow.keras.layers import Add, Dense, Dropout, Embedding, Flatten, \
                                    Input, Layer, \
                                    LayerNormalization, MultiHeadAttention, \
                                    Reshape, ZeroPadding3D
from typing import Any, Callable, List, Tuple

from .model import Model


class VisionTransformer(Model):
    def __init__(self, *, input_shape: Tuple[int], depth: int, dimensions: int,
                 mlp_dimensions: List[int], patch_size: int, heads: int,
                 prediction_head: Callable[[tf.Tensor], tf.Tensor],
                 intermediate_dropout: float = .0, dropout: float = .0,
                 name: str = 'VisionTransformer'):

            if isinstance(input_shape, list):
                input_shape = tuple(input_shape)

            num_patches = int(np.prod(np.ceil(np.asarray(input_shape) /
                                              patch_size)))

            inputs = Input(input_shape, name=f'{name}/inputs')
            x = inputs

            x = Reshape(input_shape + (1,), name=f'{name}/expand_dims')(x)

            rests = np.asarray(input_shape) % patch_size

            # If the image size is not divisible by the patch size, voxels
            # are dropped by the patch layer. Thus, we pad the image to
            # an appropriate size
            if np.sum(rests) != 0:
                padding = patch_size - rests
                first = np.ceil(padding / 2)
                last = np.floor(padding / 2)
                padding = tuple(zip(first.astype(int), last.astype(int)))
                x = ZeroPadding3D(padding, name=f'{name}/padding')(x)


            x = Patches(patch_size, name=f'{name}/patches')(x)
            x = PatchEncoder(num_patches=num_patches, dimensions=dimensions,
                             name=f'{name}/patch_encoder')(x)

            for i in range(depth):
                skip = x
                x = LayerNormalization(name=f'{name}/block_{i+1}/norm1')(x)

                attention = MultiHeadAttention(
                    num_heads=heads,
                    key_dim=dimensions,
                    dropout=intermediate_dropout,
                    name=f'{name}/block_{i+1}/attention'
                )(x, x)

                x = Add(name=f'{name}/block_{i+1}/add_1')([attention, skip])
                skip = x
                x = LayerNormalization(name=f'{name}/block_{i+1}/norm2')(x)

                for j in range(len(mlp_dimensions)):
                    x = Dense(mlp_dimensions[j],
                              name=f'{name}/block_{i+1}/dense_{j+1}')(x)
                    x = Dropout(intermediate_dropout,
                                name=f'{name}/block_{i+1}/dropout_{j+1}')(x)

                x = Add(name=f'{name}/block_{i+1}/add_2')([x, skip])

            x = LayerNormalization(name=f'{name}/head/norm')(x)
            x = Flatten(name=f'{name}/head/flatten')(x)
            x = Dropout(dropout, name=f'{name}/head/dropout')(x)

            x = prediction_head(x)

            super().__init__(inputs, x)

class Patches(Layer):
    def __init__(self, patch_size: int, name: str):
        super().__init__(name=name)

        self.patch_size = patch_size

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.extract_volume_patches(
            images,
            ksizes=[1, self.patch_size, self.patch_size, self.patch_size,1],
            strides=[1, self.patch_size, self.patch_size, self.patch_size,1],
            padding='VALID',
            name=f'{self.name}/extract_volume_patches'
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims],
                             name=f'{self.name}/reshape')
        return patches

class PatchEncoder(Layer):
    def __init__(self, num_patches: int, dimensions: int, name: str):
        super().__init__(name=name)

        self.num_patches = num_patches
        self.projection = Dense(dimensions,
                                name=f'{self.name}/projection')
        self.position_embedding = Embedding(input_dim=num_patches,
                                            output_dim=dimensions,
                                            name=f'{self.name}/embedding')

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1,
                             name=f'{self.name}/positions')
        print(self.num_patches)
        print(self.projection(patch).shape)
        print(self.position_embedding(positions).shape)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded

class RegressionVisionTransformerMini(VisionTransformer):
    def __init__(self, *, input_shape: Tuple[int], dropout: float = .0,
                 activation: Any = tf.nn.gelu, name: str = 'RegressionVisionTransformerMini', **kwargs):
        def prediction_head(x: tf.Tensor):
            x = Dense(256, activation=activation,
                      name=f'{name}/head/dense_1')(x)
            x = Dropout(dropout, name=f'{name}/head/dropout_1')(x)
            x = Dense(64, activation=activation,
                      name=f'{name}/head/dense_2')(x)
            x = Dropout(dropout, name=f'{name}/dropout_2')(x)
            x = Dense(1, name=f'{name}/head/prediction')(x)

            return x

        super().__init__(input_shape=input_shape, depth=12,
                         dimensions=128, mlp_dimensions=[256, 128],
                         heads=12, patch_size=7,
                         prediction_head=prediction_head)
