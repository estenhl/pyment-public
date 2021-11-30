from pyment.models.model import Model

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from typing import List


def gaussian_blur(img: tf.Tensor, kernel_size: int = 3, 
                  sigma: float = 1.) -> tf.Tensor:
    def gauss_kernel(channels: List[int], kernel_size: int, 
                     sigma: float) -> tf.Tensor:
        ax = tf.range(-kernel_size // 2 + 1.0, kernel_size // 2 + 1.0)
        xx, yy = tf.meshgrid(ax, ax)
        kernel = tf.exp(-(xx ** 2 + yy ** 2) / (2.0 * sigma ** 2))
        kernel = kernel / tf.reduce_sum(kernel)
        kernel = tf.tile(kernel[..., tf.newaxis], [1, 1, channels])

        return kernel

    gaussian_kernel = gauss_kernel(tf.shape(img)[-1], kernel_size, sigma)
    gaussian_kernel = gaussian_kernel[..., tf.newaxis]

    return tf.nn.depthwise_conv2d(img, gaussian_kernel, [1, 1, 1, 1],
                                  padding='SAME', data_format='NHWC')


def maximize_feature_activation(model: Model, *, layer: str, index: int,
                                iterations: int = 100, step_size: float = 10,
                                initial: np.ndarray = None, 
                                l2_decay: float = None,
                                blur_every: int = None,
                                blur_width: float = None,
                                norm_threshold: float = None,
                                contribution_threshold: float = None) -> np.ndarray:
    @tf.function
    def _loss(inputs: tf.Tensor) -> tf.Tensor:
        features = extractor(inputs)
        feature = features[..., index]

        return tf.reduce_mean(feature)

    @tf.function
    def _gradient_ascent(inputs: tf.Tensor):
        with tf.GradientTape() as tape:
            tape.watch(inputs)
            l = _loss(inputs)

        gradients = tape.gradient(l, inputs)
        gradients = tf.math.l2_normalize(gradients)

        return l, gradients 
    
    layer = model.get_layer(layer)
    extractor = Model(model.inputs, layer.output)

    if initial is None:
        raise NotImplementedError()

    img = tf.Variable(np.expand_dims(initial, axis=0))

    for i in range(iterations):
        loss, gradients = _gradient_ascent(img)
        img = img + gradients * step_size


        if l2_decay is not None and l2_decay > 0:
            img = (1 - l2_decay) * img

        if blur_every is not None and i % blur_every == 0:
            if blur_width is None:
                raise ValueError(('If blur_every is given, a blur_width is '
                                  'also necessary'))
    
            img = gaussian_blur(img, kernel_size=blur_width)

        if norm_threshold is not None and norm_threshold > 0:
            for channel in range(img.shape[-1]):
                values = img[...,channel]
                threshold = tfp.stats.percentile(values, norm_threshold)
                img = tf.maximum(img, threshold)

        if contribution_threshold is not None and contribution_threshold > 0:
            contributions = tf.math.multiply(img, gradients)
            #contributions = tf.math.reduce_sum(contributions, axis=-1)
            threshold = tfp.stats.percentile(contributions, 
                                             contribution_threshold)
            zeroes = tf.zeros_like(img)
            img = tf.where(contributions < threshold, zeroes, img)

    return img.numpy()


