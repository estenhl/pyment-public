import logging

import tensorflow as tf

from ..sfcn import MultiTaskSFCN


logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s: %(message)s',
    level=logging.DEBUG
)
logger = logging.getLogger(__name__)

def load_select_pretrained_weights(
    model: tf.keras.Model, 
    weights: str, 
    target: str = None
) -> tf.keras.Model:
    logger.info('Loading pretrained weights from %s', weights)

    backbone = MultiTaskSFCN(input_shape=(224, 192, 224), pooling='max')
    checkpoint = tf.train.Checkpoint(backbone)

    checkpoint.restore(weights).expect_partial()

    conv_layers = [2, 6, 10, 14, 18, 22]
    norm_layers = [3, 7, 11, 15, 19, 23]

    for idx in conv_layers + norm_layers:
        model.layers[idx].set_weights(backbone.layers[idx].get_weights())

    # Loading weights from the specific dense-layer corresponding to the 
    # given prediction-task in the multi-task model
    if target == 'age':
        logger.info('Loaded age weights for the prediction head')
        model.layers[27].set_weights(backbone.layers[27].get_weights())
    elif target == 'sex':
        logger.info('Loaded sex weights for the prediction head')
        model.layers[27].set_weights(backbone.layers[28].get_weights())
    else:
        logger.warning(
            'Unknown target %s. Not loading weights for prediction layer',
            target
        )

    return model