from tensorflow.keras.layers import Conv3D, GlobalAveragePooling3D, GlobalMaxPooling3D

from pyment.models import SFCN


def test_base_sfcn_number_of_convolutions():
    """ Tests that the base SFCN has the right number of convolutional
    layers. """

    model = SFCN()
    layers = model.layers
    conv_layers = [layer for layer in layers if isinstance(layer, Conv3D)]

    assert 6 == len(conv_layers), \
        'Base SFCN does not have the correct number of conv layers'

def test_base_sfcn_handles_missing_channel():
    """ Tests that the base SFCN adds a channel-dimension if this is
    missing from the specified input_shape. """

    try:
        model = SFCN(input_shape=(168, 212, 160))
    except Exception:
        assert False, \
            'Base SFCN does not handle missing channel dimension'

def test_base_sfcn_global_avg_pool():
    model = SFCN(pooling='avg')

    assert isinstance(model.layers[-1], GlobalAveragePooling3D), \
        ('Base SFCN with pooling=\'avg\' does not result in a global '
         'average pooling layer')

def test_base_sfcn_global_max_pool():
    model = SFCN(pooling='max')

    assert isinstance(model.layers[-1], GlobalMaxPooling3D), \
        ('Base SFCN with pooling=\'avg\' does not result in a global '
         'average pooling layer')
