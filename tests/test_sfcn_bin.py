from tensorflow.keras.layers import Dense

from pyment.models import BinarySFCN


def test_sfcn_bin_output():
    model = BinarySFCN()

    assert isinstance(model.layers[-1], Dense), \
        'Binary SFCN does not have a fully connected prediction layer'

    assert 1 == model.output.shape[-1], \
        'Binary SFCN does not have a single output neuron'

    assert 'sigmoid' == model.layers[-1].activation.__name__, \
        'Binary SFCN does not have a sigmoid activation on its output'

