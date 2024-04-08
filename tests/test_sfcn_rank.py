from tensorflow.keras.layers import Dense

from pyment.models import RankingSFCN


def test_sfcn_bin_output():
    prediction_range = (0, 10)
    model = RankingSFCN(prediction_range=prediction_range)

    assert isinstance(model.layers[-1], Dense), \
        'Ranking SFCN does not have a fully connected prediction layer'

    bins = prediction_range[1] - prediction_range[0] - 1
    assert bins == model.output.shape[-1], \
        'Ranking SFCN does not have an output neuron per age (-1)'

    assert 'sigmoid' == model.layers[-1].activation.__name__, \
        'Ranking SFCN does not have a sigmoid activation on its output'

