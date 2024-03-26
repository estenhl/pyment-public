from pyment.models import RegressionSFCN


def test_sfcn_reg_output():
    model = RegressionSFCN()
    outputs = model.layers[-1]

    assert 1 == outputs.output.shape[-1], \
        'Regression SFCN does not predict a single value per data point'
    #assert 'linear' == outputs.activation.__name__,

    print(outputs.activation.__name__)

def test_sfcn_reg_weights():
    model = RegressionSFCN(weights='brain-age-2022')
