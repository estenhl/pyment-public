from pyment.models import get, RegressionSFCN


def test_get_model_regressionsfcn():
    model = get('regressionsfcn')

    assert isinstance(model, RegressionSFCN), ('models.get(regression-sfcn) '
                                               'does not return a '
                                               'RegressionSFCN')


def test_get_model_sfcnreg():
    model = get('sfcn-reg')

    assert isinstance(model, RegressionSFCN), ('models.get(sfcn-reg) does not '
                                               'return a RegressionSFCN')

def test_get_model_sfcnreg_kwargs():
    model = get('sfcn-reg', name='Test')
    
    assert 'Test' == model.name, ('models.get does not pass on keyword '
                                  'arguments')