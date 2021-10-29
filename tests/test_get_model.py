from pyment.models import get, RankingSFCN, RegressionSFCN, \
                          SoftClassificationSFCN


def test_get_model_regressionsfcn():
    model = get('regressionsfcn')

    assert isinstance(model, RegressionSFCN), ('models.get(regressionsfcn) '
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

def test_get_model_soft_classification_sfcn():
    model = get('softclassificationsfcn')

    assert isinstance(model, SoftClassificationSFCN), ('models.get('
                                                       'softclassificationsfcn'
                                                       ') does not return a '
                                                       'SoftClassification'
                                                       'SFCN')

def test_get_model_sfcnsm():
    model = get('sfcn-sm')

    assert isinstance(model, SoftClassificationSFCN), ('models.get(sfcn-sm) '
                                                       'does not return a '
                                                       'SoftClassification'
                                                       'SFCN')

def test_get_model_ranking_sfcn():
    model = get('rankingsfcn')

    assert isinstance(model, RankingSFCN), ('models.get('
                                                       'rankingsfcn'
                                                       ') does not return a '
                                                       'RankingSFCN')

def test_get_model_sfcnsm():
    model = get('sfcn-rank')

    assert isinstance(model, RankingSFCN), ('models.get(sfcn-rank) '
                                                      'does not return a '
                                                      'RankingSFCN')