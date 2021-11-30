from pyment.models import RankingSFCN, RegressionSFCN, \
                          SoftClassificationSFCN


def test_has_ranking_weights():
    try:
        RankingSFCN(weights='brain-age')
    except Exception:
        assert False, 'Unable to load pretrained RankingSFCN for brain age'

def test_has_ranking_weights_no_top():
    try:
        RankingSFCN(weights='brain-age')
    except Exception:
        assert False, \
            ('Unable to load pretrained RankingSFCN for brain age without top '
             'layers')

def test_has_regression_weights():
    try:
        RegressionSFCN(weights='brain-age')
    except Exception:
        assert False, 'Unable to load pretrained RegressionSFCN for brain age'

def test_has_regression_weights_no_top():
    try:
        RegressionSFCN(weights='brain-age')
    except Exception:
        assert False, \
            ('Unable to load pretrained RegressionSFCN for brain age without '
             'top layers')

def test_has_soft_classification_brain_age_weights():
    try:
        SoftClassificationSFCN(weights='brain-age')
    except Exception:
        assert False, \
            'Unable to load pretrained SoftClassificationSFCN for brain age'

def test_has_soft_classification_brain_age_weights_no_top():
    try:
        SoftClassificationSFCN(weights='brain-age')
    except Exception:
        assert False, \
            ('Unable to load pretrained SoftClassificationSFCN for brain age '
             'without top layers')