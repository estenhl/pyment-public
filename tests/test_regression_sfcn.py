from pyment.models import RegressionSFCN


def test_regression_sfcn_initialization():
    exception = False

    try:
        RegressionSFCN()
    except Exception:
        exception = True

    assert not exception, 'Unable to instantiate regression SFCN'
