from pyment.models import RegressionVGG11


def test_regression_vgg11_initialization():
    exception = False

    try:
        RegressionVGG11()
    except Exception:
        exception = True

    assert not exception, 'Unable to instantiate VGG11'
