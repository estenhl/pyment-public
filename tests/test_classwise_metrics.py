import numpy as np

from pyment.metrics import ClasswiseAUC, ClasswisePrecision, ClasswiseRecall


def test_classwise_precision():
    metric = ClasswisePrecision(index=2)

    y = np.asarray([
        [0., 0., 1.],
        [0., 1., 0.],
        [1., 0., 0.],
        [0., 0., 1.],
        [0., 1., 0.],
        [1., 0., 0.],
        [0., 0., 1.],
        [0., 1., 0.],
        [1., 0., 0.],
        [0., 1., 0]
    ])

    yhat = np.asarray([
        [0., 0., 1.],
        [0., 0., 1.],
        [0., 0., 1.],
        [0., 1., 0.],
        [0., 1., 0.],
        [0., 1., 0.],
        [1., 0., 0.],
        [1., 0., 0.],
        [1., 0., 0.],
        [0., 0., 1.]
    ])

    assert 1/4 == metric(y, yhat), \
        'Classwise Precision does not return the correct value'


def test_classwise_recall():
    metric = ClasswiseRecall(index=2)

    y = np.asarray([
        [0., 0., 1.],
        [0., 1., 0.],
        [1., 0., 0.],
        [0., 0., 1.],
        [0., 1., 0.],
        [1., 0., 0.],
        [0., 0., 1.],
        [0., 1., 0.],
        [1., 0., 0.],
        [0., 0., 1.]
    ])

    yhat = np.asarray([
        [0., 0., 1.],
        [0., 0., 1.],
        [0., 0., 1.],
        [0., 1., 0.],
        [0., 1., 0.],
        [0., 1., 0.],
        [1., 0., 0.],
        [1., 0., 0.],
        [1., 0., 0.],
        [0., 1., 0.]
    ])

    assert 1/4 == metric(y, yhat), \
        'Classwise Precision does not return the correct value'

def test_classwise_auc():
    metric = ClasswiseAUC(index=2)

    y = np.asarray([
        [0., 0., 1.],
        [0., 1., 0.],
        [1., 0., 0.],
        [0., 0., 1.],
        [0., 1., 0.],
        [1., 0., 0.],
        [0., 0., 1.],
        [0., 1., 0.],
        [1., 0., 0.],
        [0., 0., 1.]
    ])

    yhat = np.asarray([
        [0., 0., 1.],
        [0., 0., 1.],
        [0., 0., 1.],
        [0., 1., 0.],
        [0., 1., 0.],
        [0., 1., 0.],
        [1., 0., 0.],
        [1., 0., 0.],
        [1., 0., 0.],
        [0., 1., 0.]
    ])

    assert np.isclose(0.45833, metric(y, yhat), atol=1e-5), \
        'Classwise AUC does not return the correct value'
