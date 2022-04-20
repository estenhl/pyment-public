import numpy as np

from pyment.utils.learning_rate import CyclicalLearningRateSchedule

def test_cyclical_learning_rate():
    minimum = 1e-4
    maximum = 1e-2
    cutoff = 100

    lr = CyclicalLearningRateSchedule(
        minimum=minimum,
        maximum=maximum,
        period=100/3,
        cutoff=cutoff
    )

    lrs = [lr(epoch) for epoch in range(100)]
    after_cutoff = [lr(epoch) for epoch in np.arange(cutoff, cutoff + 5)]

    assert np.amax(lrs) == maximum, \
        'CyclicalLearningRate does not have the correct max learning rate'
    assert np.amin(lrs) == minimum, \
        'CyclicalLearningRate does not have the correct min learning rate'
    assert np.all(np.asarray(after_cutoff) == minimum), \
        ('CyclicalLearningRate does not return the minimum learning rate '
         'after the cutoff')
