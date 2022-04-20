import numpy as np

from .learning_rate_schedule import LearningRateSchedule


class CyclicalLearningRateSchedule(LearningRateSchedule):
    def __init__(self, minimum: float, maximum: float, period: float,
                 cutoff: float, factor: float = None):

        def _compute(epoch: int):
            lr = np.sin((epoch * (2 * np.pi)) / period - np.pi / 2)
            lr += 1
            lr /= 2

            if factor:
                lr = lr * factor ** ((epoch  - period / 2) / period)

            lr *= (maximum - minimum)
            lr += minimum

            return lr

        epochs = np.arange(cutoff).astype(int)
        lrs = [_compute(epoch) for epoch in epochs]

        steps = {epochs[i]: lrs[i] for i in range(len(lrs))}
        steps[int(cutoff)] = minimum

        super().__init__(steps)
