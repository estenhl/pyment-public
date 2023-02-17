"""Configures a cyclical LearningRateSchedule

Example usage:
    python scripts/configure_cyclical_learning_rate.py \
        --minimum 1e-4 \
        --maximum 1e-1 \
        --period 33 \
        --factor 0.5 \
        --cutoff 100 \
        --destination /path/to/destination.json \
        --visualize
"""

import argparse

from utils import configure_environment

configure_environment()

from pyment.utils.learning_rate import CyclicalLearningRateSchedule


def configure_cyclical_learning_rate(*, minimum: float, maximum: float,
                                     period: float, cutoff: float,
                                     factor: float = None, destination: str,
                                     visualize: bool = False,) -> CyclicalLearningRateSchedule:
    schedule = CyclicalLearningRateSchedule(minimum=minimum, maximum=maximum,
                                            period=period, cutoff=cutoff,
                                            factor=factor)

    schedule.save(destination)

    if visualize:
        import matplotlib.pyplot as plt
        import numpy as np

        epochs = np.arange(cutoff + 5)
        lrs = [schedule(epoch) for epoch in epochs]

        plt.plot(epochs, lrs)
        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(('Configures and saves a cyclical '
                                      'LearningRateSchedule'))

    parser.add_argument('-min', '--minimum', required=True, type=float,
                        help='Minimum learning rate')
    parser.add_argument('-max', '--maximum', required=True, type=float,
                        help='Maximum learning rate')
    parser.add_argument('-p', '--period', required=True, type=float,
                        help='Period between two peaks of the cycle')
    parser.add_argument('-c', '--cutoff', required=True, type=float,
                        help=('Cutoff point, after which the minimum learning '
                              'rate is used'))
    parser.add_argument('-f', '--factor', required=False, default=None,
                        type=float,
                        help=('Optional argument, which if used scales the '
                              'learning rates between the peaks of the cycle '
                              'by this factor'))
    parser.add_argument('-d', '--destination', required=True,
                        help='Path where schedule is stored')
    parser.add_argument('-v', '--visualize', action='store_true',
                        help='If set, plots the learning rate')

    args = parser.parse_args()

    configure_cyclical_learning_rate(minimum=args.minimum,
                                     maximum=args.maximum,
                                     period=args.period,
                                     cutoff=args.cutoff,
                                     factor=args.factor,
                                     destination=args.destination,
                                     visualize=args.visualize)

