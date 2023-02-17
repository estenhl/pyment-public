"""Configures a stepwise LearningRateSchedule

Example usage:
    python scripts/configure_stepwise_learning_rate.py \
        --steps 0=0.1 20=0.01 40=1e-3 \
        --destination /path/to/destination.json

"""

import argparse

from typing import List, Tuple

from utils import configure_environment

configure_environment()

from pyment.utils.learning_rate import LearningRateSchedule


def _parse_step(step: str) -> Tuple[int, float]:
    assert '=' in step, \
        ('Unable to parse step where epoch and learning rate are not '
        'separated by \'=\'')

    tokens = step.split('=')

    try:
        epoch = int(tokens[0])
    except Exception:
        raise ValueError(f'Unable to parse {tokens[0]} as epoch')

    try:
        lr = float(tokens[1])
    except Exception:
        raise ValueError(f'Unable to parse {tokens[1]} as learning rate')

    return epoch, lr


def _parse_schedule(steps: List[str]):
    steps = [_parse_step(step) for step in steps]
    steps = {step[0]: step[1] for step in steps}

    return steps


def configure_stepwise_learning_rate(steps: List[str],
                                     destination: str) -> LearningRateSchedule:
    schedule = _parse_schedule(steps)
    schedule = LearningRateSchedule(schedule)

    schedule.save(destination)

    return schedule


if __name__ == '__main__':
    parser = argparse.ArgumentParser(('Configures and saves a learning rate '
                                      'schedule'))

    parser.add_argument('-s', '--steps', required=True, nargs='+',
                        help=('List of steps for the schedule. Each step '
                              'should be on the form '
                              '<epoch>=<learning_rate>. Must contain an entry '
                              'for epoch 0.'))
    parser.add_argument('-d', '--destination', required=True,
                        help='Path where schedule is stored')

    args = parser.parse_args()

    configure_stepwise_learning_rate(steps=args.steps,
                                     destination=args.destination)
