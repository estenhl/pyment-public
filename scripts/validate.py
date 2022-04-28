"""Performs validation across a given set of models, configurations
and folds.

Example usage:
    python scripts/validate.py \
        --models /path/to/model1/folder \
                 /path/to/model2/folder \
        --trainimg /path/to/training/fold_0.json \
                /path/to/training/fold_1.json \
                /path/to/training/fold_2.json \
        --validation /path/to/validation.json \
        --preprocessor /path/to/preprocessor.json \
        --augmenters /path/to/augmenter1.json \
                     /path/to/augmenter2.json \
        --batch_size 32 \
        --num_threads 8 \
        --loss mse \
        --metrics mae \
        --learning_rate_schedules /path/to/learning_rate_schedule1.json \
                                  /path/to/learning_rate_schedule2.json \
        --epochs 5 \
        --destination /path/to/destination/folder
"""


import argparse
import json
import logging
import os
import numpy as np

from typing import Any, Dict, List

from fit_model import fit_model

from utils import configure_environment

configure_environment()

from pyment.utils.decorators import json_serialize_object


logformat = '%(asctime)s - %(levelname)s - %(name)s: %(message)s'
logging.basicConfig(format=logformat, level=logging.INFO)
logger = logging.getLogger(__name__)

def _create_configurations(models: List[str], augmenters: List[str],
                           learning_rate_schedules: List[str]) -> List[Dict[str, Any]]:
    configurations = []

    for model in models:
        for augmenter in augmenters:
            for learning_rate_schedule in learning_rate_schedules:
                configurations.append({
                    'model': model,
                    'augmenter': augmenter,
                    'learning_rate_schedule': learning_rate_schedule
                })

    return configurations


def validate(*, models: List[str], training: List[str], validation: str,
             preprocessor: str = None, augmenters: List[str] = None,
             batch_size: int, num_threads: int, loss: str, metrics: str,
             learning_rate_schedules: List[str], epochs: int,
             destination: str):
    if os.path.isdir(destination):
        raise ValueError(f'Folder {destination} already exists')

    os.mkdir(destination)

    configurations = _create_configurations(models, augmenters,
                                            learning_rate_schedules)
    logger.info((f'Found {len(configurations)} configurations yielding'))

    losses = []

    for i in range(len(configurations)):
        logger.info((f'Fitting configuration {i+1}/{len(configurations)}'))

        configuration = configurations[i]
        run_destination = os.path.join(destination, f'run_{i}')

        model = configuration['model']
        augmenter = configuration['augmenter']
        learning_rate_schedule = configuration['learning_rate_schedule']

        history = fit_model(model=model,
                            training=training,
                            validation=validation,
                            preprocessor=preprocessor,
                            augmenter=augmenter,
                            batch_size=batch_size,
                            num_threads=num_threads,
                            loss=loss,
                            metrics=metrics,
                            learning_rate_schedule=learning_rate_schedule,
                            epochs=epochs,
                            destination=run_destination)

        val_losses = history['val_loss']
        best_val_loss = np.amin(val_losses)
        losses.append(best_val_loss)

    best_loss = np.amin(losses)
    best_configuration = configurations[np.argmin(losses)]

    results = {
        'configurations': configurations,
        'losses': losses,
        'best_loss': best_loss,
        'best_configuration': best_configuration,
        'best_run': np.argmin(losses)
    }

    with open(os.path.join(destination, 'results.json'), 'w') as f:
        json.dump(json_serialize_object(results), f, indent=4)

    return results

if __name__ == '__main__':
    parser = argparse.ArgumentParser(('Runs validation across a set of '
                                      'given models and data folds.'))

    parser.add_argument('-m', '--models', required=True, nargs='+',
                        help='Paths to models')
    parser.add_argument('-t', '--training', required=True, nargs='+',
                        help='Path to training data')
    parser.add_argument('-v', '--validation', required=True, nargs='+',
                        help='Path to validation data')
    parser.add_argument('-p', '--preprocessor', required=False, default=None,
                        help='Optional json-file containing a preprocessor')
    parser.add_argument('-a', '--augmenters', required=False, nargs='+',
                        help=('(Optional) json-files containing augmenters. '
                              'If multiple are given, these will be '
                              'validated over'))
    parser.add_argument('-bs', '--batch_size', required=True, type=int,
                        help='Batch size used by the models')
    parser.add_argument('-nt', '--num_threads', required=True, type=int,
                        help=('Number of threads used for loading images in '
                              'the generators'))
    parser.add_argument('-l', '--loss', required=True,
                        help='Loss function used for optimizing the model')
    parser.add_argument('-e', '--metrics', required=False, default=[],
                        nargs='+',
                        help='Metrics logged during model training')
    parser.add_argument('-lr', '--learning_rate_schedules', required=True,
                        nargs='+',
                        help=('Learning rate used by optimizers. If multiple '
                              'are given, these are validated over'))
    parser.add_argument('-n', '--epochs', required=True, type=int,
                        help='Number of epochs to run training for')
    parser.add_argument('-d', '--destination', required=True,
                        help='Folder where results are stored')

    args = parser.parse_args()

    validate(models=args.models,
             training=args.training,
             validation=args.validation,
             preprocessor=args.preprocessor,
             augmenters=args.augmenters,
             batch_size=args.batch_size,
             num_threads=args.num_threads,
             loss=args.loss,
             metrics=args.metrics,
             learning_rate_schedules=args.learning_rate_schedules,
             epochs=args.epochs,
             destination=args.destination)
