# python scripts/crossvalidate.py -m ~/projects/crossvalidation/models/dropout\=0.5-weight_decay\=1e-3/ ~/projects/crossvalidation/models/dropout\=0.5-weight_decay\=1e-4 -f ~/projects/crossvalidation/data/fold_0.json ~/projects/crossvalidation/data/fold_1.json ~/projects/crossvalidation/data/fold_2.json -p ~/projects/crossvalidation/preprocessor.json -a ~/projects/crossvalidation/augmenter.json -b 32 -nt 8 -l mse -e mse -lr ~/projects/crossvalidation/cyclical_learning_rate.json ~/projects/crossvalidation/stepwise_learning_rate_schedule.json -n 5 -d ~/projects/crossvalidation/crossvalidation
"""Performs crossvalidation across a given set of models, configurations
and folds.

Example usage:
    python scripts/crossvalidate.py \
        --models /path/to/model1/folder \
                 /path/to/model2/folder \
        --folds /path/to/fold_0.json \
                /path/to/fold_1.json \
                /path/to/fold_2.json \
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


def crossvalidate(*, models: List[str], folds: List[str],
                  preprocessor: str = None, augmenters: List[str] = None,
                  batch_size: int, num_threads: int, loss: str, metrics: str,
                  learning_rate_schedules: List[str], epochs: int,
                  destination: str):
    if os.path.isdir(destination):
        raise ValueError(f'Folder {destination} already exists')

    os.mkdir(destination)

    configurations = _create_configurations(models, augmenters,
                                            learning_rate_schedules)
    logger.info((f'Found {len(configurations)} configurations yielding '
                 f'{len(configurations) * len(folds)} total runs'))

    results = []

    for i in range(len(folds)):
        training = [folds[j] for j in range(len(folds)) if j != i]
        validation = [folds[i]]

        fold_destination = os.path.join(destination, f'fold_{i}')
        os.mkdir(fold_destination)

        fold_results = []

        for j in range(len(configurations)):
            configuration = configurations[j]
            run_destination = os.path.join(fold_destination, f'run_{j}')

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
            fold_results.append(best_val_loss)

        best_loss = np.amin(fold_results)
        best_configuration = configurations[np.argmin(fold_results)]

        fold_results = {
            'configurations': configurations,
            'results': fold_results,
            'best_loss': best_loss,
            'best_configuration': best_configuration
        }

        with open(os.path.join(fold_destination, 'results.json'), 'w') as f:
            json.dump(fold_results, f)

        results.append(fold_results)

    results = np.asarray([result['results'] for result in results])
    results = results.T

    means = np.mean(results, axis=-1)
    best_mean = np.amin(means)
    best_configuration = configurations[np.argmin(means)]

    results = {
        'configurations': configurations,
        'results': results,
        'means': means,
        'best_mean': best_mean,
        'best_configuration': best_configuration
    }

    with open(os.path.join(destination, 'results.json'), 'w') as f:
        json.dump(json_serialize_object(results), f)

    return results

if __name__ == '__main__':
    parser = argparse.ArgumentParser(('Runs crossvalidation across a set of '
                                      'given models and data folds.'))

    parser.add_argument('-m', '--models', required=True, nargs='+',
                        help='Paths to models')
    parser.add_argument('-f', '--folds', required=True, nargs='+',
                        help='Path to folds')
    parser.add_argument('-p', '--preprocessor', required=False, default=None,
                        help='Optional json-file containing a preprocessor')
    parser.add_argument('-a', '--augmenters', required=False, nargs='+',
                        help=('(Optional) json-files containing augmenters. '
                              'If multiple are given, these will be '
                              'cross-validated over'))
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
                              'are given, these are cross-validated over'))
    parser.add_argument('-n', '--epochs', required=True, type=int,
                        help='Number of epochs to run training for')
    parser.add_argument('-d', '--destination', required=True,
                        help='Folder where results are stored')

    args = parser.parse_args()

    crossvalidate(models=args.models, folds=args.folds,
                  preprocessor=args.preprocessor, augmenters=args.augmenters,
                  batch_size=args.batch_size, num_threads=args.num_threads,
                  loss=args.loss, metrics=args.metrics,
                  learning_rate_schedules=args.learning_rate_schedules,
                  epochs=args.epochs, destination=args.destination)
