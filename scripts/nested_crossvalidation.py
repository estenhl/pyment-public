import argparse
import json
import logging
import os

from tensorflow.keras.models import load_model
from typing import List

from utils import configure_environment

configure_environment()

from pyment.models.utils import find_latest_weights

from crossvalidate import crossvalidate
from fit_model import fit_model


logformat = '%(asctime)s - %(levelname)s - %(name)s: %(message)s'
logging.basicConfig(format=logformat, level=logging.INFO)
logger = logging.getLogger(__name__)

def nested_crossvalidation(*, models: List[str], folds: List[str],
                           preprocessor: str = None, augmenters: List[str] = None,
                           batch_size: int, num_threads: int, loss: str, metrics: str,
                           learning_rate_schedules: List[str], epochs: int,
                           destination: str):
    if os.path.isdir(destination):
        raise ValueError(f'Folder {destination} already exists')

    os.mkdir(destination)

    for i in range(len(folds)):
        logger.info(f'Performing outer loop for fold {i+1}/{len(folds)}')

        training = [folds[j] for j in range(len(folds)) if j != i]
        testing = [folds[i]]

        fold_destination = os.path.join(destination, f'fold_{i}')
        os.mkdir(fold_destination)

        results = crossvalidate(models=models, folds=training,
                                preprocessor=preprocessor, augmenters=augmenters,
                                batch_size=batch_size, num_threads=num_threads,
                                loss=loss, metrics=metrics,
                                learning_rate_schedules=learning_rate_schedules,
                                epochs=epochs,
                                destination=os.path.join(fold_destination,
                                                         'crossvalidation'))
        best_configuration = results['best_configuration']

        with open(os.path.join(fold_destination, 'configuration.json'), 'w') as f:
            json.dump(best_configuration, f, indent=4)

        best_model = best_configuration['model']
        best_augmenter = best_configuration['augmenter']
        best_learning_rate_schedule = best_configuration['learning_rate_schedule']

        logger.info((f'Refitting best model for fold {i+1}/{len(folds)} from '
                    f'model {best_model} with augmenter {best_augmenter} '
                    'and learning rate schedule'
                    f'{best_learning_rate_schedule}'))

        fit_model(model=best_model, training=training, validation=testing,
                  preprocessor=preprocessor, augmenter=best_augmenter,
                  batch_size=batch_size, num_threads=num_threads, loss=loss,
                  metrics=metrics,
                  learning_rate_schedule=best_learning_rate_schedule,
                  epochs=epochs,
                  destination=os.path.join(fold_destination, 'best_model'))


        checkpoints = os.path.join(fold_destination, 'best_model',
                                   'checkpoints')
        weights = find_latest_weights(checkpoints)
        model = load_model(os.path.join(checkpoints, weights))
        model.save(os.path.join(fold_destination, 'trained_model'))


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

    nested_crossvalidation(models=args.models, folds=args.folds,
                           preprocessor=args.preprocessor,
                           augmenters=args.augmenters,
                           batch_size=args.batch_size,
                           num_threads=args.num_threads,
                           loss=args.loss, metrics=args.metrics,
                           learning_rate_schedules=args.learning_rate_schedules,
                           epochs=args.epochs, destination=args.destination)
