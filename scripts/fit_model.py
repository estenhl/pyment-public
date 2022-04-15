import argparse
import json
import logging
import os
import re
import numpy as np
import pandas as pd
import tensorflow as tf

from functools import reduce
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras.callbacks import LearningRateScheduler
from tensorflow.random import set_seed
from typing import Any, List

from utils import configure_environment

configure_environment()

from pyment.callbacks import Resetter
from pyment.data import load_dataset_from_jsonfile
from pyment.data.augmenters import NiftiAugmenter
from pyment.data.generators import AsyncNiftiGenerator, \
    SingleDomainAsyncNiftiGenerator
from pyment.data.preprocessors import NiftiPreprocessor
from pyment.utils.decorators import json_serialize_object
from pyment.utils.learning_rate import LearningRateSchedule


logformat = '%(asctime)s - %(levelname)s - %(name)s: %(message)s'
logging.basicConfig(format=logformat, level=logging.INFO)
logger = logging.getLogger(__name__)

def fit_model(model: str, *, training: List[str], validation: List[str],
              preprocessor: str, augmenter: str, batch_size: int,
              num_threads: int, loss: str, metrics: str,
              learning_rate_schedule: Any = 1e-3, epochs: int,
              domain: str, destination: str):
    set_seed(42)

    with tf.distribute.MirroredStrategy().scope():
        checkpoints = os.path.join(destination, 'checkpoints')

        if os.path.isdir(destination):
            if not model.startswith(destination):
                raise ValueError(f'Folder {destination} already exists')
        else:
            os.mkdir(destination)
            os.mkdir(checkpoints)

        initial_epoch = 0
        match = re.fullmatch('epoch=(\d+).*', model)

        if match:
            initial_epoch = int(match.groups(0)[0])

        model = load_model(model)

        training = [load_dataset_from_jsonfile(filename) \
                    for filename in training]
        training = reduce(lambda x, y: x + y, training)

        training = training[np.where(~np.isnan(training.y))[0]]

        logger.info((f'Training on dataset with {len(training)} samples with '
                    f'y ranging from {round(np.amin(training.y), 2)} '
                    f'to {round(np.amax(training.y), 2)} '
                    f'(mean {round(np.mean(training.y), 2)})'))

        validation = [load_dataset_from_jsonfile(filename) \
                    for filename in validation]
        validation = reduce(lambda x, y: x + y, validation)

        validation = validation[np.where(~np.isnan(validation.y))[0]]

        logger.info((f'Validating on dataset with {len(validation)} samples '
                    f'with y ranging from {round(np.amin(validation.y), 2)} '
                    f'to {round(np.amax(validation.y), 2)} '
                    f'(mean {round(np.mean(validation.y), 2)})'))

        if preprocessor is not None:
            preprocessor = NiftiPreprocessor.from_file(preprocessor)

        if augmenter is not None:
            augmenter = NiftiAugmenter.from_file(augmenter)

        if domain is None:
            training_generator = AsyncNiftiGenerator(
                training,
                preprocessor=preprocessor,
                augmenter=augmenter,
                batch_size=batch_size,
                threads=num_threads,
                shuffle=True,
                infinite=True
            )

            validation_generator = AsyncNiftiGenerator(
                validation,
                preprocessor=preprocessor,
                batch_size=batch_size,
                threads=num_threads,
                shuffle=False,
                infinite=True
            )
        else:
            training_generator = SingleDomainAsyncNiftiGenerator(
                training,
                domain=domain,
                preprocessor=preprocessor,
                augmenter=augmenter,
                batch_size=batch_size,
                threads=num_threads,
                shuffle=True,
                infinite=True
            )

            validation_generator = SingleDomainAsyncNiftiGenerator(
                validation,
                domain=domain,
                preprocessor=preprocessor,
                batch_size=batch_size,
                threads=num_threads,
                shuffle=False,
                infinite=True
            )

        checkpoints = os.path.join(checkpoints,
                                   'epoch={epoch:d}-' + \
                                   'loss={loss:.3f}-' + \
                                   '-'.join([metric + '={' + f'{metric}:.3f' + '}' \
                                             for metric in metrics]) + '-' + \
                                   'val_loss={val_loss:.3f}-' + \
                                   '-'.join(['val_' + metric + '={' + f'val_{metric}:.3f' + '}' \
                                             for metric in metrics]))

        callbacks = [Resetter(training_generator),
                     ModelCheckpoint(checkpoints, save_best_only=True)]

        learning_rate = learning_rate_schedule

        if isinstance(learning_rate, str):
            try:
                learning_rate = float(learning_rate)
            except Exception:
                if os.path.isfile(learning_rate):
                    schedule = \
                        LearningRateSchedule.from_jsonfile(learning_rate)
                    callbacks.append(LearningRateScheduler(schedule))
                    learning_rate = schedule(0)

        model.compile(loss=loss, metrics=metrics,
                      optimizer=Adam(learning_rate))

        history = model.fit(training_generator,
                            validation_data=validation_generator,
                            steps_per_epoch=training_generator.batches,
                            validation_steps=validation_generator.batches,
                            epochs=epochs, callbacks=callbacks,
                            initial_epoch=initial_epoch)
        history = json_serialize_object(history.history)

        with open(os.path.join(destination, 'history.json'), 'w') as f:
            json.dump(history, f, indent=4)

    return history

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Fits a model on the given dataset')

    parser.add_argument('-m', '--model', required=True,
                        help='Path to the folder where the model is stored')
    parser.add_argument('-t', '--training', required=True, nargs='+',
                        help='Path to JSON files containing training data')
    parser.add_argument('-v', '--validation', required=True, nargs='+',
                        help='Path to JSON files containing validation data')
    parser.add_argument('-p', '--preprocessor', required=False, default=None,
                        help='Optional json-file containing a preprocessor')
    parser.add_argument('-a', '--augmenter', required=False, default=None,
                        help='Optional json-file containing an augmenter')
    parser.add_argument('-bs', '--batch_size', required=True, type=int,
                        help=('Batch size used by the generators for training '
                              'and validation'))
    parser.add_argument('-nt', '--num_threads', required=True, type=int,
                        help=('Number of threads used for loading images '
                              'in the generators'))
    parser.add_argument('-l', '--loss', required=True,
                        help='Loss function used for optimizing the model')
    parser.add_argument('-e', '--metrics', required=False, default=[],
                        nargs='+',
                        help='Metrics logged during model training')
    parser.add_argument('-lr', '--learning_rate_schedule', required=False,
                        default=1e-3, help='Learning rate used by optimizer')
    parser.add_argument('-n', '--epochs', required=True, type=int,
                        help='Number of epochs to run training for')
    parser.add_argument('-o', '--domain', required=False, default=None,
                        help=('Optional variable encoding domain. If used, '
                              'this variable is passed as input to the model '
                              'together with the regular images'))
    parser.add_argument('-d', '--destination', required=True,
                        help='Folder where results are stored')

    args = parser.parse_args()

    fit_model(args.model, training=args.training, validation=args.validation,
              preprocessor=args.preprocessor, augmenter=args.augmenter,
              batch_size=args.batch_size, num_threads=args.num_threads,
              loss=args.loss, metrics=args.metrics,
              learning_rate_schedule=args.learning_rate_schedule,
              epochs=args.epochs, domain=args.domain,
              destination=args.destination)
