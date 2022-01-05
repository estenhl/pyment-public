import argparse
import json
import logging
import os
import numpy as np
import pandas as pd
import tensorflow as tf

from functools import reduce
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras.callbacks import LearningRateScheduler
from tensorflow.random import set_seed
from typing import Any, List

from utils import configure_environment

configure_environment()

from pyment.callbacks import Resetter
from pyment.data import load_dataset_from_jsonfile
from pyment.data.generators.async_nifti_generator import AsyncNiftiGenerator
from pyment.labels import Label
from pyment.models import get as get_model
from pyment.models import get_model_names, ModelType
from pyment.utils.decorators import json_serialize_object
from pyment.utils.learning_rate import LearningRateSchedule


logformat = '%(asctime)s - %(levelname)s - %(name)s: %(message)s'
logging.basicConfig(format=logformat, level=logging.INFO)
logger = logging.getLogger(__name__)

def fit_model(*, model: str, model_kwargs: str = '{}', training: List[str],
              validation: List[str], target: str = None, batch_size: int,
              num_threads: int, loss: str, metrics: str,
              learning_rate_schedule: Any = 1e-3, epochs: int,
              destination: str):
    set_seed(42)

    with tf.distribute.MirroredStrategy().scope():
        if os.path.isdir(destination):
            raise ValueError(f'Folder {destination} already exists')

        model_kwargs = json.loads(model_kwargs)

        model = get_model(model, **model_kwargs)

        training = [load_dataset_from_jsonfile(filename) \
                    for filename in training]
        training = reduce(lambda x, y: x + y, training)

        if target is not None:
            training.target = target

        training = training[np.where(~np.isnan(training.y))[0]]

        logger.info((f'Training on dataset with {len(training)} samples with '
                    f'y ranging from {round(np.amin(training.y), 2)} '
                    f'to {round(np.amax(training.y), 2)} '
                    f'(mean {round(np.mean(training.y), 2)})'))

        validation = [load_dataset_from_jsonfile(filename) \
                    for filename in validation]
        validation = reduce(lambda x, y: x + y, validation)

        if target is not None:
            validation.target = target

        validation = validation[np.where(~np.isnan(validation.y))[0]]

        logger.info((f'Validating on dataset with {len(validation)} samples '
                    f'with y ranging from {round(np.amin(validation.y), 2)} '
                    f'to {round(np.amax(validation.y), 2)} '
                    f'(mean {round(np.mean(validation.y), 2)})'))

        preprocessor = lambda x: x / 255.

        training_generator = AsyncNiftiGenerator(training,
                                                preprocessor=preprocessor,
                                                batch_size=batch_size,
                                                threads=num_threads,
                                                shuffle=True,
                                                infinite=True)

        validation_generator = AsyncNiftiGenerator(validation,
                                                preprocessor=preprocessor,
                                                batch_size=batch_size,
                                                threads=num_threads,
                                                shuffle=False,
                                                infinite=True)

        os.mkdir(destination)
        checkpoints = os.path.join(destination, 'checkpoints')

        callbacks = [Resetter(training_generator),
                    ModelCheckpoint(checkpoints, save_best_only=True),
                    EarlyStopping(patience=5, restore_best_weights=True)]

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
                            epochs=epochs, callbacks=callbacks)

        with open(os.path.join(destination, 'history.json'), 'w') as f:
            json.dump(json_serialize_object(history.history), f)

        model.save(os.path.join(destination, 'model'))

        training_generator.reset()
        training_generator.infinite = False
        training_ids = training_generator.dataset.ids

        logger.info((f'Predicting for {training_generator.batches} training '
                    'batches'))

        training_predictions, training_labels = \
            model.predict(training_generator, return_labels=True)

        validation_generator.reset()
        validation_generator.infinite = False
        validation_ids = validation_generator.dataset.ids

        logger.info((f'Predicting for {validation_generator.batches} '
                    'validation batches'))

        validation_predictions, \
            validation_labels = model.predict(validation_generator,
                                            return_labels=True)

        if isinstance(training.target, Label):
            label = training.target
            decoded_training_predictions = label.revert(training_predictions)
            decoded_training_labels = label.revert(training_labels)
            decoded_validation_predictions = \
                label.revert(validation_predictions)
            decoded_validation_labels = label.revert(validation_labels)

        if model.type == ModelType.REGRESSION:
            training_predictions = np.squeeze(training_predictions)
            validation_predictions = np.squeeze(validation_predictions)

            if isinstance(training.target, Label):
                decoded_training_predictions = \
                    np.squeeze(decoded_training_predictions)
                decoded_validation_predictions = \
                    np.squeeze(decoded_validation_predictions)

            training_mae = np.mean(np.abs(training_predictions - \
                                          training_labels))
            logger.info(f'Training MAE: {round(training_mae, 4)}')

            validation_mae = np.mean(np.abs(validation_predictions - \
                                            validation_labels))
            logger.info(f'Validation MAE: {round(validation_mae, 4)}')

            results = {
                'training_mae': training_mae,
                'validation_mae': validation_mae
            }

            training_df = pd.DataFrame({
                'id': training_ids,
                'label': training_labels,
                'prediction': training_predictions
            })

            validation_df = pd.DataFrame({
                'id': validation_ids,
                'label': validation_labels,
                'prediction': validation_predictions
            })

            if isinstance(training.target, Label):
                decoded_training_mae = \
                    np.mean(np.abs(decoded_training_predictions - \
                                   decoded_training_labels))
                logger.info(f'Decoded training MAE: {decoded_training_mae}')

                decoded_validation_mae = \
                    np.mean(np.abs(decoded_validation_predictions - \
                                   decoded_validation_labels))
                logger.info(f'Decoded validation MAE: {decoded_validation_mae}')

                results['decoded_training_mae'] = decoded_training_mae
                results['decoded_validation_mae'] = decoded_validation_mae

                training_df['decoded_label'] = decoded_training_labels
                training_df['decoded_prediction'] = \
                    decoded_training_predictions
                validation_df['decoded_label'] = decoded_validation_labels
                validation_df['decoded_prediction'] = \
                    decoded_validation_predictions

            with open(os.path.join(destination, 'results.json'), 'w') as f:
                json.dump(results, f)

            training_df.to_csv(os.path.join(destination,
                                            'training_predictions.csv'),
                            index=False)
            validation_df.to_csv(os.path.join(destination,
                                            'validation_predictions.csv'),
                                index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Fits a model on the given dataset')

    parser.add_argument('-m', '--model', required=True,
                        choices=get_model_names(),
                        help='Name of the model that is used')
    parser.add_argument('-mk', '--model_kwargs', required=False, default='{}',
                        help=('JSON string with keyword arguments which is '
                              'passed to the model on initialization'))
    parser.add_argument('-t', '--training', required=True, nargs='+',
                        help='Path to JSON files containing training data')
    parser.add_argument('-v', '--validation', required=True, nargs='+',
                        help='Path to JSON files containing validation data')
    parser.add_argument('-x', '--target', required=False, default=None,
                        help='Target used by the model')
    parser.add_argument('-bs', '--batch_size', required=True, type=int,
                        help=('Batch size used by the generators for training '
                              'and validation'))
    parser.add_argument('-nt', '--num_threads', required=True, type=int,
                        help=('Number of threads used for loading images '
                              'in the generators'))
    parser.add_argument('-l', '--loss', required=True,
                        help='Loss function used for optimizing the model')
    parser.add_argument('-e', '--metrics', required=False, default=None,
                        help='Metrics logged during model training')
    parser.add_argument('-lr', '--learning_rate_schedule', required=False,
                        default=1e-3, help='Learning rate used by optimizer')
    parser.add_argument('-n', '--epochs', required=True, type=int,
                        help='Number of epochs to run training for')
    parser.add_argument('-d', '--destination', required=True,
                        help='Folder where results are stored')

    args = parser.parse_args()

    fit_model(model=args.model, model_kwargs=args.model_kwargs,
              training=args.training, validation=args.validation,
              target=args.target, batch_size=args.batch_size,
              num_threads=args.num_threads, loss=args.loss,
              metrics=args.metrics,
              learning_rate_schedule=args.learning_rate_schedule,
              epochs=args.epochs, destination=args.destination)
