"""Finetune an SFCN model from a JSON configuration."""

import argparse
import json
import logging
import os
from typing import Any

import pandas as pd
import tensorflow as tf

from pyment.configurations import (
    CategoricalTargetConfiguration,
    FinetuningConfiguration,
    compile_target_encoder,
)
from pyment.models.sfcn import MultiTaskSFCN

logger = logging.getLogger(__name__)


def _resolve_optimizer(optimizer: str) -> tf.keras.optimizers.Optimizer:
    _optimizers = {
        'adam': tf.keras.optimizers.Adam,
        'sgd': tf.keras.optimizers.SGD,
        'rmsprop': tf.keras.optimizers.RMSprop,
    }

    if not isinstance(optimizer, str):
        raise ValueError('Optimizer name must be a string')

    if optimizer not in _optimizers:
        raise KeyError(f'Unknown optimizer name {optimizer}')

    return _optimizers[optimizer]()


def finetune_from_configuration(raw_configuration: dict[str, Any]) -> None:
    """Finestunes a multi-task model towards a new task from a raw
     training-configuration dict.

    Validates ``raw_configuration`` against ``FinetuningConfiguration``,
    builds dataset and model, fits, then writes ``model/``,
    ``history.json``, and ``predictions.csv`` under
    ``configuration.destination``.

    Parameters
    ----------
    raw_configuration : dict[str, Any]
        A dict that conforms to the ``FinetuningConfiguration`` schema.
    """

    configuration = FinetuningConfiguration.model_validate(raw_configuration)

    os.makedirs(configuration.destination, exist_ok=True)

    dataset = configuration.dataset.build(
        target=configuration.target.name,
        target_encoder=compile_target_encoder(configuration.target),
    )
    train, validation = configuration.data_split.split(dataset)

    model = configuration.model.build()

    backbone = MultiTaskSFCN(weights=configuration.pretrained_multitask_weights)
    backbone.transfer_weights_to_single_task_model(
        model, target=configuration.target.name
    )

    model.compile(
        loss=configuration.loss,
        metrics=configuration.metrics,
        optimizer=_resolve_optimizer(configuration.optimizer),
    )

    train_generator = train.to_tensorflow_generator(
        batch_size=configuration.batch_size,
        num_threads=configuration.num_threads,
        shuffle=True,
    )
    validation_generator = validation.to_tensorflow_generator(
        batch_size=configuration.batch_size,
        num_threads=configuration.num_threads,
    )

    history = model.fit(
        train_generator,
        validation_data=validation_generator,
        epochs=configuration.epochs,
    )

    model.save(os.path.join(configuration.destination, 'model.keras'))

    history_serializable = {
        k: [float(v) for v in vals] for k, vals in history.history.items()
    }
    with open(
        os.path.join(configuration.destination, 'history.json'), 'w'
    ) as f:
        json.dump(history_serializable, f, indent=4)

    subsets = [('train', train), ('validation', validation)]

    if isinstance(configuration.target, CategoricalTargetConfiguration):
        prediction_columns = [
            f'prediction_{label}' for label in configuration.target.labels
        ]
    else:
        prediction_columns = ['prediction']

    predictions = pd.DataFrame(
        [], columns=['image_id', 'ground_truth', 'subset'] + prediction_columns
    )
    for subset_name, subset in subsets:
        pred_generator = subset.to_tensorflow_generator(
            batch_size=configuration.batch_size,
            num_threads=configuration.num_threads,
        )

        subset_predictions = pd.DataFrame(
            {
                'image_id': subset.labels['image_id'].values,
                'ground_truth': subset.labels[configuration.target.name].values,
                'subset': subset_name,
            }
        )

        if isinstance(configuration.target, CategoricalTargetConfiguration):
            subset_predictions[prediction_columns] = model.predict(
                pred_generator
            )
        else:
            subset_predictions['prediction'] = model.predict(
                pred_generator
            ).flatten()

        predictions = pd.concat([predictions, subset_predictions])

    predictions.to_csv(
        os.path.join(configuration.destination, 'predictions.csv'), index=False
    )


def main() -> None:
    """Entry point for the ``pyment-finetune`` CLI."""

    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s: %(message)s',
        level=logging.DEBUG,
    )

    parser = argparse.ArgumentParser(
        'Finetunes a model from a configuration file'
    )

    parser.add_argument('configuration', help='Path to configuration file')

    args = parser.parse_args()

    with open(args.configuration, 'r') as f:
        configuration = json.load(f)

    finetune_from_configuration(configuration)


if __name__ == '__main__':
    main()
