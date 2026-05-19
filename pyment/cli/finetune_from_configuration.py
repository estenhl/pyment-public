"""Finetune an SFCN model from a JSON configuration."""

import argparse
import json
import logging
import os
from typing import Any

import pandas as pd
import tensorflow as tf

from pyment.configurations import (
    FinetuningConfiguration,
    compile_target_encoder,
)
from pyment.models.sfcn import MultiTaskSFCN

logger = logging.getLogger(__name__)


def _resolve_optimizer(optimizer: str) -> tf.keras.optimizers.legacy.Optimizer:
    _legacy_optimizers = {
        'adam': tf.keras.optimizers.legacy.Adam,
        'sgd': tf.keras.optimizers.legacy.SGD,
        'rmsprop': tf.keras.optimizers.legacy.RMSprop,
    }

    if not isinstance(optimizer, str):
        raise ValueError('Optimizer name must be a string')

    if optimizer not in _legacy_optimizers:
        raise KeyError(f'Unknown optimizer name {optimizer}')

    return _legacy_optimizers[optimizer]()


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

    model.save(os.path.join(configuration.destination, 'model'))

    history_serializable = {
        k: [float(v) for v in vals] for k, vals in history.history.items()
    }
    with open(
        os.path.join(configuration.destination, 'history.json'), 'w'
    ) as f:
        json.dump(history_serializable, f, indent=4)

    subsets = [('train', train), ('validation', validation)]
    prediction_rows = []
    for subset_name, subset in subsets:
        pred_generator = subset.to_tensorflow_generator(
            batch_size=configuration.batch_size,
            num_threads=configuration.num_threads,
        )
        predictions = model.predict(pred_generator).flatten()
        ground_truths = subset.labels[configuration.target.name].values
        image_ids = subset.labels['image_id'].values
        for image_id, gt, pred in zip(image_ids, ground_truths, predictions):
            prediction_rows.append(
                {
                    'image_id': image_id,
                    'ground_truth': gt,
                    'prediction': float(pred),
                    'subset': subset_name,
                }
            )

    pd.DataFrame(prediction_rows).to_csv(
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
