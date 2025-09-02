import argparse
import json
import logging
import os
import pandas as pd
from typing import Any, Callable, Dict, List, Tuple

import tensorflow as tf
from tensorflow_neuroimaging.preprocessing import center_crop_or_pad
from tensorflow_neuroimaging.loaders.mgh import load_mgh

from pyment.configurations import DatasetConfiguration, FinetuningConfiguration
from pyment.factories import loss_factory, metric_factory, optimizer_factory
from pyment.models.sfcn import sfcn_factory
from pyment.models.utils.load_select_pretrained_weights import (
    load_select_pretrained_weights
)
from pyment.utils.json_serialize import json_serialize


logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s: %(message)s',
    level=logging.DEBUG
)
logger = logging.getLogger(__name__)

def _create_tensorflow_dataset(
    df: pd.DataFrame, *,
    target: str,
    input_shape: Tuple[int, int, int],
    batch_size: str,
    shuffle: bool = False
) -> tf.data.Dataset:
    input_shape = tf.constant(input_shape)

    df = df.copy()
    df = df.sample(frac=1.)

    dataset = tf.data.Dataset.from_tensor_slices((df['path'], df[target]))

    if shuffle:
        dataset = dataset.shuffle(buffer_size=5*batch_size)
    
    dataset = dataset.map(
        lambda path, label: (load_mgh(path), label),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    dataset = dataset.map(
        lambda image, label: (center_crop_or_pad(image, input_shape), label),
        num_parallel_calls=tf.data.AUTOTUNE
    )

    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset

def _create_checkpointing_callback(
    destination: str, 
    metrics: List[tf.keras.metrics.Metric] = None
):
    os.mkdir(destination)

    train_metrics = []
    val_metrics = []
    
    if metrics is not None:
        for metric in metrics:
            name = metric.name.replace('_', '-')
            train_metrics.append(f'{name}={{{metric.name}:.2f}}')
            val_metrics.append(f'val-{name}={{val_{metric.name}:.2f}}')

    terms = [
        'epoch={epoch:03d}',
        'loss={loss:.2f}'
    ] + train_metrics + [
        'val-loss={val_loss:.2f}'
    ] + val_metrics
    filename = '_'.join(terms) + '.hdf5'
    filepath = os.path.join(destination, filename)

    return tf.keras.callbacks.ModelCheckpoint(
        filepath,
        monitor='val_loss',
        save_best_only=True,
        save_weights_only=True
    )

def finetune(
    model_type: str,
    model_constructor_arguments: Dict[str, Any],
    weights: str, 
    input_shape: Tuple[int, int, int],
    target: str,
    loss: Callable[[tf.Tensor, tf.Tensor], tf.Tensor],
    metrics: List[tf.keras.metrics.Metric],
    optimizer: tf.optimizers.Optimizer,
    learning_rate_scheduler: tf.keras.callbacks.Callback,
    training: pd.DataFrame, 
    validation: pd.DataFrame,
    batch_size: int,
    epochs: int,
    destination: str
):
    if destination is not None:
        if os.path.isdir(destination):
            raise ValueError(f'Destination {destination} already exists')

        logger.info('Creating destination folder %s', destination)
        os.mkdir(destination)

    model_class = sfcn_factory(model_type)
    model = model_class(
        input_shape=input_shape,
        **model_constructor_arguments
    )
    load_select_pretrained_weights(model, weights, target=target)

    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

    training_dataset = _create_tensorflow_dataset(
        training, 
        input_shape=input_shape,
        target=target, 
        batch_size=batch_size,
        shuffle=True
    )
    validation_dataset = _create_tensorflow_dataset(
        validation,
        input_shape=input_shape,
        target=target,
        batch_size=batch_size,
        shuffle=False
    )

    callbacks = [
        _create_checkpointing_callback(
            os.path.join(destination, 'checkpoints'),
            metrics=metrics
        ),
        learning_rate_scheduler
    ]

    history = model.fit(
        training_dataset, 
        validation_data=validation_dataset,
        epochs=epochs,
        callbacks=callbacks
    )

    with open(os.path.join(destination, 'history.json'), 'w') as f:
        json.dump(json_serialize(history.history), f)

def finetune_from_configuration(configuration: str):
    with open(configuration, 'r') as f:
        configuration = json.load(f)
    
    configuration = FinetuningConfiguration.model_validate(configuration)

    training, validation = DatasetConfiguration.parse(
        configuration.data,
        target=configuration.training.target
    )

    # strategy = tf.distribute.MirroredStrategy()

    # with strategy.scope():

    loss_cls = loss_factory(configuration.training.loss)
    loss = loss_cls()

    optimizer_cls = optimizer_factory(configuration.training.optimizer)
    optimizer = optimizer_cls(configuration.training.learning_rate)

    metrics = None
    
    if configuration.training.metrics is not None:
        metrics = [
            metric_factory(metric)
            for metric in configuration.training.metrics
        ]

    learning_rate_scheduler = None

    if configuration.training.learning_rate_schedule:
        learning_rate_scheduler = (
            configuration.training.learning_rate_schedule.instantiate()
        )

    finetune(
        model_type=configuration.model.type,
        model_constructor_arguments=configuration.model.hyperparameters,
        weights=configuration.model.weights,
        input_shape=configuration.data.input_shape,
        target=configuration.training.target,
        loss=loss,
        metrics=metrics,
        optimizer=optimizer,
        learning_rate_scheduler=learning_rate_scheduler,
        training=training,
        validation=validation,
        batch_size=configuration.training.batch_size,
        epochs=configuration.training.epochs,
        destination=configuration.training.destination
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        'Finetunes a multi-task SFCN according to the given configuration'
    )

    parser.add_argument('configuration', help='Path to configuration JSON')

    args = parser.parse_args()

    finetune_from_configuration(args.configuration)