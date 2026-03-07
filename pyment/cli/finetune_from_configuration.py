import argparse
import json
import logging
import os

import pandas as pd

from pyment.configurations import compile_target_encoder, TrainingConfiguration


logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s: %(message)s',
    level=logging.DEBUG
)
logger = logging.getLogger(__name__)

def finetune_from_configuration(configuration: str):
    configuration = TrainingConfiguration.model_validate(configuration)

    os.makedirs(configuration.destination, exist_ok=True)

    dataset = configuration.dataset.build(
        target=configuration.target.name,
        target_encoder=compile_target_encoder(configuration.target)
    )
    train, validation = configuration.data_split.split(dataset)

    model = configuration.model.build()
    model.compile(
        loss=configuration.loss,
        metrics=configuration.metrics,
        optimizer=configuration.optimizer
    )

    # Remove batch dimension
    target_shape = model.input_shape[1:]

    train_generator = train.to_tensorflow_generator(
        target_shape=target_shape,
        batch_size=configuration.batch_size,
        num_threads=configuration.num_threads,
        shuffle=True
    )
    validation_generator = validation.to_tensorflow_generator(
        target_shape=target_shape,
        batch_size=configuration.batch_size,
        num_threads=configuration.num_threads
    )

    history = model.fit(
        train_generator,
        validation_data=validation_generator,
        epochs=configuration.epochs
    )

    model.save(os.path.join(configuration.destination, 'model'))

    history_serializable = {
        k: [float(v) for v in vals]
        for k, vals in history.history.items()
    }
    with open(os.path.join(configuration.destination, 'history.json'), 'w') as f:
        json.dump(history_serializable, f, indent=4)

    subsets = [('train', train), ('validation', validation)]
    prediction_rows = []
    for subset_name, subset in subsets:
        pred_generator = subset.to_tensorflow_generator(
            target_shape=target_shape,
            batch_size=configuration.batch_size,
            num_threads=configuration.num_threads
        )
        predictions = model.predict(pred_generator).flatten()
        ground_truths = subset.labels[configuration.target.name].values
        image_ids = subset.labels['image_id'].values
        for image_id, gt, pred in zip(image_ids, ground_truths, predictions):
            prediction_rows.append({
                'image_id': image_id,
                'ground_truth': gt,
                'prediction': float(pred),
                'subset': subset_name
            })

    pd.DataFrame(prediction_rows).to_csv(
        os.path.join(configuration.destination, 'predictions.csv'),
        index=False
    )

def main():
    parser = argparse.ArgumentParser(
        'Finetunes a model from a configuration file'
    )

    parser.add_argument(
        'configuration',
        help='Path to configuration file'
    )

    args = parser.parse_args()

    with open(args.configuration, 'r') as f:
        configuration = json.load(f)

    finetune_from_configuration(configuration)

if __name__ == '__main__':
    main()
