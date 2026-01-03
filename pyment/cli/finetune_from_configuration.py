import argparse
import json
import logging

from pyment.configurations import compile_target_encoder, TrainingConfiguration


logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s: %(message)s',
    level=logging.DEBUG
)
logger = logging.getLogger(__name__)

def finetune_from_configuration(configuration: str):
    configuration = TrainingConfiguration.model_validate(configuration)

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

    model.fit(
        train_generator,
        validation_data=validation_generator,
        epochs=configuration.epochs
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
