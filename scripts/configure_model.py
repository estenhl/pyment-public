"""Configures a keras model.

Example usage:
    python scripts/configure_model.py \
        -m regressionvgg11 \
        -k "{\"input_shape\": [43, 54, 41], \
             \"dropout\": 0.5, \
             \"weight_decay\": 1e-3}" \
        -c "{\"loss\": \"mse\", \
             \"optimizer\": \"sgd\", \
             \"metrics\": [\"mae\"]}" \
        -d /path/to/destination/folder
"""


import argparse
import json

from typing import Any, Dict

from utils import configure_environment

configure_environment()

from pyment.models import get, get_model_names


def configure_model(*, model: str, kwargs: str = '{}',
                    compile_kwargs: str = '{}', destination: str):
    kwargs = json.loads(kwargs)
    compile_kwargs = json.loads(compile_kwargs)

    model = get(model, **kwargs)
    model.compile(**compile_kwargs)

    model.save(destination)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(('Configures and stores a model '
                                      'according to the given parameters'))

    parser.add_argument('-m', '--model', required=True,
                        choices=get_model_names(),
                        help='Name of the model to use')
    parser.add_argument('-k', '--kwargs', required=False, default='{}',
                        help=('JSON string with additional parameters '
                              'passed on to the model for initialization'))
    parser.add_argument('-c', '--compile_kwargs', required=False, default='{}',
                        help=('JSON string with parameters passed to the '
                              'model.compile call'))
    parser.add_argument('-d', '--destination', required=True,
                        help='Path where json containing model is stored')

    args = parser.parse_args()

    configure_model(model=args.model, kwargs=args.kwargs,
                    compile_kwargs=args.compile_kwargs,
                    destination=args.destination)
