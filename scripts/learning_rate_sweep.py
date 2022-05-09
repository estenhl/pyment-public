"""Performs a learning rate sweep for a given model on a given dataset.

Example usage:
    python scripts/learning_rate_sweep.py \
        --model /path/to/model/folder \
        --loss mse \
        --learning_rates 1e-6 1 \
        --steps 5000 \
        --dataset /path/to/dataset1.json \
                  /path/to/dataset2.json \
        --preprocessor /path/to/preprocessor.json \
        --batch_size 32 \
        --folder /path/to/results/folder
"""

import argparse
import json
import os
import matplotlib.pyplot as plt
import numpy as np

from functools import reduce
from multiprocessing import cpu_count
from tensorflow.keras.models import load_model
from typing import Tuple
from tqdm import tqdm

from utils import configure_environment

configure_environment()

from pyment.data import load_dataset_from_jsonfile
from pyment.data.preprocessors import NiftiPreprocessor
from pyment.data.generators import AsyncNiftiGenerator


def learning_rate_sweep(*, model: str, loss: str, learning_rates: Tuple[float],
                        steps: int, window: int = 1, datasets: str,
                        preprocessor: str = None, batch_size: int,
                        num_threads: int = None, folder: str):
    if os.path.isdir(folder):
        raise ValueError(f'Folder {folder} already exists')

    os.mkdir(folder)

    model = load_model(model)
    model.compile(loss=loss, optimizer='SGD')

    datasets = [load_dataset_from_jsonfile(dataset) for dataset in datasets]
    dataset = reduce(lambda x, y: x + y, datasets)
    preprocessor = NiftiPreprocessor.from_file(preprocessor) \
                   if preprocessor is not None else None
    num_threads = num_threads if num_threads is not None \
                  else cpu_count()
    generator = AsyncNiftiGenerator(dataset,
                                    preprocessor=preprocessor,
                                    batch_size=batch_size,
                                    shuffle=True,
                                    infinite=True,
                                    threads=num_threads,
                                    avoid_singular_batches=True)
    generator = iter(generator)

    min_lr, max_lr = learning_rates
    factor = (max_lr / min_lr) ** (1 / steps)
    lr = min_lr
    model.optimizer.lr.assign(lr)
    batch = []
    lrs = []
    losses = []

    for step in tqdm(range(steps * window)):
        X, y = next(generator)
        batch_loss = model.train_on_batch(X, y)
        batch.append(batch_loss)

        if len(batch) == window:
            lrs.append(lr)
            losses.append(np.mean(batch))
            batch = []
            lr = lr * factor
            model.optimizer.lr.assign(lr)

    results = {'learning_rates': lrs, 'losses': losses}

    with open(os.path.join(folder, 'results.json'), 'w') as f:
        json.dump(results, f)

    fig = plt.figure(figsize=(15, 15))
    plt.plot(lrs, losses)
    plt.gca().set_xscale('log')
    plt.gca().set_yscale('log')
    plt.savefig(os.path.join(folder, 'results.png'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(('Performs a learning rate sweep for a '
                                      'given model, between a given min and '
                                      'max learning rate, for a given number '
                                      'of steps'))

    parser.add_argument('-m', '--model', required=True,
                        help='Path to saved model')
    parser.add_argument('-l', '--loss', required=True,
                        help='Loss used by the model')
    parser.add_argument('-lr', '--learning_rates', nargs=2, type=float,
                        help='Min and max learning rates to test')
    parser.add_argument('-s', '--steps', required=True, type=int,
                        help='Number of steps to run the sweep for')
    parser.add_argument('-w', '--window', required=False, default=1,
                        type=int,
                        help=('The number of batches which is combined into '
                              'each iteration'))
    parser.add_argument('-d', '--datasets', required=True, nargs='+',
                        help='JSON file containing the dataset')
    parser.add_argument('-p', '--preprocessor', required=False, default=None,
                        help='JSON file containing a NiftiPreprocessor')
    parser.add_argument('-bs', '--batch_size', required=True, type=int,
                        help='Batch sized used by the generator')
    parser.add_argument('-nt', '--num_threads', required=False, default=None,
                        help=('Number of threads used by the generator to '
                              'load data. If not set, the available number '
                              'of cores is used'))
    parser.add_argument('-f', '--folder', required=True,
                        help='Path to folder where results are stored')

    args = parser.parse_args()

    learning_rate_sweep(model=args.model,
                        loss=args.loss,
                        learning_rates=args.learning_rates,
                        steps=args.steps,
                        window=args.window,
                        datasets=args.datasets,
                        preprocessor=args.preprocessor,
                        batch_size=args.batch_size,
                        num_threads=args.num_threads,
                        folder=args.folder)
