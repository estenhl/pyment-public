"""Contains tests for the single-domain async nifti generator."""

import math
import numpy as np

from collections import Counter
from copy import copy
from mock import patch

from pyment.data import NiftiDataset
from pyment.data.generators import SingleDomainAsyncNiftiGenerator

from test_nifti_generator import mock_read


def test_generator_batches_are_singular():
    """Tests that a SingleDomainAsyncNiftiGenerator serves batches
    with a single domain while still being consistent.
    """
    with patch('pyment.data.io.nifti_loader.NiftiLoader.load',
               wraps=mock_read):
        np.random.seed(42)

        labels = np.random.randint(0, 3, size=30)
        domains = copy(labels)
        paths = [f'{x}.nii.gz' for x in labels]

        labels = {
            'y': labels,
            'domain': domains
        }

        dataset = NiftiDataset(paths, labels=labels, target='y')
        generator = SingleDomainAsyncNiftiGenerator(dataset, domain='domain',
                                                    threads=2, batch_size=4)
        seen = None

        for X, y in generator:
            assert 2 == len(X), \
                ('SingleDomainAsyncNiftiGenerator does not return a tuple of '
                 'the correct length')
            inputs, domain = X
            assert 1 == len(np.unique(domain)), \
                ('SingleDomainAsyncNiftiGenerator does not yield batches from '
                 'a single domain')

            seen = y if seen is None else np.concatenate([seen, y])

            x = inputs[0][0][0][0]
            domain = domain[0]
            y = y[0]

            assert x == domain, \
                ('SingleDomainAsyncNiftiGenerator mixes up relationship '
                 'between images and domains')
            assert x == y, \
                ('SingleDomainAsyncNiftiGenerator mixes up relationship '
                 'between images and labels')

        seen = seen.squeeze()

        assert Counter(labels['y']) == Counter(seen), \
            'SingleDomainAsyncNiftiGenerator does not return all datapoints'

def test_single_domain_generator_shuffle():
    """Tests that a SingleDomainAsyncNiftiGenerator shuffles between
    batches.
    """
    with patch('pyment.data.io.nifti_loader.NiftiLoader.load',
               wraps=mock_read):
        np.random.seed(42)

        labels = np.concatenate([np.ones(10) * i for i in range(3)])
        domains = copy(labels)
        paths = [f'{x}.nii.gz' for x in labels]

        labels = {
            'y': labels,
            'domain': domains
        }

        dataset = NiftiDataset(paths, labels=labels, target='y')
        generator = SingleDomainAsyncNiftiGenerator(dataset, domain='domain',
                                                    threads=2, batch_size=4,
                                                    shuffle=True, infinite=True)
        runs = [None, None]

        for i in range(2):
            for _ in range(generator.batches):
                X, y = next(generator)
                _, domain = X
                assert 1 == len(np.unique(domain)), \
                    ('SingleDomainAsyncNiftiGenerator does not yield batches from '
                    'a single domain')

                runs[i] = y if runs[i] is None \
                          else np.concatenate([runs[i], y])

            runs[i] = runs[i].squeeze()

        assert Counter(labels['y']) == Counter(runs[0]) == Counter(runs[1]), \
            ('SingleDomainAsyncNiftiGenerator with shuffle set does not '
             'contain all datapoints for each epoch')

        assert not np.array_equal(runs[0], labels['y']), \
            ('SingleDomainAsyncNiftiGenerator with shuffle set does not '
             'perform an initial shuffle')

        assert not np.array_equal(runs[0], runs[1]), \
            ('SingleDomainAsyncNiftiGenerator with shuffle set does not '
             'shuffle between epochs')
