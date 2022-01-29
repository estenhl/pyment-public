import numpy as np

from mock import patch
from time import sleep

from pyment.data import NiftiDataset
from pyment.data import AsyncNiftiGenerator

from test_nifti_generator import mock_read


@patch('pyment.data.io.nifti_loader.NiftiLoader.load')
def test_generator(mock):
    mock.return_value = np.ones((3, 3, 3))
    paths = ['1.nii.gz', '2.nii.gz', '3.nii.gz', '4.nii.gz']
    labels = {
        'y1': [1, 2, 3, 4]
    }
    dataset = NiftiDataset(paths, labels=labels, target='y1')

    generator = AsyncNiftiGenerator(dataset, batch_size=2, threads=2,
                                    shuffle=False, infinite=False)

    # Sleep to allow the generator to finish preloading initial batch
    sleep(0.1)

    # Picks out any samples that are preloaded before the generator is
    # reinitialized
    preloads = len(mock.call_args_list)

    for _ in generator:
        call_args = [mock.call_args_list[i][0][0] \
                     for i in range(len(mock.call_args_list))]

    call_args = call_args[preloads:]

    assert 4 == len(call_args), \
        ('AsyncNiftiGenerator does not try to load all images when '
         'infinite=False')

    assert set(paths) == set(call_args), \
        ('AsyncNiftiGenerator does not try to load all images when '
         'infinite=False')

@patch('pyment.data.io.nifti_loader.NiftiLoader.load')
def test_generator_shuffle(mock):
    np.random.seed(42)

    mock.return_value = np.ones((3, 3, 3))
    paths = ['1.nii.gz', '2.nii.gz', '3.nii.gz', '4.nii.gz']
    labels = {
        'y1': [1, 2, 3, 4]
    }
    dataset = NiftiDataset(paths, labels=labels, target='y1')

    generator = AsyncNiftiGenerator(dataset, batch_size=2, threads=2,
                                    shuffle=True, infinite=False)

    sleep(0.1)
    preloads = len(mock.call_args_list)

    for _ in generator:
        call_args = [mock.call_args_list[i][0][0] \
                     for i in range(len(mock.call_args_list))]

    call_args = call_args[preloads:]

    assert 4 == len(call_args), \
        ('AsyncNiftiGenerator does not try to load all images when '
         'infinite=False')

    assert set(paths) == set(call_args), \
        ('AsyncNiftiGenerator does not try to load all images when '
         'infinite=False and shuffle=True')

    assert paths != call_args, \
        'AsyncNiftiGenerator does not shuffle for first epoch'

def test_generator_infinite():
    with patch('pyment.data.io.nifti_loader.NiftiLoader.load',
               wraps=mock_read):
        paths = ['1.nii.gz', '2.nii.gz', '3.nii.gz', '4.nii.gz']
        labels = {
            'y1': [1, 2, 3, 4]
        }
        dataset = NiftiDataset(paths, labels=labels, target='y1')
        generator = AsyncNiftiGenerator(dataset, batch_size=2, threads=2,
                                        shuffle=False, infinite=True)

        epochs = [[], []]

        for epoch in range(2):
            for _ in range(2):
                X, y = next(generator)
                for i in range(len(X)):
                    assert X[i][0][0][0] == y[i], \
                        'AsyncNiftiGenerator mixes up images and labels'
                    epochs[epoch].append(y[i])
            generator.reset()

        assert epochs[0] == epochs[1] == [1, 2, 3, 4], \
            ('AsyncNiftiGenerator with infinite=True does not return all '
             'in the same order for epochs split by reset()')

def test_generator_shuffle_infinite():
    np.random.seed(42)

    with patch('pyment.data.io.nifti_loader.NiftiLoader.load',
               wraps=mock_read):
        paths = ['1.nii.gz', '2.nii.gz', '3.nii.gz', '4.nii.gz']
        labels = {
            'y1': [1, 2, 3, 4]
        }
        dataset = NiftiDataset(paths, labels=labels, target='y1')
        generator = AsyncNiftiGenerator(dataset, batch_size=2, threads=2,
                                        shuffle=True, infinite=True)

        epochs = [[], []]

        for epoch in range(2):
            for batch in range(2):
                X, y = next(generator)
                for i in range(len(X)):
                    assert X[i][0][0][0] == y[i], \
                        ('AsyncNiftiGenerator mixes up images and labels when '
                         'shuffle=True')
                    epochs[epoch].append(y[i])
            generator.reset()

        assert set(epochs[0]) == set(epochs[1]) == set([1, 2, 3, 4]), \
            ('AsyncNiftiGenerator with infinite=True and shuffle=True does '
             'not return all images for epochs split by reset()')

        assert epochs[0] != epochs[1], \
            ('AsyncNiftiGenerator with infinite=True and shuffle=True does '
             'not shuffle batches between epochs')

def test_nifti_generator_additional_inputs():
    with patch('pyment.data.io.nifti_loader.NiftiLoader.load',
               wraps=mock_read):
        paths = ['0.nii.gz', '1.nii.gz', '2.nii.gz', '3.nii.gz']
        labels = {
            'y1': [0, 1, 2, 3],
            'additional': ['a', 'b', 'c', 'd']
        }
        dataset = NiftiDataset(paths, labels=labels, target='y1')

        generator = AsyncNiftiGenerator(dataset, batch_size=2, threads=2,
                                        additional_inputs=['additional'])

        datapoint = generator.get_datapoint(1)

        assert 'image' in datapoint.keys(), \
            ('NiftiGenerator.get_datapoint with additional inputs does not '
             'return image')
        assert 'label' in datapoint.keys(), \
            ('NiftiGenerator.get_datapoint with additional inputs does not '
             'return label')
        assert 'additional' in datapoint.keys(), \
            ('NiftiGenerator.get_datapoint with additional inputs does not '
             'return additional variables')
        assert 'b' == datapoint['additional'], \
            ('NiftiGenerator.get_datapoint with additional inputs does not '
             'return correct value for the additional inputs')

def test_nifti_generator_next_additional_inputs():
    with patch('pyment.data.io.nifti_loader.NiftiLoader.load',
               wraps=mock_read):
        paths = ['0.nii.gz', '1.nii.gz', '2.nii.gz', '3.nii.gz']
        labels = {
            'y1': [0, 1, 2, 3],
            'additional': ['a', 'b', 'c', 'd']
        }
        dataset = NiftiDataset(paths, labels=labels, target='y1')

        generator = AsyncNiftiGenerator(dataset, batch_size=3, threads=2,
                                        additional_inputs=['additional'])
        iter(generator)
        batch = next(generator)

        X, _ = batch

        assert 2 == len(X), \
            ('NiftiGenerator.get_batch with additional inputs does not return '
             'a tuple for X')

        X, additional = X
        assert np.array_equal(np.zeros((3, 3, 3)), X[0]), \
            'NiftiGenerator.get_batch does not return the correct images'
        assert np.array_equal(np.ones((3, 3, 3)), X[1]), \
            'NiftiGenerator.get_batch does not return the correct images'
        assert np.array_equal(np.ones((3, 3, 3)) * 2, X[2]), \
            'NiftiGenerator.get_batch does not return the correct images'
        assert np.array_equal(['a', 'b', 'c'], additional), \
            'NiftiGenerator.get_batch does not return the correct labels'

def test_nifti_generator_next_additional_inputs_shuffle():
    np.random.seed(42)

    with patch('pyment.data.io.nifti_loader.NiftiLoader.load',
               wraps=mock_read):
        paths = ['0.nii.gz', '1.nii.gz', '2.nii.gz', '3.nii.gz']
        labels = {
            'y1': [0, 1, 2, 3],
            'additional': ['0', '1', '2', '3']
        }
        dataset = NiftiDataset(paths, labels=labels, target='y1')

        generator = AsyncNiftiGenerator(dataset, batch_size=3, shuffle=True,
                                        threads=2,
                                        additional_inputs=['additional'])
        iter(generator)
        (X, additional), _ = next(generator)

        assert not np.array_equal(['1', '2', '3'], additional), \
            'NiftiGenerator.get_batch does not shuffle additional inputs'

        for i in range(3):
            assert str(int(X[i][0][0][0])) == additional[i], \
                ('NiftiGenerator.get_batch with shuffle does not retain '
                  'relationship between image and additional inputs')

def test_nifti_generator_additional_inputs_length():
    with patch('pyment.data.io.nifti_loader.NiftiLoader.load',
               wraps=mock_read):
        paths = [f'{x}.nii.gz' for x in range(30)]
        labels = {
            'y1': np.arange(30),
            'additional': np.arange(30)
        }
        dataset = NiftiDataset(paths, labels=labels, target='y1')

        generator = AsyncNiftiGenerator(dataset, batch_size=4, threads=2,
                                        additional_inputs=['additional'])

        seen = None

        for X, y in generator:
            seen = y if seen is None else np.concatenate([seen, y])

        assert len(seen) == 30, \
            ('AsyncNiftiGenerator with additional inputs does not '
             'contain the correct number of datapoints')
