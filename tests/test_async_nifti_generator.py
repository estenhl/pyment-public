import numpy as np

from mock import patch

from pyment.data import NiftiDataset
from pyment.data import AsyncNiftiGenerator


@patch('pyment.data.io.nifti_loader.NiftiLoader.load')
def test_generator(mock):
    mock.return_value = np.ones((3, 3, 3))
    paths = ['1.nii.gz', '2.nii.gz', '3.nii.gz', '4.nii.gz']
    labels = {
        'y1': [1, 2, 3, 4]
    }
    dataset = NiftiDataset(paths, labels, target='y1')

    generator = AsyncNiftiGenerator(dataset, batch_size=2, threads=2,
                                    shuffle=False, infinite=False)

    for X, y in generator:
        pass

    assert 4 <= mock.call_count, \
        ('AsyncNiftiGenerator does not try to load all images when '
         'infinite=False')

    print(mock.call_args_list)

    call_args = [mock.call_args_list[i][0][0] for i in range(4)]

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
    dataset = NiftiDataset(paths, labels, target='y1')

    generator = AsyncNiftiGenerator(dataset, batch_size=2, threads=2,
                                    shuffle=True, infinite=False)

    for X, y in generator:
        pass

    assert 4 <= mock.call_count, \
        ('AsyncNiftiGenerator does not try to load all images when '
         'infinite=False and shuffle=True')

    call_args = [mock.call_args_list[i][0][0] for i in range(4)]

    assert set(paths) == set(call_args), \
        ('AsyncNiftiGenerator does not try to load all images when '
         'infinite=False and shuffle=True')
    
    assert paths != call_args, \
        'AsyncNiftiGenerator does not shuffle for first epoch'

def mock_image(_, path):
    id = int(path.split('.')[0])

    return np.ones((3, 3, 3)) * id

def test_generator_infinite():
    with patch('pyment.data.io.nifti_loader.NiftiLoader.load', mock_image) \
         as mock:
        paths = ['1.nii.gz', '2.nii.gz', '3.nii.gz', '4.nii.gz']
        labels = {
            'y1': [1, 2, 3, 4]
        }
        dataset = NiftiDataset(paths, labels, target='y1')
        generator = AsyncNiftiGenerator(dataset, batch_size=2, threads=2,
                                        shuffle=False, infinite=True)

        epochs = [[], []]

        for epoch in range(2):
            for batch in range(2):
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
    with patch('pyment.data.io.nifti_loader.NiftiLoader.load', mock_image) \
         as mock:
        paths = ['1.nii.gz', '2.nii.gz', '3.nii.gz', '4.nii.gz']
        labels = {
            'y1': [1, 2, 3, 4]
        }
        dataset = NiftiDataset(paths, labels, target='y1')
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