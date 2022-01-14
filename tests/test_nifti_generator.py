import numpy as np

from mock import patch

from pyment.data.datasets import NiftiDataset
from pyment.data.generators import NiftiGenerator

from utils import assert_exception

def mock_read(filename):
    id = int(filename.split('.')[0])

    return np.ones((3, 3, 3)) * id

def test_nifti_generator_batches():
    paths = ['0.nii.gz', '1.nii.gz', '2.nii.gz', '3.nii.gz']
    labels = {
        'y1': [0, 1, 2, 3]
    }
    dataset = NiftiDataset(paths, labels=labels, target='y1')

    generator = NiftiGenerator(dataset, batch_size=2)

    assert 2 == generator.batches, \
        'NiftiGenerator does not report correct number of batches'

def test_nifti_generator_batches_uneven():
    paths = ['0.nii.gz', '1.nii.gz', '2.nii.gz', '3.nii.gz', '4.nii.gz']
    labels = {
        'y1': [0, 1, 2, 3, 4]
    }
    dataset = NiftiDataset(paths, labels=labels, target='y1')

    generator = NiftiGenerator(dataset, batch_size=2)

    assert 3 == generator.batches, \
        'NiftiGenerator does not report correct number of batches'

def test_nifti_generator_get_image():
    with patch('pyment.data.io.nifti_loader.NiftiLoader.load',
               wraps=mock_read):
        paths = ['0.nii.gz', '1.nii.gz', '2.nii.gz', '3.nii.gz']
        labels = {
            'y1': [0, 1, 2, 3]
        }
        dataset = NiftiDataset(paths, labels=labels, target='y1')

        generator = NiftiGenerator(dataset, batch_size=2)
        image = generator.get_image(2)

        assert np.array_equal(np.ones((3, 3, 3)) * 2, image), \
            'NiftiGenerator.get_image returns wrong image'

def test_nifti_generator_get_label():
    paths = ['0.nii.gz', '1.nii.gz', '2.nii.gz', '3.nii.gz']
    labels = {
        'y1': [0, 1, 2, 3]
    }
    dataset = NiftiDataset(paths, labels=labels, target='y1')

    generator = NiftiGenerator(dataset, batch_size=2)
    label = generator.get_label(2)

    assert np.array_equal(2, label), \
        'NiftiGenerator.get_image returns wrong label'

def test_nifti_generator_get_datapoint():
    with patch('pyment.data.io.nifti_loader.NiftiLoader.load',
               wraps=mock_read):
        paths = ['0.nii.gz', '1.nii.gz', '2.nii.gz', '3.nii.gz']
        labels = {
            'y1': [0, 1, 2, 3]
        }
        dataset = NiftiDataset(paths, labels=labels, target='y1')

        generator = NiftiGenerator(dataset, batch_size=2)
        datapoint = generator.get_datapoint(1)

        assert 'image' in datapoint, \
            'NiftiGenerator.get_datapoint does not return an image'
        assert np.array_equal(np.ones((3, 3, 3)), datapoint['image']), \
            'NiftiGenerator.get_image returns wrong image'
        assert 'label' in datapoint, \
            'NiftiGenerator.get_datapoint does not return a label'
        assert np.array_equal(1, datapoint['label']), \
            'NiftiGenerator.get_image returns wrong label'

def test_nifti_generator_get_batch():
    with patch('pyment.data.io.nifti_loader.NiftiLoader.load',
               wraps=mock_read):
        paths = ['0.nii.gz', '1.nii.gz', '2.nii.gz', '3.nii.gz']
        labels = {
            'y1': [0, 1, 2, 3]
        }
        dataset = NiftiDataset(paths, labels=labels, target='y1')

        generator = NiftiGenerator(dataset, batch_size=2)
        batch = generator.get_batch(1, 3)

        assert len(batch) == 2, \
            'NiftiGenerator.get_batch does not return a tuple with two entries'

        X, y = batch

        assert 2 == len(X), \
            ('NiftiGenerator.get_batch does not return the correct number of '
             'images')
        assert np.array_equal(np.ones((3, 3, 3)), X[0]), \
            'NiftiGenerator.get_batch does not return the correct images'
        assert np.array_equal(np.ones((3, 3, 3)) * 2, X[1]), \
            'NiftiGenerator.get_batch does not return the correct images'
        assert np.array_equal([1, 2], y), \
            'NiftiGenerator.get_batch does not return the correct labels'

def test_nifti_generator_uninitialized():
    with patch('pyment.data.io.nifti_loader.NiftiLoader.load',
               wraps=mock_read):
        paths = ['0.nii.gz', '1.nii.gz', '2.nii.gz', '3.nii.gz']
        labels = {
            'y1': [0, 1, 2, 3]
        }
        dataset = NiftiDataset(paths, labels=labels, target='y1')

        generator = NiftiGenerator(dataset, batch_size=2)

        assert_exception(lambda: next(generator), exception=RuntimeError,
                         message=('Calling next on an NiftiGenerator without '
                                  'initializing it does not raise an error'))

def test_nifti_generator_next():
    with patch('pyment.data.io.nifti_loader.NiftiLoader.load',
               wraps=mock_read):
        paths = ['0.nii.gz', '1.nii.gz', '2.nii.gz', '3.nii.gz']
        labels = {
            'y1': [0, 1, 2, 3],
        }
        dataset = NiftiDataset(paths, labels=labels, target='y1')

        generator = NiftiGenerator(dataset, batch_size=2)
        iter(generator)
        batch = next(generator)

        assert len(batch) == 2, \
            'NiftiGenerator.get_batch does not return a tuple with two entries'

        X, y = batch

        assert 2 == len(X), \
            ('NiftiGenerator.get_batch does not return the correct number of '
             'images')
        assert np.array_equal(np.zeros((3, 3, 3)), X[0]), \
            'NiftiGenerator.get_batch does not return the correct images'
        assert np.array_equal(np.ones((3, 3, 3)), X[1]), \
            'NiftiGenerator.get_batch does not return the correct images'
        assert np.array_equal([0, 1], y), \
            'NiftiGenerator.get_batch does not return the correct labels'

def test_nifti_generator_additional_inputs():
    with patch('pyment.data.io.nifti_loader.NiftiLoader.load',
               wraps=mock_read):
        paths = ['0.nii.gz', '1.nii.gz', '2.nii.gz', '3.nii.gz']
        labels = {
            'y1': [0, 1, 2, 3],
            'additional': ['a', 'b', 'c', 'd']
        }
        dataset = NiftiDataset(paths, labels=labels, target='y1')

        generator = NiftiGenerator(dataset, batch_size=2,
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
            'y1': np.asarray([0, 1, 2, 3]),
            'additional': np.asarray(['a', 'b', 'c', 'd'])
        }
        dataset = NiftiDataset(paths, labels=labels, target='y1')

        generator = NiftiGenerator(dataset, batch_size=3,
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

        generator = NiftiGenerator(dataset, batch_size=3, shuffle=True,
                                   additional_inputs=['additional'])
        iter(generator)
        (X, additional), _ = next(generator)

        assert not np.array_equal(['1', '2', '3'], additional), \
            'NiftiGenerator.get_batch does not shuffle additional inputs'

        for i in range(3):
            assert str(int(X[i][0][0][0])) == additional[i], \
                ('NiftiGenerator.get_batch with shuffle does not retain '
                 'relationship between image and additional inputs')

