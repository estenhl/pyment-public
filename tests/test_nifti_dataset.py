"""Contains tests for testing the NiftiDataset."""

import json
import os
import numpy as np

from collections import Counter
from shutil import rmtree

from pyment.data import load_dataset_from_jsonfile, NiftiDataset
from pyment.labels import BinaryLabel, CategoricalLabel
from pyment.utils.io import encode_object_as_json

from utils import assert_exception


def test_dataset_length():
    paths = ['tmp/path1.nii.gz', 'tmp/path2.nii.gz', 'tmp/path3.nii.gz']
    data = NiftiDataset(paths)

    assert 3 == len(data), 'NiftiDataset does not report correct length'

def test_dataset_paths():
    paths = ['tmp/path1.nii.gz', 'tmp/path2.nii.gz', 'tmp/path3.nii.gz']
    data = NiftiDataset(paths)

    assert np.array_equal(paths, data.paths), \
        'NiftiDataset does not report correct paths'

def test_dataset_filenames():
    paths = ['tmp/path1.nii.gz', 'tmp/path2.nii.gz', 'tmp/path3.nii.gz']
    data = NiftiDataset(paths)

    filenames = ['path1.nii.gz', 'path2.nii.gz', 'path3.nii.gz']

    assert np.array_equal(filenames, data.filenames), \
        'NiftiDataset does not report correct filenames'

def test_dataset_ids():
    paths = ['tmp/path1.nii.gz', 'tmp/path2.nii.gz', 'tmp/path3.nii.gz']
    data = NiftiDataset(paths)

    ids = ['path1', 'path2', 'path3']

    assert np.array_equal(ids, data.ids), \
        'NiftiDataset does not report correct ids'

def test_dataset_init_target():
    paths = ['tmp/path1.nii.gz', 'tmp/path2.nii.gz', 'tmp/path3.nii.gz']
    data = NiftiDataset(paths, target='id')

    assert 'id' == data.target, \
        'Setting NiftiDataset target via init does not work'

def test_dataset_set_target():
    paths = ['tmp/path1.nii.gz', 'tmp/path2.nii.gz', 'tmp/path3.nii.gz']
    data = NiftiDataset(paths)
    data.target = 'id'

    assert 'id' == data.target, \
        'Setting NiftiDataset target via init does not work'

def test_dataset_id_target():
    paths = ['tmp/path1.nii.gz', 'tmp/path2.nii.gz', 'tmp/path3.nii.gz']
    data = NiftiDataset(paths, target='id')

    ids = ['path1', 'path2', 'path3']

    assert np.array_equal(ids, data.y), \
        'NiftiDataset with target=id does not return ids as labels'

def test_dataset_variables():
    paths = ['tmp/path1.nii.gz', 'tmp/path2.nii.gz', 'tmp/path3.nii.gz']
    labels = {
        'y1': [1, 2, 3],
        'y2': ['a', 'b', 'c']
    }
    data = NiftiDataset(paths, labels=labels)

    assert ['y1', 'y2'] == data.variables, ('NiftiDataset variables does not '
                                            'return the correct labels')

def test_dataset_labels():
    paths = ['tmp/path1.nii.gz', 'tmp/path2.nii.gz', 'tmp/path3.nii.gz']
    labels = {
        'y1': [1, 2, 3],
        'y2': ['a', 'b', 'c']
    }
    data = NiftiDataset(paths, labels=labels, target='y1')

    assert np.array_equal([1, 2, 3], data.y), \
        'NiftiDataset does not return the correct labels'

def test_dataset_invalid_target():
    paths = ['tmp/path1.nii.gz', 'tmp/path2.nii.gz', 'tmp/path3.nii.gz']
    labels = {
        'y1': [1, 2, 3],
        'y2': ['a', 'b', 'c']
    }
    data = NiftiDataset(paths, labels=labels)

    def _set_target():
        data.target = 'y3'

    assert_exception(_set_target,
                     message=('NiftiDataset does not raise an exception if '
                              'setting an invalid target'))

def test_dataset_json_paths():
    paths = ['tmp/path1.nii.gz', 'tmp/path2.nii.gz', 'tmp/path3.nii.gz']
    data = NiftiDataset(paths)

    assert 'paths' in data.json, ('NiftiDataset.json does not contain a field '
                                  'for paths')
    assert np.array_equal(paths, data.json['paths']), \
        'NiftiDataset.json does not contain correct paths'

def test_dataset_json_labels():
    paths = ['tmp/path1.nii.gz', 'tmp/path2.nii.gz', 'tmp/path3.nii.gz']
    labels = {
        'y1': [1, 2, 3],
        'y2': ['a', 'b', 'c']
    }
    data = NiftiDataset(paths, labels=labels)

    assert 'labels' in data.json, \
        'NiftiDataset.json does not contain a field for labels'
    assert 'y1' in data.labels and \
            np.array_equal([1, 2, 3], data.labels['y1']), \
        'NiftiDataset.json does not contain correct labels'
    assert 'y2' in data.labels and \
            np.array_equal(['a', 'b', 'c'], data.labels['y2']), \
        'NiftiDataset.json does not contain correct labels'

def test_dataset_json_target():
    paths = ['tmp/path1.nii.gz', 'tmp/path2.nii.gz', 'tmp/path3.nii.gz']
    labels = {
        'y1': [1, 2, 3],
        'y2': ['a', 'b', 'c']
    }
    data = NiftiDataset(paths, labels=labels, target='y1')

    assert 'target' in data.json, \
        'NiftiDataset.json does not contain a field for target'
    assert 'y1' == data.json['target'], \
        'NiftiDataset.json does not contain correct target'

def test_dataset_json_handles_numpy_labels():
    paths = ['tmp/path1.nii.gz', 'tmp/path2.nii.gz', 'tmp/path3.nii.gz']
    labels = {
        'y1': np.asarray([1, 2, 3], dtype=np.int64)
    }
    data = NiftiDataset(paths, labels=labels)

    exception = False

    try:
        json.dumps(data.json)
    except Exception:
        exception = True

    assert not exception, 'NiftiDataset.json does not handle numpy arrays'

def test_dataset_equal():
    paths = ['tmp/path1.nii.gz', 'tmp/path2.nii.gz', 'tmp/path3.nii.gz']
    labels = {
        'y1': np.asarray([1, 2, 3], dtype=np.int64)
    }
    d1 = NiftiDataset(paths, labels=labels, target='y1')
    d2 = NiftiDataset(paths, labels=labels, target='y1')

    assert d1 == d2, 'Two equal NiftiDatasets are not considered equal'

def test_dataset_unequal_paths():
    paths1 = ['tmp1/path1.nii.gz', 'tmp1/path2.nii.gz']
    paths2 = ['tmp2/path1.nii.gz', 'tmp2/path2.nii.gz']
    labels = {
        'y1': np.asarray([1, 2], dtype=np.int64)
    }
    d1 = NiftiDataset(paths1, labels=labels, target='y1')
    d2 = NiftiDataset(paths2, labels=labels, target='y1')

    assert d1 != d2, \
        ('Two NiftiDatasets with different paths are not considered different '
         'by !=')

def test_dataset_unequal_labels():
    paths = ['tmp/path1.nii.gz', 'tmp/path2.nii.gz']
    labels1 = {
        'y1': np.asarray([1, 2], dtype=np.int64)
    }
    labels2 = {
        'y2': np.asarray([1, 2], dtype=np.int64)
    }
    d1 = NiftiDataset(paths, labels=labels1)
    d2 = NiftiDataset(paths, labels=labels2)

    assert d1 != d2, \
        'Two NiftiDatasets with different labels are considered equal'

def test_dataset_unequal_target():
    paths = ['tmp/path1.nii.gz', 'tmp/path2.nii.gz']
    labels = {
        'y1': np.asarray([1, 2], dtype=np.int64),
        'y2': np.asarray([1, 2], dtype=np.int64)
    }
    d1 = NiftiDataset(paths, labels=labels, target='y1')
    d2 = NiftiDataset(paths, labels=labels, target='y2')

    assert d1 != d2, \
        'Two NiftiDatasets with different targets are considered equal'

def test_dataset_jsonstring():
    paths = ['tmp/path1.nii.gz', 'tmp/path2.nii.gz', 'tmp/path3.nii.gz']
    labels = {
        'y1': np.asarray([1, 2, 3], dtype=np.int64)
    }
    data = NiftiDataset(paths, labels=labels, target='y1')
    obj = json.loads(data.jsonstring)

    assert 'paths' in obj, 'NiftiDataset.jsonstring does not contain paths'
    assert np.array_equal(paths, obj['paths']), \
        'NiftiDataset.jsonstring contains wrong paths'
    assert 'labels' in obj, 'NiftiDataset.jsonstring does not contain labels'
    assert ['y1'] == list(obj['labels'].keys()), \
        'NiftiDataset.jsonstring contains wrong labels'
    assert np.array_equal(labels['y1'], obj['labels']['y1']), \
        'NiftiDataset.jsonstring contains wrong labels'

def test_dataset_target():
    paths = ['tmp/path1.nii.gz', 'tmp/path2.nii.gz', 'tmp/path3.nii.gz']
    labels = {
        'y1': np.asarray([1, 2, 3], dtype=np.int64)
    }
    data = NiftiDataset(paths, labels=labels, target='y1')
    obj = json.loads(data.jsonstring)

    assert 'target' in obj, 'NiftiDataset.jsonstring does not contain target'
    assert 'y1' == obj['target'], \
        'NiftiDataset.jsonstring contains wrong target'


def test_dataset_from_json():
    obj = {
        'paths': ['tmp/path1.nii.gz', 'tmp/path2.nii.gz', 'tmp/path3.nii.gz'],
        'labels': {
            'y1': np.asarray([1, 2, 3], dtype=np.int64)
        },
        'target': 'y1'
    }

    data = NiftiDataset.from_json(obj)

    assert isinstance(data, NiftiDataset), \
        'NiftiDataset.from_json does not return a NiftiDataset'

    assert np.array_equal(obj['paths'], data.paths), \
        'NiftiDataset.from_json returns object with wrong paths'
    assert ['y1'] == data.variables, \
        'NiftiDataset.from_json returns object with wrong variables'
    assert np.array_equal(obj['labels']['y1'], data.y), \
        'NiftiDataset.from_json returns object with wrong labels'

def test_dataset_to_from_json():
    paths = ['tmp/path1.nii.gz', 'tmp/path2.nii.gz', 'tmp/path3.nii.gz']
    labels = {
        'y1': np.asarray([1, 2, 3], dtype=np.int64)
    }
    data1 = NiftiDataset(paths, labels=labels, target='y1')
    data2 = NiftiDataset.from_json(data1.json)

    assert data1 == data2, \
        'NiftiDataset to and from json does not yield an equivalent object'


def test_dataset_to_from_jsonstring():
    paths = ['tmp/path1.nii.gz', 'tmp/path2.nii.gz', 'tmp/path3.nii.gz']
    labels = {
        'y1': np.asarray([1, 2, 3], dtype=np.int64)
    }
    data1 = NiftiDataset(paths, labels=labels, target='y1')
    data2 = NiftiDataset.from_jsonstring(data1.jsonstring)

    assert data1 == data2, \
        ('NiftiDataset to and from jsonstring does not yield an equivalent '
         'object')

def test_dataset_save_load():
    try:
        paths = ['tmp/path1.nii.gz', 'tmp/path2.nii.gz', 'tmp/path3.nii.gz']
        labels = {
            'y1': np.asarray([1, 2, 3], dtype=np.int64)
        }
        data1 = NiftiDataset(paths, labels=labels, target='y1')
        data1.save('tmp.json')
        data2 = load_dataset_from_jsonfile('tmp.json')

        assert data1 == data2, \
            'NiftiDataset to and from file does not yield an equivalent object'
    finally:
        if os.path.isfile('tmp.json'):
            os.remove('tmp.json')

def test_dataset_from_folder_kwargs():
    try:
        os.makedirs(os.path.join('tmp', 'images'))

        with open(os.path.join('tmp', 'labels.csv'), 'w') as f:
            f.write('id,age\n')
            f.write('1,1\n')
            f.write('2,2\n')

        for i in range(1, 3):
            with open(os.path.join('tmp', 'images', f'{i}.nii.gz'), 'w') as f:
                f.write('test')

        dataset = NiftiDataset.from_folder('tmp', target='age')

        assert 'age' == dataset.target, \
            'NiftiDataset.from_folder does not retain optional kwargs'
    finally:
        rmtree('tmp')

def test_dataset_invalid_init_target():
    paths = ['tmp/path1.nii.gz', 'tmp/path2.nii.gz', 'tmp/path3.nii.gz']
    labels = {
        'y1': [1, 2, 3],
        'y2': ['a', 'b', 'c']
    }

    assert_exception(lambda: NiftiDataset(paths, labels=labels, target='y3'),
                     message=('Instatiating NiftiDataset with invalid target '
                              'does not raise an error'))

def test_dataset_multitarget():
    paths = ['tmp/path1.nii.gz', 'tmp/path2.nii.gz', 'tmp/path3.nii.gz']
    labels = {
        'y1': np.asarray([1, 2, 3], dtype=np.int64),
        'y2': np.asarray(['a', 'b', 'c'])
    }
    data = NiftiDataset(paths, labels=labels, target=['y1', 'y2'])

    assert ['y1', 'y2'] == data.target, \
        'Unable to set multiple targets for NiftiDataset'

    y = np.asarray([
        [1, 'a'],
        [2, 'b'],
        [3, 'c']
    ])

    assert np.array_equal(y, data.y), \
        'NiftiDataset does not return correct y with multiple targets'


def test_niftidataset_add_unknown_other():
    paths = ['tmp/path1.nii.gz', 'tmp/path2.nii.gz', 'tmp/path3.nii.gz']
    labels = {
        'y1': np.asarray([1, 2, 3], dtype=np.int64)
    }
    data = NiftiDataset(paths, labels=labels, target='y1')

    assert_exception(lambda: data + 5,
                     message=('Adding an int to a NiftiDataset does not raise '
                              'an error'))

def test_niftidataset_add_paths():
    paths = ['tmp/path1.nii.gz', 'tmp/path2.nii.gz', 'tmp/path3.nii.gz']
    data = NiftiDataset(paths) + NiftiDataset(paths)

    assert np.array_equal(np.concatenate([paths, paths]), data.paths), \
        'Two added NiftiDatasets does not produce the correct paths'

def test_niftidataset_add_labels():
    paths = ['tmp/path1.nii.gz', 'tmp/path2.nii.gz', 'tmp/path3.nii.gz']
    labels = {
        'y1': np.asarray([1, 2, 3], dtype=np.int64)
    }

    data = NiftiDataset(paths, labels=labels) + NiftiDataset(paths, labels=labels)

    assert 'y1' in data.labels, \
        'Two added NiftiDatasets does not have the right label keys'
    assert np.array_equal([1, 2, 3, 1, 2, 3], data.labels['y1']), \
        'Two added NiftiDatasets does not have the correct labels'

def test_niftidataset_add_labels_none_other():
    paths = ['tmp/path1.nii.gz', 'tmp/path2.nii.gz', 'tmp/path3.nii.gz']
    labels = {
        'y1': np.asarray([1, 2, 3], dtype=np.int64)
    }

    data = NiftiDataset(paths, labels=labels) + NiftiDataset(paths)

    assert 'y1' in data.labels, \
        ('Two added NiftiDatasets does not have the right label keys '
         'when the other dataset has no labels')
    assert np.array_equal([1, 2, 3, np.nan, np.nan, np.nan], data.labels['y1'],
                          equal_nan=True), \
        ('Two added NiftiDatasets does not have the correct labels '
         'when the other dataset has no labels')

def test_niftidataset_add_labels_none_self():
    paths = ['tmp/path1.nii.gz', 'tmp/path2.nii.gz', 'tmp/path3.nii.gz']
    labels = {
        'y1': np.asarray([1, 2, 3], dtype=np.int64)
    }

    data = NiftiDataset(paths) + NiftiDataset(paths, labels=labels)

    assert 'y1' in data.labels, \
        ('Two added NiftiDatasets does not have the right label keys '
         'when self has no labels')
    assert np.array_equal([np.nan, np.nan, np.nan, 1, 2, 3], data.labels['y1'],
                          equal_nan=True), \
        ('Two added NiftiDatasets does not have the correct labels '
         'when self has no labels')

def test_niftidataset_add_labels_missing():
    paths = ['tmp/path1.nii.gz', 'tmp/path2.nii.gz', 'tmp/path3.nii.gz']
    labels1 = {
        'y1': np.asarray([1, 2, 3], dtype=np.int64)
    }
    labels2 = {
        'y2': np.asarray([1, 2, 3], dtype=np.int64)
    }

    data = NiftiDataset(paths, labels=labels1) + \
           NiftiDataset(paths, labels=labels2)

    assert set(['y1', 'y2']) == set(data.labels.keys()), \
        ('Two added NiftiDatasets does not have the right label keys '
         'when they have disjunct label sets')
    assert np.array_equal([1, 2, 3, np.nan, np.nan, np.nan], data.labels['y1'],
                          equal_nan=True), \
        ('Two added NiftiDatasets does not have the correct labels '
         'label is missing from other')
    assert np.array_equal([np.nan, np.nan, np.nan, 1, 2, 3], data.labels['y2'],
                          equal_nan=True), \
        ('Two added NiftiDatasets does not have the correct labels '
         'label is missing from self')

def test_niftidataset_add_equal_string_target():
    paths = ['tmp/path1.nii.gz', 'tmp/path2.nii.gz', 'tmp/path3.nii.gz']
    labels = {
        'y1': np.asarray([1, 2, 3], dtype=np.int64)
    }
    data = NiftiDataset(paths, labels=labels, target='y1') + \
           NiftiDataset(paths, labels=labels, target='y1')

    assert 'y1' == data.target, \
        ('Adding two NiftiDatasets with the same string target does not '
         'produce a new NiftiDataset with the same target')

def test_dataset_slice():
    paths = ['path1.nii.gz', 'path2.nii.gz', 'path3.nii.gz']
    labels = {
        'y1': np.asarray([1, 2, 3], dtype=np.int64),
        'y2': np.asarray(['A', 'B', 'C'])
    }

    data = NiftiDataset(paths, labels=labels, target='y1')
    data = data[:2]

    assert len(data) == 2, \
        'Slicing a NiftiDataset does not remove indexes'
    assert np.array_equal(['path1.nii.gz', 'path2.nii.gz'], data.paths), \
        'Slicing a NiftiDataset yields the wrong paths'
    assert set(['y1', 'y2']) == set(data.labels.keys()), \
        'Slicing a NiftiDataset does not retain all labels'
    assert np.array_equal([1, 2], data.labels['y1']), \
        'Slicing a NiftiDataset retains the wrong labels'
    assert np.array_equal(['A', 'B'], data.labels['y2']), \
        'Slicing a NiftiDataset retains the wrong labels'
    assert 'y1' == data.target, \
        'Slicing a NiftiDataset changes the target'

def test_dataset_stratified_categorical():
    paths = [f'{i}.nii.gz' for i in range(12)]
    labels = {
        'dataset': ['d1'] * 6 + ['d2'] * 6,
        'sex': np.random.permutation(['F'] * 6 + ['M'] * 6),
        'age': np.random.uniform(0, 1, 12)
    }
    data = NiftiDataset(paths, labels=labels)
    folds = data.stratified_folds(2, ['dataset'])

    assert len(folds) == 2, \
        'NiftiDataset.stratified_folds(k) does not return k folds'

    counts1 = Counter(folds[0].labels['dataset'])
    counts2 = Counter(folds[1].labels['dataset'])

    assert counts1 == counts2, \
        ('Stratifying a NiftiDataset on a categorical variable does not yield '
         'an even split')

def test_dataset_stratified_continuous():
    np.random.seed(42)

    paths = [f'{i}.nii.gz' for i in range(100)]
    labels = {
        'dataset': ['d1'] * 50 + ['d2'] * 50,
        'sex': np.random.permutation(['F'] * 50 + ['M'] * 50),
        'age': np.random.uniform(0, 1, 100)
    }
    data = NiftiDataset(paths, labels=labels)
    folds = data.stratified_folds(2, ['age'])

    assert abs(np.mean(folds[0].labels['age']) - \
               np.mean(folds[1].labels['age'])) < 1e-2, \
        ('Stratifying a NiftiDataset on a continuous variables does not yield '
         'resembling means')

def test_dataset_stratified_multiple():
    np.random.seed(42)

    paths = [f'{i}.nii.gz' for i in range(1000)]
    labels = {
        'dataset': np.random.permutation(['d1'] * 500 + ['d2'] * 500),
        'sex': np.random.permutation(['F'] * 500 + ['M'] * 500),
        'age': np.random.uniform(0, 1, 1000)
    }
    data = NiftiDataset(paths, labels=labels)
    folds = data.stratified_folds(2, ['dataset', 'sex', 'age'])

    dataset_counts1 = Counter(folds[0].labels['dataset'])
    dataset_counts2 = Counter(folds[1].labels['dataset'])

    assert dataset_counts1 == dataset_counts2, \
        ('Stratifying a NiftiDataset on multiple variables does not yield '
         'equivalent splits for the first variable')

    M_counts1 = Counter(folds[0].labels['sex'])['M']
    M_counts2 = Counter(folds[1].labels['sex'])['M']

    assert abs(M_counts1 - M_counts2) < 10, \
        ('Stratifying a NiftiDataset on multiple variables does not yield '
         'approximately similar splits for the second variable')

    assert abs(np.mean(folds[0].labels['age']) - \
               np.mean(folds[1].labels['age'])) < 1e-2, \
        ('Stratifying a NiftiDataset on multiple variables does not yield '
         'approximately similar splits for the third variable')

def test_dataset_none_target():
    paths = ['tmp/path1.nii.gz', 'tmp/path2.nii.gz', 'tmp/path3.nii.gz']
    labels = {
        'y1': np.asarray([1, 2, 3], dtype=np.int64)
    }
    data = NiftiDataset(paths, labels=labels, target=None)

    assert np.array_equal([None, None, None], data.y), \
        'NiftiDataset with None target does not return list of Nones as label'

def test_dataset_get_property():
    paths = ['tmp/path1.nii.gz', 'tmp/path2.nii.gz', 'tmp/path3.nii.gz']
    labels = {
        'y1': np.asarray([1, 2, 3], dtype=np.int64)
    }
    data = NiftiDataset(paths, labels=labels, target=None)

    y = data.get_property('y1')

    assert np.array_equal(labels['y1'], y), \
        'NiftiDataset.get_property does not return the correct values'

def test_dataset_get_indexed_property():
    paths = ['tmp/path1.nii.gz', 'tmp/path2.nii.gz', 'tmp/path3.nii.gz']
    labels = {
        'y1': np.asarray([1, 2, 3], dtype=np.int64)
    }
    data = NiftiDataset(paths, labels=labels, target=None)

    y = data.get_property('y1')[1]

    assert np.array_equal(2, y), \
        'NiftiDataset.get_property does not return the correct indexed value'

def test_dataset_binary_label():
    paths = ['tmp/path1.nii.gz', 'tmp/path2.nii.gz', 'tmp/path3.nii.gz']
    labels = {
        'y1': np.asarray(['A', 'B', 'A'])
    }
    encoder = BinaryLabel(mapping={'A': 0, 'B': 1}, name='age')
    encoder.fit(labels['y1'])
    data = NiftiDataset(paths, labels=labels, target=None, encoders={'y1': encoder})
    y = data.get_property('y1')

    assert np.array_equal([0, 1, 0], y), \
        'NiftiDataset does not apply given BinaryLabel encoder'

def test_dataset_binary_label_from_json():
    paths = ['tmp/path1.nii.gz', 'tmp/path2.nii.gz', 'tmp/path3.nii.gz']
    labels = {
        'y1': np.asarray(['A', 'B', 'A'])
    }
    encoder = BinaryLabel(mapping={'A': 0, 'B': 1}, name='age')
    encoder.fit(labels['y1'])
    encoder = encode_object_as_json(encoder)
    data = NiftiDataset(paths, labels=labels, target=None,
                        encoders={'y1': encoder})
    y = data.get_property('y1')

    assert np.array_equal([0, 1, 0], y), \
        'NiftiDataset does not apply given BinaryLabel encoder'

def test_dataset_binary_label_from_file():
    try:
        paths = ['tmp/path1.nii.gz', 'tmp/path2.nii.gz', 'tmp/path3.nii.gz']
        labels = {
            'y1': np.asarray(['A', 'B', 'A'])
        }
        encoder = BinaryLabel(mapping={'A': 0, 'B': 1}, name='age')
        encoder.fit(labels['y1'])
        encoder.save('tmp.json')
        data = NiftiDataset(paths, labels=labels, target=None,
                            encoders={'y1': 'tmp.json'})
        y = data.get_property('y1')

        assert np.array_equal([0, 1, 0], y), \
            'NiftiDataset does not apply given BinaryLabel encoder'
    finally:
        if os.path.isfile('tmp.json'):
            os.remove('tmp.json')

def test_dataset_binary_label_to_from_json():
    paths = ['tmp/path1.nii.gz', 'tmp/path2.nii.gz', 'tmp/path3.nii.gz']
    labels = {
        'y1': np.asarray(['A', 'B', 'A'])
    }
    encoder = BinaryLabel(mapping={'A': 0, 'B': 1}, name='age')
    encoder.fit(labels['y1'])
    encoder = encode_object_as_json(encoder)
    data = NiftiDataset(paths, labels=labels, target=None,
                        encoders={'y1': encoder})

    data = NiftiDataset.from_json(data.json)
    y = data.get_property('y1')

    assert np.array_equal([0, 1, 0], y), \
        'NiftiDataset does not apply given BinaryLabel encoder'

def test_dataset_binary_label_to_from_jsonstring():
    paths = ['tmp/path1.nii.gz', 'tmp/path2.nii.gz', 'tmp/path3.nii.gz']
    labels = {
        'y1': np.asarray(['A', 'B', 'A'])
    }
    encoder = BinaryLabel(mapping={'A': 0, 'B': 1}, name='age')
    encoder.fit(labels['y1'])
    encoder = encode_object_as_json(encoder)
    data = NiftiDataset(paths, labels=labels, target=None,
                        encoders={'y1': encoder})

    data = NiftiDataset.from_jsonstring(data.jsonstring)
    y = data.get_property('y1')

    assert np.array_equal([0, 1, 0], y), \
        'NiftiDataset does not apply given BinaryLabel encoder'

def test_dataset_binary_label_to_from_file():
    try:
        paths = ['tmp/path1.nii.gz', 'tmp/path2.nii.gz', 'tmp/path3.nii.gz']
        labels = {
            'y1': np.asarray(['A', 'B', 'A'])
        }
        encoder = BinaryLabel(mapping={'A': 0, 'B': 1}, name='age')
        encoder.fit(labels['y1'])
        encoder = encode_object_as_json(encoder)
        data = NiftiDataset(paths, labels=labels, target=None,
                            encoders={'y1': encoder})

        data.save('tmp.json')
        data = load_dataset_from_jsonfile('tmp.json')
        y = data.get_property('y1')

        assert np.array_equal([0, 1, 0], y), \
            'NiftiDataset does not apply given BinaryLabel encoder'
    finally:
        if os.path.isfile('tmp.json'):
            os.remove('tmp.json')

def test_nifti_dataset_slicing_inherits_encoders():
    encoder = CategoricalLabel('y2', encoding='index')
    encoder.fit(['A', 'A', 'B', 'B', 'C'])
    paths = ['path1.nii.gz', 'path2.nii.gz', 'path3.nii.gz']
    labels = {
        'y1': np.asarray([1, 2, 3], dtype=np.int64),
        'y2': np.asarray(['A', 'B', 'C'])
    }

    data = NiftiDataset(paths, labels=labels, target='y1',
                        encoders={'y2': encoder})
    data = data[:2]

    assert 'y2' in data.encoders

def test_nifti_dataset_slicing_get_single_item():
    paths = ['path1.nii.gz', 'path2.nii.gz', 'path3.nii.gz']
    labels = {
        'y1': np.asarray([1, 2, 3], dtype=np.int64),
        'y2': np.asarray(['A', 'B', 'C'])
    }

    data = NiftiDataset(paths, labels=labels, target='y1')
    assert data[1, 'y1'] == 2, \
        ('Slicing a NiftiDataset with a tuple where the first element is an '
         'index does not return the correct value')

def test_nifti_dataset_slicing_get_single_item():
    paths = ['path1.nii.gz', 'path2.nii.gz', 'path3.nii.gz']
    labels = {
        'y1': np.asarray([1, 2, 3], dtype=np.int64),
        'y2': np.asarray(['A', 'B', 'C'])
    }

    data = NiftiDataset(paths, labels=labels, target='y1')
    assert np.array_equal(data[np.arange(2), 'y1'], np.arange(1, 3)), \
        ('Slicing a NiftiDataset with a tuple where the first element is an '
         'array does not return the correct value')

def test_nifti_dataset_slicing_applies_encoder():
    encoder = CategoricalLabel('test', encoding='index')
    encoder.fit(['A', 'B', 'C'])
    paths = ['path1.nii.gz', 'path2.nii.gz', 'path3.nii.gz']
    labels = {
        'y1': np.asarray([1, 2, 3], dtype=np.int64),
        'y2': np.asarray(['A', 'B', 'C'])
    }

    data = NiftiDataset(paths, labels=labels, target='y1',
                        encoders={'y2': encoder})
    assert data[1, 'y2'] == 1, \
        'Slicing a NiftiDataset does not apply the given encoding'


