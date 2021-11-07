import json
import logging
import os
import numpy as np

from shutil import rmtree

from pyment.data import load_dataset_from_jsonfile, NiftiDataset
from pyment.labels import BinaryLabel, ContinuousLabel

from utils import assert_exception


def test_dataset_length():
    paths = ['tmp/path1.nii.gz', 'tmp/path2.nii.gz', '/tmp/path3.nii.gz']
    data = NiftiDataset(paths)

    assert 3 == len(data), 'NiftiDataset does not report correct length'

def test_dataset_paths():
    paths = ['tmp/path1.nii.gz', 'tmp/path2.nii.gz', '/tmp/path3.nii.gz']
    data = NiftiDataset(paths)

    assert np.array_equal(paths, data.paths), \
        'NiftiDataset does not report correct paths'

def test_dataset_filenames():
    paths = ['tmp/path1.nii.gz', 'tmp/path2.nii.gz', '/tmp/path3.nii.gz']
    data = NiftiDataset(paths)

    filenames = ['path1.nii.gz', 'path2.nii.gz', 'path3.nii.gz']

    assert np.array_equal(filenames, data.filenames), \
        'NiftiDataset does not report correct filenames'

def test_dataset_ids():
    paths = ['tmp/path1.nii.gz', 'tmp/path2.nii.gz', '/tmp/path3.nii.gz']
    data = NiftiDataset(paths)

    ids = ['path1', 'path2', 'path3']

    assert np.array_equal(ids, data.ids), \
        'NiftiDataset does not report correct ids'

def test_dataset_init_target():
    paths = ['tmp/path1.nii.gz', 'tmp/path2.nii.gz', '/tmp/path3.nii.gz']
    data = NiftiDataset(paths, target='id')

    assert 'id' == data.target, \
        'Setting NiftiDataset target via init does not work'

def test_dataset_set_target():
    paths = ['tmp/path1.nii.gz', 'tmp/path2.nii.gz', '/tmp/path3.nii.gz']
    data = NiftiDataset(paths)
    data.target = 'id'

    assert 'id' == data.target, \
        'Setting NiftiDataset target via init does not work'

def test_dataset_id_target():
    paths = ['tmp/path1.nii.gz', 'tmp/path2.nii.gz', '/tmp/path3.nii.gz']
    data = NiftiDataset(paths, target='id')

    ids = ['path1', 'path2', 'path3']

    assert np.array_equal(ids, data.y), \
        'NiftiDataset with target=id does not return ids as labels'

def test_dataset_variables():
    paths = ['tmp/path1.nii.gz', 'tmp/path2.nii.gz', '/tmp/path3.nii.gz']
    labels = {
        'y1': [1, 2, 3],
        'y2': ['a', 'b', 'c']
    }
    data = NiftiDataset(paths, labels)

    assert ['y1', 'y2'] == data.variables, ('NiftiDataset variables does not '
                                            'return the correct labels')

def test_dataset_labels():
    paths = ['tmp/path1.nii.gz', 'tmp/path2.nii.gz', '/tmp/path3.nii.gz']
    labels = {
        'y1': [1, 2, 3],
        'y2': ['a', 'b', 'c']
    }
    data = NiftiDataset(paths, labels, target='y1')

    assert np.array_equal([1, 2, 3], data.y), \
        'NiftiDataset does not return the correct labels'

def test_dataset_invalid_target():
    paths = ['tmp/path1.nii.gz', 'tmp/path2.nii.gz', '/tmp/path3.nii.gz']
    labels = {
        'y1': [1, 2, 3],
        'y2': ['a', 'b', 'c']
    }
    data = NiftiDataset(paths, labels)

    exception = False

    try:
        data.target = 'y3'
    except Exception:
        exception = True

    assert exception, \
        'NiftiDataset does not raise an exception if setting an invalid target'

def test_dataset_json_paths():
    paths = ['tmp/path1.nii.gz', 'tmp/path2.nii.gz', '/tmp/path3.nii.gz']
    data = NiftiDataset(paths)
    
    assert 'paths' in data.json, ('NiftiDataset.json does not contain a field '
                                  'for paths')
    assert np.array_equal(paths, data.json['paths']), \
        'NiftiDataset.json does not contain correct paths'

def test_dataset_json_labels():
    paths = ['tmp/path1.nii.gz', 'tmp/path2.nii.gz', '/tmp/path3.nii.gz']
    labels = {
        'y1': [1, 2, 3],
        'y2': ['a', 'b', 'c']
    }
    data = NiftiDataset(paths, labels)

    assert 'labels' in data.json, \
        'NiftiDataset.json does not contain a field for labels'
    assert 'y1' in data.labels and \
            np.array_equal([1, 2, 3], data.labels['y1']), \
        'NiftiDataset.json does not contain correct labels'
    assert 'y2' in data.labels and \
            np.array_equal(['a', 'b', 'c'], data.labels['y2']), \
        'NiftiDataset.json does not contain correct labels'
        
def test_dataset_json_target():
    paths = ['tmp/path1.nii.gz', 'tmp/path2.nii.gz', '/tmp/path3.nii.gz']
    labels = {
        'y1': [1, 2, 3],
        'y2': ['a', 'b', 'c']
    }
    data = NiftiDataset(paths, labels, target='y1')

    assert 'target' in data.json, \
        'NiftiDataset.json does not contain a field for target'
    assert 'y1' == data.json['target'], \
        'NiftiDataset.json does not contain correct target'

def test_dataset_json_handles_numpy_labels():
    paths = ['tmp/path1.nii.gz', 'tmp/path2.nii.gz', '/tmp/path3.nii.gz']
    labels = {
        'y1': np.asarray([1, 2, 3], dtype=np.int64)
    }
    data = NiftiDataset(paths, labels)

    exception = False

    try:
        json.dumps(data.json)
    except Exception:
        exception = True

    assert not exception, 'NiftiDataset.json does not handle numpy arrays'

def test_dataset_equal():
    paths = ['tmp/path1.nii.gz', 'tmp/path2.nii.gz', '/tmp/path3.nii.gz']
    labels = {
        'y1': np.asarray([1, 2, 3], dtype=np.int64)
    }
    d1 = NiftiDataset(paths, labels, target='y1')
    d2 = NiftiDataset(paths, labels, target='y1')

    assert d1 == d2, 'Two equal NiftiDatasets are not considered equal'

def test_dataset_unequal_paths():
    paths1 = ['tmp1/path1.nii.gz', 'tmp1/path2.nii.gz']
    paths2 = ['tmp2/path1.nii.gz', 'tmp2/path2.nii.gz']
    labels = {
        'y1': np.asarray([1, 2], dtype=np.int64)
    }
    d1 = NiftiDataset(paths1, labels, target='y1')
    d2 = NiftiDataset(paths2, labels, target='y1')

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
    d1 = NiftiDataset(paths, labels1)
    d2 = NiftiDataset(paths, labels2)

    assert d1 != d2, \
        'Two NiftiDatasets with different labels are considered equal'

def test_dataset_unequal_target():
    paths = ['tmp/path1.nii.gz', 'tmp/path2.nii.gz']
    labels = {
        'y1': np.asarray([1, 2], dtype=np.int64),
        'y2': np.asarray([1, 2], dtype=np.int64)
    }
    d1 = NiftiDataset(paths, labels, target='y1')
    d2 = NiftiDataset(paths, labels, target='y2')

    assert d1 != d2, \
        'Two NiftiDatasets with different targets are considered equal'

def test_dataset_jsonstring():
    paths = ['tmp/path1.nii.gz', 'tmp/path2.nii.gz', '/tmp/path3.nii.gz']
    labels = {
        'y1': np.asarray([1, 2, 3], dtype=np.int64)
    }
    data = NiftiDataset(paths, labels, target='y1')
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
    paths = ['tmp/path1.nii.gz', 'tmp/path2.nii.gz', '/tmp/path3.nii.gz']
    labels = {
        'y1': np.asarray([1, 2, 3], dtype=np.int64)
    }
    data = NiftiDataset(paths, labels, target='y1')
    obj = json.loads(data.jsonstring)

    assert 'target' in obj, 'NiftiDataset.jsonstring does not contain target'
    assert 'y1' == obj['target'], \
        'NiftiDataset.jsonstring contains wrong target'


def test_dataset_from_json():
    obj = {
        'paths': ['tmp/path1.nii.gz', 'tmp/path2.nii.gz', '/tmp/path3.nii.gz'],
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
    paths = ['tmp/path1.nii.gz', 'tmp/path2.nii.gz', '/tmp/path3.nii.gz']
    labels = {
        'y1': np.asarray([1, 2, 3], dtype=np.int64)
    }
    data1 = NiftiDataset(paths, labels, target='y1')
    data2 = NiftiDataset.from_json(data1.json)

    assert data1 == data2, \
        'NiftiDataset to and from json does not yield an equivalent object'


def test_dataset_to_from_jsonstring():
    paths = ['tmp/path1.nii.gz', 'tmp/path2.nii.gz', '/tmp/path3.nii.gz']
    labels = {
        'y1': np.asarray([1, 2, 3], dtype=np.int64)
    }
    data1 = NiftiDataset(paths, labels, target='y1')
    data2 = NiftiDataset.from_jsonstring(data1.jsonstring)

    assert data1 == data2, \
        ('NiftiDataset to and from jsonstring does not yield an equivalent '
         'object')

def test_dataset_save_load():
    try:
        paths = ['tmp/path1.nii.gz', 'tmp/path2.nii.gz', '/tmp/path3.nii.gz']
        labels = {
            'y1': np.asarray([1, 2, 3], dtype=np.int64)
        }
        data1 = NiftiDataset(paths, labels, target='y1')
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
    paths = ['tmp/path1.nii.gz', 'tmp/path2.nii.gz', '/tmp/path3.nii.gz']
    labels = {
        'y1': [1, 2, 3],
        'y2': ['a', 'b', 'c']
    }

    exception = False

    try:
        data = NiftiDataset(paths, labels, target='y3')
    except Exception:
        exception = True

    assert exception, \
        'Instatiating NiftiDataset with invalid target does not raise an error'

def test_dataset_multitarget():
    paths = ['tmp/path1.nii.gz', 'tmp/path2.nii.gz', '/tmp/path3.nii.gz']
    labels = {
        'y1': np.asarray([1, 2, 3], dtype=np.int64),
        'y2': np.asarray(['a', 'b', 'c'])
    }
    data = NiftiDataset(paths, labels, target=['y1', 'y2'])
    
    assert ['y1', 'y2'] == data.target, \
        'Unable to set multiple targets for NiftiDataset'
    
    y = np.asarray([
        [1, 'a'],
        [2, 'b'],
        [3, 'c']
    ])

    assert np.array_equal(y, data.y), \
        'NiftiDataset does not return correct y with multiple targets'

def test_dataset_label_target():
    paths = ['tmp/path1.nii.gz', 'tmp/path2.nii.gz', '/tmp/path3.nii.gz']
    labels = {
        'y1': np.asarray([0, 1, 0], dtype=np.int64)
    }

    exception = False

    try:
        data = NiftiDataset(paths, labels, target=BinaryLabel('y1'))
    except Exception as e:
        exception = True


    assert not exception, 'NiftiDataset does not allow BinaryLabel as target'
    assert BinaryLabel('y1') == data.target, \
        'NiftiDataset with BinaryLabel does not have the correct target'


def test_dataset_label_target_transform():
    paths = ['tmp/path1.nii.gz', 'tmp/path2.nii.gz', '/tmp/path3.nii.gz']
    labels = {
        'y1': np.asarray(['A', 'B', 'C'])
    }
    label = BinaryLabel('y1', encoding={'A': 0, 'B': 1})
    label.fit(np.asarray(['A', 'B', 'C']))

    data = NiftiDataset(paths, labels, target=label)

    assert np.array_equal([0., 1., np.nan], data.y, equal_nan=True), \
        'NiftiDataset does not apply transform from BinaryLabel'

def test_dataset_label_target_unfitted_warning(caplog):
    paths = ['tmp/path1.nii.gz', 'tmp/path2.nii.gz', '/tmp/path3.nii.gz']
    labels = {
        'y1': np.asarray(['A', 'B', 'C'])
    }
    label = BinaryLabel('y1', encoding={'A': 0, 'B': 1})

    data = NiftiDataset(paths, labels, target=label)

    with caplog.at_level(logging.WARNING):
        data.y
        assert caplog.text != '', \
            'NiftiDataset with unfitted BinaryLabel does not raise a warning'

def test_dataset_to_from_json_with_binary_label():
    paths = ['tmp/path1.nii.gz', 'tmp/path2.nii.gz', '/tmp/path3.nii.gz']
    labels = {
        'y1': np.asarray(['A', 'B', 'C'])
    }
    label = BinaryLabel('y1', encoding={'A': 0, 'B': 1})
    label.fit(np.asarray(['A', 'B', 'C', 'B']))

    data1 = NiftiDataset(paths, labels, target=label)
    data2 = NiftiDataset.from_json(data1.json)

    assert data1.target == data2.target, \
        'NiftiDataset to and from json does not retain BinaryLabel target'

def test_dataset_to_from_json_with_binary_label_y():
    paths = ['tmp/path1.nii.gz', 'tmp/path2.nii.gz', '/tmp/path3.nii.gz']
    labels = {
        'y1': np.asarray(['A', 'B', 'C'])
    }
    label = BinaryLabel('y1', encoding={'A': 0, 'B': 1})
    label.fit(np.asarray(['A', 'B', 'C', 'B']))

    data1 = NiftiDataset(paths, labels, target=label)
    data2 = NiftiDataset.from_json(data1.json)

    assert np.array_equal(data1.y, data2.y, equal_nan=True), \
        ('NiftiDataset to and from json does apply transform equally before '
         'and after')

def test_dataset_target_from_file():
    try:
        label = BinaryLabel('y1', encoding={'A': 0, 'B': 1})
        label.fit(['A', 'B', 'C', 'B'])
        label.save('tmp.json')

        paths = ['tmp/path1.nii.gz', 'tmp/path2.nii.gz', '/tmp/path3.nii.gz']
        labels = {
            'y1': np.asarray(['A', 'B', 'C', 'B'])
        }
        data = NiftiDataset(paths, labels, target='tmp.json')

        assert np.array_equal([0, 1, np.nan, 1], data.y, equal_nan=True), \
            'NiftiDataset does not apply label from file'
    finally:
        if os.path.isfile('tmp.json'):
            os.remove('tmp.json')

def test_dataset_targets_from_files():
    try:
        label1 = BinaryLabel('y1', encoding={'A': 0, 'B': 1})
        label1.fit(['A', 'B', 'C', 'B'])
        label1.save('tmp1.json')

        label2 = BinaryLabel('y2', encoding={'A': 1, 'B': 0})
        label2.fit(['A', 'B', 'C', 'B'])
        label2.save('tmp2.json')

        paths = ['tmp/path1.nii.gz', 'tmp/path2.nii.gz', '/tmp/path3.nii.gz']
        labels = {
            'y1': np.asarray(['A', 'B', 'C', 'B']),
            'y2': np.asarray(['A', 'B', 'C', 'B'])
        }
        data = NiftiDataset(paths, labels, target=['tmp1.json', 'tmp2.json'])

        expected = np.asarray([
            [0, 1],
            [1, 0],
            [np.nan, np.nan],
            [1, 0]
        ])

        assert np.array_equal(expected, data.y, equal_nan=True), \
            'NiftiDataset does not apply label from file'
    finally:
        if os.path.isfile('tmp1.json'):
            os.remove('tmp1.json')
        if os.path.isfile('tmp2.json'):
            os.remove('tmp2.json')

def test_niftidataset_add_unknown_other():
    paths = ['tmp/path1.nii.gz', 'tmp/path2.nii.gz', '/tmp/path3.nii.gz']
    labels = {
        'y1': np.asarray([1, 2, 3], dtype=np.int64)
    }
    data = NiftiDataset(paths, labels, target='y1')
    
    assert_exception(lambda: data + 5,
                     message=('Adding an int to a NiftiDataset does not raise '
                              'an error'))

def test_niftidataset_add_paths():
    paths = ['tmp/path1.nii.gz', 'tmp/path2.nii.gz', '/tmp/path3.nii.gz']
    data = NiftiDataset(paths) + NiftiDataset(paths)
    
    assert np.array_equal(np.concatenate([paths, paths]), data.paths), \
        'Two added NiftiDatasets does not produce the correct paths'

def test_niftidataset_add_labels():
    paths = ['tmp/path1.nii.gz', 'tmp/path2.nii.gz', '/tmp/path3.nii.gz']
    labels = {
        'y1': np.asarray([1, 2, 3], dtype=np.int64)
    }

    data = NiftiDataset(paths, labels) + NiftiDataset(paths, labels)

    assert 'y1' in data.labels, \
        'Two added NiftiDatasets does not have the right label keys'
    assert np.array_equal([1, 2, 3, 1, 2, 3], data.labels['y1']), \
        'Two added NiftiDatasets does not have the correct labels'

def test_niftidataset_add_labels_none_other():
    paths = ['tmp/path1.nii.gz', 'tmp/path2.nii.gz', '/tmp/path3.nii.gz']
    labels = {
        'y1': np.asarray([1, 2, 3], dtype=np.int64)
    }

    data = NiftiDataset(paths, labels) + NiftiDataset(paths)

    assert 'y1' in data.labels, \
        ('Two added NiftiDatasets does not have the right label keys '
         'when the other dataset has no labels')
    assert np.array_equal([1, 2, 3, np.nan, np.nan, np.nan], data.labels['y1'],
                          equal_nan=True), \
        ('Two added NiftiDatasets does not have the correct labels '
         'when the other dataset has no labels')

def test_niftidataset_add_labels_none_self():
    paths = ['tmp/path1.nii.gz', 'tmp/path2.nii.gz', '/tmp/path3.nii.gz']
    labels = {
        'y1': np.asarray([1, 2, 3], dtype=np.int64)
    }

    data = NiftiDataset(paths) + NiftiDataset(paths, labels)

    assert 'y1' in data.labels, \
        ('Two added NiftiDatasets does not have the right label keys '
         'when self has no labels')
    assert np.array_equal([np.nan, np.nan, np.nan, 1, 2, 3], data.labels['y1'],
                          equal_nan=True), \
        ('Two added NiftiDatasets does not have the correct labels '
         'when self has no labels')

def test_niftidataset_add_labels_missing():
    paths = ['tmp/path1.nii.gz', 'tmp/path2.nii.gz', '/tmp/path3.nii.gz']
    labels1 = {
        'y1': np.asarray([1, 2, 3], dtype=np.int64)
    }
    labels2 = {
        'y2': np.asarray([1, 2, 3], dtype=np.int64)
    }

    data = NiftiDataset(paths, labels1) + NiftiDataset(paths, labels2)

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
    paths = ['tmp/path1.nii.gz', 'tmp/path2.nii.gz', '/tmp/path3.nii.gz']
    labels = {
        'y1': np.asarray([1, 2, 3], dtype=np.int64)
    }
    data = NiftiDataset(paths, labels, 'y1') + \
           NiftiDataset(paths, labels, 'y1')

    assert 'y1' == data.target, \
        ('Adding two NiftiDatasets with the same string target does not '
         'produce a new NiftiDataset with the same target')

def test_niftidataset_add_equal_label_target():
    paths = ['tmp/path1.nii.gz', 'tmp/path2.nii.gz', '/tmp/path3.nii.gz']
    labels = {
        'y1': np.asarray([1, 2, 3], dtype=np.int64)
    }
    data = NiftiDataset(paths, labels, ContinuousLabel('y1')) + \
           NiftiDataset(paths, labels, ContinuousLabel('y1'))

    assert ContinuousLabel('y1') == data.target, \
        ('Adding two NiftiDatasets with the same Label target does not '
         'produce a new NiftiDataset with the same target')

def test_niftidataset_add_different_targets(caplog):
    paths = ['tmp/path1.nii.gz', 'tmp/path2.nii.gz', '/tmp/path3.nii.gz']
    labels = {
        'y1': np.asarray([1, 2, 3], dtype=np.int64),
        'y2': np.asarray([1, 2, 3])
    }


    with caplog.at_level(logging.WARNING):
        data = NiftiDataset(paths, labels, ContinuousLabel('y1')) + \
               NiftiDataset(paths, labels, ContinuousLabel('y2'))
        assert caplog.text != '', \
            ('Adding two NiftiDatasets with different targets does not raise '
             'a warning')

def test_niftidataset_stratified():
    pass
