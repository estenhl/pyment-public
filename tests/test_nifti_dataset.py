from pyment.data import NiftiDataset


def test_dataset_length():
    paths = ['tmp/path1.nii.gz', 'tmp/path2.nii.gz', '/tmp/path3.nii.gz']
    data = NiftiDataset(paths)

    assert 3 == len(data), 'NiftiDataset does not report correct length'

def test_dataset_paths():
    paths = ['tmp/path1.nii.gz', 'tmp/path2.nii.gz', '/tmp/path3.nii.gz']
    data = NiftiDataset(paths)

    assert paths == data.paths, 'NiftiDataset does not report correct paths'

def test_dataset_filenames():
    paths = ['tmp/path1.nii.gz', 'tmp/path2.nii.gz', '/tmp/path3.nii.gz']
    data = NiftiDataset(paths)

    filenames = ['path1.nii.gz', 'path2.nii.gz', 'path3.nii.gz']

    assert filenames == data.filenames, ('NiftiDataset does not report '
                                         'correct filenames')

def test_dataset_ids():
    paths = ['tmp/path1.nii.gz', 'tmp/path2.nii.gz', '/tmp/path3.nii.gz']
    data = NiftiDataset(paths)

    ids = ['path1', 'path2', 'path3']

    assert ids == data.ids, 'NiftiDataset does not report correct ids'

def test_dataset_init_target():
    paths = ['tmp/path1.nii.gz', 'tmp/path2.nii.gz', '/tmp/path3.nii.gz']
    data = NiftiDataset(paths, target='id')

    assert 'id' == data.target, ('Setting NiftiDataset target via init does '
                                 'not work')

def test_dataset_set_target():
    paths = ['tmp/path1.nii.gz', 'tmp/path2.nii.gz', '/tmp/path3.nii.gz']
    data = NiftiDataset(paths)
    data.target = 'id'

    assert 'id' == data.target, ('Setting NiftiDataset target via init does '
                                 'not work')

def test_dataset_id_target():
    paths = ['tmp/path1.nii.gz', 'tmp/path2.nii.gz', '/tmp/path3.nii.gz']
    data = NiftiDataset(paths, target='id')

    ids = ['path1', 'path2', 'path3']

    assert ids == data.y, ('NiftiDataset with target=id does not return ids '
                           'as labels')

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

    assert [1, 2, 3] == data.y, ('NiftiDataset does not return the correct '
                                 'labels')

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

    assert exception, ('NiftiDataset does not raise an exception if setting '
                       'an invalid target')