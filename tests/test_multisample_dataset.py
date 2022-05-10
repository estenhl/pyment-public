import os
import nibabel as nib
import numpy as np
import pandas as pd

from copy import copy
from functools import reduce
from shutil import rmtree

from pyment.data.datasets import MultisampleNiftiDataset
from pyment.data.generators import AsyncNiftiGenerator


def test_multisample_dataset_has_subjects():
    paths = [f'{i}.nii.gz' for i in range(10)]
    subjects = [str(i) for i in range(10)]
    dataset = MultisampleNiftiDataset(subjects=subjects, paths=paths)

    assert set(dataset.subjects) == set(subjects), \
        'MultisampleNiftiDataset does not have subjects'


def test_multisample_dataset_from_folder():
    try:
        os.makedirs(os.path.join('tmp', 'images'))

        subjects = []
        ids = []
        labels = []
        for i in range(3):
            for j in range(4):
                subject = i
                id = f'{i}_{j}'
                filename = f'{id}.nii.gz'
                path = os.path.join('tmp', 'images', filename)

                with open(path, 'a') as f:
                    os.utime(path)

                subjects.append(subject)
                ids.append(id)
                labels.append(2**i+3**j)

        df = pd.DataFrame({'subject': subjects, 'id': ids, 'y': labels})
        df.to_csv(os.path.join('tmp', 'labels.csv'), index=False)

        dataset = MultisampleNiftiDataset.from_folder('tmp')

        assert set(dataset.subjects) == set(subjects), \
            ('MultisampleNiftiDataset.from_folder does not fetch subjects '
             'correctly')
    finally:
        if os.path.isdir('tmp'):
           rmtree('tmp')

def test_multisample_dataset_length():
    subjects = reduce(lambda x, y: x + y,
                      [[i for j in range(4)] for i in range(3)])
    paths = reduce(lambda x, y: x + y,
                   [[f'{i}_{j}.nii.gz' for j in range(4)] for i in range(3)])

    dataset = MultisampleNiftiDataset(paths=paths, subjects=subjects)

    assert len(dataset) == 3, \
        ('MultisampleNiftiDataset does not have a length which corresponds '
         'to the number of unique subjects')

def test_multisample_dataset_gives_one_path_per_subject():
    subjects = reduce(lambda x, y: x + y,
                      [[i for j in range(4)] for i in range(3)])
    paths = reduce(lambda x, y: x + y,
                   [[f'{i}_{j}.nii.gz' for j in range(4)] for i in range(3)])

    dataset = MultisampleNiftiDataset(subjects=subjects, paths=paths)

    assert len(dataset.paths) == 3, \
        'MultisampleNiftiDataset does not have the correct number of paths'
    ids = [p.split('_')[0] for p in dataset.paths]
    assert np.array_equal(ids, ['0', '1', '2']), \
        ('MultisampleNiftiDataset does not have one path per subject in the '
         'correct order')

def test_multisample_dataset_get_first():
    subjects = reduce(lambda x, y: x + y,
                      [[i for j in range(4)] for i in range(3)])
    paths = reduce(lambda x, y: x + y,
                   [[f'{i}_{j}.nii.gz' for j in range(4)] for i in range(3)])

    dataset = MultisampleNiftiDataset(subjects=subjects, paths=paths,
                                      strategy='first')

    assert len(dataset.paths) == 3, \
        'MultisampleNiftiDataset does not have the correct number of paths'
    ids = [p.split('_')[1].split('.')[0] for p in dataset.paths]
    assert np.array_equal(ids, ['0', '0', '0']), \
        ('MultisampleNiftiDataset with \'first\' strategy does not return '
         'the first sample for all subjects')

def test_multisample_dataset_get_last():
    subjects = reduce(lambda x, y: x + y,
                      [[i for j in range(4)] for i in range(3)])
    paths = reduce(lambda x, y: x + y,
                   [[f'{i}_{j}.nii.gz' for j in range(4)] for i in range(3)])

    dataset = MultisampleNiftiDataset(subjects=subjects, paths=paths,
                                      strategy='last')

    assert len(dataset.paths) == 3, \
        'MultisampleNiftiDataset does not have the correct number of paths'
    ids = [p.split('_')[1].split('.')[0] for p in dataset.paths]
    assert np.array_equal(ids, ['3', '3', '3']), \
        ('MultisampleNiftiDataset with \'first\' strategy does not return '
         'the last sample for all subjects')

def test_multisample_dataset_get_random():
    np.random.seed(42)

    subjects = reduce(lambda x, y: x + y,
                      [[i for j in range(4)] for i in range(3)])
    paths = reduce(lambda x, y: x + y,
                   [[f'{i}_{j}.nii.gz' for j in range(4)] for i in range(3)])

    dataset = MultisampleNiftiDataset(subjects=subjects, paths=paths,
                                      strategy='random')

    assert len(dataset.paths) == 3, \
        'MultisampleNiftiDataset does not have the correct number of paths'
    ids = [p.split('_')[1].split('.')[0] for p in dataset.paths]
    assert 1 < len(set(ids)), \
        ('MultisampleNiftiDataset with \'random\' strategy does not return '
         'a random sample for all subjects')


def test_multisample_random_labels():
    subjects = reduce(lambda x, y: x + y,
                      [[i for j in range(4)] for i in range(3)])
    paths = reduce(lambda x, y: x + y,
                   [[f'{i}_{j}.nii.gz' for j in range(4)] for i in range(3)])
    labels = copy(subjects)

    dataset = MultisampleNiftiDataset(paths=paths, subjects=subjects,
                                      labels={'y': labels}, target='y',
                                      strategy='random')

    assert np.array_equal([0, 1, 2], dataset.y), \
        ('MultisampleNiftiDataset does not retain connection between labels '
         'and samples when using strategy=\'random\'')


def test_multisample_json():
    subjects = reduce(lambda x, y: x + y,
                      [[i for j in range(4)] for i in range(3)])
    paths = reduce(lambda x, y: x + y,
                   [[f'{i}_{j}.nii.gz' for j in range(4)] for i in range(3)])
    labels = copy(subjects)

    dataset = MultisampleNiftiDataset(paths=paths, subjects=subjects,
                                      labels={'y': labels}, target='y',
                                      strategy='random')

    assert 'subjects' in dataset.json, \
        'MultisampleNiftiDataset.json does not contain a \'subject\' entry'
    assert np.array_equal(subjects, dataset.json['subjects']), \
        'MultisampleNiftiDataset.json does not contain the correct subjects'
    assert 'strategy' in dataset.json, \
        'MultisampleNiftiDataset.json does not contain a \'strategy\' entry'
    assert 'random' == dataset.json['strategy'], \
        'MultisampleNiftiDataset.json contains the wrong strategy'

def test_multisample_add():
    s1 = reduce(lambda x, y: x + y,
         [[f'd=0_s={i}' for j in range(4)] for i in range(3)])
    p1 = reduce(lambda x, y: x + y,
                [[f'd=0_s={i}_t={j}.nii.gz' for j in range(4)] \
                 for i in range(3)])
    l1 = copy(s1)

    d1 = MultisampleNiftiDataset(paths=p1, subjects=s1,
                                 labels={'y': l1}, target='y',
                                 strategy='random')

    s2 = reduce(lambda x, y: x + y,
         [[f'd=1_s={i}' for j in range(4)] for i in range(3)])
    p2 = reduce(lambda x, y: x + y,
                [[f'd=1_s={i}_t={j}.nii.gz' for j in range(4)] \
                 for i in range(3)])
    l2 = copy(s2)

    d2 = MultisampleNiftiDataset(paths=p2, subjects=s2,
                                 labels={'y': l2}, target='y',
                                 strategy='random')

    dataset = d1 + d2

    expected = ['d=0_s=0', 'd=0_s=1', 'd=0_s=2', 'd=1_s=0', 'd=1_s=1',
                'd=1_s=2']

    assert np.array_equal(expected, dataset.subjects), \
        'Adding two MultisampleNiftiDatasets mess up the subjects'

    assert np.array_equal(expected, dataset.y), \
        'Adding two MultisampleNiftiDatasets mess up the labels'

def test_multisample_shuffle():
    paths = [f'{i}.nii.gz' for i in range(10)]
    subjects = np.arange(10).astype(str)
    labels = np.arange(10).astype(str)

    dataset = MultisampleNiftiDataset(subjects=subjects, paths=paths,
                                      labels={'y': labels}, target='y')

    dataset = dataset.shuffled
    subjects = dataset.subjects
    ids = [p.split('.')[0] for p in dataset.paths]
    labels = dataset.y

    assert np.array_equal(subjects, ids), \
        ('Shuffling a MultisampleNiftiDataset breaks the link between '
         'subjects and paths')
    assert np.array_equal(subjects, labels), \
        ('Shuffling a MultisampleNiftiDataset breaks the link between '
         'subjects and labels')

def test_multisample_complex_generator():
    np.random.seed(42)

    try:
        os.mkdir('tmp')
        os.makedirs(os.path.join('tmp', 'd0', 'images'))
        os.makedirs(os.path.join('tmp', 'd1', 'images'))

        datasets = [[], []]

        for i in range(30):
            subject = i
            for j in range(5):
                id = f'{i}_{j}'
                data = np.ones((3, 3, 3)) * (i+(j/10.))
                image = nib.Nifti1Image(data, affine=np.eye(4))
                label = i % 3

                dataset = int(np.round(np.random.uniform(0, 1)))

                nib.save(image, os.path.join('tmp', f'd{dataset}', 'images',
                                             f'{id}.nii.gz'))
                datasets[dataset].append({
                    'subject': subject,
                    'id': id,
                    'label': label
                })

        for i in range(len(datasets)):
            df = pd.DataFrame(datasets[i])
            df.to_csv(os.path.join('tmp', f'd{i}', 'labels.csv'),
                      index=False)

        datasets = [
            MultisampleNiftiDataset.from_folder(os.path.join('tmp', f'd{i}'),
                                                strategy='random',
                                                target='label') \
            for i in range(2)
        ]
        dataset = reduce(lambda x, y: x + y, datasets)
        generator = AsyncNiftiGenerator(dataset, batch_size=5, shuffle=True,
                                        infinite=False, threads=2)

        orders = []
        subjects = {}

        for _ in range(10):
            order = []
            for X, y in generator:
                for i in range(len(X)):
                    img = X[i]
                    value = np.unique(img)[0]
                    label = y[i]
                    subject = int(value)
                    session = int(np.round(value % 1, 1) * 10)

                    assert subject % 3 == label, \
                        ('MultisampleNiftiDataset with generator loses '
                         'relationship between images and labels')

                    order.append(subject)

                    if not subject in subjects:
                        subjects[subject] = []

                    subjects[subject].append(session)

            orders.append(order)

        for order in orders:
            assert np.array_equal(sorted(order), np.arange(30)), \
                ('MultisampleNiftiDataset with generator does not return one '
                 'datapoint per subject per iteration')

        for i in range(len(orders)):
            for j in range(len(orders)):
                if i == j:
                    continue

                assert not np.array_equal(orders[i], orders[j]), \
                    ('Using a generator with shuffle=True and a '
                     'MultisampleNiftiDataset does not yield subjects in '
                     'different order between epochs')

        for subject in subjects:
            assert 1 < len(np.unique(subjects[subject])), \
                ('MultisampleNiftiDataset with strategy=\'random\' does not '
                 'return different datapoints for the same subject in '
                 'different iterations')

    finally:
        if os.path.isdir(os.path.join('tmp')):
            rmtree(os.path.join('tmp'))

