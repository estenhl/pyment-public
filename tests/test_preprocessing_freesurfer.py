import os

from mock import patch, MagicMock
from shutil import rmtree

from pyment.utils.preprocessing import autorecon1, autorecon1_folder, \
                                       convert_mgz_to_nii_gz, \
                                       convert_mgz_to_nii_gz_folder

@patch('pyment.utils.preprocessing.freesurfer.run')
def test_autorecon1(mock):
    try:
        os.mkdir('tmp')
        autorecon1('mock.nii.gz', subject='mock', subjects_dir='tmp')

        assert 1 == mock.call_count, 'autorecon1 does not call run function'
        cmd = mock.call_args[0][0]
        expected_cmd = ('recon-all -s mock -sd tmp -i mock.nii.gz '
                        '-autorecon1 -no-isrunning')
        assert expected_cmd == cmd, ('autorecon1 does not call run with '
                                     'correct command')
    finally:
        rmtree('tmp')

@patch('pyment.utils.preprocessing.freesurfer.run')
def test_autorecon1_disable_noisrunning(mock):
    try:
        os.mkdir('tmp')
        autorecon1('mock.nii.gz', subject='mock', subjects_dir='tmp',
                   noisrunning=False)

        cmd = mock.call_args[0][0]
        expected_cmd = 'recon-all -s mock -sd tmp -i mock.nii.gz -autorecon1'
        assert expected_cmd == cmd, ('autorecon1 noisrunning does not toggle '
                                     '-noisrunning in freesurfer call')
    finally:
        rmtree('tmp')

@patch('pyment.utils.preprocessing.freesurfer.run')
def test_autorecon1_disable_silence(mock):
    try:
        os.mkdir('tmp')
        autorecon1('mock.nii.gz', subject='mock', subjects_dir='tmp',
                   silence=False)

        assert not mock.call_args[1]['silence'], ('autorecon1 silence kwargs '
                                                  'does not affect run '
                                                  'arguments')
    finally:
        rmtree('tmp')

@patch('pyment.utils.preprocessing.freesurfer.run')
def test_autorecon1_folder(mock):
    try:
        os.mkdir('tmp')
        src = os.path.join('tmp', 'src')
        os.mkdir(src)
        dest = os.path.join('tmp', 'dest')
    
        for i in range(5):
            with open(os.path.join('tmp', 'src', f'sub{i}.mgz'), 'w') as f:
                f.write('test')

        autorecon1_folder(src, dest)

        assert os.path.isdir(dest), ('autorecon1_folder does not create '
                                     'destination folder')

        assert 5 == mock.call_count, ('autorecon1_folder does not call run '
                                      'for every subject')
    finally:
        rmtree('tmp')

@patch('pyment.utils.preprocessing.freesurfer.run')
def test_convert_mgz_to_nii_gz(mock):
    convert_mgz_to_nii_gz('src.mgz', 'dest.nii.gz')

    assert 1 == mock.call_count, ('convert_mgz_to_nii_gz does not call '
                                    'run function')
    cmd = mock.call_args[0][0]
    expected_cmd = 'mri_convert src.mgz dest.nii.gz -ot nii'
    assert expected_cmd == cmd, ('autorecon1 does not call run with '
                                    'correct command')

@patch('pyment.utils.preprocessing.freesurfer.run')
def test_convert_mgz_to_nii_gz_invalid_file(mock):
    exception = False

    try:
        convert_mgz_to_nii_gz('src.nii.gz', 'dest.nii.gz')
    except Exception as e:
        exception = True

    assert exception, ('Calling convert_mgz_to_nii_gz with a non-mgz file '
                       'does not raise an error')

@patch('pyment.utils.preprocessing.freesurfer.run')
def test_convert_mgz_to_nii_gz_folder(mock):
    try:
        os.mkdir('tmp')
        src = os.path.join('tmp', 'src')
        os.mkdir(src)
        dest = os.path.join('tmp', 'dest')
    
        for i in range(5):
            with open(os.path.join('tmp', 'src', f'sub{i}.mgz'), 'w') as f:
                f.write('test')

        convert_mgz_to_nii_gz_folder(src, dest)

        assert os.path.isdir(dest), ('convert_mgz_to_nii_gz_folder does not '
                                     'create destination folder')

        assert 5 == mock.call_count, ('convert_mgz_to_nii_gz_folder does not '
                                      'call run for every subject')
    finally:
        rmtree('tmp')