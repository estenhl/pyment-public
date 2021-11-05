import os

from mock import patch, MagicMock
from shutil import rmtree

from pyment.utils.preprocessing import flirt, flirt_folder, reorient2std, \
                                       reorient2std_folder


@patch('pyment.utils.preprocessing.fsl.run')
def test_autorecon1(mock):
    reorient2std('src.nii.gz', 'dest.nii.gz')

    assert 1 == mock.call_count, 'reorient2std does not call run function'
    cmd = mock.call_args[0][0]
    expected_cmd = 'fslreorient2std src.nii.gz dest.nii.gz'
    assert expected_cmd == cmd, \
        'fslreorient2std does not call run with correct command'

@patch('pyment.utils.preprocessing.fsl.run')
def test_reorient2std_folder(mock):
    try:
        os.mkdir('tmp')
        src = os.path.join('tmp', 'src')
        os.mkdir(src)
        dest = os.path.join('tmp', 'dest')
    
        for i in range(5):
            with open(os.path.join('tmp', 'src', f'sub{i}.mgz'), 'w') as f:
                f.write('test')

        reorient2std_folder(src, dest)

        assert os.path.isdir(dest), \
            'reorient2std_folder does not create destination folder'

        assert 5 == mock.call_count, \
            'reorient2std_folder does not call run for every subject'
    finally:
        rmtree('tmp')

@patch('pyment.utils.preprocessing.fsl.run')
def test_flirt(mock):
    flirt('src.nii.gz', 'dest.nii.gz', template='template.nii.gz')

    assert 1 == mock.call_count, 'flirt does not call run function'
    cmd = mock.call_args[0][0]
    expected_cmd = ('flirt -in src.nii.gz -out dest.nii.gz -ref '
                    'template.nii.gz -dof 6')
    assert expected_cmd == cmd, 'flirt does not call run with correct command'

@patch('pyment.utils.preprocessing.fsl.run')
def test_flirt_degrees_of_freedom(mock):
    flirt('src.nii.gz', 'dest.nii.gz', template='template.nii.gz',
          degrees_of_freedom=9)

    assert 1 == mock.call_count, 'flirt does not call run function'
    cmd = mock.call_args[0][0]
    expected_cmd = ('flirt -in src.nii.gz -out dest.nii.gz -ref '
                    'template.nii.gz -dof 9')
    assert expected_cmd == cmd, 'flirt does not call run with correct command'

@patch('pyment.utils.preprocessing.fsl.run')
def test_flirt_folder(mock):
    try:
        os.mkdir('tmp')
        src = os.path.join('tmp', 'src')
        os.mkdir(src)
        dest = os.path.join('tmp', 'dest')
    
        for i in range(5):
            with open(os.path.join('tmp', 'src', f'sub{i}.mgz'), 'w') as f:
                f.write('test')

        flirt_folder(src, dest, template='template.nii.gz')

        assert os.path.isdir(dest), \
            'flirt_folder does not create destination folder'

        assert 5 == mock.call_count, \
            'flirt_folder does not call run for every subject'
    finally:
        rmtree('tmp')

@patch('pyment.utils.preprocessing.fsl.flirt')
def test_flirt_folder_degrees_of_freedom(mock):
    try:
        os.mkdir('tmp')
        src = os.path.join('tmp', 'src')
        os.mkdir(src)
        dest = os.path.join('tmp', 'dest')
    
        for i in range(5):
            with open(os.path.join('tmp', 'src', f'sub{i}.mgz'), 'w') as f:
                f.write('test')

        flirt_folder(src, dest, template='template.nii.gz', 
                     degrees_of_freedom=9)

        assert 'degrees_of_freedom' in mock.call_args[1], \
            'flirt_folder does not use degrees_of_freedom kwargs'

        degrees_of_freedom = mock.call_args[1]['degrees_of_freedom']

        assert 9 == degrees_of_freedom, \
            'flirt_folder does not pass on correct degrees of freedom'
    finally:
        rmtree('tmp')