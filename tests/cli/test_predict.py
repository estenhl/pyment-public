import pytest

from pyment.cli.predict import _detect_format, _parse_folder_name, predict


@pytest.mark.parametrize(
    'name, expected',
    [
        (
            'sub-S001_ses-01_run-1_T1w',
            ('S001', '01', '1', 'T1w'),
        ),
        (
            'sub-S001_ses-01_run-1',
            ('S001', '01', '1', None),
        ),
        (
            'sub-S001_ses-01_T1w',
            ('S001', '01', None, 'T1w'),
        ),
        (
            'sub-S001_ses-01',
            ('S001', '01', None, None),
        ),
        (
            'sub-S001_T1w',
            ('S001', None, None, 'T1w'),
        ),
        (
            'sub-S001',
            ('S001', None, None, None),
        ),
        (
            'sub-abc-xyz_ses-test_run-2_T1w',
            ('abc-xyz', 'test', '2', 'T1w'),
        ),
        (
            'sub-abc-xyz_ses-test_run-2',
            ('abc-xyz', 'test', '2', None),
        ),
    ],
    ids=[
        'full-bids-with-modality',
        'full-bids-no-modality',
        'no-run-with-modality',
        'no-run-no-modality',
        'no-session-with-modality',
        'subject-only',
        'subject-with-dash-and-modality',
        'subject-with-dash-no-modality',
    ],
)
def test_parse_folder_name_returns_groups(name, expected):
    assert _parse_folder_name(name) == expected


@pytest.mark.parametrize(
    'name',
    ['', 'random', 'subject_session_run'],
    ids=['empty', 'random', 'wrong-format'],
)
def test_parse_folder_name_raises_on_invalid(name):
    with pytest.raises(ValueError, match='Unable to match'):
        _parse_folder_name(name)


def test_detect_format_flat_folder(tmp_path):
    (tmp_path / 'sub-01.mgz').touch()
    (tmp_path / 'sub-02.mgz').touch()

    assert _detect_format(str(tmp_path)) == 'flat', (
        'Expected _detect_format to return flat for a folder of files'
    )


def test_detect_format_fastsurfer_folder(tmp_path):
    (tmp_path / 'sub-01' / 'mri').mkdir(parents=True)
    (tmp_path / 'sub-02' / 'mri').mkdir(parents=True)

    assert _detect_format(str(tmp_path)) == 'fastsurfer', (
        'Expected _detect_format to return fastsurfer for a folder of '
        "subdirectories that each contain an 'mri' subdirectory"
    )


def test_detect_format_raises_on_empty_folder(tmp_path):
    with pytest.raises(ValueError, match='empty'):
        _detect_format(str(tmp_path))


def test_detect_format_raises_on_mixed_contents(tmp_path):
    (tmp_path / 'sub-01' / 'mri').mkdir(parents=True)
    (tmp_path / 'sub-02.mgz').touch()

    with pytest.raises(ValueError, match='mix of files and subdirectories'):
        _detect_format(str(tmp_path))


def test_detect_format_raises_on_subfolders_without_mri(tmp_path):
    (tmp_path / 'sub-01').mkdir()
    (tmp_path / 'sub-02').mkdir()

    with pytest.raises(ValueError, match="not all contain an 'mri'"):
        _detect_format(str(tmp_path))


def test_predict_raises_on_unknown_format(tmp_path):
    with pytest.raises(ValueError, match='Unknown format'):
        predict(source=str(tmp_path), format='bogus')
