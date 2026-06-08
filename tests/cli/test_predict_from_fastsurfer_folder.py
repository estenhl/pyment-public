import pytest

from pyment.cli.predict_from_fastsurfer_folder import _parse_folder_name


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
