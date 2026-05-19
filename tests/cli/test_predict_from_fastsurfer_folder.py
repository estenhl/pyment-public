import pytest

from pyment.cli.predict_from_fastsurfer_folder import _parse_folder_name


@pytest.mark.parametrize(
    'name, expected',
    [
        ('sub-S001_ses-01_run-1', ('S001', '01', '1')),
        ('sub-abc-xyz_ses-test_run-2', ('abc-xyz', 'test', '2')),
        ('sub-S001_ses-01_run-1_extra_stuff', ('S001', '01', '1')),
    ],
    ids=['simple', 'subject-with-dash', 'with-trailing'],
)
def test_parse_folder_name_returns_groups(name, expected):
    assert _parse_folder_name(name) == expected


@pytest.mark.parametrize(
    'name',
    ['', 'random', 'sub-S001', 'sub-S001_ses-01', 'subject_session_run'],
    ids=['empty', 'random', 'missing-ses-run', 'missing-run', 'wrong-format'],
)
def test_parse_folder_name_raises_on_invalid(name):
    with pytest.raises(ValueError, match='Unable to match'):
        _parse_folder_name(name)
