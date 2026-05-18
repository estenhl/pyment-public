from unittest.mock import patch

import pytest

from pyment.models.utils.ensure_weights import IDENTIFIERS, ensure_weights


def test_returns_identifier_when_local_files_exist(tmp_path):
    prefix = str(tmp_path / 'weights')
    (tmp_path / 'weights.index').touch()
    (tmp_path / 'weights.data-00000-of-00001').touch()

    result = ensure_weights(prefix)

    assert result == prefix, (
        'Expected ensure_weights to return the identifier unchanged when '
        'both local weight files exist'
    )


def test_returns_identifier_when_single_local_file_exists(tmp_path):
    path = str(tmp_path / 'weights.h5')
    (tmp_path / 'weights.h5').touch()

    result = ensure_weights(path)

    assert result == path, (
        'Expected ensure_weights to return the identifier unchanged when '
        'a single local file exists'
    )


def test_raises_for_unknown_identifier():
    with pytest.raises(NotImplementedError):
        ensure_weights('not-a-real-identifier')


@pytest.mark.parametrize('identifier', list(IDENTIFIERS.keys()))
def test_downloads_known_identifier(identifier, tmp_path):
    with (
        patch(
            'pyment.models.utils.ensure_weights.download_file'
        ) as mock_download,
        patch('os.path.isfile', return_value=False),
    ):
        result = ensure_weights(identifier, local_cache=str(tmp_path))

    assert result == str(tmp_path / identifier), (
        'Expected ensure_weights to return a path under local_cache for a '
        'known identifier'
    )
    assert mock_download.call_count == 2, (
        'Expected ensure_weights to call download_file twice (data + index)'
    )


@pytest.mark.parametrize('identifier', list(IDENTIFIERS.keys()))
def test_skips_download_when_cached(identifier, tmp_path):
    (tmp_path / f'{identifier}.index').touch()
    (tmp_path / f'{identifier}.data-00000-of-00001').touch()

    with patch(
        'pyment.models.utils.ensure_weights.download_file'
    ) as mock_download:
        result = ensure_weights(identifier, local_cache=str(tmp_path))

    mock_download.assert_not_called()
    assert result == str(tmp_path / identifier), (
        'Expected ensure_weights to return cached path without downloading '
        'when both weight files are already present'
    )
