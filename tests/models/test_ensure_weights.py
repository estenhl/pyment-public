from unittest.mock import patch

import pytest

from pyment.models.utils.ensure_weights import IDENTIFIERS, ensure_weights


def test_returns_identifier_when_local_file_exists(tmp_path):
    path = str(tmp_path / 'weights.h5')
    (tmp_path / 'weights.h5').touch()

    result = ensure_weights(path)

    assert result == path, (
        'Expected ensure_weights to return the identifier unchanged when '
        'a local weight file exists'
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

    assert result == str(tmp_path / f'{identifier}.h5'), (
        'Expected ensure_weights to return a .h5 path under local_cache for '
        'a known identifier'
    )
    assert mock_download.call_count == 1, (
        'Expected ensure_weights to call download_file once'
    )


@pytest.mark.parametrize('identifier', list(IDENTIFIERS.keys()))
def test_skips_download_when_cached(identifier, tmp_path):
    (tmp_path / f'{identifier}.h5').touch()

    with patch(
        'pyment.models.utils.ensure_weights.download_file'
    ) as mock_download:
        result = ensure_weights(identifier, local_cache=str(tmp_path))

    mock_download.assert_not_called()
    assert result == str(tmp_path / f'{identifier}.h5'), (
        'Expected ensure_weights to return cached path without downloading '
        'when the weight file is already present'
    )
