import base64
import json
from unittest.mock import MagicMock, patch

import pytest

from pyment.utils.download_file import download_file


def make_mock_response(chunks: list[bytes], content_length: int = 0):
    response = MagicMock()
    response.headers.get.return_value = content_length
    response.iter_content.return_value = iter(chunks)
    response.__enter__ = lambda s: s
    response.__exit__ = MagicMock(return_value=False)
    return response


@pytest.fixture
def mock_response(tmp_path):
    response = make_mock_response([b'hello', b' world'])
    with patch('requests.get', return_value=response):
        yield response


def test_download_file_writes_content_to_destination(mock_response, tmp_path):
    destination = str(tmp_path / 'file.bin')

    download_file(url='http://example.com', destination=destination)

    with open(destination, 'rb') as f:
        assert f.read() == b'hello world', (
            'Expected download_file to write all response chunks to destination'
        )


def test_download_file_calls_raise_for_status(mock_response, tmp_path):
    download_file(
        url='http://example.com', destination=str(tmp_path / 'file.bin')
    )

    mock_response.raise_for_status.assert_called_once()


def test_download_file_http_error_propagates(tmp_path):
    response = make_mock_response([])
    response.raise_for_status.side_effect = Exception('404')

    with patch('requests.get', return_value=response):
        with pytest.raises(Exception, match='404'):
            download_file(
                url='http://example.com',
                destination=str(tmp_path / 'file.bin'),
            )


def test_download_file_missing_content_length_does_not_raise(tmp_path):
    response = make_mock_response([b'data'])
    response.headers.get.return_value = 0

    with patch('requests.get', return_value=response):
        download_file(
            url='http://example.com', destination=str(tmp_path / 'file.bin')
        )


def test_download_file_decode_github_writes_decoded_content(tmp_path):
    raw = b'decoded content'
    payload = json.dumps({'content': base64.b64encode(raw).decode()}).encode()
    response = make_mock_response([payload])

    with patch('requests.get', return_value=response):
        destination = str(tmp_path / 'file.bin')
        download_file(
            url='http://example.com',
            destination=destination,
            decode_github=True,
        )

    with open(destination, 'rb') as f:
        assert f.read() == raw, (
            'Expected download_file to write base64-decoded content to '
            'destination when decode_github is True'
        )


def test_download_file_decode_github_false_writes_raw_content(tmp_path):
    content = b'raw bytes'
    response = make_mock_response([content])

    with patch('requests.get', return_value=response):
        destination = str(tmp_path / 'file.bin')
        download_file(
            url='http://example.com',
            destination=destination,
            decode_github=False,
        )

    with open(destination, 'rb') as f:
        assert f.read() == content, (
            'Expected download_file to write raw response bytes to destination '
            'when decode_github is False'
        )
