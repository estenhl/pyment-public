"""Utility for downloading files over HTTP with optional GitHub blob
decoding."""

import base64
import json
import math

import requests
from tqdm import tqdm


def download_file(
    url: str,
    destination: str,
    description: str | None = None,
    decode_github: bool = False,
) -> bool:
    """Downloads a file from url and writes it to destination.

    If decode_github is True the response is assumed to be a GitHub blob
    API JSON response; the base64-encoded content field is decoded and
    written back to destination in place of the raw response.

    Parameters
    ----------
    url : str
        URL to download from.
    destination : str
        Local path to write the downloaded content to.
    description : str, optional
        Label shown in the tqdm progress bar.
    decode_github : bool
        If True, treat the response as a GitHub blob API JSON response
        and base64-decode the content field before writing.

    Returns
    -------
    bool
        Always True on success; raises on HTTP error.
    """
    with requests.get(url, stream=True) as response:
        response.raise_for_status()
        total_size = int(response.headers.get('content-length', 0))

        # 1 MB chunks
        chunk_size = 1 << 20

        progress_bar = tqdm(
            response.iter_content(chunk_size=chunk_size),
            total=int(math.ceil(total_size / chunk_size)),
            unit='mb',
            unit_scale=True,
            unit_divisor=1024,
            desc=description,
        )
        progress_bar.format_dict['rate'] = 'mb/s'

        with open(destination, 'wb') as f:
            for chunk in progress_bar:
                f.write(chunk)

    if decode_github:
        # Assumes a JSON file downloaded from GitHub
        with open(destination, 'rb') as f:
            data = json.load(f)

        data = base64.b64decode(data['content'])

        with open(destination, 'wb') as f:
            f.write(data)

    return True
