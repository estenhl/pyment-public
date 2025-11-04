import base64
import json
import math
import requests
from tqdm import tqdm


def download_file(
    url: str,
    destination: str,
    description: str = None,
    decode_github: bool = False
) -> str:
    with requests.get(url, stream=True) as response:
        response.raise_for_status()
        total_size = int(response.headers.get('content-length', 0))

        # 1 MB chunks
        chunk_size = 1<<20

        progress_bar = tqdm(
            response.iter_content(chunk_size=chunk_size),
            total=int(math.ceil(total_size / chunk_size)),
            unit='mb',
            unit_scale=True,
            unit_divisor=1024,
            desc=description
        )
        progress_bar.format_dict['rate'] = f'mb/s'

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
