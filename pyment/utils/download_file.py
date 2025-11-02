import math
import requests
from tqdm import tqdm


def download_file(
    url: str,
    destination: str,
    description: str = None
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
