import logging
import math
import requests

from tqdm import tqdm


logformat = '%(asctime)s - %(levelname)s - %(name)s: %(message)s'
logging.basicConfig(format=logformat, level=logging.INFO)
logger = logging.getLogger(__name__)

def download(url: str, filename: str, chunksize: int = 2**16) -> None:
    logger.info(f'Downloading {url} to {filename}')
    
    resp = requests.get(url, stream=True)

    size = int(resp.headers.get('content-length')) \
           if 'content-length' in resp.headers else None

    if size is None:
        logger.warning(('Unable to get header \'content-length\'. ' 
                        'Downloading without progress bar'))
    
    with open(filename, 'wb') as f:
        downloaded = 0
        for chunk in tqdm(resp.iter_content(chunk_size=chunksize), total=size):
            f.write(chunk)