import logging
import os
import numpy as np

from threading import Thread
from typing import Any, Dict, List

from .utils import run


logformat = '%(asctime)s - %(levelname)s - %(name)s: %(message)s'
logging.basicConfig(format=logformat, level=logging.INFO)
logger = logging.getLogger(__name__)

def autorecon1(path: str, *, subject: str, subjects_dir: str, 
               noisrunning: bool = True, silence: bool = True) -> None:
    logger.debug(f'Running autorecon1 on {path}')

    if not os.path.isdir(subjects_dir):
        os.mkdir(subjects_dir)

    cmd = f'recon-all -s {subject} -sd {subjects_dir} -i {path} -autorecon1'

    if noisrunning:
        cmd += ' -no-isrunning'

    run(cmd, silence=silence)


def _brainmask_exists(subjects_dir: str, filename: str) -> bool:
    subject = filename.split('.')[0]
    return os.path.isfile(os.path.join(subjects_dir, subject, 'mri', 
                                       'brainmask.mgz'))


def autorecon1_folder(src: str, dest: str, *, threads: int = 1, 
                      noisrunning: bool = True, 
                      silence: bool = True) -> None:
    filenames = os.listdir(src)
    remaining = [f for f in filenames if not _brainmask_exists(dest, f)]

    logger.info((f'Skipping {len(filenames) - len(remaining)} ' 
                 'existing subjects'))

    filenames = remaining

    params = [{
        'path': os.path.join(src, filename),
        'subject': filename.split('.')[0],
        'subjects_dir': dest,
        'noisrunning': noisrunning,
        'silence': silence
    } for filename in filenames]

    class Worker(Thread):
        def __init__(self, params: List[Dict[str, Any]]):
            super().__init__()
            self.params = params

        def run(self):
            for params in self.params:
                autorecon1(**params)

    params = np.array_split(params, threads)
    workers = [Worker(p) for p in params]

    for w in workers:
        w.start()

    for w in workers:
        w.join()


def convert_mgz_to_nii_gz(src: str, dest: str, *, 
                          silence: bool = True) -> None:
    logger.debug(f'Running convert on {src}')

    cmd = f'mri_convert {src} {dest} -ot nii'

    run(cmd, silence=silence)


def convert_mgz_to_nii_gz_folder(src: str, dest: str, *, 
                                 silence: bool = True) -> None:
    for filename in os.listdir(src):
        target = filename.split('.')[0] + '.nii.gz'
        convert_mgz_to_nii_gz(os.path.join(src, filename), 
                              os.path.join(dest, target))