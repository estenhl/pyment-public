import logging
import os
import subprocess

from shutil import copyfile


logformat = '%(asctime)s - %(levelname)s - %(name)s: %(message)s'
logging.basicConfig(format=logformat, level=logging.INFO)
logger = logging.getLogger(__name__)

def run(cmd: str, *, silence: bool = True):
    stdout = subprocess.DEVNULL if silence else None

    process = subprocess.Popen(cmd.split(' '), stdout=stdout)
    process.communicate()

def extract_brainmasks_from_recon(recon: str, destination: str, symlink: bool = False):
    copy = os.symlink if symlink else copyfile

    if not os.path.isdir(destination):
        os.makedirs(destination)
    
    for subject in os.listdir(recon):
        brainmask = os.path.abspath(os.path.join(recon, subject, 'mri', 
                                                 'brainmask.mgz'))

        if not os.path.isfile(brainmask):
            logger.warning(f'Skipping {subject}: Missing brainmask')
            continue

        path = os.path.join(destination, f'{subject}.mgz')

        if os.path.isfile(path):
            logger.info(f'Skipping {subject}. {path} already exists')
            continue
        
        copy(brainmask, path)