import logging
import os

from .utils import run


logformat = '%(asctime)s - %(levelname)s - %(name)s: %(message)s'
logging.basicConfig(format=logformat, level=logging.INFO)
logger = logging.getLogger(__name__)


def reorient2std(src: str, dest: str, *, silence: bool = True):
    logger.debug(f'Running reorient on {src}')

    cmd = f'fslreorient2std {src} {dest}'

    run(cmd, silence=silence)


def reorient2std_folder(src: str, dest: str, *, silence: bool = True):
    if not os.path.isdir(dest):
        os.makedirs(dest)

    for filename in os.listdir(src):
        path = os.path.join(dest, filename)

        if os.path.isfile(path):
            logger.info(f'Skipping {filename}: Already exists')
            continue
    
        reorient2std(os.path.join(src, filename), path,
                     silence=silence)


def flirt(src: str, dest: str, *, template: str, 
          degrees_of_freedom: int = 6, silence: bool = True):
    logger.debug(f'Running flirt on {src} with template {template}')

    cmd = (f'flirt -in {src} -out {dest} -ref {template} '
           f'-dof {degrees_of_freedom}')

    run(cmd, silence=silence)


def flirt_folder(src: str, dest: str, *, template: str, 
                 degrees_of_freedom: int = 6, silence: bool = True):
    if not os.path.isdir(dest):
        os.makedirs(dest)
    
    for filename in os.listdir(src):
        path = os.path.join(dest, filename)

        if os.path.isfile(path):
            logger.info(f'Skipping {filename}: Already exists')
            continue
    
        flirt(os.path.join(src, filename), path, template=template, 
              silence=silence, degrees_of_freedom=degrees_of_freedom)