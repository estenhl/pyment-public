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
    for filename in os.listdir(src):
        reorient2std(os.path.join(src, filename), os.path.join(dest, filename))


def flirt(src: str, dest: str, *, template: str, silence: bool = True):
    logger.debug(f'Running flirt on {src} with template {template}')

    cmd = f'flirt -in {src} -out {dest} -ref {template} -dof 6'

    run(cmd, silence=silence)


def flirt_folder(src: str, dest: str, *, template: str, silence: bool = True):
    for filename in os.listdir(src):
        flirt(os.path.join(src, filename), os.path.join(dest, filename),
              template=template, silence=silence)