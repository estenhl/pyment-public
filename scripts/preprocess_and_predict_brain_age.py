import argparse
import logging
import os

from shutil import copyfile, rmtree, which

from pyment.utils.preprocessing import autorecon1_folder, \
                                       convert_mgz_to_nii_gz_folder, \
                                       crop_folder, \
                                       extract_brainmasks_from_recon, \
                                       flirt_folder, \
                                       reorient2std_folder
from predict_brain_age import predict_brain_age


logformat = '%(asctime)s - %(levelname)s - %(name)s: %(message)s'
logging.basicConfig(format=logformat, level=logging.INFO)
logger = logging.getLogger(__name__)

def preprocess_and_predict_brain_age(*, folder: str, model: str, 
                                     weights: str = None,
                                     batch_size: int, threads: int = None, 
                                     normalize: bool = False, destination: str,
                                     temporary_folder: str = '.', 
                                     remove_temporary_folders: bool = False,
                                     verbose: bool = False, 
                                     mni152_template: str):
    for tool in ['recon-all', 'mri_convert', 'fslreorient2std', 'flirt']:
        assert which(tool) is not None, ('Unable to locate required tool '
                                         f'\'{tool}\'')

    labelsfile = os.path.join(folder, 'labels.csv')
    
    assert os.path.isfile(labelsfile), ('Folder {folder} must have a '
                                        'labels.csv file')

    if not os.path.isdir(temporary_folder):
        os.mkdir(temporary_folder)

    recon = os.path.join(temporary_folder, 'recon')

    autorecon1_folder(os.path.join(folder, 'images'), recon, threads=threads,
                      silence=not verbose)
    logger.info(f'Finished autorecon for {len(os.listdir(recon))} subjects')

    brainmasks = os.path.join(temporary_folder, 'brainmasks', 'images')
    extract_brainmasks_from_recon(recon, brainmasks, symlink=True)
    logger.info(f'Found {len(os.listdir(brainmasks))} brainmasks')

    nii = os.path.join(temporary_folder, 'nifti', 'images')
    convert_mgz_to_nii_gz_folder(brainmasks, nii, silence=not verbose)
    logger.info(f'Converted {len(os.listdir(nii))} images to nifti')

    reoriented = os.path.join(temporary_folder, 'reoriented', 'images')
    reorient2std_folder(nii, reoriented, silence=not verbose)
    logger.info(f'Reoriented {len(os.listdir(reoriented))} images')

    mni152 = os.path.join(temporary_folder, 'mni152', 'images')
    flirt_folder(reoriented, mni152, template=mni152_template, 
                 silence=not verbose)
    logger.info((f'Registered {len(os.listdir(mni152))} images to MNI152 '
                 'space'))

    cropped = os.path.join(temporary_folder, 'cropped', 'images')
    crop_folder(mni152, cropped, bounds=((6, 173), (2, 214), (0, 160)))
    logger.info(f'Cropped {len(os.listdir(cropped))} images')
    cropped = os.path.join(temporary_folder, 'cropped')

    copyfile(labelsfile, os.path.join(cropped, 'labels.csv'))

    predict_brain_age(folder=cropped, model=model, weights=weights, 
                      batch_size=batch_size, threads=threads, 
                      normalize=normalize, destination=destination)

    if remove_temporary_folders:
        rmtree(os.path.join(temporary_folder, 'recon'))
        rmtree(os.path.join(temporary_folder, 'brainmasks'))
        rmtree(os.path.join(temporary_folder, 'nifti'))
        rmtree(os.path.join(temporary_folder, 'reoriented'))
        rmtree(os.path.join(temporary_folder, 'mni152'))
        rmtree(os.path.join(temporary_folder, 'cropped'))

        if len(os.listdir(temporary_folder)) == 0:
            os.rmdir(temporary_folder)
                                
if __name__ == '__main__':
    parser = argparse.ArgumentParser(('Estimates brain age for images from a '
                                      'given folder using a given model with '
                                      'given weights. Before estimation the '
                                      'default preprocessing is applied to '
                                      'the images'))
    
    parser.add_argument('-f', '--folder', required=True,
                        help=('Folder containing images. Should have a '
                              'csv-file called \'labels.csv\' with columns '
                              'id and age, and a subfolder \'images\' '
                              'containing nifti files'))
    parser.add_argument('-m', '--model', required=True,
                        help='Name of the model to use (e.g. sfcn-reg)')
    parser.add_argument('-w', '--weights', required=False, default=None,
                        help='Weights to load in the model')
    parser.add_argument('-b', '--batch_size', required=True, type=int,
                        help='Batch size to use while predicting')
    parser.add_argument('-t', '--threads', required=False, default=None, 
                        type=int, help=('Number of threads to use for reading '
                                        'data. If not set, a synchronous '
                                        'generator will be used'))
    parser.add_argument('-n', '--normalize', action='store_true',
                        help=('If set, images will be normalized to range '
                              '(0, 1) before prediction'))
    parser.add_argument('-d', '--destination', required=True,
                        help=('Path where CSV containing ids, labels '
                              'and predictions are stored'))
    parser.add_argument('-e', '--temp_folder', required=False, default='.',
                        help=('Path to folder where temporary directories '
                              'containing partially stored images are stored. '
                              'If not set, the current directory will be '
                              'used'))
    parser.add_argument('-r', '--remove_temporary_folders', 
                        action='store_true', help=('If set, temporary folders '
                                                   'are removed (Not the '
                                                   'root).'))
    parser.add_argument('-v', '--verbose', action='store_true',
                        help=('If set, logs from underlying processes '
                              '(e.g. FreeSurfer) are shown'))
    parser.add_argument('-i', '--mni152_template', required=True,
                        help=('Path to MNI152 template used for FLIRT '
                              'registration'))

    args = parser.parse_args()

    preprocess_and_predict_brain_age(folder=args.folder, model=args.model, 
                                     weights=args.weights, 
                                     batch_size=args.batch_size,
                                     threads=args.threads, 
                                     normalize=args.normalize,
                                     destination=args.destination,
                                     temporary_folder=args.temp_folder,
                                     remove_temporary_folders=\
                                         args.remove_temporary_folders,
                                     verbose=args.verbose,
                                     mni152_template=args.mni152_template)