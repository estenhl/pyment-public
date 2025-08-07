import argparse

from pyment.models import MultiTaskSFCN

def predict_from_fastsurfer_folder(folder: str, output: str, weights: str):
    model = MultiTaskSFCN()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        'Generates predictions from SFCN multi and a given weights file for '
        'all images in a fastsurfer folder'
    )

    parser.add_argument('folder', help='Path to fastsurfer folder')
    parser.add_argument(
        'output', help='Path where CSV with predictions are written'
    )
    parser.add_argument('-w', '--weights', help='Path to weights file')

    args = parser.parse_args()

    predict_from_fastsurfer_folder(
        folder=args.folder,
        output=args.output,
        weights=args.weights
    )

