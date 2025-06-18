import argparse

from FastSurferCNN.run_prediction import main


def preprocess(source_path: str, destination_path: str):
    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        'Preprocesses a given MRI image with FastSurfer'
    )
    parser.add_argument('-i', '--input', type=str, required=True,
                        help='Path to the MRI image to preprocess')
    parser.add_argument('-o', '--output', type=str, required=True,
                        help='Path where the preprocessed image will be saved')

    args = parser.parse_args()

    preprocess(args.input, args.output)
