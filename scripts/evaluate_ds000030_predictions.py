import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def evaluate_predictions(
    labels_path: str,
    predictions_path: str,
    visualize: bool = False,
    destination: str | None = None,
):
    labels = pd.read_csv(labels_path)
    predictions = pd.read_csv(predictions_path)

    predictions['image_id'] = predictions['source'].apply(
        lambda path: path.split('/')[-1]
    )
    predictions = predictions.rename(columns={'age': 'age_prediction'})
    predictions = pd.merge(
        predictions[['image_id', 'age_prediction']],
        labels[['image_id', 'age']],
        on='image_id',
        how='left',
    )

    mae = np.mean(np.abs(predictions['age'] - predictions['age_prediction']))
    print(f'MAE: {mae:.2f}')

    if visualize or destination is not None:
        fig, ax = plt.subplots()

        bounds: dict[str, float] = {
            key: f(predictions[['age', 'age_prediction']].values)
            for key, f in [('min', np.amin), ('max', np.amax)]
        }
        ax.set_xlim((bounds['min'] - 5, bounds['max'] + 5))
        ax.set_ylim((bounds['min'] - 5, bounds['max'] + 5))
        ax.plot(
            [bounds['min'] - 5, bounds['max'] + 5],
            [bounds['min'] - 5, bounds['max'] + 5],
        )
        ax.scatter(predictions['age'], predictions['age_prediction'])
        ax.set_xlabel('True age')
        ax.set_ylabel('Predicted age')
        ax.set_title('Age prediction')

        if destination is not None:
            fig.savefig(destination)

        if visualize:
            plt.show()

        plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        'Evaluates age predictions for the ds000030 dataset'
    )
    parser.add_argument(
        '-l',
        '--labels',
        required=True,
        help='Path to labels CSV',
    )
    parser.add_argument(
        '-p',
        '--predictions',
        required=True,
        help='Path to predictions CSV',
    )
    parser.add_argument(
        '--visualize',
        action='store_true',
        default=False,
        help='Display the scatter plot interactively',
    )
    parser.add_argument(
        '-d',
        '--destination',
        required=False,
        default=None,
        help='Path to write the scatter plot image',
    )

    args = parser.parse_args()

    evaluate_predictions(
        labels_path=args.labels,
        predictions_path=args.predictions,
        visualize=args.visualize,
        destination=args.destination,
    )
