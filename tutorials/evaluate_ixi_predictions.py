import argparse
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

from download_ixi import DEFAULT_DESTINATION


def evaluate_ixi_predictions(
    labels: str,
    predictions: str
) -> None:
    labels = pd.read_excel(labels)
    predictions = pd.read_csv(predictions)

    predictions['IXI_ID'] = predictions['source'].apply(
        lambda path: int(path.split('/')[-1][3:6])
    )
    predictions['age_prediction'] = predictions['age']
    predictions = pd.merge(
        predictions[['IXI_ID', 'age_prediction']],
        labels[['IXI_ID', 'AGE']],
        on='IXI_ID',
        how='left'
    )

    mae = np.mean(np.abs(predictions['AGE'] - predictions['age_prediction']))
    print(f'MAE: {mae:.2f}')

    plt.scatter(predictions['AGE'], predictions['age_prediction'])
    plt.xlabel('True age')
    plt.ylabel('Predicted age')
    plt.title('Age prediction')
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        'Evaluates predictions for the IXI dataset'
    )
    parser.add_argument(
        '-l', '--labels',
        required=False,
        default=os.path.join(DEFAULT_DESTINATION, 'IXI.xls'),
        help='Path to XLSX containing labels'
    )
    parser.add_argument(
        '-p', '--predictions',
        required=False,
        default=os.path.join(
            DEFAULT_DESTINATION,
            'outputs',
            'predictions.csv'
        ),
        help='Path to CSV containing predictions'
    )

    args = parser.parse_args()

    evaluate_ixi_predictions(
        labels=args.labels,
        predictions=args.predictions
    )
