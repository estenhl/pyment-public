import argparse
import pandas as pd

from pyment.labels import Label


def configure_label(*, name: str, type: str, filename: pd.DataFrame = None,
                    column: str = None, destination: str):
    label = Label.from_type(type, name=name)

    if filename is not None:
        if column is None:
            raise ValueError('Must supply a column alongside the CSV')

        df = pd.read_csv(filename)
        values = df[column].values

        label.fit(values)

    label.save(destination)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(('Configures a label according to given '
                                      'parameters. If the variable requires '
                                      'fitting (e.g. finding min- and '
                                      'max-values for normalization), an '
                                      'optional dataframe with values can '
                                      'be supplied'))

    parser.add_argument('-n', '--name', required=True,
                        help='Name of the variable')
    parser.add_argument('-t', '--type', required=True, choices=Label.types,
                        help=('Type of the variable. Possible options are'
                              f'{Label.types}'))
    parser.add_argument('-f', '--filename', required=False, default=None,
                        help=('Optional CSV containing data used for fitting '
                              'the variable preprocessing'))
    parser.add_argument('-c', '--column', required=False, default=None,
                        help=('Optional column containing the values of the '
                              'CSV used for fitting variable preprocessing'))
    parser.add_argument('-d', '--destination', required=True,
                        help='Path where json configuring the label is stored')

    args = parser.parse_args()

    configure_label(name=args.name, type=args.type, filename=args.filename,
                    column=args.column, destination=args.destination)