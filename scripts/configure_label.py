import argparse
import json
import numpy as np
import pandas as pd

from functools import reduce

from utils import configure_environment

configure_environment()

from pyment.labels import Label


def configure_label(*, name: str, type: str, filenames: pd.DataFrame = None,
                    columns: str = None, destination: str, kwargs: str = '{}'):
    kwargs = json.loads(kwargs)
    
    label = Label.from_type(type, name=name, **kwargs)

    if len(filenames) is not None:
        assert len(filenames) == len(columns), \
            'Must provide one column per filename'

        values = reduce(lambda x, y: np.concatenate([x, y]),
                        [pd.read_csv(filenames[i])[columns[i]].values \
                         for i in range(len(filenames))])

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
    parser.add_argument('-f', '--filenames', required=False, default=[], 
                        nargs='+', help=('Optional CSVs containing data used '
                                         'for fitting the variable '
                                         'preprocessing'))
    parser.add_argument('-c', '--columns', required=False, default=[], 
                        nargs='+', help=('Optional columns containing the '
                                         'corresponding to the CSVs for '
                                         'fitting variable preprocessing'))
    parser.add_argument('-d', '--destination', required=True,
                        help='Path where json configuring the label is stored')
    parser.add_argument('-k', '--kwargs', required=False, default='{}',
                        help=('Optional jsonstring containing keyword'
                              'arguments for instantiating the label'))

    args = parser.parse_args()

    configure_label(name=args.name, type=args.type, filenames=args.filenames,
                    columns=args.columns, destination=args.destination,
                    kwargs=args.kwargs)