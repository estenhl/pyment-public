import argparse
import os
import re
import numpy as np
import pandas as pd

from functools import reduce
from typing import Callable, List, Union


def _locate(lines: List[str], pattern: str, idx: Union[int, str] = 0):
    lines = [i for i in range(len(lines)) if re.fullmatch(pattern, lines[i])]

    if idx == 'last':
        return lines[-1]
    elif idx == 'first':
        return lines[0]
    elif isinstance(idx, int):
        return lines[idx]
    else:
        raise ValueError(f'Unknown idx {idx}')

def _locate_section(lines: List[str], name: str):
    return _locate(lines, f'.*# {name}.*')

def _locate_table(lines: List[str]):
    table_start = _locate(lines, '\| .*', idx=0)
    table_end = _locate(lines, '\| .*', idx=-1)

    return table_start, table_end

def _to_markup_table_row(tokens: List[str]):
    return '| ' + ' | '.join(tokens) + ' |'

def _create_markup_table_heading(names: List[str], sep: List[str] = None):
    sep = sep if sep is not None else ['---'] * len(names)

    return [_to_markup_table_row(names), _to_markup_table_row(sep)]

def generate_publications_table(table: pd.DataFrame):
    lines = _create_markup_table_heading(['Title', 'Abbreviation',
                                          'Publication year',
                                          'Corresponding author'],
                                          ['---', ':-:', ':-:', ':-:'])

    for _, row in table.iterrows():
        name = f'[{row["name"]}](https://doi.org/{row["doi"]})'
        author = f'[{row["author"]}](mailto:{row["email"]})'
        row = [name, row['abbreviation'], str(row['year']), author]
        lines.append(_to_markup_table_row(row))

    return lines

def generate_architectures_table(table: pd.DataFrame):
    lines = _create_markup_table_heading(['Name', 'Abbreviation',
                                          'Description'],
                                         ['---', ':-:', '---'])

    for _, row in table.iterrows():
        row = [row['name'], row['abbreviation'], row['description']]
        lines.append(_to_markup_table_row(row))

    return lines

def combine_duplicate_rows(table: pd.DataFrame):
    masked_names = {}
    for idx, row in table.iterrows():
        name = row['name']
        architecture = row['architecture']
        masked = reduce(lambda n, i: n.replace(f'fold-{i}', 'fold-X'),
                     np.arange(0, 10), name)
        key = (architecture, masked)

        if not key in masked_names:
            masked_names[key] = []

        masked_names[key].append(idx)

    for (architecture, name), entries in masked_names.items():
        if entries == 1:
            continue

        shas = table.loc[entries, 'sha'].values.tolist()
        description = table.loc[entries[0], 'description']
        description = reduce(lambda n, i: n.replace(f'fold {i}', 'fold X'),
                             np.arange(0, 10), description)

        table.at[entries[0], 'name'] = name
        table.at[entries[0], 'description'] = description
        table.at[entries[0], 'sha'] = shas
        table = table.drop(entries[1:])

    return table

def generate_models_table(table: pd.DataFrame,
                          publications: pd.DataFrame):
    lines = _create_markup_table_heading(['Name', 'Architecture',
                                          'Source publication',
                                          'Description',
                                          'Training sample size',
                                          'Expected out-of-sample error',
                                          'URL'],
                                          [':-:', ':-:', ':-:', '---', ':-:',
                                           ':-:', ':-:'])

    table = combine_duplicate_rows(table)

    for _, row in table.iterrows():
        publication = \
            publications[publications['abbreviation'] == row['publication']]
        assert len(publication) == 1, \
            ('Models table referring to unknown publication '
            f'{row["publication"]}')
        publication = f'http://doi.org/{publication["doi"].values[0]}'
        publication = f'[{row["publication"]}]({publication})'

        url = 'https://api.github.com/repos/estenhl/pyment-public/git/blobs'
        sha = row['sha']
        print(sha)

        if isinstance(sha, str):
            url = f'[link]({url}/{sha})'
        elif isinstance(sha, list) and len(sha) == 1:
            url = f'[link]({url}/{sha[0]})'
        elif isinstance(sha, list):
            urls = [f'[fold {idx}]({url}/{value})<br />' \
                    for idx, value in enumerate(sha)]
            url = ' '.join(urls)

        error = row['error'] if isinstance(row['error'], str) else ''

        row = [row['name'], row['architecture'], publication,
               row['description'], str(row['sample']), error, url]
        lines.append(_to_markup_table_row(row))

    return lines

def replace_table(lines: List[str], start: int, end: int, table: pd.DataFrame,
                  generator: Callable[[pd.DataFrame], pd.DataFrame]):
    relevant_lines = lines[start:end]

    table_start, table_end = \
        _locate_table(relevant_lines)
    table_start += start#
    table_end += start + 1
    table = generator(table)

    return lines[:table_start] + table + lines[table_end:]

def validate_models_table(publications: pd.DataFrame,
                          architectures: pd.DataFrame,
                          models: pd.DataFrame):
    used_publications = set(models['publication'].values)
    known_publications = set(publications['abbreviation'].values)
    unknown_publications = used_publications - known_publications

    if len(unknown_publications) > 0:
        raise ValueError('Models table referring to unknown publications: '
                         f'{unknown_publications}')

    used_architectures = set(models['architecture'].values)
    known_architectures = set(architectures['abbreviation'].values)
    unknown_architectures = used_architectures - known_architectures

    if len(unknown_architectures) > 0:
        raise ValueError('Models table referring to unknown publications: '
                         f'{unknown_architectures}')

def generate_main_readme(path: str, publications: pd.DataFrame,
                         architectures: pd.DataFrame, models: pd.DataFrame):
    with open(path, 'r') as f:
        lines = [line.strip() for line in f.readlines()]

    publications_section_start = _locate_section(lines, 'Publications')
    architectures_section_start = _locate_section(lines, 'Architectures')
    models_section_start = _locate_section(lines, 'Models')

    lines = replace_table(lines,
                          start=publications_section_start,
                          end=architectures_section_start,
                          table=publications,
                          generator=generate_publications_table)
    lines = replace_table(lines,
                          start=architectures_section_start,
                          end=models_section_start,
                          table=architectures,
                          generator=generate_architectures_table)

    validate_models_table(publications, architectures, models)

    models_table_generator = lambda x: generate_models_table(x, publications)

    lines = replace_table(lines,
                          start=models_section_start,
                          end=len(lines),
                          table=models,
                          generator=models_table_generator)


    with open(path, 'w') as f:
        f.write('\n'.join(lines))

# def generate_images_table(table: pd.DataFrame):


# def generate_docker_readme(path: str, table: pd.DataFrame):
#     with open(path, 'r') as f:
#         lines = [line.strip() for line in f.readlines()]

#     images_section_start = _locate_section(lines, 'Images')

#     lines = replace_table(lines,
#                           start=images_section_start,
#                           end=len(lines),
#                           table=table,
#                           generator=generate_images_table)

def generate_readmes(data: str, main_readme: str, docker_readme: str,
                     preprocessing_readme: str):
    tables = {name: pd.read_csv(os.path.join(data, f'{name}.csv')) \
              for name in ['architectures', 'images', 'models',
                           'preprocessing', 'publications']}

    generate_main_readme(main_readme, tables['publications'],
                         tables['architectures'], tables['models'])
    #generate_docker_readme(docker_readme, tables['images'])

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Generates READMEs based on the current '
                                     'data')

    scripts_folder = os.path.dirname(os.path.abspath(__file__))
    root_folder = os.path.join(scripts_folder, os.pardir)

    parser.add_argument('-d', '--data', required=False,
                        default=os.path.join(root_folder, '.data'),
                        help='Folder containing tables with updated data')
    parser.add_argument('-mr', '--main_readme', required=False,
                        default=os.path.join(root_folder, 'README.md'),
                        help='Path to main README.md file')
    parser.add_argument('-dr', '--docker_readme', required=False,
                        default=os.path.join(root_folder, 'docker',
                                             'README.md'),
                        help='Path to docker README.md file')
    parser.add_argument('-pr', '--preprocessing_readme', required=False,
                        default=os.path.join(root_folder, 'preprocessing',
                                             'README.md'),
                        help='Path to preprocessing README.md file')

    args = parser.parse_args()

    generate_readmes(args.data,
                     main_readme=args.main_readme,
                     docker_readme=args.docker_readme,
                     preprocessing_readme=args.preprocessing_readme)
