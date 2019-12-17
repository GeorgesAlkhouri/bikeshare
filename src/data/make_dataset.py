# -*- coding: utf-8 -*-
import click
import logging
import requests
import os
import zipfile
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import pandas as pd

URL = 'https://archive.ics.uci.edu/ml/machine-learning-databases/' \
    '00275/Bike-Sharing-Dataset.zip'


def get_raw_dataset(input_filepath):
    df = pd.read_csv(os.path.join(input_filepath, 'hour.csv'))
    return df


def download_dataset(input_filepath):
    output_path = os.path.join(input_filepath, 'Bike-Sharing-Dataset.zip')

    zip_file = requests.get(URL)
    open(output_path, 'wb').write(zip_file.content)

    assert os.path.isfile(output_path), 'File not downloaded to %s' % output_path

    with zipfile.ZipFile(output_path, 'r') as zip_ref:
        zip_ref.extractall(input_filepath)


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)

    download_dataset(input_filepath)
    logger.info('Raw data downloaded to % s', input_filepath)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
