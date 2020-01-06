import click
import requests
import os
import zipfile
import pandas as pd
from src.conf import config


def get_raw_dataset(input_filepath, file_name='hour.csv'):

    raw_df_path = os.path.join(input_filepath, file_name)

    assert os.path.isfile(raw_df_path), 'File %s not found in %s' % (file_name, input_filepath)

    df = pd.read_csv(raw_df_path)
    return df


def download_dataset(input_filepath):
    output_path = os.path.join(input_filepath, 'Bike-Sharing-Dataset.zip')

    zip_file = requests.get(config.data_url)
    open(output_path, 'wb').write(zip_file.content)

    assert os.path.isfile(output_path), 'File not downloaded to %s' % output_path

    with zipfile.ZipFile(output_path, 'r') as zip_ref:
        zip_ref.extractall(input_filepath)


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
def main(input_filepath):
    download_dataset(input_filepath)
    print('Raw data downloaded to', input_filepath)


if __name__ == '__main__':
    main()
