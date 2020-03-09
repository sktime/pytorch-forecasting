import os
import wget
import pyunpack
import pandas as pd
import numpy as np

def download_from_url(url, output_path):
    """Downloads a file froma url."""

    print('Pulling data from {} to {}'.format(url, output_path))
    wget.download(url, output_path)
    print('done')

def recreate_folder(path):
    """Deletes and recreates folder."""

    shutil.rmtree(path)
    os.makedirs(path)

def unzip(zip_path, output_file, data_folder):
    """Unzips files and checks successful completion."""

    print('Unzipping file: {}'.format(zip_path))
    pyunpack.Archive(zip_path).extractall(data_folder)

    # Checks if unzip was successful
    if not os.path.exists(output_file):
        raise ValueError(
            'Error in unzipping process! {} not found.'.format(output_file))

def download_and_unzip(url, zip_path, csv_path, data_folder):
    """Downloads and unzips an online csv file.
    Args:
    url: Web address
    zip_path: Path to download zip file
    csv_path: Expected path to csv file
    data_folder: Folder in which data is stored.
    """

    download_from_url(url, zip_path)

    unzip(zip_path, csv_path, data_folder)

    print('Done.')

def download_electricity(config):
    """Downloads electricity dataset from UCI repository."""

    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00321/LD2011_2014.txt.zip'

    data_folder = config.data_folder
    csv_path = os.path.join(data_folder, 'LD2011_2014.txt')
    zip_path = csv_path + '.zip'

    download_and_unzip(url, zip_path, csv_path, data_folder)

    print('Aggregating to hourly data')

    df = pd.read_csv(csv_path, index_col=0, sep=';', decimal=',')
    df.index = pd.to_datetime(df.index)
    df.sort_index(inplace=True)

    # Used to determine the start and end dates of a series
    output = df.resample('1h').mean().replace(0., np.nan)

    earliest_time = output.index.min()

    df_list = []
    for label in output:
        print('Processing {}'.format(label))
        srs = output[label]

        start_date = min(srs.fillna(method='ffill').dropna().index)
        end_date = max(srs.fillna(method='bfill').dropna().index)

        active_range = (srs.index >= start_date) & (srs.index <= end_date)
        srs = srs[active_range].fillna(0.)

        tmp = pd.DataFrame({'power_usage': srs})
        date = tmp.index
        tmp['t'] = (date - earliest_time).seconds / 60 / 60 + (
            date - earliest_time).days * 24
        tmp['days_from_start'] = (date - earliest_time).days
        tmp['categorical_id'] = label
        tmp['date'] = date
        tmp['id'] = label
        tmp['hour'] = date.hour
        tmp['day'] = date.day
        tmp['day_of_week'] = date.dayofweek
        tmp['month'] = date.month

        df_list.append(tmp)

    output = pd.concat(df_list, axis=0, join='outer').reset_index(drop=True)

    output['categorical_id'] = output['id'].copy()
    output['hours_from_start'] = output['t']
    output['categorical_day_of_week'] = output['day_of_week'].copy()
    output['categorical_hour'] = output['hour'].copy()

    # Filter to match range used by other academic papers
    output = output[(output['days_from_start'] >= 1096)
                  & (output['days_from_start'] < 1346)].copy()

    output.to_csv(config.data_csv_path)

    print('Done.')
    
class Config():
    def __init__(self, data_folder, csv_path):
        self.data_folder = data_folder
        self.data_csv_path = csv_path