# -*- coding: utf-8 -*-

import logging

import click
import pandas as pd
from pandas import DataFrame
from dotenv import find_dotenv, load_dotenv
from sklearn.model_selection import train_test_split
from typing import Tuple

import src.data.preprocessing as preprocessing
from src.jutils.data import DataUtils


def make_dataset(df: DataFrame, porcentaje_entrenamiento) -> Tuple[DataFrame, DataFrame]:
    df = df.copy()
    if porcentaje_entrenamiento < 1:
        df_train_test, df_validation = train_test_split(df, train_size=porcentaje_entrenamiento, random_state=1)
    else:
        df_train_test = df
        df_validation = df[[False] * df.shape[0]]
    return df_train_test, df_validation


def main(data_folder_path, input_filename, porcentaje_entrenamiento):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    input_filename_stem = input_filename.split('.')[0]
    input_filename = input_filename_stem + '.parquet'
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')
    du = DataUtils(
        data_folder_path=data_folder_path,
        input_file_name=input_filename,
        y_name='price',
        load_data=lambda path: pd.read_parquet(path),
        save_data=lambda df, path: df.to_parquet(path)
    )
    if not du.preprocessed_file_path.exists():
        preprocessing.main(data_folder_path, input_filename)
    du.data = du.load_data(du.preprocessed_file_path)

    df_train_test, df_validation = make_dataset(du.data, porcentaje_entrenamiento)

    du.save_data(
        df_train_test,
        du.raw_train_test_path
    )
    du.save_data(
        df_validation,
        du.raw_validation_path
    )


@click.command()
@click.argument('data_folder_path', type=click.types.Path(file_okay=False))
@click.argument('input_filename', type=click.types.STRING)
@click.argument('porcentaje_entrenamiento', type=click.types.FLOAT)
def main_terminal(data_folder_path, input_filename, porcentaje_entrenamiento):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    main(data_folder_path, input_filename, porcentaje_entrenamiento)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main_terminal()
