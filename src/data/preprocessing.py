# -*- coding: utf-8 -*-

import logging
from pathlib import Path

import click
import pandas as pd
from dotenv import find_dotenv, load_dotenv
from pandas import DataFrame
import numpy as np

from src.data.procesamiento_datos import LimpiezaCalidad
from src.jutils.data import DataUtils


def preprocessing(df: DataFrame) -> DataFrame:
    _columnas_numericas = [columna for columna in df.columns if columna != 'date']
    li = LimpiezaCalidad(_columnas_numericas)
    if 'index' not in df.columns:
        cant_filas = df.shape[0]
        df['index'] = np.linspace(1, cant_filas, cant_filas)
    return li.transform(df)


def main(data_folder_path, input_filename: str):
    """ Runs data processing scripts. to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    input_filename_stem = input_filename.split('.')[0]
    input_filename = input_filename_stem + '.parquet'
    logger = logging.getLogger(__name__)
    logger.info('preprocesando el dataset')
    du = DataUtils(
        data_folder_path=data_folder_path,
        input_file_name=input_filename,
        y_name='price',
        load_data=lambda path: pd.read_parquet(path),
        save_data=lambda df, path: df.to_parquet(path)
    )
    input_filename_stem = input_filename.split('.')[0]
    ruta = du.raw_path.joinpath(input_filename_stem+'.csv')
    df_raw = pd.read_csv(ruta, sep=',', index_col=0)
    du.data = preprocessing(df_raw)
    du.save_data(
        du.data,
        du.preprocessed_file_path
    )


@click.command()
@click.argument('data_folder_path', type=click.types.Path(file_okay=False))
@click.argument('input_filename', type=click.types.STRING)
def main_terminal(data_folder_path, input_filename):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    main(data_folder_path, input_filename)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main_terminal()
