# -*- coding: utf-8 -*-
import os
import sys

import click
import logging
from pathlib import Path

import pandas as pd
from pandas import DataFrame
from dotenv import find_dotenv, load_dotenv
from src.jutils.data import DataUtils
from src.data.procesamiento_datos import LimpiezaCalidad, ProcesamientoDatos


def make_dataset(df: DataFrame) -> DataFrame:
    _columnas_numericas = [columna for columna in df.columns if columna != 'date']
    _columnas_a_logaritmo = ['sqft_above', 'sqft_living15', 'sqft_lot', 'sqft_lot15', 'sqft_living']
    _columnas_a_categoricas = ['sqft_lot', 'sqft_lot15']
    li = LimpiezaCalidad(_columnas_numericas)
    pda = ProcesamientoDatos(_columnas_a_categoricas, _columnas_a_logaritmo)
    return df.pipe(li.transform).pipe(pda.fit_transform)


@click.command()
@click.argument('input_filename', type=click.types.STRING)
def main(input_filename):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')
    du = DataUtils(
        data_folder_path=Path('../../data').resolve().absolute(),
        input_file_name=input_filename,
        y_name='price',
        load_data=lambda path: pd.read_parquet(path),
        save_data=lambda df, path: df.to_parquet(path)
    )
    df_raw = pd.read_csv(du.raw_path.joinpath(du.input_file_name), index_col=0, sep=',')
    du.data = make_dataset(df_raw)
    du.save_data(
        du.data,
        du.processed_path.joinpath(Path(du.input_file_name).with_suffix('.parquet').name)
    )


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
