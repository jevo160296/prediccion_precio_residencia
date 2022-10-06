# -*- coding: utf-8 -*-

import logging
from pathlib import Path

import click
import numpy as np
from dotenv import find_dotenv, load_dotenv
from pandas import DataFrame

from src.core.variables_globales import deprecated
from src.core.steps import Steps
from src.data.procesamiento_datos import LimpiezaCalidad


@deprecated
def preprocessing(df: DataFrame) -> DataFrame:
    _columnas_numericas = [columna for columna in df.columns if columna != 'date']
    li = LimpiezaCalidad(_columnas_numericas)
    if 'index' not in df.columns:
        cant_filas = df.shape[0]
        df['index'] = np.linspace(1, cant_filas, cant_filas)
    return li.transform(df)


def main(steps: Steps = None):
    """ Runs data processing scripts. to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]
    if steps is None:
        steps = Steps.build(str(project_dir), logger)
    df_transformado = steps.etl(False)

    ruta_guardado = steps.du.interim_path.joinpath('etl.parquet')

    steps.du.save_data(
        df_transformado,
        ruta_guardado)
    logger.info(f'Resultado de preprocessing guardado exitosamente en {ruta_guardado}')
    return df_transformado


@click.command()
def main_terminal():
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    main()


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main_terminal()
