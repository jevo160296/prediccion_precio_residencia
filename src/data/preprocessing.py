# -*- coding: utf-8 -*-

import logging
import click
from dotenv import find_dotenv, load_dotenv
from src.core.steps import Steps


def main(steps=None):
    """ Runs data processing scripts. to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    if steps is None:
        steps = Steps.build(logger)
    df_preprocesado = steps.etl()
    ruta_guardado = steps.du.interim_path.joinpath('etl.parquet')
    steps.du.save_data(
        df_preprocesado,
        ruta_guardado
    )
    logger.info(f'Resultado de preprocessing guardado exitosamente en {ruta_guardado}')
    return df_preprocesado


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
