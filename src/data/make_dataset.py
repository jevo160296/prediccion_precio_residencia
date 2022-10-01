# -*- coding: utf-8 -*-

import logging
from typing import Tuple

import click
from dotenv import find_dotenv, load_dotenv
from pandas import DataFrame

from src.core.steps import Steps


def main(steps: Steps = None, porcentaje_entrenamiento=0.7) -> Tuple[DataFrame, DataFrame]:
    """
    Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).

    :return:  df_train_test, df_validation
    """
    logger = logging.getLogger(__name__)
    if steps is None:
        steps = Steps.build(logger)
    df_train_test, df_validation = steps.make_dataset(porcentaje_entrenamiento)
    steps.du.save_data(
        df_train_test,
        steps.du.raw_train_test_path
    )
    logger.info(f'Resultado de la partición de training en make_dataset guardado exitosamente en '
                f'{steps.du.raw_train_test_path}')
    steps.du.save_data(
        df_validation,
        steps.du.raw_validation_path
    )
    logger.info(f'Resultado de la partición de validation en make_dataset guardado exitosamente en '
                f'{steps.du.raw_validation_path}')
    return df_train_test, df_validation


@click.command()
@click.argument('porcentaje_entrenamiento', type=click.types.FLOAT, default=0.7)
def main_terminal(porcentaje_entrenamiento):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    main(porcentaje_entrenamiento=porcentaje_entrenamiento)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main_terminal()
