import logging

import click
import pandas as pd
from dotenv import find_dotenv, load_dotenv
from pandas import DataFrame

import src.data.make_dataset as make_dataset
from src.core.steps import Steps


def main(steps=None, porcentaje_entrenamiento=0.7):
    logger = logging.getLogger(__name__)

    if steps is None:
        steps = Steps.build(logger)
    df_train_test_transformed, _ = steps.feature_engineering(porcentaje_entrenamiento)

    steps.du.save_data(
        df_train_test_transformed,
        steps.du.transformed_train_test_path
    )
    logger.info(f'Resultado de transformar el train set guardado en {steps.du.transformed_train_test_path}')
    return df_train_test_transformed


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
