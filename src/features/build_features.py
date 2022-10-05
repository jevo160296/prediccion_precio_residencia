import logging
from pathlib import Path

import click
from dotenv import find_dotenv, load_dotenv
from pandas import DataFrame

from src.core.variables_globales import deprecated
from src.core.steps import Steps
from src.data.procesamiento_datos import ProcesamientoDatos


@deprecated
def build_features(df: DataFrame) -> DataFrame:
    pda = ProcesamientoDatos()
    return pda.fit_transform(df)


def main(steps: Steps = None, porcentaje_entrenamiento=0.7):
    logger = logging.getLogger(__name__)
    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]
    if steps is None:
        steps = Steps.build(str(project_dir), logger)
    df_train_test_transformed, df_validacion = steps.feature_engineering(porcentaje_entrenamiento, False)

    steps.du.save_data(
        df_train_test_transformed,
        steps.du.transformed_train_test_path
    )
    logger.info(f'Resultado de transformar el train set guardado en {steps.du.transformed_train_test_path}')
    return df_train_test_transformed


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
