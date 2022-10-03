import logging

import click
import pandas as pd
from dotenv import find_dotenv, load_dotenv
from pandas import DataFrame

import src.data.make_dataset as make_dataset
from src.data.procesamiento_datos import ProcesamientoDatos
from src.jutils.data import DataUtils


def build_features(df: DataFrame) -> DataFrame:
    pda = ProcesamientoDatos()
    return pda.fit_transform(df)


def main(data_folder_path, input_filename, porcentaje_entrenamiento, set_train_validacion):
    input_filename_stem = input_filename.split('.')[0]
    input_filename = input_filename_stem + '.parquet'
    logger = logging.getLogger(__name__)
    logger.info('Buiding features')
    du = DataUtils(
        data_folder_path=data_folder_path,
        input_file_name=input_filename,
        y_name='price',
        load_data=lambda path: pd.read_parquet(path),
        save_data=lambda df, path: df.to_parquet(path)
    )
    if not (du.raw_train_test_path.exists() and du.raw_validation_path.exists()):
        make_dataset.main(data_folder_path, input_filename, porcentaje_entrenamiento)
    if set_train_validacion == 'Entrenamiento':
        du.data = du.load_data(du.raw_train_test_path)
        ruta_guardado = du.transformed_train_test_path
    elif set_train_validacion == 'Validacion':
        du.data = du.load_data(du.raw_validation_path)
        ruta_guardado = du.transformed_validation_path
    else:
        raise AttributeError('El atributo set_train_validacion solo puede ser Entrenamiento o Validacion')
    df_transformado = build_features(du.data)
    du.save_data(df_transformado, ruta_guardado)


@click.command()
@click.argument('data_folder_path', type=click.types.Path(file_okay=False))
@click.argument('input_filename', type=click.types.STRING)
@click.argument('porcentaje_entrenamiento', type=click.types.FLOAT)
@click.argument('set_train_validacion', type=click.types.Choice(['Entrenamiento', 'Validacion']))
def main_terminal(data_folder_path, input_filename, porcentaje_entrenamiento, set_train_validacion):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    main(data_folder_path, input_filename, porcentaje_entrenamiento, set_train_validacion)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main_terminal()
