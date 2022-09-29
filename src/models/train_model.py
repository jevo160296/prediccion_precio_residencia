import logging
from pathlib import Path

import click
import pandas as pd
from pandas import DataFrame

import src.features.build_features as build_features
from src.jutils.data import DataUtils
from src.models.modelo import Modelo


def entrenar(df: DataFrame) -> Modelo:
    modelo = Modelo()
    modelo.fit(df, df['price'])
    return modelo


def main(data_folder_path, input_filename, porcentaje_entrenamiento):
    input_filename_stem = input_filename.split('.')[0]
    input_filename = input_filename_stem + '.parquet'
    logger = logging.getLogger(__name__)
    logger.info('Entrenando el modelo')
    du = DataUtils(
        data_folder_path=data_folder_path,
        input_file_name=input_filename,
        y_name='price',
        load_data=lambda path: pd.read_parquet(path),
        save_data=lambda _df, path: _df.to_parquet(path)
    )
    if not du.transformed_train_test_path.exists():
        build_features.main(data_folder_path, input_filename, porcentaje_entrenamiento, 'Entrenamiento')
    df = du.load_data(du.transformed_train_test_path)
    modelo = entrenar(df)
    du.model = modelo


@click.command()
@click.argument('data_folder_path', type=click.types.Path(file_okay=False))
@click.argument('input_filename', type=click.types.STRING)
@click.argument('porcentaje_entrenamiento', type=click.types.FLOAT)
def main_terminal(data_folder_path, input_filename, porcentaje_entrenamiento):
    main(data_folder_path, input_filename, porcentaje_entrenamiento)


if __name__ == '__main__':
    main_terminal()
