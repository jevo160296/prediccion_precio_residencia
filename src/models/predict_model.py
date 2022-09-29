import logging
from pathlib import Path

import click
import pandas as pd
import sklearn.utils.validation
from pandas import DataFrame
from sklearn.utils.validation import check_is_fitted

from src.models.modelo import Modelo
from src.jutils.data import DataUtils
import src.features.build_features as build_features


def predecir(df: DataFrame, modelo: Modelo) -> DataFrame:
    check_is_fitted(modelo, msg='El modelo no está entrenado aún, debe ejecutar el entrenamiento primero.')
    df['price_predict'] = modelo.predict(df)
    return df


def main(data_folder_path, input_filename):
    input_filename_stem = input_filename.split('.')[0]
    input_filename = input_filename_stem + '.parquet'
    logger = logging.getLogger(__name__)
    logger.info('realizando predicción.')
    du = DataUtils(
        data_folder_path=data_folder_path,
        input_file_name=input_filename,
        y_name='price',
        load_data=lambda path: pd.read_parquet(path),
        save_data=lambda _df, path: _df.to_parquet(path)
    )
    ruta_carga = du.transformed_train_test_path
    if not ruta_carga.exists():
        build_features.main(data_folder_path, input_filename, 1, "Entrenamiento")
    df_preprocesado = du.load_data(ruta_carga)
    if not du.model_path.exists():
        raise sklearn.exceptions.NotFittedError('Primero se debe ejecutar el entrenamiento del modelo.')
    modelo: Modelo = du.model
    df = predecir(df_preprocesado, modelo)
    ruta_guardado = du.processed_path.joinpath(
        du.input_file_path.
        with_stem(du.input_file_path.stem + '_predicho').
        with_suffix('.parquet').
        name)
    du.save_data(df, ruta_guardado)
    logger.info(f'Predicción guardada en {ruta_guardado}')


@click.command()
@click.argument('data_folder_path', type=click.types.Path(file_okay=False))
@click.argument('input_filename', type=click.types.STRING)
def main_terminal(data_folder_path, input_filename):
    main(data_folder_path, input_filename)


if __name__ == '__main__':
    project_dir = Path(__file__).resolve().parents[2]
    main_terminal()
