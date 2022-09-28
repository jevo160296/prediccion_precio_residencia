import sklearn.utils.validation

from modelo import Modelo
import logging
from pandas import DataFrame
import click
from src.data.make_dataset import make_dataset
from sklearn.utils.validation import check_is_fitted
from src.jutils.data import DataUtils
from pathlib import Path
import pandas as pd


def predecir(df: DataFrame, modelo: Modelo) -> DataFrame:
    check_is_fitted(modelo, msg='El modelo no está entrenado aún, debe ejecutar el entrenamiento primero.')
    df_ = make_dataset(df)
    df_['price_predict'] = modelo.predict(df_)
    return df_


@click.command()
@click.argument('input_filename', type=click.types.STRING)
def main(input_filename):
    logger = logging.getLogger(__name__)
    logger.info('realizando predicción.')
    du = DataUtils(
        data_folder_path=Path('../../data').resolve().absolute(),
        input_file_name=input_filename,
        y_name='price',
        load_data=lambda path: pd.read_parquet(path),
        save_data=lambda _df, path: _df.to_parquet(path)
    )
    df = pd.read_csv(du.input_file_path, index_col=0, sep=',')
    if not du.models_path.exists():
        raise sklearn.exceptions.NotFittedError('Primero se debe ejecutar el entrenamiento del modelo.')
    modelo: Modelo = du.model
    df = predecir(df, modelo)
    df.to_csv(
        du.processed_path.joinpath(
            du.input_file_path.
            with_stem(du.input_file_path.stem + '_predicho').
            with_suffix('.csv').
            name),
        sep=';',
        index=True,
        decimal=','
    )


if __name__ == '__main__':
    main()
