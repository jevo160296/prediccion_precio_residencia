from typing import Tuple, Callable

from pandas import DataFrame
from sklearn.model_selection import train_test_split
from pathlib import Path


def validar_duplicados(df: DataFrame):
    filas = df.shape[0]
    cant_duplicados = df.duplicated().sum()
    print(
        f'De {filas} registros hay {cant_duplicados} filas duplicadas, representando el {cant_duplicados / filas:.2%}')


def eliminar_duplicados(df: DataFrame):
    # Eliminando duplicados
    df = df.drop_duplicates(keep='first')
    filas = df.shape[0]
    print(f'Después de la eliminación de duplicados, el conjunto de datos queda con {filas} filas.')
    return df


def validar_index_duplicados(df: DataFrame):
    # Validando duplicados de index
    son_duplicados = df['index'].duplicated()
    cant_duplicados = son_duplicados.sum()
    filas = df.shape[0]
    print(
        f'De {filas} registros, hay {cant_duplicados} registros con index duplicado, que representan el '
        f'{cant_duplicados / filas:.2%}.')
    return son_duplicados


def split_train_test_validation(
        df: DataFrame,
        save_data: Callable[[DataFrame, Path], None],
        train_test_path: Path,
        validation_path: Path,
        test_size: float = 0.2,
        random_state: int = 1) -> Tuple[DataFrame, DataFrame]:
    train_test, validation = train_test_split(df, test_size, random_state)
    save_data(train_test, train_test_path)
    save_data(validation, validation_path)
    return train_test, validation
