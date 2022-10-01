from typing import Tuple, Callable

import sklearn.exceptions
from pandas import DataFrame
from sklearn.model_selection import train_test_split
from pathlib import Path
import pandas as pd
import numpy as np


def convertir_tipos(df: DataFrame, columnas_a_convertir):
    _df = df.copy()
    total_filas = _df.shape[0]
    for columna_numerica in set(columnas_a_convertir).intersection(df.columns):
        if not pd.api.types.is_numeric_dtype(_df[columna_numerica]):
            # son_numericos = _df[columna_numerica].apply(pd.api.types.is_number)
            # cant = son_numericos.sum()
            # print(
            #    f'La columna {columna_numerica} tiene {cant} valores no numéricos, {cant / total_filas:.2%}, '
            #    f'se reemplazarán por nan.')
            _df[columna_numerica] = pd.to_numeric(_df[columna_numerica], errors='coerce')
    return _df


def validar_duplicados(df: DataFrame):
    filas = df.shape[0]
    cant_duplicados = df.duplicated().sum()
    print(
        f'De {filas} registros hay {cant_duplicados} filas duplicadas, representando el {cant_duplicados / filas:.2%}')


def eliminar_duplicados(df: DataFrame, index_dup_ok=True):
    # Eliminando duplicados
    # print(f'Antes de la eliminación de duplicados, el conjunto de datos tiene {df.shape[0]} filas.')
    df = df.drop_duplicates(keep='first')
    hay_index_duplicados = sum(validar_index_duplicados(df)) > 0
    if hay_index_duplicados and not index_dup_ok:
        raise sklearn.exceptions.ChangedBehaviorWarning('Después de la eliminación de duplicados exactos aún quedan '
                                                        'index duplicados.')
    # print(f'Después de la eliminación de duplicados, el conjunto de datos queda con {filas} filas.')
    return df


def convertir_col_date_a_date(df: DataFrame) -> DataFrame:
    _df = df.copy()
    if 'date' in df.columns:
        _df['date'] = pd.to_datetime(_df['date'], errors='coerce')
    return _df


def reemplazar_valores_extremos(df: DataFrame, columnas_numericas) -> DataFrame:
    _df = df.copy()
    columnas_numericas = list(set(columnas_numericas).intersection(df.columns))
    _df[columnas_numericas] = _df[columnas_numericas].where(lambda x: x > -1e+10, other=np.nan).where(
        lambda x: x < 1e+10, other=np.nan)
    return _df


def reemplazar_nulos_por_la_media(df: DataFrame, columnas_numericas) -> DataFrame:
    # Se reemplazan los valores nulos por la media Nota: No se considera que haya data leakage pues los valores
    # reemplazados son entre registros con el mismo index y como al final se va a dejar un dataset con index únicos,
    # no hay riesgo que estén tanto en el set de entrenamiento como en el de test
    _df = df.copy()
    columnas_numericas = list(set(columnas_numericas).intersection(df.columns))
    for columna_numerica in columnas_numericas:
        _df[columna_numerica] = _df[columna_numerica].fillna(
            _df.groupby('index')[columna_numerica].transform('median'))
    return _df


def reemplazar_fechas_nulas(df: DataFrame) -> DataFrame:
    _df = df.copy()
    # Reemplazando fechas nulas por la primera fecha no nula
    if 'date' in df.columns:
        _df['date'] = _df['date'].fillna(
            _df.groupby(['index'], sort=False)['date'].apply(lambda x: x.ffill().bfill()))
    return _df


def reemplazar_ceros_por_nulos(df: DataFrame) -> DataFrame:
    _df = df.copy()
    # Reemplazando ceros por valores nulos
    if 'sqft_basement' in _df.columns:
        _df['sqft_basement'] = _df['sqft_basement'].replace(0, np.nan)
    if 'yr_renovated' in df.columns:
        _df['yr_renovated'] = _df['yr_renovated'].replace(0, np.nan)
    return _df


def validar_index_duplicados(df: DataFrame):
    # Validando duplicados de index
    son_duplicados = df['index'].duplicated()
    # cant_duplicados = son_duplicados.sum()
    # filas = df.shape[0]
    # print(
    #    f'De {filas} registros, hay {cant_duplicados} registros con index duplicado, que representan el '
    #    f'{cant_duplicados / filas:.2%}.')
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
