from typing import Tuple, Callable

import sklearn.exceptions
from pandas import DataFrame
from sklearn.model_selection import train_test_split
from pathlib import Path
import pandas as pd
import numpy as np
from typing import List, Dict


def convertir_tipos(df: DataFrame, columnas_a_convertir):
    _df = df.copy()
    total_filas = _df.shape[0]
    for columna_numerica in columnas_a_convertir:
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


def eliminar_duplicados(df: DataFrame):
    # Eliminando duplicados
    # print(f'Antes de la eliminación de duplicados, el conjunto de datos tiene {df.shape[0]} filas.')
    df = df.drop_duplicates(keep='first')
    filas = df.shape[0]
    # print(f'Después de la eliminación de duplicados, el conjunto de datos queda con {filas} filas.')
    return df


def convertir_col_date_a_date(df: DataFrame) -> DataFrame:
    _df = df.copy()
    _df['date'] = pd.to_datetime(_df['date'], errors='coerce')
    return _df


def reemplazar_valores_extremos(df: DataFrame, columnas_numericas) -> DataFrame:
    _df = df.copy()
    _df[columnas_numericas] = _df[columnas_numericas].where(lambda x: x > -1e+10, other=np.nan).where(
        lambda x: x < 1e+10, other=np.nan)
    return _df


def reemplazar_nulos_por_la_media(df: DataFrame, columnas_numericas) -> DataFrame:
    # Se reemplazan los valores nulos por la media Nota: No se considera que haya data leakage pues los valores
    # reemplazados son entre registros con el mismo index y como al final se va a dejar un dataset con index únicos,
    # no hay riesgo que estén tanto en el set de entrenamiento como en el de test
    _df = df.copy()
    for columna_numerica in columnas_numericas:
        _df[columna_numerica] = _df[columna_numerica].fillna(
            _df.groupby('index')[columna_numerica].transform('median'))
    return _df


def reemplazar_fechas_nulas(df: DataFrame) -> DataFrame:
    _df = df.copy()
    # Reemplazando fechas nulas por la primera fecha no nula
    _df['date'] = _df['date'].fillna(
        _df.groupby(['index'], sort=False)['date'].apply(lambda x: x.ffill().bfill()))
    return _df


def reemplazar_ceros_por_nulos(df: DataFrame) -> DataFrame:
    _df = df.copy()
    # Reemplazando ceros por valores nulos
    _df[['sqft_basement', 'yr_renovated']] = _df[['sqft_basement', 'yr_renovated']].replace(0, np.nan)
    return _df


def validar_index_duplicados(df: DataFrame, is_duplicated_ok=True) -> DataFrame:
    # Validando duplicados de index
    son_duplicados = df['index'].duplicated()
    if not is_duplicated_ok and son_duplicados.sum() > 0:
        msg_error = 'Después de realizar la limpieza inicial, aún quedan índices duplicados, esto no sucedió con el ' \
                    'conjunto inicial de entrenamiento, se deben validar nuevamente los datos '
        raise Exception(msg_error)
    # cant_duplicados = son_duplicados.sum()
    # filas = df.shape[0]
    # print(
    #    f'De {filas} registros, hay {cant_duplicados} registros con index duplicado, que representan el '
    #    f'{cant_duplicados / filas:.2%}.')
    return df


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


def calcular_mediana_recortada(df: DataFrame, column: str, isoutlier: pd.Series):
    df = df.copy()
    mediana_recortada = df[column][~isoutlier].median()
    return mediana_recortada


def mediana_recortada_imputacion(df: DataFrame, columns: List[str],
                                 isoutlier_funcs: Dict[str, Callable[[DataFrame], pd.Series]],
                                 mediana_recortada: Dict[str, float]) -> DataFrame:
    df = df.copy()
    for column in columns:
        isoutlier = isoutlier_funcs[column](df)
        df.loc[isoutlier, column] = mediana_recortada[column]
    return df


def validar_datos_nulos(df: DataFrame) -> DataFrame:
    df = df.copy()
    cant_nulos_por_columna = df.drop(columns='price').isnull().sum()
    columnas_con_nulos = cant_nulos_por_columna[cant_nulos_por_columna > 0]
    cant_columnas_con_nulos = len(columnas_con_nulos)
    if cant_columnas_con_nulos > 0:
        raise Exception(f'Después de la imputación y eliminación de nulos, quedaron valores nulos en algunas columnas,'
                        'esto no sucedió en los datos originales de entrenamiento, se debe revisar los datos '
                        f'nuevamente. Columnas con nulos: {str(columnas_con_nulos)}')
    return df


def clasificar_columnas(df: DataFrame, clasificacion_columnas: Dict[str, List[str]]):
    categoricas_ordinal = list(set(clasificacion_columnas['categorica_ordinal']).intersection(df.columns))
    numericas_continua = list(set(clasificacion_columnas['numerica_continua']).intersection(df.columns))
    numerica_discreta = list(set(clasificacion_columnas['numerica_discreta']).intersection(df.columns))

    df[categoricas_ordinal] = df[categoricas_ordinal].astype(int)
    df[numericas_continua] = df[numericas_continua].astype(float)
    df[numerica_discreta] = np.floor(df[numerica_discreta], where=lambda x: ~np.isnan(x))
    return df


def seleccionar_columnas(df: DataFrame, columnas: List[str]) -> DataFrame:
    columnas = list(set(columnas).intersection(df.columns))
    return df.copy()[columnas]


def calculo_variables_adicionales(df: DataFrame) -> DataFrame:
    df = df.copy()
    df['fue_renovada'] = (~df['yr_renovated'].isna()).astype(int)
    df['yr_date'] = df['date'].dt.year
    df['antiguedad_venta'] = df['yr_date'] - df['yr_built']
    return df
