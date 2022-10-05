from typing import List, Dict, Callable

import pandas as pd
from pandas import DataFrame
import numpy as np


def z_score_outliers(_df, _column):
    """
    Returns:
        zscore, outlier
    """
    # Adaptado de https://www.kaggle.com/code/shweta2407/regression-on-housing-data-accuracy-87
    # creating lists to store zscore and outliers
    zscore = []
    isoutlier = []
    # for zscore generally taken thresholds are 2.5, 3 or 3.5 hence i took 3
    threshold = 3
    # calculating the mean of the passed column
    mean = np.mean(_df[_column])
    # calculating the standard deviation of the passed column
    std = np.std(_df[_column])
    for i in _df[_column]:
        z = (i - mean) / std
        zscore.append(z)
        # if the zscore is greater than threshold = 3 that means it is an outlier
        isoutlier.append(np.abs(z) > threshold)
    return zscore, isoutlier


def z_score_outliers_delete(df: DataFrame, columns: list) -> DataFrame:
    df = df.copy()
    for column in columns:
        df = df[~pd.Series(z_score_outliers(df, column)[1], index=df.index)]
    return df


def eliminacionOutliers(data: pd.DataFrame, columns: list) -> pd.DataFrame:
    data = data.copy()
    data = z_score_outliers_delete(data, columns)
    return data


def eliminacion_outliers_custom_function(df: DataFrame, columns: List[str],
                                         isoutlier_funcs: Dict[str, Callable[[DataFrame], pd.Series]]) -> DataFrame:
    df = df.copy()
    for column in columns:
        isoutlier = isoutlier_funcs[column](df)
        df = df[~isoutlier]
    return df


def conversionTipoDatos(data: pd.DataFrame) -> pd.DataFrame:
    variables_categoricas = ['grade', 'view', 'waterfront', 'condition', 'zipcode']
    data[variables_categoricas] = data[variables_categoricas].astype('category')

    return data


def calculoVariablesAdicionales(data: pd.DataFrame) -> pd.DataFrame:
    data['yr_date'] = data['date'].dt.year
    data['antiguedad_venta'] = data['yr_date'] - data['yr_built']
    data.drop(columns=['yr_date', 'date', 'yr_built'], inplace=True)

    return data


def eliminacionColumnas(data: pd.DataFrame) -> pd.DataFrame:
    columnas_a_eliminar = {'lat', 'yr_renovated', 'long', 'jhygtf'}.intersection(data.columns)
    data.drop(columns=columnas_a_eliminar, inplace=True)

    return data
