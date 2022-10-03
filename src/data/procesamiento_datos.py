from typing import Union, List, Callable

import pandas as pd
from pandas import DataFrame
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import make_column_transformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from src.data.funciones_base import convertir_tipos, eliminar_duplicados, convertir_col_date_a_date, \
    reemplazar_valores_extremos, reemplazar_nulos_por_la_media, reemplazar_fechas_nulas, reemplazar_ceros_por_nulos, \
    validar_index_duplicados, calcular_mediana_recortada, mediana_recortada_imputacion, validar_datos_nulos, \
    clasificar_columnas, seleccionar_columnas, calculo_variables_adicionales
from src.features.limpiezaDatos1 import eliminacionOutliers, eliminacion_outliers_custom_function
from src.core.variables_globales import columnas_entrada


def outlier_nan(column: str) -> Callable[[DataFrame], pd.Series]:
    def inner(X: DataFrame):
        return X[column].isna()

    return inner


class LimpiezaCalidad(BaseEstimator, TransformerMixin):
    def __init__(self, columnas_numericas: List[str]):
        self._pipeline: Union[None, Pipeline] = None
        self.columnas_numericas: List[str] = columnas_numericas

    @property
    def pipeline(self) -> Pipeline:
        if self._pipeline is None:
            self._pipeline = Pipeline([
                ('convertir_tipos', FunctionTransformer(self._convertir_tipos)),
                ('eliminar_duplicados', FunctionTransformer(eliminar_duplicados)),
                ('col_date_a_date', FunctionTransformer(convertir_col_date_a_date)),
                ('valores_extremos', FunctionTransformer(self._reemplazar_valores_extremos)),
                ('nulos_por_media', FunctionTransformer(self._reemplazar_nulos_por_la_media)),
                ('fechas_nulas', FunctionTransformer(reemplazar_fechas_nulas)),
                ('ceros_por_nulos', FunctionTransformer(reemplazar_ceros_por_nulos)),
                ('eliminar_duplicados2', FunctionTransformer(eliminar_duplicados)),
                ('calculo_variables_adicionales', FunctionTransformer(calculo_variables_adicionales)),
                ('validar_indices_duplicados', FunctionTransformer(validar_index_duplicados,
                                                                   kw_args={'is_duplicated_ok': False})),
                ('seleccionar_columnas', FunctionTransformer(seleccionar_columnas,
                                                             kw_args={'columnas': columnas_entrada + ['price']}))
            ]
            )
        return self._pipeline

    def _convertir_tipos(self, df: DataFrame) -> DataFrame:
        return convertir_tipos(df, self.columnas_numericas)

    def _reemplazar_valores_extremos(self, df: DataFrame) -> DataFrame:
        return reemplazar_valores_extremos(df, self.columnas_numericas)

    def _reemplazar_nulos_por_la_media(self, df: DataFrame) -> DataFrame:
        return reemplazar_nulos_por_la_media(df, self.columnas_numericas)

    def fit(self, X: DataFrame, y=None):
        return self.pipeline

    def transform(self, X: DataFrame, y=None) -> DataFrame:
        return self.pipeline.transform(X)


class Preprocesamiento(BaseEstimator, TransformerMixin):
    """
    Esta clase tiene los métodos necesarios para realizar el preprocesamiento de los datos de entrenamiento (Esto no se
    realiza para los datos de validación o predicción.
    """

    def __init__(self, columnas_z_score, columnas_drop_na):
        self._pipeline: Union[None, Pipeline] = None
        self.columnas_z_score = columnas_z_score
        self.columnas_drop_na = columnas_drop_na
        self._isoutliers_definitions = {
            'bathrooms': self.outlier_bathrooms,
            'bedrooms': self.outlier_bedrooms,
            'sqft_living': outlier_nan('sqft_living'),
            'sqft_lot': outlier_nan('sqft_lot'),
            'floors': outlier_nan('floors'),
            'waterfront': outlier_nan('waterfront'),
            'view': outlier_nan('view'),
            'grade': outlier_nan('grade'),
            'sqft_above': outlier_nan('sqft_above'),
            'lat': outlier_nan('lat'),
            'sqft_living15': outlier_nan('sqft_living15'),
            'price': outlier_nan('price'),
            'antiguedad_venta': outlier_nan('antiguedad_venta')
        }

    @staticmethod
    def outlier_bathrooms(X: DataFrame) -> pd.Series:
        return (X['bathrooms'] == 0) | (X['bathrooms'] > 4)

    @staticmethod
    def outlier_bedrooms(X: DataFrame) -> pd.Series:
        return (X['bedrooms'] == 0) | (X['bedrooms'] > 5)

    @property
    def pipeline(self) -> Pipeline:
        if self._pipeline is None:
            self._pipeline = Pipeline([
                ('eliminar_outliers', FunctionTransformer(eliminacionOutliers,
                                                          kw_args={'columns': self.columnas_z_score})),
                ('eliminar_outliers_custom', FunctionTransformer(self._eliminacion_outliers_custom_function))
            ]
            )
        return self._pipeline

    def _eliminacion_outliers_custom_function(self, df):
        return eliminacion_outliers_custom_function(df, self.columnas_drop_na, self._isoutliers_definitions)

    def fit(self, X: DataFrame, y: DataFrame = None):
        return self.pipeline.fit(X)

    def transform(self, X: DataFrame, y=None) -> DataFrame:
        return self.pipeline.transform(X)


class ProcesamientoDatos(BaseEstimator, TransformerMixin):
    def __init__(self):
        self._pipeline: Union[None, Pipeline] = None
        self._medianas_recortadas = None
        self._columnas_mediana_recortada_impute = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors',
                                                   'waterfront', 'view', 'grade', 'sqft_living15', 'zipcode',
                                                   'sqft_lot15', 'condition', 'bathrooms', 'bedrooms',
                                                   'antiguedad_venta']

        self._isoutliers_definitions = {
            'bathrooms': self.outlier_bathrooms,
            'bedrooms': self.outlier_bedrooms,
            'sqft_living': outlier_nan('sqft_living'),
            'sqft_lot': outlier_nan('sqft_lot'),
            'floors': outlier_nan('floors'),
            'waterfront': outlier_nan('waterfront'),
            'view': outlier_nan('view'),
            'grade': outlier_nan('grade'),
            'sqft_above': outlier_nan('sqft_above'),
            'lat': outlier_nan('lat'),
            'sqft_living15': outlier_nan('sqft_living15'),
            'zipcode': outlier_nan('zipcode'),
            'sqft_lot15': outlier_nan('sqft_lot15'),
            'condition': outlier_nan('condition'),
            'fue_renovada': outlier_nan('fue_renovada'),
            'antiguedad_venta': outlier_nan('antiguedad_venta')
        }
        self._clasificacion_columnas = {
            'categorica_ordinal': ['zipcode', 'grade', 'view', 'waterfront', 'condition', 'lat', 'long'],
            'fecha': ['date'],
            'id': ['index'],
            'numerica_continua': ['sqft_basement', 'sqft_above', 'sqft_living15', 'sqft_lot', 'price', 'sqft_lot15',
                                  'sqft_living'],
            'numerica_discreta': ['bathrooms', 'bedrooms', 'yr_renovated', 'yr_built', 'jhygtf', 'yr_date',
                                  'antiguedad_venta', 'floors']
        }

    @staticmethod
    def outlier_bathrooms(X: DataFrame) -> pd.Series:
        return (X['bathrooms'] == 0) | (X['bathrooms'] > 4) | (X['bathrooms'].isna())

    @staticmethod
    def outlier_bedrooms(X: DataFrame) -> pd.Series:
        return (X['bedrooms'] == 0) | (X['bedrooms'] > 5) | (X['bedrooms'].isna())


    @property
    def pipeline(self) -> Pipeline:
        if self._pipeline is None:
            self._pipeline = Pipeline([
                ('imputar_nulos_mediana', FunctionTransformer(self._mediana_recortada_imputacion)),
                ('validar nulos', FunctionTransformer(validar_datos_nulos)),
                ('conversion_tipos', FunctionTransformer(self._clasificar_columnas))
            ]
            )
        return self._pipeline

    def _mediana_recortada_imputacion(self, df: DataFrame) -> DataFrame:
        return mediana_recortada_imputacion(df, self._columnas_mediana_recortada_impute, self._isoutliers_definitions,
                                            self._medianas_recortadas)

    def _clasificar_columnas(self, df: DataFrame) -> DataFrame:
        return clasificar_columnas(df, self._clasificacion_columnas)

    def fit(self, X: DataFrame, y: DataFrame = None):
        self._medianas_recortadas = {}
        for column in self._columnas_mediana_recortada_impute:
            self._medianas_recortadas[column] = calcular_mediana_recortada(X, column,
                                                                           self._isoutliers_definitions[column](X))
        return self.pipeline.fit(X)

    def transform(self, X: DataFrame, y=None) -> DataFrame:
        return self.pipeline.transform(X)
