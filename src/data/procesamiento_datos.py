from typing import Union, List

from pandas import DataFrame
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, PowerTransformer, KBinsDiscretizer

from src.data.funciones_base import convertir_tipos, eliminar_duplicados, convertir_col_date_a_date, \
    reemplazar_valores_extremos, reemplazar_nulos_por_la_media, reemplazar_fechas_nulas, reemplazar_ceros_por_nulos
from src.features.limpiezaDatos1 import conversionTipoDatos, eliminacionColumnas, eliminacionOutliers, \
    calculoVariablesAdicionales
from src.features.procesamiento_datos import transformacion_logaritmica_y, entrenar_logaritmica, \
    entrenar_logaritmica_y, transformacion_logaritmica, numericas_a_binarias, procesamiento_datos_faltantes, \
    numericas_a_categoricas, entrenar_numericas_a_categoricas


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
                ('passtrhough', None)
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
        return self.pipeline.fit(X)

    def transform(self, X: DataFrame, y=None) -> DataFrame:
        return self.pipeline.transform(X)


class ProcesamientoDatos(BaseEstimator, TransformerMixin):
    def __init__(self, columnas_a_categoricas, columnas_a_logaritmo):
        self._pipeline: Union[None, Pipeline] = None
        self._pt: Union[None, PowerTransformer] = None
        self._pty: Union[None, PowerTransformer] = None
        self._kbd: Union[None, KBinsDiscretizer] = None
        self.columnas_a_categoricas = columnas_a_categoricas
        self.columnas_a_logaritmo = columnas_a_logaritmo

    @property
    def pipeline(self) -> Pipeline:
        if self._pipeline is None:
            self._pipeline = Pipeline([
                ('eliminar_outliers', FunctionTransformer(eliminacionOutliers)),
                ('conversion_tipos', FunctionTransformer(conversionTipoDatos)),
                ('calculo_variables_adicionales', FunctionTransformer(calculoVariablesAdicionales)),
                # ('eliminacion_columnas', FunctionTransformer(eliminacionColumnas)),
                ('procesamiento_datos_faltantes', FunctionTransformer(procesamiento_datos_faltantes)),
                # ('numericas_a_binarias', FunctionTransformer(numericas_a_binarias)),

                ('passtrhough', None)
            ]
            )
        return self._pipeline

    def _transformacion_logaritmica(self, df: DataFrame) -> DataFrame:
        if self._pt is None:
            self._pt = entrenar_logaritmica(df)
        return transformacion_logaritmica(df, self._pt)

    def _numericas_a_categoricas(self, df: DataFrame) -> DataFrame:
        if self._kbd is None:
            self._kbd = entrenar_numericas_a_categoricas(df, self.columnas_a_categoricas)
        return numericas_a_categoricas(df, self._kbd, self.columnas_a_categoricas)

    def _transformacion_logaritmica_y(self, df: DataFrame) -> DataFrame:
        if self._pty is None:
            self._pty = entrenar_logaritmica_y(df)
        return transformacion_logaritmica_y(df, self._pty)

    def fit(self, X: DataFrame, y: DataFrame = None):
        _ = self.pipeline.fit_transform(X)
        return self.pipeline.fit(X)

    def transform(self, X: DataFrame, y=None) -> DataFrame:
        return self.pipeline.transform(X)
