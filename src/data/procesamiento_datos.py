from typing import Union, List

from pandas import DataFrame
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import FunctionTransformer

from src.data.funciones_base import convertir_tipos, eliminar_duplicados, convertir_col_date_a_date, \
    reemplazar_valores_extremos, reemplazar_nulos_por_la_media, reemplazar_fechas_nulas, reemplazar_ceros_por_nulos


class LimpiezaCalidad(BaseEstimator, TransformerMixin):
    def __init__(self, columnas_numericas: List[str]):
        self._pipeline: Union[None, Pipeline] = None
        self._columnas_numericas: List[str] = columnas_numericas

    @property
    def pipeline(self) -> Pipeline:
        if self._pipeline is None:
            self._pipeline = make_pipeline([
                FunctionTransformer(self._convertir_tipos),
                FunctionTransformer(eliminar_duplicados),
                FunctionTransformer(convertir_col_date_a_date),
                FunctionTransformer(self._reemplazar_valores_extremos),
                FunctionTransformer(self._reemplazar_nulos_por_la_media),
                FunctionTransformer(reemplazar_fechas_nulas),
                FunctionTransformer(reemplazar_ceros_por_nulos),
                FunctionTransformer(eliminar_duplicados)
            ])
        return self._pipeline

    def _convertir_tipos(self, df: DataFrame) -> DataFrame:
        return convertir_tipos(df, self._columnas_numericas)

    def _reemplazar_valores_extremos(self, df: DataFrame) -> DataFrame:
        return reemplazar_valores_extremos(df, self._columnas_numericas)

    def _reemplazar_nulos_por_la_media(self, df: DataFrame) -> DataFrame:
        return reemplazar_nulos_por_la_media(df, self._columnas_numericas)

    def fit(self, X: DataFrame, y=None):
        self.pipeline.fit(X)

    def transform(self, X: DataFrame, y=None) -> DataFrame:
        return self.pipeline.transform(X)
