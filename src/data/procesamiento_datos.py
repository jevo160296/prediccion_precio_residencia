from typing import Union

from pandas import DataFrame
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import make_pipeline, Pipeline


class LimpiezaCalidad(BaseEstimator, TransformerMixin):
    def __init__(self):
        self._pipeline: Union[None, Pipeline] = None

    @property
    def pipeline(self) -> Pipeline:
        if self._pipeline is None:
            self._pipeline = make_pipeline([])
        return self._pipeline

    def fit(self, X: DataFrame, y=None):
        self.pipeline.fit(X)

    def transform(self, X: DataFrame, y=None) -> DataFrame:
        return self.pipeline.transform(X)
