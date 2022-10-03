from typing import Union

from sklearn.base import RegressorMixin, BaseEstimator
from sklearn.compose import make_column_transformer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from src.core.variables_globales import columnas_entrada


class Modelo(BaseEstimator, RegressorMixin):
    def __init__(self):
        super(Modelo, self).__init__()
        self._pipeline: Union[None, Pipeline] = None
        self._is_fitted = False

    def __sklearn_is_fitted__(self):
        return self._is_fitted

    @property
    def pipeline(self):
        if self._pipeline is None:
            self._pipeline = make_pipeline(
                make_column_transformer(
                    ('passthrough', columnas_entrada)
                ),
                PolynomialFeatures(degree=2),
                LinearRegression()
            )
        return self._pipeline

    def fit(self, X, y):
        fitted = self.pipeline.fit(X, y)
        self._is_fitted = True
        return fitted

    def predict(self, X):
        return self.pipeline.predict(X)
