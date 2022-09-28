from sklearn.base import RegressorMixin, BaseEstimator
from sklearn.compose import make_column_transformer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.linear_model import Ridge
from typing import Union


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
                    ('passthrough', ['grade', 'view', 'bathrooms', 'bedrooms', 'sqft_above', 'sqft_living15',
                                     'waterfront', 'floors', 'sqft_lot', 'condition', 'sqft_lot15', 'sqft_living',
                                     'tiene_sotano'])
                ),
                Ridge()
            )
        return self._pipeline

    def fit(self, X, y):
        fitted = self.pipeline.fit(X, y)
        self._is_fitted = True
        return fitted

    def predict(self, X):
        return self.pipeline.predict(X)
