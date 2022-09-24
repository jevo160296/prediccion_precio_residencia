from pathlib import Path
from typing import Union, Callable
from pandas import DataFrame
import joblib
import pandas as pd


class DataUtils:
    def __init__(self,
                 data_folder_path: Path,
                 input_file_name: str,
                 load_data: Callable[[Path], DataFrame] = lambda input_filepath: pd.read_csv(input_filepath, sep=';'),
                 save_data: Callable[[DataFrame, Path], None] = lambda df, filepath: df.to_csv(filepath, sep=';',
                                                                                               index=False)
                 ):
        self.data_folder_path = data_folder_path
        self.input_file_name = input_file_name
        self._X_names: Union[None, list] = None
        self._data: Union[None, DataFrame] = None
        self._input_data: Union[None, DataFrame] = None
        self._train_test_data: Union[None, DataFrame] = None
        self._validation_data: Union[None, DataFrame] = None
        self._model = None
        self.load_data = load_data
        self.save_data = save_data

    @property
    def X_names(self):
        return list(set.difference(set(self.data.columns), {self.y_names}))

    @property
    def y_names(self):
        return 'precio_kg'

    @property
    def data(self) -> DataFrame:
        return self._data

    @data.setter
    def data(self, data):
        self._data = data

    @property
    def input_data(self) -> DataFrame:
        if self._input_data is None:
            self._input_data = self.load_data(self.input_file_path)
        return self._input_data.copy()

    @property
    def train_test_data(self) -> DataFrame:
        if self._train_test_data is None:
            self._train_test_data = self.load_data(self.transformed_train_test_path)
        return self._train_test_data.copy()

    @property
    def validation_data(self) -> DataFrame:
        if self._validation_data is None:
            self._validation_data = self.load_data(self.transformed_validation_path)
        return self._validation_data.copy()

    @property
    def model(self):
        if self._model is None:
            self._model = joblib.load(self.models_path)
        return self._model

    @model.setter
    def model(self, model):
        joblib.dump(model, self.models_path)

    @property
    def input_file_path(self):
        return self.data_folder_path.joinpath('raw', self.input_file_name)

    @property
    def raw_validation_path(self):
        path = self.data_folder_path.joinpath('interim', self.input_file_name)
        return path.with_stem(path.stem + '_validation')

    @property
    def raw_train_test_path(self):
        path = self.data_folder_path.joinpath('interim', self.input_file_name)
        return path.with_stem(path.stem + '_train_test')

    @property
    def transformed_validation_path(self):
        path = self.raw_validation_path.with_stem(self.raw_validation_path.stem + '_processed')
        name = path.name
        parent = path.parent.parent.joinpath('processed')
        return parent.joinpath(name)

    @property
    def transformed_train_test_path(self):
        path = self.raw_train_test_path.with_stem(self.raw_train_test_path.stem + '_processed')
        name = path.name
        parent = path.parent.parent.joinpath('processed')
        return parent.joinpath(name)

    @property
    def models_path(self):
        path = self.data_folder_path.parent.joinpath('models/')
        return path

    @property
    def external_path(self):
        return self.data_folder_path.joinpath('external/')

    @property
    def interim_path(self):
        return self.data_folder_path.joinpath('interim/')

    @property
    def processed_path(self):
        return self.data_folder_path.joinpath('processed/')

    @property
    def raw_path(self):
        return self.data_folder_path.joinpath('raw/')
