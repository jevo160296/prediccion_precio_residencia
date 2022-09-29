from pathlib import Path
from typing import Union, Callable
from pandas import DataFrame
import joblib
import pandas as pd
from functools import wraps


def create_parent_if_not_exists(funcion):
    @wraps(funcion)
    def wrapper(*args, **kwargs):
        path: Path = funcion(*args, **kwargs)
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    return wrapper


class DataUtils:
    def __init__(self,
                 data_folder_path: Path,
                 input_file_name: str,
                 y_name: str = '',
                 load_data: Callable[[Path], DataFrame] = lambda input_filepath: pd.read_csv(input_filepath, sep=';'),
                 save_data: Callable[[DataFrame, Path], None] = lambda df, filepath: df.to_csv(filepath, sep=';',
                                                                                               index=False)
                 ):
        self.data_folder_path = Path(data_folder_path)
        self.input_file_name = input_file_name
        self._y_name = y_name
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
        return self._y_name

    @property
    def data(self) -> DataFrame:
        return self._data

    @data.setter
    def data(self, data):
        self._data = data

    @property
    def model(self):
        if self._model is None:
            self._model = joblib.load(self.model_path)
        return self._model

    @model.setter
    def model(self, model):
        joblib.dump(model, self.model_path)

    @property
    @create_parent_if_not_exists
    def input_file_path(self):
        return self.data_folder_path.joinpath('raw', self.input_file_name)

    @property
    @create_parent_if_not_exists
    def preprocessed_file_path(self):
        ruta_guardado = self.interim_path.joinpath(Path(self.input_file_name).with_suffix('.parquet').name)
        return ruta_guardado.with_stem(ruta_guardado.stem + '_preprocesado')

    @property
    @create_parent_if_not_exists
    def raw_validation_path(self):
        path = self.interim_path.joinpath(self.input_file_name)
        return path.with_stem(path.stem + '_validation')

    @property
    @create_parent_if_not_exists
    def raw_train_test_path(self):
        path = self.interim_path.joinpath(self.input_file_name)
        return path.with_stem(path.stem + '_train_test')

    @property
    @create_parent_if_not_exists
    def transformed_validation_path(self):
        path = self.raw_validation_path.with_stem(self.raw_validation_path.stem + '_processed')
        name = path.name
        return self.processed_path.joinpath(name)

    @property
    @create_parent_if_not_exists
    def transformed_train_test_path(self):
        path = self.raw_train_test_path.with_stem(self.raw_train_test_path.stem + '_processed')
        name = path.name
        return self.processed_path.joinpath(name)

    @property
    @create_parent_if_not_exists
    def model_path(self):
        path = self.data_folder_path.parent.joinpath('models/model.joblib')
        return path

    @property
    @create_parent_if_not_exists
    def external_path(self):
        return self.data_folder_path.joinpath('external/')

    @property
    @create_parent_if_not_exists
    def interim_path(self):
        return self.data_folder_path.joinpath('interim/')

    @property
    @create_parent_if_not_exists
    def processed_path(self):
        return self.data_folder_path.joinpath('processed/')

    @property
    @create_parent_if_not_exists
    def raw_path(self):
        return self.data_folder_path.joinpath('raw/')
