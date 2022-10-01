from pandas import DataFrame
import numpy as np
from sklearn.model_selection import train_test_split
from src.data.procesamiento_datos import LimpiezaCalidad, Preprocesamiento, ProcesamientoDatos, PostProcesamiento
from src.models.modelo import Modelo
from typing import Tuple, Callable, Union
from sklearn.metrics import r2_score
from src.jutils.data import DataUtils, DataAccess
from pathlib import Path
import zipfile
import pandas as pd
import logging


class Steps:
    def __init__(
            self,
            get_raw_df: Callable[[], DataFrame],
            du: DataUtils,
            X_columns: list,
            y_column: str,
            logger: Union[None, logging.Logger] = None
    ):
        self._get_raw_df = get_raw_df
        self.du = du
        self.X_columns = X_columns
        self.y_column = y_column
        self.logger = logger

        self._columnas_numericas = None
        self._columnas_a_logaritmo = None
        self._columnas_a_categoricas = None
        self._raw_df = None
        self._li = None
        self._preprocessing = None
        self._processing = None
        self._postprocessing = None
        self._modelo = None

    @classmethod
    def build(cls, logger: Union[None, logging.Logger] = None):
        du = DataUtils(Path(r'C:\Users\jevo1\Documents\Python Scripts\trabajo_ciencia_de_datos_1\data'),
                       'kc_house_dataDS.csv', 'price')

        def process_raw_file(path_to_downloaded_file: Path):
            with zipfile.ZipFile(path_to_downloaded_file, 'r') as zip_ref:
                zip_ref.extractall(path_to_downloaded_file.parent)
            path_to_processed_file = path_to_downloaded_file.parent.joinpath(r'HouseKing\kc_house_dataDS.csv')
            return path_to_processed_file

        da = DataAccess(r'https://bit.ly/3orsN0U', du, lambda path: pd.read_csv(path, sep=',', index_col=0),
                        process_raw_file)
        return cls(
            da.get_df,
            du=du,
            X_columns=['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'waterfront', 'view',
                       'grade', 'sqft_above', 'lat', 'sqft_living15'],
            y_column='price',
            logger=logger
        )

    @property
    def raw_df(self):
        if self._raw_df is None:
            self._raw_df = self._get_raw_df()
        return self._raw_df

    @property
    def columnas_a_logaritmo(self):
        if self._columnas_a_logaritmo is None:
            self._columnas_a_logaritmo = ['sqft_above', 'sqft_living15', 'sqft_lot', 'sqft_lot15', 'sqft_living']
        return self._columnas_a_logaritmo

    @property
    def columnas_a_categoricas(self):
        if self._columnas_a_categoricas is None:
            self._columnas_a_categoricas = ['sqft_lot', 'sqft_lot15']
        return self._columnas_a_categoricas

    @property
    def columnas_numericas(self):
        if self._columnas_numericas is None:
            self._columnas_numericas = list(set(self.raw_df.columns).difference(['date']))
        return self._columnas_numericas

    @property
    def li(self) -> LimpiezaCalidad:
        if self._li is None:
            self._li = LimpiezaCalidad(self.columnas_numericas)
        return self._li

    @property
    def preprocessing(self):
        if self._preprocessing is None:
            self._preprocessing = Preprocesamiento()
        return self._preprocessing

    @property
    def processing(self):
        if self._processing is None:
            self._processing = ProcesamientoDatos(self.columnas_a_categoricas, self.columnas_a_logaritmo)
        return self._processing

    @property
    def postprocessing(self):
        if self._postprocessing is None:
            self._postprocessing = PostProcesamiento()
        return self._postprocessing

    @property
    def modelo(self):
        if self._modelo is None:
            if not self.du.model_path.exists():
                self._modelo = Modelo()
            else:
                self._modelo = self.du.model
        return self._modelo

    def predict_model_one(self, bedrooms, bathrooms, sqft_living, sqft_lot, floors, waterfront, view, grade, sqft_above,
                          lat, sqft_living15):
        dictionary = {
            'bedrooms': [bedrooms],
            'bathrooms': [bathrooms],
            'sqft_living': [sqft_living],
            'sqft_lot': [sqft_lot],
            'floors': [floors],
            'waterfront': [waterfront],
            'view': [view],
            'grade': [grade],
            'sqft_above': [sqft_above],
            'lat': [lat],
            'sqft_living15': [sqft_living15]
        }
        df = pd.DataFrame.from_dict(dictionary)
        return self.predict_model_many(df)

    def predict_model_many(self, df: DataFrame):
        if 'index' not in df.columns:
            rows = df.shape[0]
            df['index'] = np.linspace(1, rows, rows).round()
        df_transformed = df.pipe(self.li.transform).pipe(self.processing.transform)
        df_predicted = self.du.model.predict(df_transformed)
        return df_predicted

    def log(self, msg: str):
        if self.logger is not None:
            self.logger.info(msg)

    def etl(self) -> DataFrame:
        self.log('Ejecutando etl.')
        df = self.raw_df
        li = self.li
        if 'index' not in df.columns:
            cant_filas = df.shape[0]
            df['index'] = np.linspace(1, cant_filas, cant_filas)
        return li.transform(df)

    def make_dataset(self, porcentaje_entrenamiento) -> Tuple[DataFrame, DataFrame]:
        """
        :return: df_train_test, df_validation
        """
        self.log('Ejecutando make_dataset.')
        df = self.etl()
        if porcentaje_entrenamiento < 1:
            df_train_test, df_validation = train_test_split(df, train_size=porcentaje_entrenamiento,
                                                            random_state=1)
        else:
            df_train_test = df
            df_validation = df[[False] * df.shape[0]]

        return df_train_test, df_validation

    def feature_engineering(self, porcentaje_entrenamiento: float) -> Tuple[DataFrame, DataFrame]:
        """
        :return: df_train_test_transformed, df_validation
        """
        self.log('Ejecutando feature_engineering.')
        df_train_test, df_validation = self.make_dataset(porcentaje_entrenamiento)
        df_train_test_transformed = df_train_test. \
            pipe(self.preprocessing.fit_transform). \
            pipe(self.processing.fit_transform). \
            pipe(self.postprocessing.fit_transform)
        return df_train_test_transformed, df_validation

    def training(self, porcentaje_entrenamiento) -> Modelo:
        self.log('Ejecutando training.')
        df_train_test_transformed, _ = self.feature_engineering(porcentaje_entrenamiento)
        self.modelo.fit(df_train_test_transformed[self.X_columns], df_train_test_transformed[self.y_column])
        return self.modelo

    def prediction(self, porcentaje_entrenamiento):
        """
        :return: y_real_train_test, y_real_validation, y_predict_train_test, y_predict_validation
        """
        self.log('Ejecutando prediction.')
        df_train_test_transformed, df_validation = self.feature_engineering(porcentaje_entrenamiento)
        df_validation_transformed = self.processing.transform(df_validation)
        y_real_train_test = df_train_test_transformed[self.y_column]
        y_real_validation = df_validation_transformed[self.y_column]
        y_predict_train_test = self.modelo.predict(df_train_test_transformed)
        y_predict_validation = self.modelo.predict(df_validation_transformed)
        return y_real_train_test, y_real_validation, y_predict_train_test, y_predict_validation

    def evaluation(self, porcentaje_entrenamiento):
        """
        :return: score_train_test, score_validation
        """
        self.log('Ejecutando evaluation.')
        y_real_train_test, y_real_validation, y_predict_train_test, y_predict_validation = \
            self.prediction(porcentaje_entrenamiento)
        score_train_test = r2_score(y_real_train_test, y_predict_train_test)
        score_validation = r2_score(y_real_validation, y_predict_validation)
        return score_train_test, score_validation
