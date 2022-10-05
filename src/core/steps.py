from pandas import DataFrame
import numpy as np
from sklearn.model_selection import train_test_split
from src.data.procesamiento_datos import LimpiezaCalidad, Preprocesamiento, ProcesamientoDatos
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
            columnas_z_score: list,
            logger: Union[None, logging.Logger] = None
    ):
        self._get_raw_df = get_raw_df
        self.du = du
        self.X_columns = X_columns
        self.y_column = y_column
        self.columnas_z_score = columnas_z_score
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
    def build(cls, folder_path: str, logger: Union[None, logging.Logger] = None):
        du = DataUtils(
            Path(folder_path + r'\data'),
            'kc_house_dataDS.parquet',
            'price',
            load_data=lambda path: pd.read_parquet(path),
            save_data=lambda df, path: df.to_parquet(path)
        )

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
            X_columns=['zipcode', 'grade', 'view', 'bathrooms', 'bedrooms', 'sqft_living15', 'waterfront', 'floors',
                       'sqft_lot', 'condition', 'sqft_lot15', 'sqft_living', 'fue_renovada', 'antiguedad_venta'],
            y_column='price',
            columnas_z_score=['price', 'sqft_lot', 'sqft_lot15'],
            logger=logger
        )

    @property
    def raw_df(self):
        if self._raw_df is None:
            self._raw_df = self._get_raw_df()
        return self._raw_df

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
            self._preprocessing = Preprocesamiento(self.columnas_z_score, [])
        return self._preprocessing

    @property
    def processing(self):
        if self._processing is None:
            self._processing = ProcesamientoDatos()
        return self._processing

    @property
    def modelo(self):
        if self._modelo is None:
            if not self.du.model_path.exists():
                self._modelo = Modelo()
            else:
                self._modelo = self.du.model
        return self._modelo

    def predict_model_one(self, zipcode, grade, view, bathrooms, bedrooms, sqft_living15, waterfront, floors,
                          sqft_lot, condition, sqft_lot15, sqft_living, fue_renovada, antiguedad_venta):
        dictionary = {
            'zipcode': [zipcode],
            'grade': [grade],
            'view': [view],
            'bathrooms': [bathrooms],
            'bedrooms': [bedrooms],
            'sqft_living15': [sqft_living15],
            'waterfront': [waterfront],
            'floors': [floors],
            'sqft_lot': [sqft_lot],
            'condition': [condition],
            'sqft_lot15': [sqft_lot15],
            'sqft_living': [sqft_living],
            'fue_renovada': [fue_renovada],
            'antiguedad_venta': [antiguedad_venta]
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

    def etl(self, modo_entrenamiento_validacion: bool) -> DataFrame:
        self.log('Ejecutando etl.')
        df = self.raw_df
        li = self.li
        if 'index' not in df.columns:
            cant_filas = df.shape[0]
            df['index'] = np.linspace(1, cant_filas, cant_filas)
        df_transformado = li.transform(df)
        if modo_entrenamiento_validacion:
            # Cuando se inicialice en modo entrenamiento o validación se deben eliminar los registros con precios nulos
            #   o por fuera de lo normal de acuerdo con el z-score.
            pval = Preprocesamiento([], ['price'])
            df_transformado = pval.fit_transform(df_transformado)

        return df_transformado

    def make_dataset(self, porcentaje_entrenamiento,
                     modo_entrenamiento_validacion: bool) -> Tuple[DataFrame, DataFrame]:
        """
        :return: df_train_test, df_validation
        """
        df = self.etl(modo_entrenamiento_validacion)
        self.log('Ejecutando make_dataset.')
        if porcentaje_entrenamiento < 1:
            df_train_test, df_validation = train_test_split(df, train_size=porcentaje_entrenamiento,
                                                            random_state=1)
        else:
            df_train_test = df
            df_validation = df[[False] * df.shape[0]]

        return df_train_test, df_validation

    def feature_engineering(self, porcentaje_entrenamiento: float,
                            modo_entrenamiento_validacion: bool) -> Tuple[DataFrame, DataFrame]:
        """
        :return: df_train_test_transformed, df_validation
        """
        df_train_test, df_validation = self.make_dataset(porcentaje_entrenamiento, modo_entrenamiento_validacion)
        self.log('Ejecutando feature_engineering.')
        df_train_test_transformed = df_train_test. \
            pipe(self.preprocessing.fit_transform). \
            pipe(self.processing.fit_transform)
        df_validation_transformed = df_validation. \
            pipe(self.preprocessing.fit_transform). \
            pipe(self.processing.fit_transform)
        return df_train_test_transformed, df_validation_transformed

    def training(self, porcentaje_entrenamiento) -> Modelo:
        modo_entrenamiento_validacion = True
        df_train_test_transformed, _ = self.feature_engineering(porcentaje_entrenamiento, modo_entrenamiento_validacion)
        self.log('Ejecutando training.')
        self.modelo.fit(df_train_test_transformed[self.X_columns], df_train_test_transformed[self.y_column])
        return self.modelo

    def prediction(self, porcentaje_entrenamiento, modo_entrenamiento_validacion: bool):
        """
        :return: y_real_train_test, y_real_validation, y_predict_train_test, y_predict_validation
        """
        df_train_test_transformed, df_validation = self.feature_engineering(porcentaje_entrenamiento,
                                                                            modo_entrenamiento_validacion)
        self.log('Ejecutando prediction.')
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
        y_real_train_test, y_real_validation, y_predict_train_test, y_predict_validation = \
            self.prediction(porcentaje_entrenamiento, True)
        self.log('Ejecutando evaluation.')
        score_train_test = r2_score(y_real_train_test, y_predict_train_test)
        score_validation = r2_score(y_real_validation, y_predict_validation)
        return score_train_test, score_validation
