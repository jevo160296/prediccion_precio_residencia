import unittest
from pathlib import Path

import pandas as pd
from pandas import DataFrame
import numpy as np

from src.data.funciones_base import (
    eliminar_duplicados, convertir_col_date_a_date, reemplazar_valores_extremos, reemplazar_nulos_por_la_media,
    reemplazar_fechas_nulas, reemplazar_ceros_por_nulos, convertir_tipos
)

from src.models.modelo import Modelo
import plotly.express as px

from src.features.procesamiento_datos import transformacion_logaritmica, entrenar_logaritmica, \
    transformacion_logaritmica_y, entrenar_logaritmica_y, numericas_a_binarias

from src.features.limpiezaDatos1 import conversionTipoDatos, eliminacionOutliers, eliminacionColumnas, \
    calculoVariablesAdicionales

from src.data.procesamiento_datos import LimpiezaCalidad, ProcesamientoDatos
from src.jutils.data import DataUtils
from src.jutils.visual import Plot


class TestsCase(unittest.TestCase):
    @classmethod
    def debugTestCase(cls):
        loader = unittest.defaultTestLoader
        testSuit = loader.loadTestsFromTestCase(cls)
        testSuit.debug()

    def setUp(self) -> None:
        data_folder_path = Path('../data').resolve().absolute()
        self._du_inicial = DataUtils(
            data_folder_path, 'kc_house_dataDS.csv', 'price', lambda path: pd.read_csv(path, sep=',', index_col=0),
            lambda df, path: df.to_parquet(path))
        self._du = DataUtils(
            data_folder_path, 'kc_house_dataDS.csv', 'price', lambda path: pd.read_parquet(path),
            lambda df, path: df.to_parquet(path)
        )
        self._df = self._du_inicial.input_data
        self._columnas_numericas = [columna for columna in self._df.columns if columna != 'date']
        self._columnas_a_logaritmo = ['sqft_above', 'sqft_living15', 'sqft_lot', 'sqft_lot15', 'sqft_living']
        self._columnas_a_categoricas = ['sqft_lot', 'sqft_lot15']

    @staticmethod
    def print_shape(df: DataFrame, msg: str = '', verbosity: int = 1) -> DataFrame:
        if verbosity > 0:
            print(f'{msg}: {df.shape=}')
        return df

    def run_limpieza_inicial(self) -> DataFrame:
        columnas_numericas = self._columnas_numericas
        df = self._df \
            .pipe(self.print_shape, 'Inicio') \
            .pipe(convertir_tipos, columnas_numericas) \
            .pipe(eliminar_duplicados) \
            .pipe(self.print_shape, 'Después de eliminar duplicados') \
            .pipe(convertir_col_date_a_date) \
            .pipe(reemplazar_valores_extremos, columnas_numericas) \
            .pipe(reemplazar_nulos_por_la_media, columnas_numericas) \
            .pipe(reemplazar_fechas_nulas) \
            .pipe(reemplazar_ceros_por_nulos) \
            .pipe(self.print_shape, 'Después de los reemplazos') \
            .pipe(eliminar_duplicados) \
            .pipe(self.print_shape, 'Después de eliminar índices duplicados')
        return df

    def test1_limpieza_inicial(self):
        df = self.run_limpieza_inicial()
        self.assertEqual(df['index'].duplicated().sum(), 0)
        print('Listo')

    def test2_limpieza_inicial_pipeline(self):
        df_pd_pipe = self.run_limpieza_inicial()
        li = LimpiezaCalidad(self._columnas_numericas)
        df_sk_pipe = li.transform(self._df)
        self.assertTrue(all(df_pd_pipe == df_sk_pipe))
        print('Listo')

    def test3_procesamiento_datos(self):
        li = LimpiezaCalidad(self._columnas_numericas)
        pda = ProcesamientoDatos(self._columnas_a_categoricas, self._columnas_a_logaritmo)
        df = li.transform(self._df)
        df = pda.fit_transform(df)
        print('Listo')

    def test4_entrenamiento_modelo(self):
        # Primero se procesarán los datos
        li = LimpiezaCalidad(self._columnas_numericas)
        pda = ProcesamientoDatos(self._columnas_a_categoricas, self._columnas_a_logaritmo)
        modelo = Modelo()
        df = self._df.pipe(li.transform).pipe(pda.fit_transform)
        modelo.fit(df, df['price'])
        print('Listo')

    def test5_prediccion_modelo(self):
        # Primero se procesará los datos
        li = LimpiezaCalidad(self._columnas_numericas)
        pda = ProcesamientoDatos(self._columnas_a_categoricas, self._columnas_a_logaritmo)
        modelo = Modelo()
        df = self._df.pipe(li.transform).pipe(pda.fit_transform)
        modelo.fit(df, df['price'])
        df_predicho: DataFrame = modelo.predict(df)
        print(f'listo: {df_predicho.__hash__=}')


if __name__ == '__main__':
    # unittest.main()
    TestsCase.debugTestCase()
