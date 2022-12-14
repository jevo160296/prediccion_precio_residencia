import unittest
from pathlib import Path

from pandas import DataFrame

from src.core.variables_globales import columnas_entrada
from src.core.steps import Steps
from src.data.funciones_base import (
    eliminar_duplicados, convertir_col_date_a_date, reemplazar_valores_extremos, reemplazar_nulos_por_la_media,
    reemplazar_fechas_nulas, reemplazar_ceros_por_nulos, convertir_tipos, calculo_variables_adicionales,
    seleccionar_columnas
)
from src.data.procesamiento_datos import LimpiezaCalidad, ProcesamientoDatos


class TestsCase(unittest.TestCase):
    @classmethod
    def debugTestCase(cls):
        loader = unittest.defaultTestLoader
        testSuit = loader.loadTestsFromTestCase(cls)
        testSuit.debug()

    def setUp(self) -> None:
        data_folder_path = Path('../data').resolve().absolute()
        self._steps = Steps.build(str(data_folder_path))
        self._df = self._steps.raw_df
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
            .pipe(self.print_shape, 'Después de eliminar índices duplicados') \
            .pipe(calculo_variables_adicionales) \
            .pipe(seleccionar_columnas, columnas=columnas_entrada + ['price', 'index'])
        return df

    def test1_limpieza_inicial(self):
        print('-'*100)
        df = self.run_limpieza_inicial()
        self.assertEqual(df['index'].duplicated().sum(), 0)
        print('Listo')
        print('-' * 100)

    def test2_limpieza_inicial_pipeline(self):
        print('-' * 100)
        df_pd_pipe = self.run_limpieza_inicial().drop(columns='index')
        li = LimpiezaCalidad(self._columnas_numericas)
        df_sk_pipe = li.transform(self._df)
        self.assertTrue(df_pd_pipe.shape == df_pd_pipe.shape)
        self.assertTrue(all(df_pd_pipe == df_sk_pipe.reindex(columns=df_pd_pipe.columns)))
        print('Listo')
        print('-' * 100)

    def test3_procesamiento_datos(self):
        print('-' * 100)
        li = LimpiezaCalidad(self._columnas_numericas)
        pda = ProcesamientoDatos()
        df = li.transform(self._df)
        df = pda.fit_transform(df)
        print('Listo')
        print('-' * 100)


if __name__ == '__main__':
    unittest.main()
    # TestsCase.debugTestCase()
