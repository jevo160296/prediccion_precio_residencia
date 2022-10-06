from pandas import DataFrame
from sklearn.preprocessing import PowerTransformer, KBinsDiscretizer


def eliminacion_datos_faltantes(df: DataFrame) -> DataFrame:
    df = df.copy()
    columnas_datosFaltantes = set(df.columns).difference(['date', 'sqft_basement', 'yr_renovated'])
    df = df.dropna(subset=columnas_datosFaltantes)
    return df


def numericas_a_binarias(df: DataFrame) -> DataFrame:
    # Se considerará que un apartamento tiene sótano si sqft_basement está nulo.
    df = df.copy()
    df['tiene_sotano'] = df['sqft_basement'].notna()
    # Reemplazando true por 1 y false por 0
    df['tiene_sotano'] = df['tiene_sotano'].astype('int').astype('category')
    df.drop(columns='sqft_basement', inplace=True)
    return df


def transformacion_logaritmica(df: DataFrame, pt: PowerTransformer) -> DataFrame:
    # Se transformarán las siguientes variables: sqft_above, sqft_living15, sqft_lot, price, sqft_lot15, sqft_living
    df = df.copy()
    columnas = ['sqft_above', 'sqft_living15', 'sqft_lot', 'sqft_lot15', 'sqft_living']
    df[columnas] = pt.transform(df[columnas])
    return df


def entrenar_logaritmica(df: DataFrame) -> PowerTransformer:
    # Se transformarán las siguientes variables: sqft_above, sqft_living15, sqft_lot, price, sqft_lot15, sqft_living
    df = df.copy()
    columnas = ['sqft_above', 'sqft_living15', 'sqft_lot', 'sqft_lot15', 'sqft_living']
    pt = PowerTransformer(method='box-cox')
    pt.fit(df[columnas])
    return pt


def numericas_a_categoricas(df: DataFrame, kbd: KBinsDiscretizer, columnas_a_categoricas) -> DataFrame:
    # Después de realizar la transformación de las variables, se encuentra que sqft_lot y sqft_lot15 siguen teniendo
    # una alta kurtosis, por lo tanto se procederá a convertirlas en variables categóricas.
    df = df.copy()
    df[columnas_a_categoricas] = kbd.transform(df[columnas_a_categoricas])
    df[columnas_a_categoricas] = df[columnas_a_categoricas].astype('category')
    return df


def entrenar_numericas_a_categoricas(df: DataFrame, columnas_a_categoricas) -> KBinsDiscretizer:
    kbd = KBinsDiscretizer(strategy='kmeans', encode='ordinal')
    kbd.fit(df[columnas_a_categoricas])
    return kbd


def transformacion_logaritmica_y(df: DataFrame, pty: PowerTransformer) -> DataFrame:
    # Se transformarán las siguientes variables: sqft_above, sqft_living15, sqft_lot, price, sqft_lot15, sqft_living
    df = df.copy()
    df[['price']] = pty.transform(df[['price']])
    return df


def entrenar_logaritmica_y(df: DataFrame) -> PowerTransformer:
    # Se transformarán las siguientes variables: sqft_above, sqft_living15, sqft_lot, price, sqft_lot15, sqft_living
    df = df.copy()
    pty = PowerTransformer(method='box-cox')
    pty.fit(df[['price']])
    return pty
