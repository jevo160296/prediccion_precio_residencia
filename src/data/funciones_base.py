def validar_duplicados(df):
    filas = df.shape[0]
    cant_duplicados = df.duplicated().sum()
    print(
        f'De {filas} registros hay {cant_duplicados} filas duplicadas, representando el {cant_duplicados / filas:.2%}')


def eliminar_duplicados(_df):
    # Eliminando duplicados
    df = _df.drop_duplicates(keep='first')
    filas = _df.shape[0]
    print(f'Después de la eliminación de duplicados, el conjunto de datos queda con {filas} filas.')
    return df


def validar_index_duplicados(_df):
    # Validando duplicados de index
    _son_duplicados = _df['index'].duplicated()
    _cant_duplicados = _son_duplicados.sum()
    _filas = _df.shape[0]
    print(
        f'De {_filas} registros, hay {_cant_duplicados} registros con index duplicado, que representan el '
        f'{_cant_duplicados / _filas:.2%}.')
    return _son_duplicados
