def validar_duplicados(df):
    filas = df.shape[0]
    cant_duplicados = df.duplicated().sum()
    print(
        f'De {filas} registros hay {cant_duplicados} filas duplicadas, representando el {cant_duplicados / filas:.2%}')


def eliminar_duplicados(df):
    # Eliminando duplicados
    df = df.drop_duplicates(keep='first')
    filas = df.shape[0]
    print(f'Después de la eliminación de duplicados, el conjunto de datos queda con {filas} filas.')
    return df


def validar_index_duplicados(df):
    # Validando duplicados de index
    son_duplicados = df['index'].duplicated()
    cant_duplicados = son_duplicados.sum()
    filas = df.shape[0]
    print(
        f'De {filas} registros, hay {cant_duplicados} registros con index duplicado, que representan el '
        f'{cant_duplicados / filas:.2%}.')
    return son_duplicados
