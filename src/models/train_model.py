from modelo import Modelo
from pandas import DataFrame
import pandas as pd
from src.data.make_dataset import make_dataset
from src.jutils.data import DataUtils
from pathlib import Path
import click
from sklearn.model_selection import train_test_split


def entrenar(df: DataFrame) -> Modelo:
    modelo = Modelo()
    df = make_dataset(df)
    modelo.fit(df, df['price'])
    return modelo


@click.command()
@click.argument('input_filename', type=click.types.STRING)
@click.argument('porcentaje_entrenamiento', type=click.types.FLOAT)
def main(input_filename, porcentaje_entrenamiento):
    du = DataUtils(
        data_folder_path=Path('../../data').resolve().absolute(),
        input_file_name=input_filename,
        y_name='price',
        load_data=lambda path: pd.read_parquet(path),
        save_data=lambda _df, path: _df.to_parquet(path)
    )
    df = pd.read_csv(du.input_file_path, index_col=0, sep=',')
    df_train_test, df_validation = train_test_split(df, train_size=porcentaje_entrenamiento, random_state=1)
    modelo = entrenar(df_train_test)
    du.save_data(df_train_test, du.raw_train_test_path)
    du.save_data(df_validation, du.raw_validation_path)
    du.model = modelo


if __name__ == '__main__':
    main()
