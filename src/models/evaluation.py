from src.jutils.data import DataUtils
from pathlib import Path
from src.models.modelo import Modelo
import pandas as pd
import click
import src.features.build_features as build_features
import src.models.train_model as train_model
from sklearn.metrics import r2_score


def main(data_folder_path, input_filename, porcentaje_entrenamiento):
    input_filename_stem = input_filename.split('.')[0]
    input_filename = input_filename_stem + '.parquet'
    du = DataUtils(
        data_folder_path=data_folder_path,
        input_file_name=input_filename,
        y_name='price',
        load_data=lambda path: pd.read_parquet(path),
        save_data=lambda _df, path: _df.to_parquet(path)
    )
    if not (du.transformed_train_test_path.exists() and du.transformed_validation_path.exists()):
        build_features.main(data_folder_path, input_filename, porcentaje_entrenamiento, "Entrenamiento")
        build_features.main(data_folder_path, input_filename, porcentaje_entrenamiento, "Validacion")
    if not du.model_path.exists():
        train_model.main(data_folder_path, input_filename, porcentaje_entrenamiento)
    df_entrenamiento = du.load_data(du.transformed_train_test_path)
    df_validacion = du.load_data(du.transformed_validation_path)
    modelo: Modelo = du.model

    print(f'Train shape: {df_entrenamiento.shape}')
    print(f'Validation shape: {df_validacion.shape}')

    y_train_predict = modelo.predict(df_entrenamiento)
    y_validation_predict = modelo.predict(df_validacion)

    train_score = r2_score(df_entrenamiento['price'], y_train_predict)
    print(f'{train_score=}')
    with du.processed_path.joinpath('train_score.txt').open('w') as f:
        f.write(f'Train R2 score: {train_score}')

    validation_score = r2_score(df_validacion['price'], y_validation_predict)
    print(f'{validation_score=}')
    with du.processed_path.joinpath('validation.txt').open('w') as f:
        f.write(f'Test R2 score: {validation_score}')


# noinspection DuplicatedCode
@click.command()
@click.argument('data_folder_path', type=click.types.Path(file_okay=False))
@click.argument('input_filename', type=click.types.STRING)
@click.argument('porcentaje_entrenamiento', type=click.types.FLOAT)
def main_terminal(data_folder_path, input_filename, porcentaje_entrenamiento):
    main(data_folder_path, input_filename, porcentaje_entrenamiento)


if __name__ == '__main__':
    main_terminal()
