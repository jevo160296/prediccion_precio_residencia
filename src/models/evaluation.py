from src.jutils.data import DataUtils
from pathlib import Path
from src.models.modelo import Modelo
import logging
import pandas as pd
import click
import src.features.build_features as build_features
import src.models.train_model as train_model
from sklearn.metrics import r2_score


def main(porcentaje_entrenamiento=0.7):
    if not (du.transformed_train_test_path.exists() and du.transformed_validation_path.exists()):
        build_features.main(data_folder_path, input_filename, porcentaje_entrenamiento, "Entrenamiento")
        build_features.main(data_folder_path, input_filename, porcentaje_entrenamiento, "Validacion")
    if not du.model_path.exists():
        train_model.main(data_folder_path, input_filename, porcentaje_entrenamiento)
    df_entrenamiento = du.load_data(du.transformed_train_test_path)
    df_validacion = du.load_data(du.transformed_validation_path)
    modelo: Modelo = du.model

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
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main_terminal()
