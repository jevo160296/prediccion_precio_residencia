import logging
from pathlib import Path

import click
from dotenv import find_dotenv, load_dotenv

import src.features.build_features as build_features
import src.data.make_dataset as make_dataset
import src.data.preprocessing as preprocessing
import src.models.evaluation as evaluation
import src.models.train_model as train_model
import src.models.predict_model as predict_model


@click.command()
@click.argument('tarea',
                type=click.types.Choice(
                    ['preprocessing', 'makedataset', 'build_features', 'train_model', 'evaluation', 'predict_model']
                ))
def main(tarea):
    def get_data_folder_path():
        return input('Ingrese la ruta a la carpeta data: ')

    def get_input_filename():
        return input('Ingrese el nombre del archivo de entrada: ')

    def get_porcentaje_entrenamiento():
        return float(
            input('Ingrese el porcentaje de entrenamiento para dividir conjunto de datos: ')
        )

    def get_set_train_validation():
        return {'s': 'Entrenamiento', 'n': 'Validacion'}[
            input('¿Desea realizar la ejecución para el set de entrenamiento? s/n: ').lower().strip()
        ]

    if tarea == 'preprocessing':
        preprocessing.main(get_data_folder_path(), get_input_filename())
    elif tarea == 'makedataset':
        make_dataset.main(get_data_folder_path(), get_input_filename(), get_porcentaje_entrenamiento())
    elif tarea == 'build_features':
        build_features.main(get_data_folder_path(), get_input_filename(), get_porcentaje_entrenamiento(),
                            get_set_train_validation())
    elif tarea == 'train_model':
        train_model.main(get_data_folder_path(), get_input_filename(), get_porcentaje_entrenamiento())
    elif tarea == 'evaluation':
        evaluation.main(get_data_folder_path(), get_input_filename(), get_porcentaje_entrenamiento())
    elif tarea == 'predict_model':
        predict_model.main(get_data_folder_path(), get_input_filename())


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
