import logging
from pathlib import Path

import click
from dotenv import find_dotenv, load_dotenv
import requests

import src.features.build_features as build_features
import src.data.make_dataset as make_dataset
import src.data.preprocessing as preprocessing
import src.models.evaluation as evaluation
import src.models.train_model as train_model
import src.models.predict_model as predict_model

from enum import Enum


class Opcion:
    def __init__(self, nombre):
        self._nombre = nombre

    @property
    def nombre(self):
        return self._nombre

    def __repr__(self):
        return self._nombre


class Opciones(Enum):
    preprocessing = Opcion('Preprocessing')
    makedataset = Opcion('MakeDataSet')
    build_features_training = Opcion('Build features training')
    build_features_validation = Opcion('Build features validation')
    train_model = Opcion('Train model')
    evaluation = Opcion('Evaluation')
    preict_model = Opcion('Predict model')

    def __repr__(self):
        return str(self.value)


def menu() -> Opcion:
    for index, opcion in enumerate(Opciones):
        print(f'{index + 1}. {opcion.value.nombre}')
    opcion = int(input('Ingrese una opción: '))
    print(f'Opcion seleccionada {list(Opciones)[opcion - 1]}')
    return list(Opciones)[opcion - 1]


def main():
    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    source_file = 'https://drive.google.com/file/d/1V2TlWi81sFWKBlEuwI5Tl48uYKnK8znh/view?usp=sharing'

    tarea = menu()

    def get_data_folder_path():
        return project_dir.joinpath('data')

    def get_input_filename():
        return 'kc_house_data_'

    def get_porcentaje_entrenamiento():
        return 0.8

    def get_file():
        return requests.get(source_file)

    if tarea == Opciones.preprocessing:
        preprocessing.main(get_data_folder_path(), get_input_filename())
    elif tarea == Opciones.makedataset:
        make_dataset.main(get_data_folder_path(), get_input_filename(), get_porcentaje_entrenamiento())
    elif tarea == Opciones.build_features_training:
        build_features.main(get_data_folder_path(), get_input_filename(), get_porcentaje_entrenamiento(),
                            'Entrenamiento')
    elif tarea == Opciones.build_features_validation:
        build_features.main(get_data_folder_path(), get_input_filename(), get_porcentaje_entrenamiento(),
                            'Validacion')
    elif tarea == Opciones.train_model:
        train_model.main(get_data_folder_path(), get_input_filename(), get_porcentaje_entrenamiento())
    elif tarea == Opciones.evaluation:
        evaluation.main(get_data_folder_path(), get_input_filename(), get_porcentaje_entrenamiento())
    elif tarea == Opciones.preict_model:
        predict_model.main(get_data_folder_path(), get_input_filename())


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
