import logging
from enum import Enum

import pyinputplus as pyip
from dotenv import find_dotenv, load_dotenv

import src.data.make_dataset as make_dataset
import src.data.preprocessing as preprocessing
import src.features.build_features as build_features
import src.models.evaluation as evaluation
import src.models.predict_model as predict_model
import src.models.train_model as train_model


class Opciones(Enum):
    preict_model = 'Predict model'
    preprocessing = 'Preprocessing'
    makedataset = 'MakeDataSet'
    build_features = 'Build features'
    train_model = 'Train model'
    evaluation = 'Evaluation'
    salir = 'Salir'


def main():
    opciones = [str(opcion.value) for opcion in Opciones]

    tarea = None
    while tarea != Opciones.salir.value:
        tarea = pyip.inputMenu(opciones, numbered=True, prompt='Ingrese una opción: \n')
        if tarea == Opciones.preprocessing.value:
            preprocessing.main()
        elif tarea == Opciones.makedataset.value:
            make_dataset.main()
        elif tarea == Opciones.build_features.value:
            build_features.main()
        elif tarea == Opciones.train_model.value:
            train_model.main()
        elif tarea == Opciones.evaluation.value:
            evaluation.main()
        elif tarea == Opciones.preict_model.value:
            predict_model.main()
        if tarea != Opciones.salir.value:
            input('Presione enter para continuar.')
            print()
            print()
            print()


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
