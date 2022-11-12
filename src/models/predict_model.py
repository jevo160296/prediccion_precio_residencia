import logging
from pathlib import Path

import click
import pyinputplus as pyip
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted
from src.core.variables_globales import zip_codes, bool_validations, float_validations, int_validations

import src.models.train_model as train_model
from src.core.steps import Steps


def get_prediction_values() -> dict:
    cur_zipcode = '98'
    while cur_zipcode not in zip_codes:
        print(f'Zipcode: {cur_zipcode}_')
        indices_disponibles = sorted(list(set(map(lambda x: x[len(cur_zipcode)],
                                                  filter(lambda zipcode: cur_zipcode == zipcode[:len(cur_zipcode)],
                                                         zip_codes)))))
        cur_zipcode += pyip.inputChoice(indices_disponibles, blank=True)
    print(f'Zipcode: {cur_zipcode}')
    resultados = {'zipcode': int(cur_zipcode)}
    # resultados = {'zipcode': pyip.inputChoice(**list_validations['zipcode'])}
    for parametro, argumentos in bool_validations.items():
        resultados[parametro] = {'si': 1, 'no': 0}[pyip.inputYesNo(yesVal='si', noVal='no', **argumentos)]
    resultados['yr_renovated'] = 1900 if resultados['fue_renovado'] == 1 else 0
    resultados.pop('fue_renovado')
    for parametro, argumentos in float_validations.items():
        resultados[parametro] = pyip.inputFloat(**argumentos)
    for parametro, argumentos in int_validations.items():
        resultados[parametro] = pyip.inputInt(**argumentos)

    return resultados


def main(steps: Steps = None
         ):
    logger = logging.getLogger(__name__)
    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]
    if steps is None:
        steps = Steps.build(str(project_dir), logger)

    # Validando que el modelo esté entrenado
    try:
        check_is_fitted(steps.modelo)
        check_is_fitted(steps.processing)
    except NotFittedError:
        logger.info('No se ha realizado el entrenamiento del modelo.')
        reentrenar = \
            pyip.inputYesNo('¿Desea realizar el entrenamiento del modelo? si/no ', yesVal='si', noVal='no') == 'si'
        if reentrenar:
            train_model.main(steps)
        else:
            logger.error('Primero se debe ejecutar el entrenamiento del modelo.')
            return

    df_predicted = steps.predict_model_one(**get_prediction_values())
    logger.info('-----------------------------------------------------')
    logger.info(f'Precio predicho: {df_predicted[0]}')
    logger.info('-----------------------------------------------------')


@click.command()
def main_terminal():
    parametros = ['zipcode',
                  'grade',
                  'view',
                  'bathrooms',
                  'bedrooms',
                  'sqft_living15',
                  'waterfront',
                  'floors',
                  'sqft_lot',
                  'condition',
                  'sqft_lot15',
                  'sqft_living',
                  'fue_renovada',
                  'antiguedad_venta']

    main()


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main_terminal()
