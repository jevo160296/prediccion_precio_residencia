import logging
from pathlib import Path

import click

from src.core.steps import Steps
import pyinputplus as pyip
import re


def main(steps: Steps = None
         ):
    logger = logging.getLogger(__name__)
    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]
    if steps is None:
        steps = Steps.build(str(project_dir), logger)

    def get_prediction_values() -> dict:
        def prompt(msg, minimo, maximo):
            return f'{msg} ({minimo} - {maximo}) '

        float_validations = {
            'sqft_lot15': {'min': 659, 'max': 57140,
                           'prompt': prompt('sqft_lot15 (Pies cuadrados del lote en el 2015) ', 659, 57140)},
            'sqft_living15': {'min': 460, 'max': 5790, 'prompt': prompt('sqft_living15 (Pies cuadrados del área '
                                                                        'habitable en el 2015) ', 460, 5790)},
            'sqft_lot': {'min': 520, 'max': 137214, 'prompt': prompt('sqft_lot (Pies cuadrados del lote en la '
                                                                     'actualidad) ', 520, 137214)},
            'sqft_living': {'min': 290, 'max': 12050, 'prompt': prompt('sqft_living (Pies cuadrados del área habitable'
                                                                       ' en la actualidad) ', 290, 12050)},
            'bathrooms': {'min': 1, 'max': 4, 'prompt': prompt('bathrooms (Cantidad de baños) ', 1, 4)},
            'bedrooms': {'min': 1, 'max': 5, 'prompt': prompt('bedrooms (Cantidad de habitaciones) ', 1, 5)},
            'floors': {'min': 1, 'max': 3, 'prompt': prompt('floors (Cantidad de pisos construidos) ', 1, 3)}
        }
        int_validations = {
            'grade': {'min': 1, 'max': 13, 'prompt': prompt('Grade ', 1, 13)},
            'view': {'min': 0, 'max': 4, 'prompt': prompt('View ', 0, 4)},
            'condition': {'min': 1, 'max': 5, 'prompt': prompt('Condition', 1, 5)},
            # 'yr_renovated': {'min': 1900, 'max': 2022},
            'yr_built': {'min': 1900, 'max': 2022, 'prompt': prompt('Año de construcción ', 1900, 2022)}
        }
        bool_validations = {
            'waterfront': {'prompt': '¿Tiene vista al mar, laguna o río? si/no '},
            'fue_renovado': {'prompt': '¿Ha sido renovado alguna vez? si/no '}
        }

        zip_codes = {'98001', '98002', '98003', '98004', '98005', '98006', '98007', '98008', '98010', '98011', '98014',
                     '98019', '98022', '98023', '98024', '98027', '98028', '98029', '98030', '98031', '98032', '98033',
                     '98034', '98038', '98039', '98040', '98042', '98045', '98052', '98053', '98055', '98056', '98058',
                     '98059', '98065', '98070', '98072', '98074', '98075', '98077', '98092', '98102', '98103', '98105',
                     '98106', '98107', '98108', '98109', '98112', '98115', '98116', '98117', '98118', '98119', '98122',
                     '98125', '98126', '98133', '98136', '98144', '98146', '98148', '98155', '98166', '98168', '98177',
                     '98178', '98188', '98198', '98199'}

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
