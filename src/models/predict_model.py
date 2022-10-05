import logging
from pathlib import Path

import click

from src.core.steps import Steps


def main(
        zipcode,
        grade,
        view,
        bathrooms,
        bedrooms,
        sqft_living15,
        waterfront,
        floors,
        sqft_lot,
        condition,
        sqft_lot15,
        sqft_living,
        fue_renovada,
        antiguedad_venta,
        steps: Steps = None
):
    logger = logging.getLogger(__name__)
    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]
    if steps is None:
        steps = Steps.build(str(project_dir), logger)

    steps.predict_model_one(
        zipcode=zipcode,
        grade=grade,
        view=view,
        bathrooms=bathrooms,
        bedrooms=bedrooms,
        sqft_living15=sqft_living15,
        waterfront=waterfront,
        floors=floors,
        sqft_lot=sqft_lot,
        condition=condition,
        sqft_lot15=sqft_lot15,
        sqft_living=sqft_living,
        fue_renovada=fue_renovada,
        antiguedad_venta=antiguedad_venta
    )


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
    kwargs = {}
    for parametro in parametros:
        kwargs[parametro] = input(f'Ingrese {parametro}')
    main(**kwargs)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main_terminal()
