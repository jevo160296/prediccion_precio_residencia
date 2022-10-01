import logging
from pathlib import Path

import click

from src.core.steps import Steps


def main(
        bedrooms,
        bathrooms,
        sqft_living,
        sqft_lot,
        floors,
        waterfront,
        view,
        grade,
        sqft_above,
        lat,
        sqft_living15,
        steps: Steps = None
):
    logger = logging.getLogger(__name__)

    if steps is None:
        steps = Steps.build(logger)

    steps.predict_model_one(
        bedrooms=bedrooms,
        bathrooms=bathrooms,
        sqft_living=sqft_living,
        sqft_lot=sqft_lot,
        floors=floors,
        waterfront=waterfront,
        view=view,
        grade=grade,
        sqft_above=sqft_above,
        lat=lat,
        sqft_living15=sqft_living15
    )


@click.command()
def main_terminal():
    parametros = ['bedrooms',
                  'bathrooms',
                  'sqft_living',
                  'sqft_lot',
                  'floors',
                  'waterfront',
                  'view',
                  'grade',
                  'sqft_above',
                  'lat',
                  'sqft_living15']
    kwargs = {}
    for parametro in parametros:
        kwargs[parametro] = input(f'Ingrese {parametro}')
    main(**kwargs)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    project_dir = Path(__file__).resolve().parents[2]
    main_terminal()
