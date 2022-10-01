import logging

import click
from pandas import DataFrame

from src.core.steps import Steps
from src.models.modelo import Modelo


def entrenar(df: DataFrame) -> Modelo:
    modelo = Modelo()
    modelo.fit(df, df['price'])
    return modelo


def main(steps: Steps = None, porcentaje_entrenamiento=0.7):
    logger = logging.getLogger(__name__)
    if steps is None:
        steps = Steps.build(logger)

    steps.du.model = steps.training(porcentaje_entrenamiento)
    logger.info(f'Modelo exitósamente entrenado, almacenado en {steps.du.model_path}')
    return steps.du.model


@click.command()
@click.option('-p', '--porcentaje-entrenamiento', type=float, required=False, default=0.7)
def main_terminal(porcentaje_entrenamiento):
    main(porcentaje_entrenamiento=porcentaje_entrenamiento)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main_terminal()
