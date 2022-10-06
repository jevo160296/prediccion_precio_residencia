import logging
from pathlib import Path

import click
import joblib
from pandas import DataFrame

from src.core.variables_globales import deprecated
from src.core.steps import Steps
from src.models.modelo import Modelo


@deprecated
def entrenar(df: DataFrame) -> Modelo:
    modelo = Modelo()
    modelo.fit(df, df['price'])
    return modelo


def main(steps: Steps = None, porcentaje_entrenamiento=0.7):
    logger = logging.getLogger(__name__)
    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]
    if steps is None:
        steps = Steps.build(str(project_dir), logger)

    steps.du.model, pda = steps.training(porcentaje_entrenamiento)
    # Guardando preprocesamiento
    joblib.dump(pda, steps.du.model_path.with_stem('pda'))
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
