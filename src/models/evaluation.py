import logging

import click

from src.core.steps import Steps


def main(steps: Steps = None, porcentaje_entrenamiento=0.7):
    logger = logging.getLogger(__name__)
    if steps is None:
        steps = Steps.build(logger)

    score_train_test, score_validation = steps.evaluation(porcentaje_entrenamiento)

    print(f'{score_train_test=}')
    with steps.du.processed_path.joinpath('train_score.txt').open('w') as f:
        f.write(f'Train R2 score: {score_train_test}')

    print(f'{score_validation=}')
    with steps.du.processed_path.joinpath('validation.txt').open('w') as f:
        f.write(f'Test R2 score: {score_validation}')


# noinspection DuplicatedCode
@click.command()
def main_terminal():
    main(porcentaje_entrenamiento=0.7)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main_terminal()
