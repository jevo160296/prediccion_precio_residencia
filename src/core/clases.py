from typing import TypedDict


class IntValidacion(TypedDict):
    min: int
    max: int
    prompt: str


class BoolValidation(TypedDict):
    prompt: str


class FloatValidations(TypedDict):
    sqft_lot15: IntValidacion
    sqft_living15: IntValidacion
    sqft_lot: IntValidacion
    sqft_living: IntValidacion
    bathrooms: IntValidacion
    bedrooms: IntValidacion
    floors: IntValidacion


class IntValidations(TypedDict):
    grade: IntValidacion
    view: IntValidacion
    condition: IntValidacion
    yr_built: IntValidacion


class BoolValidations(TypedDict):
    waterfront: BoolValidation
    fue_renovado: BoolValidation
