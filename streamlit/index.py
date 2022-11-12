import logging
from pathlib import Path

import streamlit as st

from src.core.steps import Steps
from src.core.variables_globales import zip_codes, bool_validations, float_validations, int_validations
from src.core.clases import IntValidacion, BoolValidation

# Inicialización de las variables
project_dir = Path(__file__).resolve().parents[1]
logger = logging.getLogger(__name__)
steps = Steps.build(str(project_dir), logger)


def slider(validacion: IntValidacion):
    return st.slider(validacion['prompt'], validacion['min'], validacion['max'])


def number_input(validacion: IntValidacion):
    return st.number_input(validacion['prompt'], validacion['min'], validacion['max'])


def checkbox(validacion: BoolValidation):
    return st.checkbox(validacion['prompt'])


st.write("""
# Predicción precio residencia
""")

col1, col2 = st.columns(2)
with col1:
    waterfront = checkbox(bool_validations['waterfront'])
with col2:
    yr_renovated = 1900 if checkbox(bool_validations['fue_renovado']) else 0
zipcode = st.selectbox('Zip code', zip_codes)
sqft_living = number_input(float_validations['sqft_living'])
sqft_lot = number_input(float_validations['sqft_lot'])
sqft_lot15 = number_input(float_validations['sqft_lot15'])
sqft_living15 = number_input(float_validations['sqft_living15'])
grade = slider(int_validations['grade'])
view = slider(int_validations['view'])
bathrooms = slider(float_validations['bathrooms'])
bedrooms = slider(float_validations['bedrooms'])
floors = slider(float_validations['floors'])
condition = slider(int_validations['condition'])

year_options = list(range(int_validations['yr_built']['min'], int_validations['yr_built']['max']))
yr_built = st.selectbox(int_validations['yr_built']['prompt'], year_options, index=year_options.index(2000))

if st.button('Calcular predicción', type='primary'):
    prediccion = steps.predict_model_one(
        zipcode=zipcode,
        grade=grade,
        view=view,
        bathrooms=bathrooms,
        bedrooms=bedrooms,
        sqft_living=sqft_living,
        waterfront=waterfront,
        floors=floors,
        sqft_lot=sqft_lot,
        condition=condition,
        sqft_lot15=sqft_lot15,
        sqft_living15=sqft_living15,
        yr_renovated=yr_renovated,
        yr_built=yr_built
    )
    st.write(f'Se estima un valor de la casa de: **${prediccion[0]:,.0f}**')
