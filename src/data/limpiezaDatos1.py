

import pandas as pd
import numpy as np
import logging
from pathlib import Path



def main(input_filepath, output_filepath):
    """ Runs data feature engineering scripts to turn interim data from (../interim) into
        cleaned data ready for machine learning (saved in ../processed).
    """

def eliminacionOutliers(data:pd.DataFrame)->pd.DataFrame:
    data = data[data['price'] <= 1130000]
    data = data[data['sqft_lot'] <= 350000]
    data = data[data['bedrooms'] <= 5]
    data = data[data['bedrooms'] > 0]
    data = data[data['bathrooms'] <= 4]
    data = data[data['bathrooms'] >= 1]

    return data



def conversionTipoDatos(data:pd.DataFrame)->pd.DataFrame:
    variables_categoricas = ['grade', 'view', 'waterfront', 'condition', 'zipcode']
    data[variables_categoricas] = data[variables_categoricas].astype('category')

    return data

def calculoVariablesAdicionales(data: pd.DataFrame)->pd.DataFrame:
    data['yr_date'] = data['date'].dt.year
    data['antiguedad_venta'] = data['yr_date'] - data['yr_built']
    data.drop(columns=['yr_date', 'date', 'yr_built'], inplace=True)

    return data


def eliminacionColumnas(data: pd.DataFrame)->pd.DataFrame:
    data.drop(columns=['lat', 'yr_renovated', 'long', 'jhygtf'], inplace=True)

    return data










