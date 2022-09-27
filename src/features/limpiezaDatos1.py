import pandas as pd


def eliminacionOutliers(data: pd.DataFrame) -> pd.DataFrame:
    data = data[data['price'] <= 1130000]
    data = data[data['sqft_lot'] <= 350000]
    data = data[data['bedrooms'] <= 5]
    data = data[data['bedrooms'] > 0]
    data = data[data['bathrooms'] <= 4]
    data = data[data['bathrooms'] >= 1]

    return data


def conversionTipoDatos(data: pd.DataFrame) -> pd.DataFrame:
    variables_categoricas = ['grade', 'view', 'waterfront', 'condition', 'zipcode']
    data[variables_categoricas] = data[variables_categoricas].astype('category')

    return data


def calculoVariablesAdicionales(data: pd.DataFrame) -> pd.DataFrame:
    data['yr_date'] = data['date'].dt.year
    data['antiguedad_venta'] = data['yr_date'] - data['yr_built']
    data.drop(columns=['yr_date', 'date', 'yr_built'], inplace=True)

    return data


def eliminacionColumnas(data: pd.DataFrame) -> pd.DataFrame:
    data.drop(columns=['lat', 'yr_renovated', 'long', 'jhygtf'], inplace=True)

    return data
