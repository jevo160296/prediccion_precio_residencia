import functools
import warnings

from src.core.clases import FloatValidations, IntValidations, BoolValidations

columnas_raw = ['zipcode', 'index', 'grade', 'sqft_basement', 'view', 'bathrooms', 'bedrooms', 'sqft_above',
                'sqft_living15', 'lat', 'waterfront', 'floors', 'date', 'yr_renovated', 'yr_built', 'long',
                'jhygtf', 'sqft_lot', 'price', 'condition', 'wertyj', 'sqft_lot15', 'sqft_living'
                ]

columnas_entrada = ['zipcode', 'grade', 'view', 'bathrooms', 'bedrooms', 'sqft_living15', 'waterfront', 'floors',
                    'sqft_lot', 'condition', 'sqft_lot15', 'sqft_living', 'fue_renovada', 'antiguedad_venta']


# Validaciones de datos
def prompt(msg, minimo, maximo):
    return f'{msg} ({minimo} - {maximo}) '


float_validations: FloatValidations = {
    'sqft_lot15': {'min': 700, 'max': 58000,
                   'prompt': prompt('sqft_lot15: Pies cuadrados del lote en el 2015 ', 700, 57000)},
    'sqft_living15': {'min': 500, 'max': 6000, 'prompt': prompt('sqft_living15: Pies cuadrados del área '
                                                                'habitable en el 2015 ', 500, 6000)},
    'sqft_lot': {'min': 600, 'max': 140000, 'prompt': prompt('sqft_lot: Pies cuadrados del lote en la '
                                                             'actualidad ', 600, 130000)},
    'sqft_living': {'min': 300, 'max': 12000, 'prompt': prompt('sqft_living: Pies cuadrados del área habitable'
                                                               ' en la actualidad ', 300, 11000)},
    'bathrooms': {'min': 1, 'max': 4, 'prompt': prompt('bathrooms: Cantidad de baños ', 1, 4)},
    'bedrooms': {'min': 1, 'max': 5, 'prompt': prompt('bedrooms: Cantidad de habitaciones ', 1, 5)},
    'floors': {'min': 1, 'max': 3, 'prompt': prompt('floors: Cantidad de pisos construidos ', 1, 3)}
}
int_validations: IntValidations = {
    'grade': {'min': 1, 'max': 13, 'prompt': prompt('Grade ', 1, 13)},
    'view': {'min': 0, 'max': 4, 'prompt': prompt('View ', 0, 4)},
    'condition': {'min': 1, 'max': 5, 'prompt': prompt('Condition', 1, 5)},
    # 'yr_renovated': {'min': 1900, 'max': 2022},
    'yr_built': {'min': 1900, 'max': 2022, 'prompt': prompt('Año de construcción ', 1900, 2022)}
}
bool_validations: BoolValidations = {
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


def deprecated(func):
    @functools.wraps(func)
    def new_func(*args, **kwargs):
        warnings.simplefilter('always', DeprecationWarning)  # turn off filter
        warnings.warn("Call to deprecated function {}.".format(func.__name__),
                      category=DeprecationWarning,
                      stacklevel=2)
        warnings.simplefilter('default', DeprecationWarning)  # reset filter
        return func(*args, **kwargs)

    return new_func
