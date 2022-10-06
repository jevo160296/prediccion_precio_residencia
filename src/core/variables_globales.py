import functools
import warnings

columnas_raw = ['zipcode', 'index', 'grade', 'sqft_basement', 'view', 'bathrooms', 'bedrooms', 'sqft_above',
                'sqft_living15', 'lat', 'waterfront', 'floors', 'date', 'yr_renovated', 'yr_built', 'long',
                'jhygtf', 'sqft_lot', 'price', 'condition', 'wertyj', 'sqft_lot15', 'sqft_living'
                ]

columnas_entrada = ['zipcode', 'grade', 'view', 'bathrooms', 'bedrooms', 'sqft_living15', 'waterfront', 'floors',
                    'sqft_lot', 'condition', 'sqft_lot15', 'sqft_living', 'fue_renovada', 'antiguedad_venta']


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
