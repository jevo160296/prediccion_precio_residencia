import functools
import warnings

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
