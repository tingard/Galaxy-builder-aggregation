import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import lib.galaxy_utilities as gu
from scipy.optimize import least_squares
from sklearn.compose import TransformedTargetRegressor
from gzbuilderspirals import metric
from gzbuilderspirals.oo import Pipeline
import warnings
from astropy.utils.exceptions import AstropyWarning
warnings.simplefilter('ignore', category=AstropyWarning)


x = np.linspace(0.1, 1, 100)

y = 5 * np.tanh(10 * x)

# clf = TransformedTargetRegressor(func=np.arctanh, inverse_func=np.tanh, check_inverse=True)
def f(p):
    return p[0]*np.tanh(p[1] * x) - y

res = least_squares(f, (1, 10))
print(res['x'])
