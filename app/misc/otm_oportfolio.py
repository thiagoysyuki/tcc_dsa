
from scipy.stats import norm, skew, kurtosis
import pandas as pd
import numpy as np

def semideviation(r):
    if isinstance(r, pd.DataFrame):
        return r.aggregate(semideviation)
    elif isinstance(r, pd.Series):
        excess = r - r.mean()
        excess_negative = excess[excess < 0]
        excess_negative_squared = excess_negative ** 2
        n_negative = (excess < 0).sum()
        return (excess_negative_squared.sum() / n_negative) ** 0.5
    else:
        raise TypeError("Input must be a pandas Series or DataFrame.")

def var_historic(r, level=0.05):

    if isinstance(r, pd.DataFrame):
        return r.aggregate(var_historic, level=level)
    elif isinstance(r, pd.Series):
        return np.percentile(r, level)
    else:
        raise TypeError("Input must be a pandas Series or DataFrame.")
    
def var_gaussian(r, level=0.05, modified=False):
    
    z = norm.ppf(level)

    if modified:
        S = skew(r)
        K = kurtosis(r)
        z = (z + (z**2 - 1) * S / 6 + (z**3 - 3 * z) * (K - 3) / 24 -
             (2 * z**3 - 5 * z) * S**2 / 36)
    return -z * r.std(ddof=0) - r.mean()


def cvar_historic(r, level=0.05):

    if isinstance(r, pd.Series):
        is_beyond = r <= var_historic(r, level=level)
        return -r[is_beyond].mean()
    elif isinstance(r, pd.DataFrame):
        return r.aggregate(cvar_historic, level=level)
    else:
        raise TypeError("Input must be a pandas Series or DataFrame.")

