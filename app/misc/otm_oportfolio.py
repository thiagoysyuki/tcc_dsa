
from scipy.stats import norm, skew, kurtosis

def var_gaussian(r, level=0.05, modified=False):
    
    z = norm.ppf(level)

    if modified:
        S = skew(r)
        K = kurtosis(r)
        z = (z + (z**2 - 1) * S / 6 + (z**3 - 3 * z) * (K - 3) / 24 -
             (2 * z**3 - 5 * z) * S**2 / 36)
    return -z * r.std(ddof=0) - r.mean()