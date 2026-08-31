import numpy as np
from scipy.spatial.distance import squareform
from scipy.stats import linregress
from tqdm.auto import tqdm

def get_residual(x, y):
    slope, intercept, r, p, se = linregress(x, y)
    return y - (slope*x + intercept)


def single_fit(x, y, rng, confound_x=None, shuffle=False, precomputed_partial_x=None):
    if len(y.shape) == 2:
        y_sq = y
    else:
        y_sq = squareform(y)

    p = np.arange(y_sq.shape[0])
    if shuffle:
        p = rng.permutation(p)

    y = squareform(y_sq[p][:,p], checks=False)

    if confound_x is not None:
        y = get_residual(confound_x, y)

        if precomputed_partial_x is not None:
            x = precomputed_partial_x
        else:
            x = get_residual(confound_x, x)

    slope, intercept, r, p, se = linregress(x, y)

    return x, y, slope, intercept, r


def mantel_test(x, y, rng, confound_x=None, n=1000, use_tqdm=False):
    """this function assumes the inputs have the same format as pdist output"""
    y_sq = squareform(y)

    precomputed_partial_x = get_residual(confound_x, x) if (confound_x is not None) else None

    test_statistic = single_fit(x, y_sq, rng, confound_x=confound_x, precomputed_partial_x=precomputed_partial_x, shuffle=False)[-1]
    range_iterator = (tqdm(range(n-1)) if use_tqdm else range(n-1))
    null_samples = [single_fit(x, y_sq, rng, confound_x=confound_x, precomputed_partial_x=precomputed_partial_x, shuffle=True)[-1] for _ in range_iterator]

    p = (np.abs(np.array([test_statistic] + null_samples)) >= np.abs(test_statistic)).mean()

    return test_statistic, null_samples, p
