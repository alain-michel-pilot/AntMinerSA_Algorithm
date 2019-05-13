from scipy import stats
import pandas as pd
import warnings
import numpy as np
from numpy import (asarray, sqrt, compress)


def _stats_wilcoxon(x, y=None, zero_method="wilcox", correction=False):
    """
    Calculate the Wilcoxon signed-rank test.
    The Wilcoxon signed-rank test tests the null hypothesis that two
    related paired samples come from the same distribution. In particular,
    it tests whether the distribution of the differences x - y is symmetric
    about zero. It is a non-parametric version of the paired T-test.
    Parameters
    ----------
    x : array_like
        The first set of measurements.
    y : array_like, optional
        The second set of measurements.  If `y` is not given, then the `x`
        array is considered to be the differences between the two sets of
        measurements.
    zero_method : string, {"pratt", "wilcox", "zsplit"}, optional
        "pratt":
            Pratt treatment: includes zero-differences in the ranking process
            (more conservative)
        "wilcox":
            Wilcox treatment: discards all zero-differences
        "zsplit":
            Zero rank split: just like Pratt, but splitting the zero rank
            between positive and negative ones
    correction : bool, optional
        If True, apply continuity correction by adjusting the Wilcoxon rank
        statistic by 0.5 towards the mean value when computing the
        z-statistic.  Default is False.
    Returns
    -------
    statistic : float
        The sum of the ranks of the differences above or below zero, whichever
        is smaller.
    pvalue : float
        The two-sided p-value for the test.
    Notes
    -----
    Because the normal approximation is used for the calculations, the
    samples used should be large.  A typical rule is to require that
    n > 20.
    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Wilcoxon_signed-rank_test
    """

    if zero_method not in ["wilcox", "pratt", "zsplit"]:
        raise ValueError("Zero method should be either 'wilcox' "
                         "or 'pratt' or 'zsplit'")

    if y is None:
        d = asarray(x)
    else:
        x, y = map(asarray, (x, y))
        if len(x) != len(y):
            raise ValueError('Unequal N in wilcoxon.  Aborting.')
        d = x - y

    if zero_method == "wilcox":
        # Keep all non-zero differences
        d = compress(np.not_equal(d, 0), d, axis=-1)

    count = len(d)
    if count < 10:
        warnings.warn("Warning: sample size too small for normal approximation.")

    r = stats.rankdata(abs(d))
    r_plus = np.sum((d > 0) * r, axis=0)
    r_minus = np.sum((d < 0) * r, axis=0)

    if zero_method == "zsplit":
        r_zero = np.sum((d == 0) * r, axis=0)
        r_plus += r_zero / 2.
        r_minus += r_zero / 2.

    T = min(r_plus, r_minus)
    mn = count * (count + 1.) * 0.25
    se = count * (count + 1.) * (2. * count + 1.)

    if zero_method == "pratt":
        r = r[d != 0]

    replist, repnum = _find_repeats(r)
    if repnum.size != 0:
        # Correction for repeated elements.
        se -= 0.5 * (repnum * (repnum * repnum - 1)).sum()

    se = sqrt(se / 24)
    correction = 0.5 * int(bool(correction)) * np.sign(T - mn)
    z = (T - mn - correction) / se
    prob = 2. * stats.norm.sf(abs(z))

    return T, z, prob


def _find_repeats(array):
    arr = np.array(array, dtype=np.float64)
    # This function assumes it may clobber its input.
    if len(arr) == 0:
        return np.array(0, np.float64), np.array(0, np.intp)

    # XXX This cast was previously needed for the Fortran implementation,
    # should we ditch it?
    arr = np.asarray(arr, np.float64).ravel()
    arr.sort()

    # Taken from NumPy 1.9's np.unique.
    change = np.concatenate(([True], arr[1:] != arr[:-1]))
    unique = arr[change]
    change_idx = np.concatenate(np.nonzero(change) + ([arr.size],))
    freq = np.diff(change_idx)
    atleast2 = freq > 1
    return unique[atleast2], freq[atleast2]


def wilcoxon(file_name):
    data = pd.read_csv(file_name, delimiter=',', header=0, index_col=False)
    official = data['official'].values
    dev = data['dev'].values
    # T_stats, p_value = stats.wilcoxon(official, dev, zero_method='zsplit')
    T_stats, z_stats, p_value = _stats_wilcoxon(official, dev, zero_method="zsplit")

    log_file = "log_Wilcoxon_statisticaltest.txt"
    f = open(log_file, "a+")
    f.write('\n\n=== WILCOXON TEST\n')
    f.write('\nFrom file: ' + repr(file_name))
    f.write('\nOfficial algorithm rates:\n' + repr(official))
    f.write('\nDeveloped algorithm rates:\n' + repr(dev))
    f.write('\n\n> T-Statistic: ' + repr(T_stats))
    f.write('\n> p-value: ' + repr(p_value))
    f.write('\n> z-Statistic: ' + repr(z_stats))
    f.close()

    print('\n\n> T-statistic: ', T_stats)
    print('\n> P-value: ', p_value)
    print('\n> Z-statistic: ', z_stats)

    return


if __name__ == '__main__':

    file = 'statistical_data/dataset_wine_accuracymeasure.csv'
    wilcoxon(file)
