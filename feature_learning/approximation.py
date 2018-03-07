import numpy as np
from scipy.stats import gamma

from matplotlib import pyplot as plt


class GammaFit(object):
    def __init__(self, data):
        self.data = data
        self.params = gamma.fit(data)

    def plot(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots(1)
        a, b = np.histogram(self.data, bins='fd', density=1)
        ax.plot(b[1:], a, label='empirical', c='blue')
        ax.plot(b[1:], self.pdf(b[1:]), label='gamma', c='orange')
        return ax

    def pdf(self, x):
        return gamma.pdf(x, *self.params)

    def cdf(self, x):
        return gamma.cdf(x, *self.params)

    def sf(self, x):
        return gamma.sf(x, *self.params)


class GammaMixture(object):
    def __init__(self, data):
        self.data = data
        self.fits = []
        self._split_and_fit()

    def _split_and_fit(self):
        median = np.median(self.data)
        mad = np.median(np.abs(self.data - median))
        self.fits.append(GammaFit(self.data[self.data <= median + mad]))
        self.fits.append(GammaFit(self.data[self.data >= median - mad]))

    def cdf(self, x):
        return np.min([f.cdf(x) for f in self.fits], axis=0)

    def sf(self, x):
        return np.max([f.sf(x) for f in self.fits], axis=0)

    def plot(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots(1)
        for f in self.fits:
            f.plot(ax)
        return ax
