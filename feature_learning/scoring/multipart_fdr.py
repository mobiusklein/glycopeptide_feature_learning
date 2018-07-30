# -*- coding: utf8 -*-
'''This module implements techniques derived from the pGlyco2
FDR estimation procedure described in:

[1] Liu, M.-Q., Zeng, W.-F., Fang, P., Cao, W.-Q., Liu, C., Yan, G.-Q., … Yang, P.-Y.
    (2017). pGlyco 2.0 enables precision N-glycoproteomics with comprehensive quality
    control and one-step mass spectrometry for intact glycopeptide identification.
    Nature Communications, 8(1), 438. https://doi.org/10.1038/s41467-017-00535-2
[2] Zeng, W.-F., Liu, M.-Q., Zhang, Y., Wu, J.-Q., Fang, P., Peng, C., … Yang, P. (2016).
    pGlyco: a pipeline for the identification of intact N-glycopeptides by using HCD-
    and CID-MS/MS and MS3. Scientific Reports, 6(April), 25102. https://doi.org/10.1038/srep25102
'''
import numpy as np

try:
    from matplotlib import pyplot as plt
except ImportError:
    pass

from .mixture import GammaMixture, GaussianMixtureWithPriorComponent


class FiniteMixtureModelFDREstimator(object):
    def __init__(self, decoy_scores, target_scores):
        self.decoy_scores = decoy_scores
        self.target_scores = target_scores
        self.gamma_mixture = None
        self.gaussian_mixture = None

    def log(self, message):
        print(message)

    def estimate_gamma(self, max_components=10):
        models = []
        bics = []
        n = len(self.decoy_scores)
        np.random.seed(n)
        if n < 10:
            self.log("Too few decoy observations")
            self.gamma_mixture = GammaMixture([1.0], [1.0], [1.0])
            return self.gamma_mixture
        for i in range(1, max_components + 1):
            self.log("Fitting %d Components" % (i,))
            model = GammaMixture.fit(self.decoy_scores, i)
            bic = model.bic(self.decoy_scores)
            models.append(model)
            bics.append(bic)
            self.log("BIC: %g" % (bic,))
        i = np.argmin(bics)
        self.log("Selected %d Components" % (i + 1,))
        self.gamma_mixture = models[i]
        return self.gamma_mixture

    def estimate_gaussian(self, max_components=10):
        models = []
        bics = []
        n = len(self.target_scores)
        np.random.seed(n)
        if n < 10:
            self.log("Too few target observations")
            self.gaussian_mixture = GaussianMixtureWithPriorComponent([1.0], [1.0], self.gamma_mixture, [0.5, 0.5])
            return self.gaussian_mixture
        for i in range(1, max_components + 1):
            self.log("Fitting %d Components" % (i,))
            model = GaussianMixtureWithPriorComponent.fit(
                self.target_scores, i, self.gamma_mixture)
            bic = model.bic(self.target_scores)
            models.append(model)
            bics.append(bic)
            self.log("BIC: %g" % (bic,))
        i = np.argmin(bics)
        self.log("Selected %d Components" % (i + 1,))
        self.gaussian_mixture = models[i]
        return self.gaussian_mixture

    def estimate_posterior_error_probability(self, X):
        return self.gaussian_mixture.prior.score(X) * self.gaussian_mixture.weights[
            -1] / self.gaussian_mixture.score(X)

    def estimate_fdr(self, X):
        X_ = np.array(sorted(X, reverse=True))
        pep = self.estimate_posterior_error_probability(X_)
        # The FDR is the expected value of PEP, or the average PEP in this case.
        # The expression below is a cumulative mean (the cumulative sum divided
        # by the number of elements in the sum)
        fdr = np.cumsum(pep) / np.arange(1, len(X_) + 1)
        # Use searchsorted on the ascending ordered version of X_
        # to find the indices of the origin values of X, then map
        # those into the ascending ordering of the FDR vector to get
        # the FDR estimates of the original X
        return fdr[::-1][np.searchsorted(X_[::-1], X)]

    def plot(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots(1)
        X = np.arange(1, max(self.target_scores), 0.1)
        ax.plot(X,
                np.exp(self.gaussian_mixture.logpdf(X)).sum(axis=1))
        for col in np.exp(self.gaussian_mixture.logpdf(X)).T:
            ax.plot(X, col, linestyle='--')
        ax.hist(self.target_scores, bins=100, density=1, alpha=0.15)
        return ax
