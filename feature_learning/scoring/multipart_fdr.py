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


from glycan_profiling.tandem.target_decoy import NearestValueLookUp


from .mixture import GammaMixture, GaussianMixtureWithPriorComponent


class FiniteMixtureModelFDREstimator(object):
    def __init__(self, decoy_scores, target_scores):
        self.decoy_scores = np.array(decoy_scores)
        self.target_scores = np.array(target_scores)
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
                self.target_scores, i, self.gamma_mixture, deterministic=True)
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
        fdr[np.isnan(fdr)] = 1.0
        fdr_descending = fdr[::-1]
        for i in range(1, fdr_descending.shape[0]):
            if fdr_descending[i - 1] < fdr_descending[i]:
                fdr_descending[i] = fdr_descending[i - 1]
        fdr = fdr_descending[::-1]
        fdr = fdr[::-1][np.searchsorted(X_[::-1], X)]
        return fdr

    def plot_mixture(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots(1)
        X = np.arange(1, max(self.target_scores), 0.1)
        ax.plot(X,
                np.exp(self.gaussian_mixture.logpdf(X)).sum(axis=1))
        for col in np.exp(self.gaussian_mixture.logpdf(X)).T:
            ax.plot(X, col, linestyle='--')
        ax.hist(self.target_scores, bins=100, density=1, alpha=0.15)
        return ax

    def plot(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots(1)
        points = np.linspace(
            min(self.target_scores.min(), self.decoy_scores.min()),
            max(self.target_scores.max(), self.decoy_scores.max()),
            10000)
        target_scores = np.sort(self.target_scores)
        target_counts = [(self.target_scores >= i).sum() for i in points]
        decoy_counts = [(self.decoy_scores >= i).sum() for i in points]
        fdr = self.estimate_fdr(target_scores)
        at_5_percent = np.where(fdr < 0.05)[0][0]
        at_1_percent = np.where(fdr < 0.01)[0][0]
        line1 = ax.plot(points, target_counts, label='Target', color='blue')
        line2 = ax.plot(points, decoy_counts, label='Decoy', color='orange')
        ax.vlines(target_scores[at_5_percent], 0, np.max(target_counts), linestyle='--', color='blue', lw=0.75)
        ax.vlines(target_scores[at_1_percent], 0, np.max(target_counts), linestyle='--', color='blue', lw=0.75)
        ax2 = ax.twinx()
        line3 = ax2.plot(target_scores, fdr, label='FDR', color='grey', linestyle='--')
        ax.legend([line1[0], line2[0], line3[0]], ['Target', 'Decoy', 'FDR'])
        return ax
