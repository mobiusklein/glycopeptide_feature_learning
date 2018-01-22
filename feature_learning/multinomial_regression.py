import six
from collections import namedtuple, defaultdict

import numpy as np
from scipy.stats import norm
from scipy.linalg import solve_triangular, pinv

from glypy.utils import Enum
from glycopeptidepy.structure.fragment import PeptideFragment

from .amino_acid_classification import AminoAcidClassification, classify_amide_bond_frank, classify_residue_frank


class FragmentSeriesClassification(Enum):
    b = 0
    y = 1
    stub_glycopeptide = 2
    unassigned = 3


FragmentSeriesClassification_max = max(FragmentSeriesClassification, key=lambda x: x[1].value)[1].value - 1


class FragmentTypeClassification(AminoAcidClassification):
    pass


FragmentTypeClassification_max = max(FragmentTypeClassification, key=lambda x: x[1].value)[1].value


FragmentCharge_max = 2
StubFragment_max_glycosylation_size = 5

_FragmentType = namedtuple(
    "FragmentType", [
        "nterm", "cterm", "series", "glycosylated", "charge", "peak_pair", "sequence"])


class FragmentType(_FragmentType):

    def is_assigned(self):
        return self.series != FragmentSeriesClassification.unassigned

    def as_feature_vector(self):
        k_ftypes = (FragmentTypeClassification_max + 1)
        k_series = (FragmentSeriesClassification_max + 1)
        k_unassigned = 1
        k_charge = FragmentCharge_max + 1
        k_charge_series = k_charge * k_series
        k_charge_cterm_pro = (FragmentSeriesClassification_max + 1)
        k_series_cterm_pro = (FragmentSeriesClassification_max + 1)
        k_glycosylated = 2
        k_glycosylated_proline = k_glycosylated
        k_glycosylated_stubs = StubFragment_max_glycosylation_size + 1
        k_sequence_composition_stubs = FragmentTypeClassification_max + 1
        k = (
            (k_ftypes * 2) + k_series + k_unassigned + k_charge + k_charge_series + k_charge_cterm_pro +
            k_series_cterm_pro + k_glycosylated + k_glycosylated_proline + k_glycosylated_stubs +
            k_sequence_composition_stubs)
        X = np.zeros(k, dtype=np.uint8)
        offset = 0

        if self.nterm is not None:
            X[self.nterm.value] = 1
        offset += k_ftypes

        if self.cterm is not None:
            X[offset + self.cterm.value] = 1
        offset += k_ftypes

        if self.is_assigned():
            X[offset + self.series.value] = 1
        offset += k_series

        # tarck the unassigned placeholder observation separately
        X[offset] = int(not self.is_assigned())
        offset += k_unassigned

        # use charge - 1 because there is no 0 charge
        if (self.series != FragmentSeriesClassification.stub_glycopeptide) and self.is_assigned():
            X[offset + (self.charge - 1)] = 1
        offset += k_charge

        if self.is_assigned():
            index = (self.series.value * k_charge) + (self.charge - 1)
            X[offset + index] = 1
        offset += k_charge_series

        if self.cterm == FragmentTypeClassification.pro:
            index = (self.charge - 1)
            X[offset + index] = 1
        offset += k_charge_cterm_pro

        if self.cterm == FragmentTypeClassification.pro:
            X[offset + self.series.value] = 1
        offset += k_series_cterm_pro

        # non-stub ion glycosylation
        if self.series != FragmentSeriesClassification.stub_glycopeptide and self.is_assigned():
            X[offset + int(self.glycosylated)] = 1
        offset += k_glycosylated

        if self.cterm == FragmentTypeClassification.pro:
            X[offset + int(self.glycosylated)] = 1
        offset += k_glycosylated_proline

        if self.series == FragmentSeriesClassification.stub_glycopeptide:
            X[offset + int(self.glycosylated)] = 1
        offset += k_glycosylated_stubs
        if self.series == FragmentSeriesClassification.stub_glycopeptide:
            ctr = classify_sequence_by_residues(self.sequence)
            for tp, c in ctr:
                X[offset + tp.value] = c
        offset += k_sequence_composition_stubs
        return X

    @staticmethod
    def feature_names():
        names = []
        for label, tp in sorted(FragmentTypeClassification, key=lambda x: x[1].value):
            if tp.value is None:
                continue
            names.append("n-term %s" % label)
        for label, tp in sorted(FragmentTypeClassification, key=lambda x: x[1].value):
            if tp.value is None:
                continue
            names.append("c-term %s" % label)

        for label, tp in sorted(FragmentSeriesClassification, key=lambda x: x[1].value):
            if tp.value is None or label == "unassigned":
                continue
            names.append("series %s" % label)
        names.append("unassigned")
        for i in range(FragmentCharge_max + 1):
            i += 1
            names.append("charge %d" % i)
        for label, tp in sorted(FragmentSeriesClassification, key=lambda x: x[1].value):
            for i in range(FragmentCharge_max + 1):
                if tp.value is None or label == 'unassigned':
                    continue
                names.append("series %s:charge %d" % (label, i + 1))
        for i in range(FragmentCharge_max + 1):
            names.append("charge %d:c-term pro" % (i + 1))
        for label, tp in sorted(FragmentSeriesClassification, key=lambda x: x[1].value):
            if tp.value is None or label == "unassigned":
                continue
            names.append("series:c-term pro %s" % label)
        for i in range(2):
            names.append("is_glycosylated %r" % (i))
        for i in range(2):
            names.append("is_glycosylated:c-term pro %r" % (i))
        for i in range(StubFragment_max_glycosylation_size + 1):
            names.append("stub glycopeptide:is_glycosylated %r" % (i))
        for label, tp in sorted(FragmentTypeClassification, key=lambda x: x[1].value):
            if tp.value is None:
                continue
            names.append("stub glycopeptide:composition %s" % (label,))
        return names

    def __str__(self):
        return '(%s, %s, %s, %r, %r)' % (
            self[0].name if self[0] else '',
            self[1].name if self[1] else '',
            self[2].name, self[3], self[4])


def classify_sequence_by_residues(sequence):
    ctr = defaultdict(int)
    for res, mods in sequence:
        ctr[classify_residue_frank(res)] += 1
    return sorted(ctr.items())


def build_fragment_intensity_matches(gpsm):
    fragment_classification = []
    intensities = []
    matched_total = 0
    total = sum(p.intensity for p in gpsm.deconvoluted_peak_set)
    counted = set()
    for peak_fragment_pair in gpsm.match().solution_map:
        peak, fragment = peak_fragment_pair
        if peak not in counted:
            matched_total += peak.intensity
            counted.add(peak)
        if fragment.series == 'oxonium_ion':
            continue
        intensities.append(peak.intensity)
        if fragment.series == 'stub_glycopeptide':
            fragment_classification.append(
                FragmentType(
                    None, None, FragmentSeriesClassification.stub_glycopeptide,
                    min(fragment.glycosylation_size, StubFragment_max_glycosylation_size),
                    peak.charge, peak_fragment_pair, gpsm.structure))
            continue
        nterm, cterm = classify_amide_bond_frank(*fragment.flanking_amino_acids)
        glycosylation = bool(fragment.glycosylation) | bool(
            set(fragment.modification_dict) & PeptideFragment.concerned_modifications)
        fragment_classification.append(
            FragmentType(
                nterm, cterm, FragmentSeriesClassification[fragment.series],
                glycosylation, peak.charge, peak_fragment_pair, gpsm.structure))

    unassigned = total - matched_total
    ft = FragmentType(None, None, FragmentSeriesClassification.unassigned, 0, 0, None, None)
    fragment_classification.append(ft)
    intensities.append(unassigned)
    return fragment_classification, np.array(intensities), total


def encode_classification(classification):
    X = []
    for i, row in enumerate(classification):
        X.append(row.as_feature_vector())
    return np.vstack(X)


def logit(x):
    return np.log(x / (1 - x))


def invlogit(x):
    return 1 / (1 + np.exp(-x))


def multinomial_control(epsilon=1e-8, maxit=25, nsamples=1000, trace=False):
    return dict(epsilon=epsilon, maxit=maxit, nsamples=nsamples, trace=trace)


def deviance(y, mu, wt):
    ys = np.array(list(map(np.sum, y)))
    ms = np.array(list(map(np.sum, mu)))
    # this is the leftover count after accounting for signal matched by
    # the fragment
    dc = np.where(wt == ys, 0, wt * (1 - ys) * np.log((1 - ys) / (1 - ms)))
    return np.sum([
        # this inner sum is the squared residuals
        a.sum() + (2 * dc[i]) for i, a in enumerate(deviance_residuals(y, mu, wt))])


def deviance_residuals(y, mu, wt):
    # returns the squared residual. The sign is lost? The sign can be regained
    # from (yi - mu[i])
    residuals = []
    for i, yi in enumerate(y):
        # this is similar to the unit deviance of the Poisson distribution?
        # "sub-residual", contributing to the total being the actual residual,
        # but these are not residuals themselves, the sum is, and that must be positive
        ym = np.where(yi == 0, 0, yi * np.log(yi / mu[i]))
        residuals.append(2 * wt[i] * ym)
    return residuals


def multinomial_fit(x, y, weights, dispersion=1, adjust_dispersion=True, prior_coef=None, prior_disp=None, **control):
    """Fit a multinomial generalized linear model to bond-type by intensity observations
    of glycopeptide fragmentation.

    Parameters
    ----------
    x : list
        list of fragment type matrices
    y : list
        list of observed intensities
    weights : list
        list of total intensities
    dispersion : int, optional
        The dispersion of the model
    adjust_dispersion : bool, optional
        Whether or not to adjust the dispersion according to
        the prior dispersion parameters
    prior_coef : dict, optional
        Description
    prior_disp : dict, optional
        Description
    **control
        Description

    Returns
    -------
    dict
    """
    control = multinomial_control(**control)

    # make a copy of y and convert each observation
    # into an ndarray.
    y = list(map(np.array, y))
    n = weights
    lengths = np.array(list(map(len, y)))
    nvars = x[0].shape[1]
    if prior_coef is None:
        prior_coef = {"mean": np.zeros(nvars), "precision": np.zeros(nvars)}
    prior_coef = {k: np.array(v) for k, v in prior_coef.items()}
    if prior_disp is None:
        prior_disp = {"df": 0, "scale": 0}
    S_inv0 = prior_coef['precision']
    beta0 = prior_coef['mean']
    beta0 = S_inv0 * beta0
    nu = prior_disp['df']
    nu_tau2 = nu * prior_disp['scale']

    phi = dispersion
    mu = [0 for _ in y]
    eta = [0 for _ in y]

    # intialize parameters
    for i in range(len(y)):
        # ensure no value is 0 by adding 0.5
        mu[i] = (y[i] + 0.5) / (1 + n[i] + 0.5 * lengths[i])
        # put on the scale of mu
        y[i] = y[i] / n[i]
        # link function
        eta[i] = np.log(mu[i]) + np.log(1 + np.exp(logit(np.sum(mu[i]))))
        assert not np.any(np.isnan(eta[i]))

    dev = deviance(y, mu, n)
    for iter_ in range(control['maxit']):
        if control['trace']:
            print("Iteration %d" % (iter_,))
        z = phi * beta0
        H = phi * np.diag(S_inv0)
        for i in range(len(y)):
            # Variance of Y_i, multinomial, e.g. covariance matrix
            # Here an additional dimension is introduced to coerce mu[i] into
            # a matrix to match the behavior of tcrossprod
            W = np.diag(mu[i]) - mu[i][:, None].dot(mu[i][:, None].T)
            # Since both mu and y are on the same scale it is convenient to multiply both by the total
            # Here, x[i] is transposed to match the behavior of crossprod
            # Working Residual analog
            z += n[i] * x[i].T.dot(y[i] - mu[i] + W.dot(eta[i]))
            # Sum of covariances, close to the Hessian (log-likelihood)
            H += n[i] * x[i].T.dot(W.dot(x[i]))
        H += np.identity(H.shape[0])
        # H = CtC, H_inv = C_inv * Ct_inv
        # get the upper triangular Cholesky decomposition, s.t. C.T.dot(C) == H
        # np.linalg.cholesky returns lower by default, so it must be transposed
        C = np.linalg.cholesky(H).T
        # Solve for updated coefficients. Use back substitution algorithm.
        beta = solve_triangular(C, solve_triangular(C, z, trans="T"))

        for i in range(len(y)):
            # linear predictor
            eta[i] = x[i].dot(beta)
            # canonical poisson inverse link
            mu[i] = np.exp(eta[i])
            # Apply a normalizing constraint for multinomial
            # inverse link to expected value from linear predictor
            mu[i] = mu[i] / (1 + mu[i].sum())

        dev_new = deviance(y, mu, n)
        if adjust_dispersion:
            phi = (nu_tau2 + dev_new) / (nu + np.sum(lengths))
        if control['trace']:
            print("[%d] deviance = %f" % (iter_, dev_new))
        rel_error = np.abs((dev_new - dev) / dev)
        # converged?
        if (not np.isinf(dev)) and (rel_error < control["epsilon"]):
            break
        dev = dev_new
    return dict(
        coef=beta, scaled_y=y, mu=mu, dispersion=dispersion, weights=np.array(n),
        covariance_unscaled=pinv(C), iterations=iter_, deviance=dev,
        H=H, C=C)


def multinomial_predict(x, weights, beta):
    yhat = (x.dot(beta))
    yhat /= (1 + yhat.sum())
    return yhat


class MultinomialRegressionFit(object):
    def __init__(self, coef, scaled_y, mu, dispersion, weights, covariance_unscaled,
                 deviance, H, **info):
        self.coef = coef
        self.scaled_y = scaled_y
        self.mu = mu
        self.weights = weights
        self.dispersion = dispersion
        self.covariance_unscaled = covariance_unscaled
        self.deviance = deviance
        self.H = H
        self.info = info

    @property
    def hessian(self):
        return self.H

    def estimate_dispersion(self):
        return deviance(self.scaled_y, self.mu, self.weights) / (len(self.scaled_y) - len(self.coef))

    def predict(self, x):
        yhat = np.exp(x.dot(self.coef))
        yhat /= (1 + yhat.sum())
        return yhat

    def parameter_intervals(self):
        return np.sqrt(np.diag(np.linalg.inv(self.hessian))) * np.sqrt((self.estimate_dispersion()))

    def residuals(self, gpsms, normalized=False):
        y = []
        total = []
        mu = []
        for gpsm in gpsms:
            c, intens, t = build_fragment_intensity_matches(gpsm)
            X = encode_classification(c)
            yhat = self.predict(X)
            y.append(intens / t)
            mu.append(yhat)
            total.append(t)
        ys = np.array(list(map(np.sum, y)))
        ms = np.array(list(map(np.sum, mu)))
        sign = (ys - ms) / np.abs(ys - ms)
        wt = np.array(total)
        # this is the leftover count after accounting for signal matched by
        # the fragment
        dc = np.where(wt == ys, 0, wt * (1 - ys) * np.log((1 - ys) / (1 - ms)))
        as_ = deviance_residuals(y, mu, wt)
        return sign * np.sqrt(np.array([
            # this inner sum is the squared residuals
            a.sum() + (2 * dc[i]) for i, a in enumerate(as_)]))

    def deviance_residuals(self, gpsm):
        c, intens, t = build_fragment_intensity_matches(gpsm)
        X = encode_classification(c)
        yhat = self.predict(X)
        return deviance_residuals([intens / t], [yhat], [t])[0]

    def test_goodness_of_fit(self, gpsm):
        c, intens, t = build_fragment_intensity_matches(gpsm)
        X = encode_classification(c)
        yhat = self.predict(X)

        # standardize intensity
        yhat *= 100
        intens = intens / t * 100

        dc = (100 - intens.sum()) * np.log((100 - intens.sum()) / (100 - yhat.sum()))

        # drop the unassigned point
        intens = (intens)
        theor = (yhat)

        ratio = np.log(intens / theor)
        mask = np.where(intens > theor, 0, 1)
        ratio[mask] = np.log((intens[mask]) / (theor[mask]))
        G = intens.dot(ratio) + dc
        return G

    def test_goodness_of_fit2(self, gpsm):
        c, intens, t = build_fragment_intensity_matches(gpsm)
        X = encode_classification(c)
        yhat = self.predict(X)

        # standardize intensity
        yhat *= 100
        intens = intens / t * 100

        # drop the unassigned point
        intens = (intens)[:-1]
        theor = (yhat)[:-1]

        relative_diff_intens = (intens - theor) / intens
        relative_diff_theor = (theor - intens) / intens

        mask = intens >= theor
        similarity = np.zeros_like(intens)
        nmask = ~mask

        nmask = nmask & (relative_diff_theor < 1)
        mask = mask & (relative_diff_intens < 1)
        similarity[nmask] = (1 - relative_diff_theor[nmask])
        similarity[mask] = np.sqrt(1 - relative_diff_intens[mask])
        score = similarity.dot(np.sqrt(theor))
        return score

    def describe(self):
        table = []
        header = ['Name', 'Value', 'SE', "Z", "p-value"]
        # compute Z-statistic (parameter value / std err) Wald test (approx) p-value = 2 * pnorm(-abs(Z))
        table.append(header)
        for name, value, se in zip(FragmentType.feature_names(), self.coef, self.parameter_intervals()):
            z = value / se
            p = 2 * norm.cdf(-abs(z))
            table.append((name, str(value), str(se), str(z), str(p)))
        columns = zip(*table)
        column_widths = [max(map(len, col)) + 2 for col in columns]
        output_buffer = six.StringIO()
        for row in table:
            for i, col in enumerate(row):
                width = column_widths[i]
                if i != 0:
                    col = col.center(width)
                    col = "|%s" % col
                else:
                    col = col.ljust(width)
                output_buffer.write(col)
            output_buffer.write("\n")
        return output_buffer.getvalue()
