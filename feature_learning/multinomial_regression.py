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


# the number of ion series to consider is one less than the total number of series
# because unassigned is a special case which no matched peaks will receive
FragmentSeriesClassification_max = max(FragmentSeriesClassification, key=lambda x: x[1].value)[1].value - 1

# the number of backbone ion series to consider is two less because the stub_glycopeptide
# series is not a backbone fragmentation series
BackboneFragmentSeriesClassification_max = FragmentSeriesClassification_max - 1


class FragmentTypeClassification(AminoAcidClassification):
    pass


FragmentTypeClassification_max = max(FragmentTypeClassification, key=lambda x: x[1].value)[1].value

# consider fragments with up to 2 monosaccharides attached to a backbone fragment
BackboneFragment_max_glycosylation_size = 2
# consider fragments of up to charge 4+
FragmentCharge_max = 3
# consider up to 10 monosaccharides of glycan still attached to a stub ion
StubFragment_max_glycosylation_size = 10

_FragmentType = namedtuple(
    "FragmentType", [
        "nterm", "cterm", "series", "glycosylated", "charge", "peak_pair", "sequence"])


def get_nterm_index_from_fragment(fragment, structure):
    size = len(structure)
    direction = fragment.series.direction
    if direction < 0:
        index = size + (fragment.series.direction * fragment.position + fragment.series.direction)
    else:
        index = fragment.position - 1
    return index


def get_cterm_index_from_fragment(fragment, structure):
    size = len(structure)
    direction = fragment.series.direction
    if direction < 0:
        index = size + (fragment.series.direction * fragment.position)
    else:
        index = fragment.position
    return index


class FragmentType(_FragmentType):

    @property
    def fragment(self):
        return self.peak_pair.fragment

    @property
    def peak(self):
        return self.peak_pair.peak

    def is_assigned(self):
        return self.series != FragmentSeriesClassification.unassigned

    def is_backbone(self):
        return (self.series != FragmentSeriesClassification.stub_glycopeptide) and self.is_assigned()

    def is_stub_glycopeptide(self):
        return (self.series == FragmentSeriesClassification.stub_glycopeptide)

    def as_feature_vector(self):
        k_ftypes = (FragmentTypeClassification_max + 1)
        k_series = (FragmentSeriesClassification_max + 1)
        k_unassigned = 1
        k_charge = FragmentCharge_max + 1
        k_charge_series = k_charge * k_series

        k_glycosylated = BackboneFragment_max_glycosylation_size + 1

        k = (
            (k_ftypes * 2) + k_series + k_unassigned + k_charge + k_charge_series +
            k_glycosylated)

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
        if self.is_backbone():
            X[offset + (self.charge - 1)] = 1
        offset += k_charge

        if self.is_assigned():
            index = (self.series.value * k_charge) + (self.charge - 1)
            X[offset + index] = 1
        offset += k_charge_series

        # non-stub ion glycosylation
        if self.is_backbone():
            X[offset + int(self.peak_pair.fragment.glycosylation_size)] = 1
        offset += k_glycosylated

        return X

    @classmethod
    def feature_names(cls):
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

        for i in range(BackboneFragment_max_glycosylation_size + 1):
            names.append("is_glycosylated %r" % (i))
        return names

    def __str__(self):
        return '(%s, %s, %s, %r, %r)' % (
            self[0].name if self[0] else '',
            self[1].name if self[1] else '',
            self[2].name, self[3], self[4])

    @classmethod
    def from_peak_peptide_fragment_pair(cls, peak_fragment_pair, gpsm):
        peak, fragment = peak_fragment_pair
        nterm, cterm = classify_amide_bond_frank(*fragment.flanking_amino_acids)
        glycosylation = bool(fragment.glycosylation) | bool(
            set(fragment.modification_dict) & PeptideFragment.concerned_modifications)
        inst = cls(
            nterm, cterm, FragmentSeriesClassification[fragment.series],
            glycosylation, min(peak.charge, FragmentCharge_max + 1),
            peak_fragment_pair, gpsm.structure)
        return inst

    @classmethod
    def build_fragment_intensity_matches(cls, gpsm):
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
                    cls(
                        None, None, FragmentSeriesClassification.stub_glycopeptide,
                        min(fragment.glycosylation_size, StubFragment_max_glycosylation_size),
                        min(peak.charge, FragmentCharge_max + 1), peak_fragment_pair, gpsm.structure))
                continue
            inst = cls.from_peak_peptide_fragment_pair(peak_fragment_pair, gpsm)
            fragment_classification.append(inst)

        unassigned = total - matched_total
        ft = cls(None, None, FragmentSeriesClassification.unassigned, 0, 0, None, None)
        fragment_classification.append(ft)
        intensities.append(unassigned)
        return fragment_classification, np.array(intensities), total

    @classmethod
    def encode_classification(cls, classification):
        X = []
        for i, row in enumerate(classification):
            X.append(row.as_feature_vector())
        return np.vstack(X)

    @classmethod
    def fit_regression(cls, gpsms, **kwargs):
        breaks = []
        matched = []
        totals = []
        for gpsm in gpsms:
            c, y, t = cls.build_fragment_intensity_matches(gpsm)
            x = cls.encode_classification(c)
            breaks.append(x)
            matched.append(y)
            totals.append(t)
        fit = multinomial_fit(breaks, matched, totals, **kwargs)
        return MultinomialRegressionFit(model_type=cls, **fit)


class ProlineSpecializingModel(FragmentType):
    def specialize_proline(self):
        k_charge_cterm_pro = (FragmentCharge_max + 1)
        k_series_cterm_pro = (BackboneFragmentSeriesClassification_max + 1)
        k_glycosylated_proline = BackboneFragment_max_glycosylation_size + 1

        k = (k_charge_cterm_pro + k_series_cterm_pro + k_glycosylated_proline)

        X = np.zeros(k, dtype=np.uint8)
        offset = 0

        if self.cterm == FragmentTypeClassification.pro:
            index = (self.charge - 1)
            X[offset + index] = 1
        offset += k_charge_cterm_pro

        if self.cterm == FragmentTypeClassification.pro:
            X[offset + self.series.value] = 1
        offset += k_series_cterm_pro

        if self.cterm == FragmentTypeClassification.pro:
            X[offset + int(self.peak_pair.fragment.glycosylation_size)] = 1
        offset += k_glycosylated_proline
        return X

    def as_feature_vector(self):
        X = super(ProlineSpecializingModel, self).as_feature_vector()
        return np.hstack((X, self.specialize_proline()))

    @classmethod
    def feature_names(cls):
        names = super(ProlineSpecializingModel, cls).feature_names()
        for i in range(FragmentCharge_max + 1):
            names.append("charge %d:c-term pro" % (i + 1))
        for label, tp in sorted(FragmentSeriesClassification, key=lambda x: x[1].value):
            if tp.value is None or label in ("unassigned", "stub_glycopeptide"):
                continue
            names.append("series:c-term pro %s" % label)
        for i in range(BackboneFragment_max_glycosylation_size + 1):
            names.append("is_glycosylated:c-term pro %r" % (i))
        return names


class StubGlycopeptideCompositionModel(ProlineSpecializingModel):

    def encode_stub_information(self):
        k_glycosylated_stubs = StubFragment_max_glycosylation_size + 1
        k_sequence_composition_stubs = FragmentTypeClassification_max + 1
        k = k_glycosylated_stubs + k_sequence_composition_stubs

        X = np.zeros(k, dtype=np.uint8)
        offset = 0

        if self.is_stub_glycopeptide():
            X[offset + int(self.glycosylated)] = 1
        offset += k_glycosylated_stubs
        if self.is_stub_glycopeptide():
            ctr = classify_sequence_by_residues(self.sequence)
            for tp, c in ctr:
                X[offset + tp.value] = c
        offset += k_sequence_composition_stubs
        return X

    def as_feature_vector(self):
        X = super(StubGlycopeptideCompositionModel, self).as_feature_vector()
        return np.hstack((X, self.encode_stub_information()))

    @classmethod
    def feature_names(self):
        names = super(StubGlycopeptideCompositionModel, self).feature_names()
        for i in range(StubFragment_max_glycosylation_size + 1):
            names.append("stub glycopeptide:is_glycosylated %r" % (i))
        for label, tp in sorted(FragmentTypeClassification, key=lambda x: x[1].value):
            if tp.value is None:
                continue
            names.append("stub glycopeptide:composition %s" % (label,))
        return names


class StubGlycopeptideFucosylationModel(StubGlycopeptideCompositionModel):
    def encode_stub_fucosylation(self):
        X = [0, 0]
        if self.is_stub_glycopeptide():
            i = int(self.peak_pair.fragment.glycosylation['Fuc'] > 0)
            X[i] = 1
        return np.array(X, dtype=np.uint8)

    def as_feature_vector(self):
        X = super(StubGlycopeptideFucosylationModel, self).as_feature_vector()
        return np.hstack((X, self.encode_stub_fucosylation()))

    @classmethod
    def feature_names(self):
        names = super(StubGlycopeptideFucosylationModel, self).feature_names()
        for i in range(2):
            names.append("stub glycopeptide:is_fucosylated %r" % (i))
        return names


class NeighboringAminoAcidsModel(StubGlycopeptideFucosylationModel):
    bond_offset_depth = 1

    def get_nterm_neighbor(self, offset=1):
        index = get_nterm_index_from_fragment(self.fragment, self.sequence)
        index -= offset
        if index < 0:
            return None
        else:
            residue = self.sequence[index][0]
            return classify_residue_frank(residue)

    def get_cterm_neighbor(self, offset=1):
        index = get_cterm_index_from_fragment(self.fragment, self.sequence)
        index += offset
        if index > len(self.sequence) - 1:
            return None
        else:
            residue = self.sequence[index][0]
            return classify_residue_frank(residue)

    def encode_neighboring_residues(self):
        k_ftypes = (FragmentTypeClassification_max + 1)
        k = (k_ftypes * 2)

        X = np.zeros(k, dtype=np.uint8)
        offset = 0

        if self.is_backbone():
            nterm = self.get_nterm_neighbor(self.bond_offset_depth)
            if nterm is not None:
                X[nterm.value] = 1
        offset += k_ftypes
        if self.is_backbone():
            cterm = self.get_cterm_neighbor(self.bond_offset_depth)
            if cterm is not None:
                X[offset + cterm.value] = 1
        offset += k_ftypes
        return X

    def as_feature_vector(self):
        X = super(NeighboringAminoAcidsModel, self).as_feature_vector()
        return np.hstack((X, self.encode_neighboring_residues()))

    @classmethod
    def feature_names(cls):
        names = super(NeighboringAminoAcidsModel, cls).feature_names()
        for i in range(1, cls.bond_offset_depth + 1):
            for label, tp in sorted(FragmentTypeClassification, key=lambda x: x[1].value):
                if tp.value is None:
                    continue
                names.append("n-term - %d %s" % (i, label))
        for i in range(1, cls.bond_offset_depth + 1):
            for label, tp in sorted(FragmentTypeClassification, key=lambda x: x[1].value):
                if tp.value is None:
                    continue
                names.append("c-term + %d %s" % (i, label))
        return names


class AmideBondCrossproductModel(StubGlycopeptideCompositionModel):
    def specialize_fragmentation_site(self):
        k_series_ftypes = ((FragmentTypeClassification_max + 1) ** 2) * (
            BackboneFragmentSeriesClassification_max + 1)
        k = k_series_ftypes

        X = np.zeros(k, dtype=np.uint8)
        offset = 0

        if self.is_backbone():
            index_series = self.series.value
            index_nterm = self.nterm.value
            index_cterm = self.cterm.value
            # the cterm-th slot in the nterm-th section
            index_ftypes = index_cterm + ((FragmentTypeClassification_max + 1) * index_nterm)
            index = index_ftypes + ((FragmentTypeClassification_max + 1) ** 2) * index_series
            X[offset + index] = 1

        offset += k_series_ftypes

        return X

    def as_feature_vector(self):
        X = super(AmideBondCrossproductModel, self).as_feature_vector()
        return np.hstack((X, self.specialize_fragmentation_site()))

    @classmethod
    def feature_names(self):
        names = super(AmideBondCrossproductModel, self).feature_names()
        for i in range(BackboneFragmentSeriesClassification_max + 1):
            for j in range(FragmentTypeClassification_max + 1):
                for k in range(FragmentTypeClassification_max + 1):
                    series = FragmentSeriesClassification[i].name
                    n_term = FragmentTypeClassification[j].name
                    c_term = FragmentTypeClassification[k].name
                    names.append("series %s:n-term %s:c-term %s" % (series, n_term, c_term))
        return names


def classify_sequence_by_residues(sequence):
    ctr = defaultdict(int)
    for res, mods in sequence:
        ctr[classify_residue_frank(res)] += 1
    return sorted(ctr.items())


build_fragment_intensity_matches = FragmentType.build_fragment_intensity_matches


encode_classification = FragmentType.encode_classification


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
        # this is similar to the unit deviance of the Poisson distribution
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


def fit_matches(gpsms, **kwargs):
    breaks = []
    matched = []
    totals = []
    for gpsm in gpsms:
        c, y, t = build_fragment_intensity_matches(gpsm)
        x = encode_classification(c)
        breaks.append(x)
        matched.append(y)
        totals.append(t)
    fit = multinomial_fit(breaks, matched, totals, **kwargs)
    return MultinomialRegressionFit(**fit)


def multinomial_predict(x, weights, beta):
    yhat = (x.dot(beta))
    yhat /= (1 + yhat.sum())
    return yhat


class MultinomialRegressionFit(object):
    def __init__(self, coef, scaled_y, mu, dispersion, weights, covariance_unscaled,
                 deviance, H, model_type=FragmentType, **info):
        self.coef = coef
        self.scaled_y = scaled_y
        self.mu = mu
        self.weights = weights
        self.dispersion = dispersion
        self.covariance_unscaled = covariance_unscaled
        self.deviance = deviance
        self.H = H
        self.info = info
        self.model_type = model_type

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
            c, intens, t = self.model_type.build_fragment_intensity_matches(gpsm)
            X = self.model_type.encode_classification(c)
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
        c, intens, t = self.model_type.build_fragment_intensity_matches(gpsm)
        X = self.model_type.encode_classification(c)
        yhat = self.predict(X)
        return deviance_residuals([intens / t], [yhat], [t])[0]

    def test_goodness_of_fit(self, gpsm):
        c, intens, t = self.model_type.build_fragment_intensity_matches(gpsm)
        X = self.model_type.encode_classification(c)
        yhat = self.predict(X)

        # standardize intensity
        yhat *= 100
        intens = intens / t * 100

        dc = (100 - intens.sum()) * np.log((100 - intens.sum()) / (100 - yhat.sum()))

        intens = (intens)
        theor = (yhat)

        ratio = np.log(intens / theor)
        G = intens.dot(ratio) + dc
        return G

    def test_goodness_of_fit2(self, gpsm):
        c, intens, t = self.model_type.build_fragment_intensity_matches(gpsm)
        X = self.model_type.encode_classification(c)
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
        for name, value, se in zip(self.model_type.feature_names(), self.coef, self.parameter_intervals()):
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
