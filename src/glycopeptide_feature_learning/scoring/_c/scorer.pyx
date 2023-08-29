# cython: embedsignature=True
cimport cython
from cython.parallel cimport prange

from cpython.mem cimport PyMem_Malloc, PyMem_Free
from cpython cimport PyTuple_GetItem, PyTuple_Size, PyList_GET_ITEM, PyList_GET_SIZE
from cpython.int cimport PyInt_AsLong
from libc.stdlib cimport malloc, calloc, free
from libc.math cimport log10, log, log2, sqrt, exp, erf

import numpy as np
cimport numpy as np

np.import_array()

from numpy.math cimport isnan, NAN

from ms_deisotope._c.peak_set cimport DeconvolutedPeak, DeconvolutedPeakSet

from glycan_profiling._c.structure.fragment_match_map cimport (
    FragmentMatchMap, PeakFragmentPair)

from glycopeptidepy._c.structure.base cimport AminoAcidResidueBase, SequencePosition
from glycopeptidepy._c.structure.sequence_methods cimport _PeptideSequenceCore
from glycopeptidepy._c.structure.fragment cimport (
    PeptideFragment, FragmentBase, IonSeriesBase, ChemicalShiftBase, StubFragment)
from glycopeptidepy._c.structure.glycan cimport GlycosylationManager

from glycopeptidepy.structure import IonSeries

from glycopeptide_feature_learning._c.model_types cimport _FragmentType
from glycopeptide_feature_learning._c.approximation cimport StepFunction

from glycopeptide_feature_learning.approximation import (
    PearsonResidualCDF as _PearsonResidualCDF)

cdef StepFunction PearsonResidualCDF = _PearsonResidualCDF


cdef:
    IonSeriesBase IonSeries_b, IonSeries_y, IonSeries_c, IonSeries_z, IonSeries_stub_glycopeptide

IonSeries_b = IonSeries.b
IonSeries_y = IonSeries.y
IonSeries_c = IonSeries.c
IonSeries_z = IonSeries.z
IonSeries_stub_glycopeptide = IonSeries.stub_glycopeptide


@cython.final
@cython.freelist(10000)
cdef class BackbonePosition(object):

    @staticmethod
    cdef BackbonePosition _create(_FragmentType match, double intensity, double predicted, double reliability):
        cdef BackbonePosition self = BackbonePosition.__new__(BackbonePosition)
        self.match = match
        self.intensity = intensity
        self.predicted = predicted
        self.reliability = reliability
        return self


cdef scalar_or_array pad(scalar_or_array x, double pad=0.5):
    return (1 - pad) * x + pad


cdef scalar_or_array unpad(scalar_or_array x, double pad=0.5):
    return (x - pad) / (1 - pad)


# A variety of sigmoidal functions to choose from


cpdef long pascal_triangle(long a, long b):
    result = 1.0
    for i in range(b):
        result *= (a - i) / (i + 1)
    return result


cpdef scalar_or_array generalized_smoothstep(long N, scalar_or_array x):
    if scalar_or_array is double:
        if x > 1.0:
            x = 1.0
        elif x < 0.0:
            x = 0.0
    else:
        x = np.clip(x, 0, 1)
    result = 0
    for n in range(N + 1):
        result += pascal_triangle(-N - 1, n) * pascal_triangle(2 * N + 1, N - n) * x ** (N + n + 1)
    return result


cdef double shifted_normalized_sigmoid_erf(double x, double shift=0.0) nogil:
    return (erf(x * 6 - shift) + 1) / 2


cdef scalar_or_array normalized_sigmoid(scalar_or_array x):
    if scalar_or_array is double:
        return ((1 / (1 + exp(-x))) - 0.5) * 2
    else:
        return ((1 / (1 + np.exp(-x))) - 0.5) * 2


cpdef double correlation_test(double[::1] x, double[::1] y):
    cdef:
        size_t n
        double result

    n = len(x)
    result = correlation(&x[0], &y[0], n)
    return result

# Consider replacing this with a single pass algorith, preferrably inlined
# into the calling function:
# https://en.wikipedia.org/wiki/Pearson_correlation_coefficient#For_a_sample
# https://stats.stackexchange.com/a/561425/59613
@cython.boundscheck(False)
@cython.cdivision(True)
cdef double correlation(double* x, double* y, size_t n) nogil:
    cdef:
        size_t i
        double xsum, ysum, xmean, ymean
        double cov, varx, vary
    if n == 0:
        return NAN
    xsum = 0.
    ysum = 0.
    for i in range(n):
        xsum += x[i]
        ysum += y[i]
    xmean = xsum / n
    ymean = ysum / n
    cov = 0.
    # The denominator shouldn't initialize to zero if all elements of x or
    # y could be equal to their mean, otherwise we get division by zero.
    varx = 1e-12
    vary = 1e-12
    for i in range(n):
        cov += (x[i] - xmean) * (y[i] - ymean)
        varx += (x[i] - xmean) ** 2
        vary += (y[i] - ymean) ** 2
    return cov / (sqrt(varx) * sqrt(vary))


@cython.binding(True)
@cython.boundscheck(False)
@cython.cdivision(True)
def calculate_peptide_score(self, double error_tolerance=2e-5, bint use_reliability=True, double base_reliability=0.5,
                            double coverage_weight=1.0, **kwargs):
    cdef:
        list c, backbones
        tuple coverage_result
        np.ndarray[np.float64_t, ndim=1] intens, yhat, reliability
        np.ndarray[np.float64_t, ndim=1, mode='c'] n_term_ions, c_term_ions
        double t, coverage_score, normalizer
        double corr_score, corr, peptide_score
        double reliability_sum, rel_i
        double temp
        double* intens_
        double* yhat_
        size_t i, n
        _FragmentType ci
        BackbonePosition pos
        FragmentBase frag
        IonSeriesBase series
        _PeptideSequenceCore target

    c, intens, t, yhat = self._get_predicted_intensities()
    if self.model_fit.reliability_model is None or not use_reliability:
        reliability = np.ones_like(yhat)
    else:
        reliability = self._get_reliabilities(c, base_reliability=base_reliability)
    backbones = []
    n = PyList_GET_SIZE(c)
    for i in range(n):
        ci = <_FragmentType>PyList_GET_ITEM(c, i)
        if ci.is_assigned():
            frag = ci.get_fragment()
            series = frag.get_series()
            if (series.int_code == IonSeries_b.int_code or
                 series.int_code == IonSeries_y.int_code or
                 series.int_code == IonSeries_c.int_code or
                 series.int_code == IonSeries_z.int_code):
                backbones.append(
                    BackbonePosition._create(
                        ci, intens[i] / t, yhat[i], reliability[i]))
    n = PyList_GET_SIZE(backbones)
    if n == 0:
        return 0

    peptide_score = 0.0
    reliability_sum = 0.0

    intens_ = <double*>PyMem_Malloc(sizeof(double) * n)
    yhat_ = <double*>PyMem_Malloc(sizeof(double) * n)
    for i in range(n):
        pos = <BackbonePosition>PyList_GET_ITEM(backbones, i)
        intens_[i] = pos.intensity
        yhat_[i] = pos.predicted
        # delta_i = (intens_[i] - yhat_[i]) ** 2
        # if intens_[i] > yhat_[i]:
        #     delta_i /= 2
        # denom_i = yhat_[i] * (1 - yhat_[i]) * pos.reliability
        # pearson_peptide_score = -log10(PearsonResidualCDF.interpolate_scalar(delta_i / denom_i) + 1e-6)
        # if isnan(pearson_peptide_score):
        #     pearson_peptide_score = 0.0

        temp = log10(intens_[i] * t)
        temp *= 1 - abs(pos.match.peak_pair.mass_accuracy() / error_tolerance) ** 4
        rel_i = unpad(pos.reliability, base_reliability)
        temp *= rel_i + 1.0
        reliability_sum += rel_i
        # the 0.17 term ensures that the maximum value of the -log10 transform of the cdf is
        # mapped to approximately 1.0 (1.02). The maximum value is guaranteed to 6.0 because
        # the minimum value returned from the CDF is 0 + 1e-6 padding, which maps to 6.
        # temp *= (0.17 * pearson_peptide_score)
        peptide_score += temp

    # peptide reliability is usually less powerful, so it does not benefit
    # us to use the normalized correlation coefficient here
    corr = correlation(intens_, yhat_, n)
    if isnan(corr):
        corr = -0.5

    # peptide fragment correlation is weaker than the glycan correlation.
    corr = (1.0 + corr) / 2.0
    corr_score = corr * 2.0 * log10(n)

    target = <_PeptideSequenceCore>self.target
    coverage_score = self._calculate_peptide_coverage()

    PyMem_Free(intens_)
    PyMem_Free(yhat_)
    peptide_score += corr_score
    peptide_score += reliability_sum
    peptide_score *= coverage_score ** coverage_weight
    return peptide_score


@cython.binding(True)
@cython.boundscheck(False)
@cython.cdivision(True)
def calculate_partial_glycan_score(self, double error_tolerance=2e-5, bint use_reliability=True, double base_reliability=0.5,
                                   double core_weight=0.4, double coverage_weight=0.6, fragile_fucose=False, extended_glycan_search=True, **kwargs):
    cdef:
        list c, stubs
        np.ndarray[np.float64_t, ndim=1] intens, yhat, reliability
        size_t i, n, n_signif_frags
        _FragmentType ci
        FragmentBase frag
        double* reliability_
        double* intens_
        double* yhat_
        double oxonium_component, coverage, glycan_prior
        double glycan_score, temp, t
        double corr, corr_score, reliability_sum
        double peptide_coverage

    c, intens, t, yhat = self._get_predicted_intensities()
    if self.model_fit.reliability_model is None or not use_reliability:
        reliability = np.ones_like(yhat)
    else:
        reliability = self._get_reliabilities(c, base_reliability=base_reliability)
    stubs = []
    n = PyList_GET_SIZE(c)
    n_signif_frags = 0
    for i in range(n):
        ci = <_FragmentType>PyList_GET_ITEM(c, i)
        if ci.is_assigned():
            frag = ci.get_fragment()

            if frag.get_series().int_code == IonSeries_stub_glycopeptide.int_code:
                if (<StubFragment>frag).get_glycosylation_size() > 1:
                    n_signif_frags += 1
                stubs.append(
                    BackbonePosition._create(
                        ci, intens[i] / t, yhat[i], reliability[i]))
    n = PyList_GET_SIZE(stubs)
    if n == 0:
        return 0
    glycan_score = 0.0
    intens_ = <double*>PyMem_Malloc(sizeof(double) * n)
    yhat_ = <double*>PyMem_Malloc(sizeof(double) * n)
    reliability_sum = 0.0
    for i in range(n):
        pos = <BackbonePosition>PyList_GET_ITEM(stubs, i)
        intens_[i] = pos.intensity
        yhat_[i] = pos.predicted
        reliability_sum += pos.reliability

        temp = log10(intens_[i] * t)
        temp *= 1 - abs(pos.match.peak_pair.mass_accuracy() / error_tolerance) ** 4
        glycan_score += temp

    if n > 1:
        corr = correlation(intens_, yhat_, n)
        if isnan(corr):
            corr = -0.5

    else:
        corr = -0.5

    peptide_coverage = self._calculate_peptide_coverage()
    corr = (1 + corr) / 2
    corr_score = corr * n_signif_frags + reliability_sum * n_signif_frags

    # corr_score *= min(peptide_coverage + 0.75, 1.0)
    # corr_score *= normalized_sigmoid(max(peptide_coverage - 0.03, 0.0) * 42)
    # corr_score *= shifted_normalized_sigmoid_erf(peptide_coverage)
    corr_score *= min(exp(peptide_coverage * 3) - 1, 1)

    glycan_prior = 0.0
    # oxonium_component = self._signature_ion_score()
    coverage = self._calculate_glycan_coverage(
        core_weight, coverage_weight, fragile_fucose=fragile_fucose,
        extended_glycan_search=extended_glycan_search)
    if coverage > 0:
        glycan_prior = self.target.glycan_prior
    glycan_score = (glycan_score + corr_score + glycan_prior) * coverage #+ oxonium_component

    PyMem_Free(intens_)
    PyMem_Free(yhat_)
    return max(glycan_score, 0)


@cython.binding(True)
@cython.boundscheck(False)
@cython.cdivision(True)
cpdef _calculate_pearson_residuals(self, bint use_reliability=True, double base_reliability=0.5):
    r"""Calculate the raw Pearson residuals of the Multinomial model

    .. math::
        \frac{y - \hat{y}}{\hat{y} * (1 - \hat{y}) * r}

    Parameters
    ----------
    use_reliability : bool, optional
        Whether or not to use the fragment reliabilities to adjust the weight of
        each matched peak
    base_reliability : float, optional
        The lowest reliability a peak may have, compressing the range of contributions
        from the model based on the experimental evidence

    Returns
    -------
    np.ndarray
        The Pearson residuals
    """
    cdef:
        list c
        np.ndarray[np.float64_t, ndim=1, mode='c'] intens, yhat, reliability
        np.ndarray[np.float64_t, ndim=1, mode='c'] pearson_residuals
        double t
        double intens_i_norm, delta_i, denom_i
        size_t i, n
        np.npy_intp knd
    c, intens, t, yhat = self._get_predicted_intensities()
    if self.model_fit.reliability_model is None or not use_reliability:
        reliability = np.ones_like(yhat)
    else:
        reliability = self._get_reliabilities(c, base_reliability=base_reliability)
    # the last positionis the unassigned term, ignore it
    n = PyList_GET_SIZE(c) - 1
    knd = n
    pearson_residuals = np.PyArray_ZEROS(1, &knd, np.NPY_FLOAT64, 0)
    for i in range(n):
        # standardize intensity
        intens_i_norm = intens[i] / t
        delta_i = (intens_i_norm - yhat[i]) ** 2
        # reduce penalty for exceeding predicted intensity
        if (intens_i_norm > yhat[i]):
            delta_i /= 2.
        denom_i = yhat[i] * (1 - yhat[i]) * reliability[i]
        pearson_residuals[i] = (delta_i / denom_i)
    return pearson_residuals


@cython.cdivision(True)
cpdef double classify_ascending_abundance_peptide_Y(spectrum_match):
    cdef:
        double abundance, ratio
        size_t size, total_size
        FragmentMatchMap solution_map
        PeakFragmentPair pfp
        FragmentBase frag
        StubFragment stub
        _PeptideSequenceCore target
        IonSeriesBase series

    solution_map = <FragmentMatchMap?>spectrum_match.solution_map
    target = <_PeptideSequenceCore?>spectrum_match.target
    if target._glycosylation_manager is None:
        return 0
    size = 0
    abundance = 0
    for obj in solution_map.members:
        pfp = <PeakFragmentPair>obj
        frag = <FragmentBase>pfp.fragment
        if frag.get_series().int_code == IonSeries_stub_glycopeptide.int_code:
            stub = <StubFragment>frag
            if pfp.peak.intensity > abundance:
                size = stub.get_glycosylation_size()
                abundance = pfp.peak.intensity

    total_size = (<GlycosylationManager?>target._glycosylation_manager).get_total_glycosylation_size()
    ratio = (<double>size) / (<double>total_size)
    return ratio


@cython.boundscheck(False)
@cython.cdivision(True)
cpdef np.ndarray[np.float64_t, ndim=1, mode='c'] predict_yhat_from_feature_and_coefs(
                                                    np.ndarray[np.uint8_t, ndim=2, mode='c'] features,
                                                    np.ndarray[np.float64_t, ndim=1, mode='c'] coefs,
                                                    long n_threads=1):
    cdef:
        Py_ssize_t n, i, num_threads
        size_t m, k, j
        np.uint8_t[:, ::1] features_view
        np.float64_t[::1] coefs_view, out_view
        np.ndarray[np.float64_t, ndim=1, mode='c'] out
        double acc, total
        np.npy_intp knd


    n = features.shape[0]
    m = features.shape[1]
    k = coefs.shape[0]

    knd = n

    out = <np.ndarray[np.float64_t, ndim=1, mode='c']>np.PyArray_EMPTY(1, &knd, np.NPY_FLOAT64, 0)
    if n == 0:
        return out
    num_threads = n_threads
    features_view = features
    coefs_view = coefs
    out_view = out
    total = 0
    with nogil:
        for i in prange(n, num_threads=num_threads, schedule='static'):
            acc = 0
            for j in range(m):
                acc += features_view[i, j] * coefs_view[j]
            acc = exp(acc)
            total += acc
            out_view[i] = acc
        total += 1
        for i in prange(n, num_threads=num_threads, schedule='static'):
            out_view[i] /= total
    return out


@cython.binding(True)
@cython.boundscheck(False)
@cython.cdivision(True)
def calculate_peptide_score_no_glycosylation(self, double error_tolerance=2e-5, bint use_reliability=True, double base_reliability=0.5,
                                             double coverage_weight=0.7, **kwargs):
    cdef:
        list c, backbones
        tuple coverage_result
        np.ndarray[np.float64_t, ndim=1] intens, yhat, reliability
        np.ndarray[np.float64_t, ndim=1, mode='c'] n_term_ions, c_term_ions
        double t, coverage_score, normalizer
        double corr_score, corr, peptide_score
        double weighted_coverage_scaler
        double reliability_sum, rel_i
        double temp
        double* intens_
        double* yhat_
        size_t i, n
        _FragmentType ci
        BackbonePosition pos
        FragmentBase frag
        IonSeriesBase series
        _PeptideSequenceCore target


    c, intens, t, yhat = self._get_predicted_intensities()
    if self.model_fit.reliability_model is None:
        use_reliability = False
    if not use_reliability:
        reliability = None
    else:
        reliability = self._get_reliabilities(c, base_reliability=base_reliability)
    backbones = []
    coverage_score = self._calculate_peptide_coverage_no_glycosylated()
    n = PyList_GET_SIZE(c)
    for i in range(n):
        ci = <_FragmentType>PyList_GET_ITEM(c, i)
        if ci.is_assigned():
            frag = ci.get_fragment()
            series = frag.get_series()
            if (series.int_code == IonSeries_b.int_code or
                series.int_code == IonSeries_y.int_code or
                series.int_code == IonSeries_c.int_code or
                series.int_code == IonSeries_z.int_code):
                backbones.append(
                    BackbonePosition._create(
                        ci, intens[i] / t, yhat[i], reliability[i] if use_reliability else 1.0))
    n = PyList_GET_SIZE(backbones)
    if n == 0:
        return 0

    peptide_score = 0.0
    reliability_sum = 0.0

    intens_ = <double*>PyMem_Malloc(sizeof(double) * n)
    yhat_ = <double*>PyMem_Malloc(sizeof(double) * n)
    for i in range(n):
        pos = <BackbonePosition>PyList_GET_ITEM(backbones, i)
        intens_[i] = pos.intensity
        yhat_[i] = pos.predicted

        frag = pos.match.get_fragment()
        if (<PeptideFragment>frag)._is_glycosylated():
            continue

        temp = log10(intens_[i] * t)
        temp *= 1 - abs(pos.match.peak_pair.mass_accuracy() / error_tolerance) ** 4
        rel_i = unpad(pos.reliability, base_reliability)
        temp *= rel_i + 1.0
        reliability_sum += rel_i
        peptide_score += temp

    corr = correlation(intens_, yhat_, n)
    if isnan(corr):
        corr = -0.5

    # peptide fragment correlation is weaker than the glycan correlation.
    corr_score = peptide_correlation_score1(corr, n)

    PyMem_Free(intens_)
    PyMem_Free(yhat_)
    peptide_score += corr_score
    peptide_score += reliability_sum
    peptide_score *= coverage_score ** coverage_weight
    return peptide_score


cdef double peptide_correlation_score1(double corr, long n):
    cdef:
        double corr_score
    corr = (1.0 + corr) / 2.0
    corr_score = corr * 2.0 * log10(n)
    return corr_score


cdef double peptide_correlation_score2(double corr, long n):
    cdef:
        double corr_score

    if corr <= 0:
        corr_score = 0
    else:
        if corr > (1 - 1e-3):
            corr = 1 - 1e-3
        corr_score = -np.log(1 - corr)
        corr_score *= n
    return corr_score


@cython.binding(True)
@cython.boundscheck(False)
@cython.cdivision(True)
def calculate_partial_glycan_score_no_glycosylated_peptide_coverage(self, double error_tolerance=2e-5, bint use_reliability=True, double base_reliability=0.5,
                                                                    double core_weight=0.4, double coverage_weight=0.6, fragile_fucose=False, extended_glycan_search=True, **kwargs):
    cdef:
        list c, stubs
        np.ndarray[np.float64_t, ndim=1] intens, yhat, reliability
        size_t i, n, n_signif_frags
        _FragmentType ci
        FragmentBase frag
        double* reliability_
        double* intens_
        double* yhat_
        double oxonium_component, coverage, glycan_prior
        double glycan_score, temp, t
        double corr, corr_score, reliability_sum
        double peptide_coverage

    c, intens, t, yhat = self._get_predicted_intensities()
    if self.model_fit.reliability_model is None or not use_reliability:
        reliability = np.ones_like(yhat)
    else:
        reliability = self._get_reliabilities(c, base_reliability=base_reliability)
    stubs = []
    n = PyList_GET_SIZE(c)
    n_signif_frags = 0
    for i in range(n):
        ci = <_FragmentType>PyList_GET_ITEM(c, i)
        if ci.is_assigned():
            frag = ci.get_fragment()

            if frag.get_series().int_code == IonSeries_stub_glycopeptide.int_code:
                if (<StubFragment>frag).get_glycosylation_size() > 1:
                    n_signif_frags += 1
                stubs.append(
                    BackbonePosition._create(
                        ci, intens[i] / t, yhat[i], reliability[i]))
    n = PyList_GET_SIZE(stubs)
    if n == 0:
        return 0
    glycan_score = 0.0
    intens_ = <double*>PyMem_Malloc(sizeof(double) * n)
    yhat_ = <double*>PyMem_Malloc(sizeof(double) * n)
    reliability_sum = 0.0
    for i in range(n):
        pos = <BackbonePosition>PyList_GET_ITEM(stubs, i)
        intens_[i] = pos.intensity
        yhat_[i] = pos.predicted
        reliability_sum += pos.reliability

        temp = log10(intens_[i] * t)
        temp *= 1 - abs(pos.match.peak_pair.mass_accuracy() / error_tolerance) ** 4
        glycan_score += temp

    if n > 1:
        corr = correlation(intens_, yhat_, n)
        if isnan(corr):
            corr = -0.5

    else:
        corr = -0.5

    if n_signif_frags > 0:
        peptide_coverage = self._calculate_peptide_coverage_no_glycosylated()
        corr_score = (((1 + corr) / 2) * n_signif_frags) + (reliability_sum * (n_signif_frags) * max(corr, 0.25))

        # corr_score *= min(peptide_coverage + 0.75, 1.0)
        # corr_score *= normalized_sigmoid(max(peptide_coverage - 0.03, 0.0) * 42)
        # corr_score *= shifted_normalized_sigmoid_erf(peptide_coverage)
        corr_score *= min(exp(peptide_coverage * 3) - 1, 1)
    else:
        corr_score = 0

    glycan_prior = 0.0
    # oxonium_component = self._signature_ion_score()
    coverage = self._calculate_glycan_coverage(
        core_weight, coverage_weight, fragile_fucose=fragile_fucose,
        extended_glycan_search=extended_glycan_search)
    if coverage > 0:
        glycan_prior = self.target.glycan_prior
    glycan_score = (glycan_score + corr_score + glycan_prior) * coverage #+ oxonium_component

    PyMem_Free(intens_)
    PyMem_Free(yhat_)
    return max(glycan_score, 0)