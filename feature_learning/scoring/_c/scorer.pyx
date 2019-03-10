# cython: embedsignature=True
# distutils: include_dirs = C:\Users\Asus\Miniconda2\lib\site-packages\numpy\core\include
cimport cython
from cpython.mem cimport PyMem_Malloc, PyMem_Free
from cpython cimport PyTuple_GetItem, PyTuple_Size, PyList_GET_ITEM, PyList_GET_SIZE
from cpython.int cimport PyInt_AsLong
from libc.stdlib cimport malloc, calloc, free
from libc.math cimport log10, log, sqrt, exp

import numpy as np
cimport numpy as np

np.import_array()

from numpy.math cimport isnan, NAN

from ms_deisotope._c.peak_set cimport DeconvolutedPeak, DeconvolutedPeakSet

from glycan_profiling._c.structure.fragment_match_map cimport (
    FragmentMatchMap, PeakFragmentPair)

from glycopeptidepy._c.structure.base cimport AminoAcidResidueBase, SequencePosition
from glycopeptidepy._c.structure.sequence_methods cimport _PeptideSequenceCore
from glycopeptidepy._c.structure.fragment cimport PeptideFragment, FragmentBase, IonSeriesBase, ChemicalShiftBase

from feature_learning._c.model_types cimport _FragmentType
from feature_learning._c.approximation cimport StepFunction

from feature_learning.multinomial_regression import (
    PearsonResidualCDF as _PearsonResidualCDF)

cdef StepFunction PearsonResidualCDF = _PearsonResidualCDF


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
    varx = 0.
    vary = 0.
    for i in range(n):
        cov += (x[i] - xmean) * (y[i] - ymean)
        varx += (x[i] - xmean) ** 2
        vary += (y[i] - ymean) ** 2
    return cov / (sqrt(varx) * sqrt(vary))


@cython.binding(True)
@cython.boundscheck(False)
@cython.cdivision(True)
def calculate_peptide_score(self, double error_tolerance=2e-5, bint use_reliability=True, double base_reliability=0.5,
                            double coverage_weight=1.0, *args, **kwargs):
    cdef:
        list c, backbones
        tuple coverage_result
        np.ndarray[np.float64_t, ndim=1] intens, yhat, reliability
        np.ndarray[np.float64_t, ndim=1] n_term_ions, c_term_ions
        double t, coverage_score, normalizer
        double corr_score, corr, peptide_score
        double temp, delta_i, denom_i
        double* peptide_score_vec
        double* reliability_
        double* intens_
        double* yhat_
        size_t i, n, size, n_isnan
        _FragmentType ci
        np.npy_intp knd
        BackbonePosition pos
        _PeptideSequenceCore target

    c, intens, t, yhat = self._get_predicted_intensities()
    if self.model_fit.reliability_model is None or not use_reliability:
        reliability = np.ones_like(yhat)
    else:
        reliability = self._get_reliabilities(c, base_reliability=base_reliability)
    series_set = ('b', 'y')
    backbones = []
    n = PyList_GET_SIZE(c)
    for i in range(n):
        intens[i] = intens[i]
        ci = <_FragmentType>PyList_GET_ITEM(c, i)
        if ci.series in series_set:
            backbones.append(
                BackbonePosition._create(
                    ci, intens[i] / t, yhat[i], reliability[i]))
    n = PyList_GET_SIZE(backbones)
    if n == 0:
        return 0

    knd = n
    c = []
    # intens = np.PyArray_ZEROS(1, &knd, np.NPY_FLOAT64, 0)
    # yhat = np.PyArray_ZEROS(1, &knd, np.NPY_FLOAT64, 0)
    # reliability = np.PyArray_ZEROS(1, &knd, np.NPY_FLOAT64, 0)
    intens_ = <double*>PyMem_Malloc(sizeof(double) * n)
    yhat_ = <double*>PyMem_Malloc(sizeof(double) * n)
    reliability_ = <double*>PyMem_Malloc(sizeof(double) * n)

    for i in range(n):
        pos = <BackbonePosition>PyList_GET_ITEM(backbones, i)
        c.append(pos.match)
        intens_[i] = pos.intensity
        yhat_[i] = pos.predicted
        reliability_[i] = pos.reliability

    # peptide reliability is usually less powerful, so it does not benefit
    # us to use the normalized correlation coefficient here
    corr = correlation(intens_, yhat_, n)
    if isnan(corr):
        corr = -0.5
    # peptide fragment correlation is weaker than the overall correlation.
    corr = (1.0 + corr) / 2.0
    corr_score = corr * 2.0 * log10(n)

    # peptide_score_vec = np.PyArray_ZEROS(1, &knd, np.NPY_FLOAT64, 0)
    peptide_score_vec = <double*>PyMem_Malloc(sizeof(double) * n)

    n_isnan = 0
    for i in range(n):
        delta_i = (intens_[i] - yhat_[i]) ** 2
        if intens_[i] > yhat_[i]:
            delta_i /= 2
        denom_i = yhat_[i] * (1 - yhat_[i]) * reliability_[i]
        peptide_score_vec[i] = -log10(PearsonResidualCDF.interpolate_scalar(delta_i / denom_i) + 1e-6)
        n_isnan += isnan(peptide_score_vec[i])

    if n_isnan == n:
        for i in range(n):
            peptide_score_vec[i] = 0.0

    target = <_PeptideSequenceCore>self.target
    size = target.get_size()
    normalizer = (2 * (size - 1))

    # peptide backbone coverage without separate term for glycosylation site parsimony
    coverage_result = self._compute_coverage_vectors()
    n_term_ions = <np.ndarray[np.float64_t, ndim=1]>PyTuple_GetItem(coverage_result, 0)
    c_term_ions = <np.ndarray[np.float64_t, ndim=1]>PyTuple_GetItem(coverage_result, 1)

    coverage_score = 0.0
    for i in range(size):
        coverage_score += n_term_ions[i] + c_term_ions[size - i - 1]
    coverage_score /= normalizer

    peptide_score = 0.0
    for i in range(n):
        ci = <_FragmentType>PyList_GET_ITEM(c, i)
        temp = log10(intens_[i] * t) 
        temp *= 1 - abs(ci.peak_pair.mass_accuracy() / error_tolerance) ** 4
        temp *= unpad(reliability_[i], base_reliability) + 0.75
        # the 0.17 term ensures that the maximum value of the -log10 transform of the cdf is
        # mapped to approximately 1.0 (1.02). The maximum value is guaranteed to 6.0 because
        # the minimum value returned from the CDF is 0 + 1e-6 padding, which maps to 6.
        temp *= (0.17 * peptide_score_vec[i])
        peptide_score += temp
    PyMem_Free(peptide_score_vec)
    PyMem_Free(reliability_)
    PyMem_Free(intens_)
    PyMem_Free(yhat_)
    peptide_score += corr_score
    peptide_score *= coverage_score ** coverage_weight
    return peptide_score
