from ms_deisotope._c.peak_set cimport DeconvolutedPeak, DeconvolutedPeakSetIndexed

cimport numpy as np

np.import_array()


cdef class RankedPeak(DeconvolutedPeak):
    cdef:
        public int rank


cpdef intensity_rank(object peak_list, double minimum_intensity=*)


cpdef DeconvolutedPeakSetIndexed build_deconvoluted_peak_set_from_arrays(np.ndarray[double, ndim=1] mz_array,
                                                                         np.ndarray[double, ndim=1] intensity_array,
                                                                         np.ndarray[long, ndim=1] charge_array)