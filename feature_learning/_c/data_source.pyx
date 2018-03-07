import operator

cimport cython

import numpy as np
cimport numpy as np

np.import_array()

from ms_deisotope._c.averagine cimport neutral_mass
from ms_deisotope._c.peak_set cimport DeconvolutedPeak, DeconvolutedPeakSetIndexed, _Index, Envelope
from ms_peak_picker._c.peak_set cimport PeakBase


cdef class RankedPeak(DeconvolutedPeak):
    cdef:
        public int rank

    def __init__(self, neutral_mass, intensity, charge, signal_to_noise, index,
                 rank=-1):
        DeconvolutedPeak.__init__(
            self, neutral_mass, intensity, charge, signal_to_noise, index, 0)
        self.rank = rank

    def __reduce__(self):
        return self.__class__, (self.neutral_mass, self.intensity, self.charge,
                                self.signal_to_noise, self.index, self.rank)

    cpdef PeakBase clone(self):
        return self.__class__(self.neutral_mass, self.intensity, self.charge,
                              self.signal_to_noise, self.index, self.rank)

    def __repr__(self):
        return "RankedPeak(%0.2f, %0.2f, %d, %0.2f, %s, %d)" % (
            self.neutral_mass, self.intensity, self.charge,
            self.signal_to_noise, self.index, self.rank)

    @staticmethod
    cdef DeconvolutedPeak _create_simple(double neutral_mass, double intensity, int charge,
                                         double score, double mz, Envelope envelope):
        cdef:
            DeconvolutedPeak inst
        inst = RankedPeak.__new__(RankedPeak)
        inst.neutral_mass = neutral_mass
        inst.intensity = intensity
        inst.charge = charge
        inst.score = score
        inst.signal_to_noise = score
        inst.mz = mz
        inst.envelope = envelope
        inst.index = _Index._create(0, 0)
        inst.full_width_at_half_max = 0
        inst.a_to_a2_ratio = 0
        inst.most_abundant_mass = 0
        inst.average_mass = 0
        inst.area = 0
        inst.rank = -1

        return inst


cdef object get_intensity = operator.attrgetter('intensity')


cpdef intensity_rank(object peak_list, double minimum_intensity=100.):
    cdef:
        list peaks
        RankedPeak p
        int i, rank, tailing
        size_t j, n
    peaks = sorted(peak_list, key=get_intensity, reverse=True)
    i = 0
    rank = 10
    tailing = 6
    n = len(peaks)
    for j in range(n):
        p = <RankedPeak>peaks[j]
        if p.intensity < minimum_intensity:
            p.rank = -1
            continue
        i += 1
        if i == 10 and rank != 0:
            if rank == 1:
                if tailing != 0:
                    i = 0
                    tailing -= 1
                else:
                    i = 0
                    rank -= 1
            else:
                i = 0
                rank -= 1
        if rank == 0:
            break
        p.rank = rank


@cython.boundscheck(False)
cpdef DeconvolutedPeakSetIndexed build_deconvoluted_peak_set_from_arrays(np.ndarray[double, ndim=1] mz_array,
                                                                         np.ndarray[double, ndim=1] intensity_array,
                                                                         np.ndarray[long, ndim=1] charge_array):
    cdef:
        list peaks
        size_t n, i
        double mz, peak_neutral_mass
        int charge
        DeconvolutedPeak peak
        DeconvolutedPeakSetIndexed peak_set

    peaks = []
    n = mz_array.shape[0]
    i = 0
    for i in range(n):
        mz = mz_array[i]
        charge = charge_array[i]
        peak_neutral_mass = neutral_mass(mz, charge)

        peak = RankedPeak._create_simple(
            peak_neutral_mass,
            intensity_array[i],
            charge_array[i],
            intensity_array[i],
            mz,
            None)
        peaks.append(peak)
    peak_set = DeconvolutedPeakSetIndexed(peaks)
    peak_set.reindex()
    return peak_set