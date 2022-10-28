
cimport cython

from ms_deisotope._c.peak_set cimport DeconvolutedPeak, DeconvolutedPeakSet

from glycopeptidepy._c.structure.base cimport (AminoAcidResidueBase, ModificationBase)
from glycopeptidepy._c.structure.fragment cimport (FragmentBase, PeptideFragment, IonSeriesBase)

from glycan_profiling._c.structure.fragment_match_map cimport (PeakFragmentPair, FragmentMatchMap)


cpdef set get_peak_index(FragmentMatchMap self)


cdef class FeatureBase(object):
    cdef:
        public str name
        public double tolerance
        public double intensity_ratio
        public int from_charge
        public int to_charge
        public object feature_type
        public object terminal

    cpdef list find_matches(self, DeconvolutedPeak peak, DeconvolutedPeakSet peak_list, object structure=*)
    cpdef bint is_valid_match(self, DeconvolutedPeak from_peak, DeconvolutedPeak to_peak,
                              FragmentMatchMap solution_map, structure=*, set peak_indices=*)


cdef class MassOffsetFeature(FeatureBase):

    cdef:
        public double offset
        public Py_hash_t _hash

    cpdef bint test(self, DeconvolutedPeak peak1, DeconvolutedPeak peak2)



cpdef list ComplementFeature_find_matches(self, DeconvolutedPeak peak, DeconvolutedPeakSet peak_list, object structure=*)
cpdef bint LinkFeature_is_valid_match(MassOffsetFeature self, DeconvolutedPeak from_peak, DeconvolutedPeak to_peak,
                                      FragmentMatchMap solution_map, structure=*, set peak_indices=*) except *


cdef class FeatureFunctionEstimatorBase(object):
    cdef:
        public FeatureBase feature_function
        public IonSeriesBase series
        public double tolerance
        public bint prepranked
        public bint track_relations
        public bint verbose
        public double total_on_series_satisfied
        public double total_off_series_satisfied
        public double total_on_series
        public double total_off_series
        public list peak_relations

    cpdef match_peaks(self, gpsm, DeconvolutedPeakSet peaks)


cdef class FittedFeatureBase(object):
    cdef:
        public FeatureBase feature
        public int from_charge
        public int to_charge
        public IonSeriesBase series
        public double on_series
        public double off_series

        public long on_count
        public long off_count
        public list relations


    cpdef list find_matches(self, DeconvolutedPeak peak, DeconvolutedPeakSet peak_list, structure=*)
    cpdef bint is_valid_match(self, DeconvolutedPeak from_peak, DeconvolutedPeak to_peak,
                              FragmentMatchMap solution_map, structure=*, set peak_indices=*)
    cpdef double _feature_probability(self, double p=*)


cdef class FragmentationFeatureBase(object):
    cdef:
        public FeatureBase feature
        public IonSeriesBase series
        public dict fits

    cpdef list find_matches(self, DeconvolutedPeak peak, DeconvolutedPeakSet peak_list, structure=*)
    cpdef bint is_valid_match(self, DeconvolutedPeak from_peak, DeconvolutedPeak to_peak,
                              FragmentMatchMap solution_map, structure=*, set peak_indices=*)


cdef class FragmentationModelBase(object):
    cdef:
        public IonSeriesBase series
        public list features
        public list feature_table
        public double error_tolerance
        public double on_frequency
        public double off_frequency
        public double prior_probability_of_match
        public double offset_probability

    cdef size_t get_size(self)
    cpdef find_matches(self, scan, FragmentMatchMap solution_map, structure)
    cpdef double _score_peak(self, DeconvolutedPeak peak, list matched_features, FragmentMatchMap solution_map, structure)


cdef class FragmentationModelCollectionBase(object):
    cdef:
        public dict models

    cpdef dict find_matches(self, scan, FragmentMatchMap solution_map, structure)
    cpdef dict score(self, scan, FragmentMatchMap solution_map, structure)


cdef class PeakRelation(object):
    cdef:
        public DeconvolutedPeak from_peak
        public DeconvolutedPeak to_peak
        public int intensity_ratio
        public object feature
        public object series
        public int from_charge
        public int to_charge

    cpdef tuple peak_key(self)

    @staticmethod
    cdef PeakRelation _create(DeconvolutedPeak from_peak, DeconvolutedPeak to_peak, feature, IonSeriesBase series)