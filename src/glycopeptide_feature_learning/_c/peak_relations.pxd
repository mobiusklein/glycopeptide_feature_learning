
cimport cython

from ms_deisotope._c.peak_set cimport DeconvolutedPeak, DeconvolutedPeakSet

from glycopeptidepy._c.structure.base cimport (AminoAcidResidueBase, ModificationBase)
from glycopeptidepy._c.structure.sequence_methods cimport _PeptideSequenceCore
from glycopeptidepy._c.structure.fragment cimport (FragmentBase, PeptideFragment, IonSeriesBase)

from glycresoft._c.structure.fragment_match_map cimport (PeakFragmentPair, FragmentMatchMap)

from numpy cimport npy_uint32 as uint32_t, npy_uint16 as uint16_t, npy_int16 as int16_t, npy_int8 as int8_t


cpdef set get_peak_index(FragmentMatchMap self)

cpdef int8_t[::1] make_index(match)


cdef class TargetProperties:
    cdef:
        public double peptide_backbone_mass

    @staticmethod
    cdef TargetProperties from_glycopeptide(_PeptideSequenceCore glycopeptide)


cdef class FeatureBase(object):
    cdef:
        public str name
        public double tolerance
        public double intensity_ratio
        public int from_charge
        public int to_charge
        public object feature_type
        public object terminal

    cpdef list find_matches(self, DeconvolutedPeak peak, DeconvolutedPeakSet peak_list, object structure=*, TargetProperties props=*)
    cpdef bint is_valid_match(self, size_t from_peak, size_t to_peak,
                              FragmentMatchMap solution_map, structure=*, set peak_indices=*)


cdef class MassOffsetFeature(FeatureBase):

    cdef:
        public double offset
        public Py_hash_t _hash

    cpdef bint test(self, DeconvolutedPeak peak1, DeconvolutedPeak peak2)
    cdef inline bint _test(self, DeconvolutedPeak peak1, DeconvolutedPeak peak2) nogil


cdef class LinkFeature(MassOffsetFeature):
    cdef:
        public object _amino_acid
        public AminoAcidResidueBase _amino_acid_residue

    cdef inline bint _amino_acid_in_fragment(self, PeptideFragment fragment)
    cdef inline bint _amino_acid_in_list(self, list aas)

    cpdef bint amino_acid_in_fragment(self, PeptideFragment fragment)
    cpdef bint is_valid_match(self, size_t from_peak, size_t to_peak,
                              FragmentMatchMap solution_map, structure=*, set peak_indices=*)

cdef class ComplementFeature(MassOffsetFeature):
    cdef inline bint _test_relative(self, DeconvolutedPeak peak1, DeconvolutedPeak peak2, double reference_mass) nogil


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

    cpdef match_peaks(self, gpsm, DeconvolutedPeakSet peaks, int min_rank=*)


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


    cpdef list find_matches(self, DeconvolutedPeak peak, DeconvolutedPeakSet peak_list, structure=*, TargetProperties props=*)
    cpdef bint is_valid_match(self, size_t from_peak, size_t to_peak,
                              FragmentMatchMap solution_map, structure=*, set peak_indices=*)
    cpdef double _feature_probability(self, double p=*)


cdef class FragmentationFeatureBase(object):
    cdef:
        public FeatureBase feature
        public IonSeriesBase series
        public dict fits

    cpdef list find_matches(self, DeconvolutedPeak peak, DeconvolutedPeakSet peak_list, structure=*, TargetProperties props=*)
    cpdef bint is_valid_match(self, size_t from_peak, size_t to_peak,
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
    cpdef find_matches(self, scan, FragmentMatchMap solution_map, structure, TargetProperties props=*)
    cpdef double _score_peak(self, DeconvolutedPeak peak, list matched_features, FragmentMatchMap solution_map, structure)


cdef class FragmentationModelCollectionBase(object):
    cdef:
        public dict models

    cpdef dict find_matches(self, scan, FragmentMatchMap solution_map, structure, TargetProperties props=*)
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


cdef struct partition_t:
    int16_t from_charge
    int16_t to_charge
    int intensity_ratio


cdef struct feature_fit_t:
    partition_t partition

    uint32_t on_count
    uint32_t off_count

    double on_series
    double off_series


cdef struct peak_relation_t:
    size_t from_peak
    size_t to_peak

    int8_t series
    feature_fit_t* feature


cdef struct feature_table_t:
    feature_fit_t* features
    size_t size


cdef struct partitioned_peak_relation_t:
    size_t from_peak
    size_t to_peak

    int8_t series
    feature_fit_t* feature

    partition_t partition


cdef struct partitioned_fit_table_t:
    feature_fit_t* fits
    uint16_t p1_charge_max
    uint16_t p2_charge_max
    size_t size


cdef int create_partitioned_fit_table(uint16_t p1_charge_max, uint16_t p2_charge_max, partitioned_fit_table_t* destination) nogil
cdef size_t compute_partition_offset(partitioned_fit_table_t* self, uint16_t from_charge, uint16_t to_charge, int16_t intensity_ratio) nogil
cdef feature_fit_t* partitioned_fit_table_get(partitioned_fit_table_t* self, uint16_t from_charge, uint16_t to_charge, int16_t intensity_ratio) nogil
cdef feature_fit_t* partitioned_fit_table_get_partition(partitioned_fit_table_t* self, partition_t* partition) nogil