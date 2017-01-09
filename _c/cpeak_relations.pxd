from glycresoft_sqlalchemy.utils.ccommon_math cimport (
    DPeak, MassOffsetFeature, ppm_error, intensity_rank,
    intensity_ratio_function, search_spectrum, MatchedSpectrum,
    PeakStruct, MSFeatureStruct, MatchedSpectrumStruct)

cdef class PeakRelation(object):
    cdef:
        public DPeak from_peak
        public DPeak to_peak
        public int from_charge
        public int to_charge
        public int intensity_ratio
        public str kind
        public MassOffsetFeature feature
        public bint same_terminal
        public MatchedSpectrum annotation

cdef class FittedFeature(object):
    cdef:
        public MassOffsetFeature feature
        public list relations


cdef public struct PeakRelationStruct:
    PeakStruct from_peak
    PeakStruct to_peak
    int from_charge
    int to_charge
    int intensity_ratio
    char* kind
    MSFeatureStruct* feature
    bint same_terminal
    MatchedSpectrumStruct* annotation


cdef public struct PeakRelationStructArray:
    PeakRelationStruct* relations
    size_t size


cdef public struct RelationSpectrumPair:
    PeakRelationStructArray* relations
    MatchedSpectrumStruct* matched_spectrum


cdef public struct RelationSpectrumPairArray:
    RelationSpectrumPair* pairs
    size_t size


cdef public struct FittedFeatureStruct:
    RelationSpectrumPairArray* relation_pairs
    MSFeatureStruct* feature
    char* kind


cdef public struct FittedFeatureStructArray:
    FittedFeatureStruct** features
    size_t size


cdef public struct PeakAnnotationStruct:
    PeakStruct* peak
    PeakRelationStructArray* annotations
