cdef char* NOISE
cdef int OUT_OF_RANGE_INT = 999

cdef bint feature_match(MSFeatureStruct* feature, PeakStruct* peak1, PeakStruct* peak2) nogil
cdef bint precursor_context(MSFeatureStruct* feature, MatchedSpectrumStruct* ms) nogil

cdef struct MSFeatureStruct:
    double offset
    double tolerance
    char* name
    bint fixed
    int intensity_ratio
    int from_charge
    int to_charge
    char* feature_type
    int min_peak_rank
    int max_peak_rank
    IonTypeDoubleMap* ion_type_matches
    IonTypeDoubleMap* ion_type_totals
    int glycan_peptide_ratio
    int peptide_mass_rank

cdef struct FragmentMatchStruct:
    double observed_mass
    double intensity
    char* key
    char* ion_type
    long peak_index

cdef struct MatchedSpectrumStruct:
    FragmentMatchStructArray* peak_match_list
    PeakStructArray* peak_list
    char* glycopeptide_sequence
    int scan_time
    int peaks_explained
    int peaks_unexplained
    int id
    double peptide_mass
    double glycan_mass
    int peptide_mass_rank
    int glycan_peptide_ratio


cdef struct MSFeatureStructArray:
    MSFeatureStruct* features
    size_t size

cdef struct FragmentMatchStructArray:
    FragmentMatchStruct* matches
    size_t size

cdef struct MatchedSpectrumStructArray:
    MatchedSpectrumStruct* matches
    size_t size

cdef struct PeakToPeakShiftMatches:
    PeakStruct* peaks
    size_t size
    double mass_shift
    PeakStruct* reference


cdef void free_fragment_match_struct_array(FragmentMatchStructArray* matches) nogil
cdef void free_matched_spectrum_struct(MatchedSpectrumStruct* ms) nogil
cdef void free_ms_feature_struct(MSFeatureStruct* feature) nogil
cdef void free_ms_feature_struct_array(MSFeatureStructArray* features) nogil
