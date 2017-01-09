cdef char* NOISE
cdef int OUT_OF_RANGE_INT = 999

cpdef double ppm_error(double x, double y)

cpdef object tol_ppm_error(double x, double y, double tolerance)


cdef inline bint feature_match(MSFeatureStruct* feature, PeakStruct* peak1, PeakStruct* peak2) nogil
cdef bint _precursor_context(MSFeatureStruct* feature, MatchedSpectrumStruct* ms) nogil

cdef int intensity_ratio_function(DPeak peak1, DPeak peak2)
cdef int _intensity_ratio_function(PeakStruct* peak1, PeakStruct* peak2) nogil


cdef void intensity_rank(list peak_list, double minimum_intensity=*)
cdef void _intensity_rank(PeakStructArray* peak_list, double minimum_intensity=*) nogil


cdef class MassOffsetFeature(object):
    cdef:
        public double offset
        public double tolerance
        public str name
        public bint fixed
        public int intensity_ratio
        public int from_charge
        public int to_charge
        public str feature_type
        public int min_peak_rank
        public int max_peak_rank
        public dict ion_type_matches
        public dict ion_type_totals
        public int glycan_peptide_ratio
        public int peptide_mass_rank

    cdef bint test(self, DPeak peak1, DPeak peak2)
    cpdef ion_type_increment(self, str name, str slot=*)
    cdef bint _precursor_context(self, MatchedSpectrum ms)

cdef class DPeak(object):
    '''
    Defines a type for holding the same relevant information that the database model Peak does
    without the non-trivial overhead of descriptor access on the mapped object to check for
    updated data.
    '''
    cdef:
        public double neutral_mass
        public long id
        public long scan_peak_index
        public int charge
        public double intensity
        public int rank
        public double mass_charge_ratio
        public list peak_relations

        cdef PeakStruct* as_struct(self)

cdef class TheoreticalFragment(object):
    cdef:
        public double neutral_mass
        public str key

cdef class MatchedSpectrum(object):
    cdef:
        public dict peak_match_map
        #: list of DPeak instances
        public list peak_list
        public str glycopeptide_sequence
        public int scan_time
        public int peaks_explained
        public int peaks_unexplained
        public int id
        public double peptide_mass
        public double glycan_mass
        public int peptide_mass_rank
        public int glycan_peptide_ratio

    cpdef set peak_explained_by(self, object peak_id)
    cpdef DPeak get_peak(self, long id)

cdef class FragmentMatch(object):
    cdef:
        public double observed_mass
        public double intensity
        public str key
        public str ion_type
        public long peak_index


# Scalar Structs

cdef struct PeakStruct:
    double neutral_mass
    long id
    long scan_peak_index
    int charge
    double intensity
    int rank
    double mass_charge_ratio

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

cdef struct IonTypeIndex:
    char** names
    size_t* indices
    size_t size

cdef struct IonTypeDoubleMap:
    IonTypeIndex* index_ref
    double* values


# Array Structs

cdef struct PeakStructArray:
    PeakStruct* peaks
    Py_ssize_t size

cdef struct MSFeatureStructArray:
    MSFeatureStruct* features
    Py_ssize_t size

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


# Convert cdef classes to structs
cdef MatchedSpectrumStruct* unwrap_matched_spectrum(MatchedSpectrum ms)
cdef MSFeatureStructArray* unwrap_feature_functions(list features)
cdef PeakStructArray* unwrap_peak_list(list)


#

cdef DPeak wrap_peak(PeakStruct* peak)
cdef MassOffsetFeature wrap_feature(MSFeatureStruct* feature)
cdef MatchedSpectrum wrap_matched_spectrum_struct(MatchedSpectrumStruct* ms)


# 

cdef FragmentMatchStructArray* matched_spectrum_struct_peak_explained_by(MatchedSpectrumStruct* ms, long peak_id) nogil


# Sort PeakStructArray
cdef void sort_by_intensity(PeakStructArray* peak_list) nogil
cdef void sort_by_neutral_mass(PeakStructArray* peak_list) nogil


# Peak Search Functions
cpdef list search_spectrum(DPeak peak, list peak_list, MassOffsetFeature feature)
cdef PeakStructArray* _search_spectrum(PeakStruct* peak, PeakStructArray* peak_list, MSFeatureStruct* feature) nogil
cdef PeakStructArray* _openmp_search_spectrum(PeakStruct* peak, PeakStructArray* peak_list, MSFeatureStruct* feature) nogil


# Ion Type Mappings

cdef IonTypeIndex* ION_TYPE_INDEX

cdef IonTypeIndex* new_ion_type_index(char** names, size_t size) nogil
cdef size_t ion_type_index(IonTypeIndex* mapper, char* name) nogil
cdef char* ion_type_name(IonTypeIndex* mapper, size_t index) nogil
cdef void ion_type_add(IonTypeIndex* mapper, char* name) nogil
cdef void print_ion_type_index(IonTypeIndex* mapper) nogil


cdef IonTypeDoubleMap* new_ion_type_double_map() nogil
cdef IonTypeDoubleMap* new_ion_type_double_map_from_dict(dict mapper)
cdef dict dict_from_ion_type_double_map(IonTypeDoubleMap* mapper)


cdef double ion_type_double_get(IonTypeDoubleMap* mapper, size_t index) nogil
cdef void ion_type_double_set(IonTypeDoubleMap* mapper, size_t index, double value) nogil
cdef void ion_type_double_inc(IonTypeDoubleMap* mapper, size_t index) nogil
cdef void ion_type_double_inc_name(IonTypeDoubleMap* mapper, char* name) nogil
cdef double sum_ion_double_map(IonTypeDoubleMap* mapper) nogil

# Struct Free Functions
cdef void free_fragment_match_struct_array(FragmentMatchStructArray* matches) nogil
cdef void free_peak_struct_array(PeakStructArray* peaks) nogil
cdef void free_matched_spectrum_struct(MatchedSpectrumStruct* ms) nogil
cdef void free_ion_type_index(IonTypeIndex* mapper) nogil
cdef void free_ion_type_double_map(IonTypeDoubleMap* mapper) nogil
cdef void free_ms_feature_struct(MSFeatureStruct* feature) nogil
cdef void free_ms_feature_struct_array(MSFeatureStructArray* features) nogil
