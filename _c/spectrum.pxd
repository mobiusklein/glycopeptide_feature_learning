cdef int intensity_ratio_function(PeakStruct* peak1, PeakStruct* peak2) nogil
cdef void intensity_rank(PeakStructArray* peak_list, double minimum_intensity=*) nogil


cdef struct PeakStruct:
    double neutral_mass
    long id
    long scan_peak_index
    int charge
    double intensity
    int rank
    double mass_charge_ratio


cdef struct PeakStructArray:
    PeakStruct* peaks
    Py_ssize_t size

cdef void sort_by_intensity(PeakStructArray* peak_list) nogil
cdef void sort_by_neutral_mass(PeakStructArray* peak_list) nogil

cdef int binary_search(PeakStructArray* peak_list, double query_mass, double error_tolerance, PeakStruct* out) nogil

cdef void free_peak_struct_array(PeakStructArray* peaks) nogil
