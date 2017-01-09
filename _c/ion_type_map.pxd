cdef struct IonTypeIndex:
    char** names
    size_t* indices
    size_t size

cdef struct IonTypeDoubleMap:
    IonTypeIndex* index_ref
    double* values


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

cdef void free_ion_type_index(IonTypeIndex* mapper) nogil
cdef void free_ion_type_double_map(IonTypeDoubleMap* mapper) nogil

cdef IonTypeIndex* ION_TYPE_INDEX
