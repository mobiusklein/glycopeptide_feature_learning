from cpython.string cimport PyString_AsString, PyString_FromString

from libc.stdlib cimport abort, malloc, free, realloc, calloc
from libc.string cimport strcmp
from libc cimport *

from constants cimport (
    OTHER, NOISE, OUT_OF_RANGE_INT)

cdef extern from * nogil:
    void qsort (void *base, unsigned short n, unsigned short w, int (*cmp_func)(void*, void*))
    int printf   (const char *template, ...)


cdef IonTypeIndex* new_ion_type_index(char** names, size_t size) nogil:
    cdef IonTypeIndex* result = <IonTypeIndex*>malloc(sizeof(IonTypeIndex))
    cdef size_t i

    result.names = <char**>malloc(sizeof(char*) * 0)
    result.indices = <size_t*>malloc(sizeof(size_t) * 0)
    result.size = 0

    for i in range(size):
        ion_type_add(result, names[i])

    return result


cdef IonTypeDoubleMap* _new_ion_type_double_map(IonTypeIndex* index) nogil:
    cdef IonTypeDoubleMap* result
    result = <IonTypeDoubleMap*>malloc(sizeof(IonTypeDoubleMap))
    result.index_ref = index
    result.values = <double*>calloc(index.size, sizeof(double))
    return result


cdef IonTypeDoubleMap* new_ion_type_double_map() nogil:
    return _new_ion_type_double_map(ION_TYPE_INDEX)


cdef IonTypeDoubleMap* new_ion_type_double_map_from_dict(dict mapper):
    cdef:
        size_t i, j
        char* name
        IonTypeDoubleMap* result
        str pname
        double pvalue

    result = new_ion_type_double_map()
    j = 0
    for  pname, pvalue in mapper.items():
        name = PyString_AsString(pname)
        if not ion_type_exists(ION_TYPE_INDEX, name):
            ion_type_add(ION_TYPE_INDEX, name)
        i = ion_type_index(ION_TYPE_INDEX, name)
        ion_type_double_set(result, i, pvalue)
        j += 1

    return result


cdef dict dict_from_ion_type_double_map(IonTypeDoubleMap* mapper):
    cdef:
        size_t i, j
        char* name
        dict result
        str pname
        double pvalue

    result = dict()
    for i in range(mapper.index_ref.size):
        pname = PyString_FromString(ion_type_name(mapper.index_ref, i))
        pvalue = ion_type_double_get(mapper, i)
        result[pname] = pvalue

    return result


cdef size_t ion_type_index(IonTypeIndex* mapper, char* name) nogil:
    cdef size_t ix
    for ix in range(mapper.size):
        if strcmp(mapper.names[ix], name) == 0:
            return mapper.indices[ix]
    return OUT_OF_RANGE_INT


cdef char* ion_type_name(IonTypeIndex* mapper, size_t index) nogil:
    if index == OUT_OF_RANGE_INT:
        return OTHER
    return mapper.names[index]


cdef void ion_type_add(IonTypeIndex* mapper, char* name) nogil:
    mapper.names = <char**>realloc(mapper.names, sizeof(char*) * (mapper.size + 1))
    mapper.indices = <size_t*>realloc(mapper.indices, sizeof(size_t) * (mapper.size + 1))
    mapper.names[mapper.size] = name
    mapper.indices[mapper.size] = mapper.size
    mapper.size += 1


cdef bint ion_type_exists(IonTypeIndex* mapper, char* name) nogil:
    return OUT_OF_RANGE_INT == ion_type_index(mapper, name)


cdef double ion_type_double_get(IonTypeDoubleMap* mapper, size_t index) nogil:
    return mapper.values[index]


cdef void ion_type_double_set(IonTypeDoubleMap* mapper, size_t index, double value) nogil:
    mapper.values[index] = value


cdef void ion_type_double_inc(IonTypeDoubleMap* mapper, size_t index) nogil:
    mapper.values[index] += 1


cdef void ion_type_double_inc_name(IonTypeDoubleMap* mapper, char* name) nogil:
    cdef:
        size_t ix
    ix = ion_type_index(mapper.index_ref, name)
    if ix == OUT_OF_RANGE_INT:
        ix = ion_type_index(mapper.index_ref, OTHER)
    mapper.values[ix] += 1


cdef void print_ion_type_double(IonTypeDoubleMap* mapper) nogil:
    cdef:
        size_t i
    for i in range(mapper.index_ref.size):
        printf("%s -> %f\n", ion_type_name(mapper.index_ref, i), ion_type_double_get(mapper, i))


cdef void print_ion_type_index(IonTypeIndex* mapper) nogil:
    cdef:
        size_t i
    for i in range(mapper.size):
        printf("%s -> %d\n", ion_type_name(mapper, i), i)


cdef double sum_ion_double_map(IonTypeDoubleMap* mapper) nogil:
    cdef:
        double total
        size_t i
    total = 0.
    for i in range(mapper.index_ref.size):
        total += ion_type_double_get(mapper, i)
    return total


cdef void free_ion_type_index(IonTypeIndex* mapper) nogil:
    free(mapper.indices)
    free(mapper.names)
    free(mapper)


cdef void free_ion_type_double_map(IonTypeDoubleMap* mapper) nogil:
    free(mapper.values)
    free(mapper)


ION_TYPE_INDEX = new_ion_type_index(["b", "y", "stub_ion", "oxonium", "c", "z", "noise", "other"], 8)