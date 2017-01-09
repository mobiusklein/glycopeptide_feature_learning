from libc.stdlib cimport abort, malloc, free, realloc, calloc
from libc.match cimport fabs
from libc cimport *

cdef extern from * nogil:
    void qsort (void *base, unsigned short n, unsigned short w, int (*cmp_func)(void*, void*))
    int printf   (const char *template, ...)


cdef int intensity_ratio_function(PeakStruct* peak1, PeakStruct* peak2) nogil:
    cdef double ratio
    ratio = peak1.intensity / (peak2.intensity)
    if ratio >= 5:
        return -4
    elif 2.5 <= ratio < 5:
        return -3
    elif 1.7 <= ratio < 2.5:
        return -2
    elif 1.3 <= ratio < 1.7:
        return -1
    elif 1.0 <= ratio < 1.3:
        return 0
    elif 0.8 <= ratio < 1.0:
        return 1
    elif 0.6 <= ratio < 0.8:
        return 2
    elif 0.4 <= ratio < 0.6:
        return 3
    elif 0.2 <= ratio < 0.4:
        return 4
    elif 0. <= ratio < 0.2:
        return 5


cdef void intensity_rank(PeakStructArray* peak_list, double minimum_intensity=100.) nogil:
    cdef:
        size_t i
        int step, rank, tailing
        PeakStruct p
    sort_by_intensity(peak_list)
    step = 0
    rank = 10
    tailing = 6
    for i in range(peak_list.size):
        p = peak_list.peaks[i]
        if p.intensity < minimum_intensity:
            p.rank = 0
            continue
        step += 1
        if step == 10 and rank != 0:
            if rank == 1:
                if tailing != 0:
                    step = 0
                    tailing -= 1
                else:
                    step = 0
                    rank -= 1
            else:
                step = 0
                rank -= 1
        if rank == 0:
            break
        p.rank = rank


cdef int binary_search(PeakStructArray* peak_list, double query_mass, double error_tolerance, PeakStruct* out) nogil:
    cdef:
        size_t i, j, n, lo, hi, mid
        PeakStruct item
        double error

    lo = 0
    hi = peak_list.size

    while hi != lo:
        mid = (hi + lo) / 2
        item = peak_list.v[mid]

        error = (query_mass - item.neutral_mass) / item.neutral_mass
        if fabs(error) < error_tolerance:
            # TODO Implement Sweep
            out = &item
            return 0
        elif (hi - lo) == 1:
            return 1
        elif item.neutral_mass > query_mass:
            hi = mid
        else:
            lo = mid
    return 2


cdef void free_peak_struct_array(PeakStructArray* peaks) nogil:
    free(peaks.peaks)
    free(peaks)


cdef void sort_by_intensity(PeakStructArray* peak_list) nogil:
    qsort(peak_list.peaks, peak_list.size, sizeof(PeakStruct), compare_by_intensity)

cdef void sort_by_neutral_mass(PeakStructArray* peak_list) nogil:
    qsort(peak_list, peak_list.size, sizeof(PeakStruct), compare_by_neutral_mass)

cdef int compare_by_neutral_mass(const void * a, const void * b) nogil:
    if (<PeakStruct*>a).neutral_mass < (<PeakStruct*>b).neutral_mass:
        return -1
    elif (<PeakStruct*>a).neutral_mass == (<PeakStruct*>b).neutral_mass:
        return 0
    elif (<PeakStruct*>a).neutral_mass > (<PeakStruct*>b).neutral_mass:
        return 1

cdef int compare_by_intensity(const void * a, const void * b) nogil:
    if (<PeakStruct*>a).intensity < (<PeakStruct*>b).intensity:
        return -1
    elif (<PeakStruct*>a).intensity == (<PeakStruct*>b).intensity:
        return 0
    elif (<PeakStruct*>a).intensity > (<PeakStruct*>b).intensity:
        return 1
