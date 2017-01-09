from cpython.ref cimport PyObject
from cpython.string cimport PyString_AsString, PyString_FromString
from libc.stdlib cimport abort, malloc, free, realloc, calloc
from libc.math cimport fabs
from libc.string cimport strcmp
from libc cimport *

cdef extern from * nogil:
    void qsort (void *base, unsigned short n, unsigned short w, int (*cmp_func)(void*, void*))
    int printf   (const char *template, ...)

from cython.parallel cimport parallel, prange # openmp must be enabled at compile time
cimport cython

from cpython.int cimport PyInt_AsLong
from cpython.float cimport PyFloat_AsDouble, PyFloat_FromDouble
from cpython.dict cimport PyDict_GetItem, PyDict_SetItem, PyDict_Values
from cpython.list cimport PyList_GET_ITEM, PyList_GET_SIZE, PyList_Append

import operator
from collections import defaultdict
from glycresoft_sqlalchemy.structure.sequence import Sequence as GlycopeptideSequence, Composition

cdef:
    object get_intensity = operator.attrgetter("intensity")
    int OUT_OF_RANGE_INT = 999
    char* OTHER = "other"
    char* NOISE = "noise"
    double WATER_MASS = Composition("H2O").mass


cpdef double ppm_error(double x, double y):
    return _ppm_error(x, y)


cdef inline double _ppm_error(double x, double y) nogil:
    return (x - y) / y


cpdef object tol_ppm_error(double x, double y, double tolerance):
    cdef double err
    err = (x - y) / y
    if abs(err) <= tolerance:
        return err
    else:
        return None


cpdef str interpolate_fragment_ion_type(str ion_key):
    if ion_key[0] in ['b', 'c', 'y', 'z']:
        return ion_key[0]
    elif "pep" in ion_key:
        return "stub_ion"
    else:
        return "oxonium"


cdef int intensity_ratio_function(DPeak peak1, DPeak peak2):
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


cdef int peptide_mass_rank(double mass):
    cdef double unit = 121.6
    cdef int ratio = <int>(mass / unit)
    if ratio < 9:
        return 1
    elif ratio < 13:
        return 2
    elif ratio < 17:
        return 3
    elif ratio < 20:
        return 4
    else:
        return 5


cdef int glycan_peptide_ratio(double glycan_mass, double peptide_mass):
    cdef double ratio = (glycan_mass / peptide_mass)
    if ratio < 0.4:
        return 0
    elif 0.4 <= ratio < 0.8:
        return 1
    elif 0.8 <= ratio < 1.2:
        return 2
    elif 1.2 <= ratio < 1.6:
        return 3
    elif 1.6 <= ratio < 2.0:
        return 4
    elif ratio >= 2.0:
        return 5


cdef int _intensity_ratio_function(PeakStruct* peak1, PeakStruct* peak2) nogil:
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


# Peak list search


cpdef list search_spectrum(DPeak peak, list peak_list, MassOffsetFeature feature):
    '''
    Search one DPeak instance against a list of DPeaks, returning all which satisfy a
    single MassOffsetFeature.
    '''
    cdef:
        list matches = []
        DPeak other_peak
        Py_ssize_t i
        object adder = matches.append
    for i in range(PyList_GET_SIZE(peak_list)):
        other_peak = <DPeak>PyList_GET_ITEM(peak_list, i)
        if feature.test(peak, other_peak):
            adder(other_peak)
    return matches


cpdef list search_spectrum_by_mass(double mass, list peak_list, double tolerance=2e-5):
    '''
    Search a mass against a list of DPeaks
    '''
    cdef:
        DPeak other_peak
        Py_ssize_t i
        list matches = []
    for i in range(PyList_GET_SIZE(peak_list)):
        other_peak = <DPeak>PyList_GET_ITEM(peak_list, i)
        if abs(ppm_error(mass, other_peak)) <= tolerance:
            PyList_Append(matches, other_peak)
    return matches


cdef inline bint feature_match(MSFeatureStruct* feature, PeakStruct* peak1, PeakStruct* peak2) nogil:
    if (feature.intensity_ratio == OUT_OF_RANGE_INT or _intensity_ratio_function(peak1, peak2) == feature.intensity_ratio) and\
       ((feature.from_charge == OUT_OF_RANGE_INT and feature.to_charge == OUT_OF_RANGE_INT) or
        (feature.from_charge == peak1.charge and feature.to_charge == peak2.charge)) and\
       ((feature.min_peak_rank == OUT_OF_RANGE_INT) or (peak1.rank >= feature.min_peak_rank)) and\
       ((feature.max_peak_rank == OUT_OF_RANGE_INT) or (peak1.rank <= feature.max_peak_rank)):
        return fabs(_ppm_error(peak1.neutral_mass + feature.offset, peak2.neutral_mass)) <= feature.tolerance
    else:
        return False


cdef bint _precursor_context(MSFeatureStruct* feature, MatchedSpectrumStruct* ms) nogil:
    return (feature.glycan_peptide_ratio == OUT_OF_RANGE_INT or feature.glycan_peptide_ratio == ms.glycan_peptide_ratio) and\
       (feature.peptide_mass_rank == OUT_OF_RANGE_INT or feature.peptide_mass_rank == ms.peptide_mass_rank)


cdef PeakStructArray* _search_spectrum(PeakStruct* peak, PeakStructArray* peak_list, MSFeatureStruct* feature) nogil:
    cdef:
        PeakStructArray* matches
        size_t i, j, n
        PeakStruct query_peak

    matches = <PeakStructArray*>malloc(sizeof(PeakStructArray))
    matches.peaks = <PeakStruct*>malloc(sizeof(PeakStruct) * peak_list.size)
    matches.size = peak_list.size
    n = 0
    for i in range(peak_list.size):
        query_peak = peak_list.peaks[i]
        if feature_match(feature, peak, &query_peak):
            matches.peaks[n] = query_peak
            n += 1
    matches.peaks = <PeakStruct*>realloc(matches.peaks, sizeof(PeakStruct) * n)
    matches.size = n
    return matches


cdef PeakStructArray* _openmp_search_spectrum(PeakStruct* peak, PeakStructArray* peak_list, MSFeatureStruct* feature) nogil:
    cdef:
        PeakStructArray* matches
        size_t i, j, n
        PeakStruct query_peak
        PeakStruct* temp
        int* did_match
        long i_p, n_p
    n = peak_list.size
    n_p = n
    did_match = <int*>malloc(sizeof(int)*n)
    for i_p in prange(n_p, schedule="guided", num_threads=8):
        if feature_match(feature, peak, &peak_list.peaks[i_p]):
            did_match[i_p] = 1
        else:
            did_match[i_p] = 0
    matches = <PeakStructArray*>malloc(sizeof(PeakStructArray))
    matches.peaks = <PeakStruct*>malloc(sizeof(PeakStruct) * peak_list.size)
    matches.size = peak_list.size

    j = 0
    for i in range(n):
        if did_match[i]:
            matches.peaks[j] = peak_list.peaks[i]
            j += 1
    matches.peaks = <PeakStruct*>realloc(matches.peaks, sizeof(PeakStruct) * j)
    matches.size = j
    free(did_match)
    return matches


cdef void _intensity_rank(PeakStructArray* peak_list, double minimum_intensity=100.) nogil:
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


cdef void intensity_rank(list peak_list, double minimum_intensity=100.):
    cdef:
        Py_ssize_t i = 0
        int step, rank, tailing
        DPeak p
    peak_list = sorted(peak_list, key=get_intensity, reverse=True)
    step = 0
    rank = 10
    tailing = 6
    for i in range(len(peak_list)):
        p = <DPeak>peak_list[i]
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


# cdef classes


cdef class MassOffsetFeature(object):

    def __init__(self, offset, tolerance, name=None, intensity_ratio=OUT_OF_RANGE_INT,
                 from_charge=OUT_OF_RANGE_INT, to_charge=OUT_OF_RANGE_INT, feature_type="",
                 min_peak_rank=OUT_OF_RANGE_INT, max_peak_rank=OUT_OF_RANGE_INT,
                 glycan_peptide_ratio=OUT_OF_RANGE_INT, peptide_mass_rank=OUT_OF_RANGE_INT,
                 fixed=False):
        if name is None:
            name = "F:" + str(offset)
            if intensity_ratio is not OUT_OF_RANGE_INT:
                name += ", %r" % (intensity_ratio if intensity_ratio > OUT_OF_RANGE_INT else '')

        self.offset = offset
        self.tolerance = tolerance
        self.name = name
        self.intensity_ratio = intensity_ratio
        self.from_charge = from_charge
        self.to_charge = to_charge
        self.feature_type = feature_type
        self.min_peak_rank = min_peak_rank
        self.max_peak_rank = max_peak_rank
        self.ion_type_matches = dict()
        self.ion_type_totals = dict()
        self.glycan_peptide_ratio = glycan_peptide_ratio
        self.peptide_mass_rank = peptide_mass_rank
        self.fixed = fixed

    def __getstate__(self):
        return {
            "offset": self.offset,
            "tolerance": self.tolerance,
            "name": self.name,
            "intensity_ratio": self.intensity_ratio,
            "from_charge": self.from_charge,
            "to_charge": self.to_charge,
            "min_peak_rank": self.min_peak_rank,
            "max_peak_rank": self.max_peak_rank,
            "ion_type_matches": self.ion_type_matches,
            "ion_type_totals": self.ion_type_totals,
            "feature_type": self.feature_type,
            "glycan_peptide_ratio": self.glycan_peptide_ratio,
            "peptide_mass_rank": self.peptide_mass_rank
        }

    def __setstate__(self, d):
        self.name = d['name']
        self.offset = d['offset']
        self.tolerance = d['tolerance']
        self.intensity_ratio = d['intensity_ratio']
        self.from_charge = d['from_charge']
        self.to_charge = d['to_charge']
        self.min_peak_rank = d["min_peak_rank"]
        self.max_peak_rank = d["max_peak_rank"]
        self.ion_type_matches = d["ion_type_matches"]
        self.ion_type_totals = d["ion_type_totals"]
        self.feature_type = d["feature_type"]
        self.glycan_peptide_ratio = d['glycan_peptide_ratio']
        self.peptide_mass_rank = d['peptide_mass_rank']

    def __reduce__(self):
        return MassOffsetFeature, (0, 0), self.__getstate__()

    def __call__(self, DPeak query, DPeak peak):
        return self.test(query, peak)        

    cdef bint test(self, DPeak peak1, DPeak peak2):
        if (self.intensity_ratio == OUT_OF_RANGE_INT or intensity_ratio_function(peak1, peak2) == self.intensity_ratio) and\
           ((self.from_charge == OUT_OF_RANGE_INT and self.to_charge == OUT_OF_RANGE_INT) or
            (self.from_charge == peak1.charge and self.to_charge == peak2.charge)) and\
           ((self.min_peak_rank == OUT_OF_RANGE_INT) or (peak1.rank >= self.min_peak_rank)) and\
           ((self.max_peak_rank == OUT_OF_RANGE_INT) or (peak1.rank <= self.max_peak_rank)):
            return abs(ppm_error(peak1.neutral_mass + self.offset, peak2.neutral_mass)) <= self.tolerance
        else:
            return False

    def precursor_context(self, ms):
        return (self.glycan_peptide_ratio == OUT_OF_RANGE_INT or self.glycan_peptide_ratio == ms.glycan_peptide_ratio) and\
           (self.peptide_mass_rank == OUT_OF_RANGE_INT or self.peptide_mass_rank == ms.peptide_mass_rank)

    cdef bint _precursor_context(self, MatchedSpectrum ms):
        return (self.glycan_peptide_ratio == OUT_OF_RANGE_INT or self.glycan_peptide_ratio == ms.glycan_peptide_ratio) and\
           (self.peptide_mass_rank == OUT_OF_RANGE_INT or self.peptide_mass_rank == ms.peptide_mass_rank)

    def __repr__(self):
        fields = {}
        fields['offset'] = self.offset
        if self.glycan_peptide_ratio != OUT_OF_RANGE_INT:
            fields['glycan_peptide_ratio'] = self.glycan_peptide_ratio
        if self.peptide_mass_rank != OUT_OF_RANGE_INT:
            fields['peptide_mass_rank'] = self.peptide_mass_rank
        if self.from_charge != OUT_OF_RANGE_INT:
            fields["from_charge"] = self.from_charge
        if self.to_charge != OUT_OF_RANGE_INT:
            fields['to_charge'] = self.to_charge
        if self.min_peak_rank != OUT_OF_RANGE_INT and self.min_peak_rank == self.max_peak_rank:
            fields['base_peak_intensity'] = self.min_peak_rank
        if self.intensity_ratio != OUT_OF_RANGE_INT:
            fields["intensity_ratio"] = self.intensity_ratio
        terms = []
        for k, v in fields.items():
            if isinstance(v, int):
                terms.append("%s=%d" % (k, v))
            elif isinstance(v, float):
                terms.append("%s=%0.4f" % (k, v))
            else:
                terms.append("%s=%r" % (k, v))
        return "<{} {}>".format(self.name, ", ".join(terms))


    def __hash__(self):
        return hash((self.glycan_peptide_ratio, self.peptide_mass_rank, self.offset,
                     self.intensity_ratio, self.from_charge,
                     self.to_charge, self.min_peak_rank, self.max_peak_rank))

    cpdef ion_type_increment(self, str name, str slot="matches"):
        cdef:
            double value
            PyObject* pvalue
            dict dist

        if slot == "matches":
            dist = self.ion_type_matches
        elif slot == "totals":
            dist = self.ion_type_totals
        pvalue = PyDict_GetItem(dist, name)
        if pvalue == NULL:
            value = 0.
        else:
            value = PyFloat_AsDouble(<object>pvalue)
        value += 1
        PyDict_SetItem(dist, name , value)


cdef class DPeak(object):
    '''
    Defines a type for holding the same relevant information that the database model Peak does
    without the non-trivial overhead of descriptor access on the mapped object to check for
    updated data.
    '''


    def __init__(self, peak=None):
        if peak is not None:
            self.neutral_mass = PyFloat_AsDouble(peak.neutral_mass)
            self.id = PyInt_AsLong(peak.id)
            self.charge = PyInt_AsLong(peak.charge)
            self.intensity = PyFloat_AsDouble(peak.intensity)
            self.scan_peak_index = PyInt_AsLong(peak.scan_peak_index) 
        self.peak_relations = []

    def __repr__(self):
        return "<DPeak {} {} {}>".format(self.neutral_mass, self.charge, self.intensity)

    cdef PeakStruct* as_struct(self):
        cdef PeakStruct* result
        result = <PeakStruct*>malloc(sizeof(PeakStruct))
        result.neutral_mass = self.neutral_mass
        result.id = self.id
        result.charge = self.charge
        result.intensity = self.intensity
        result.rank = self.rank
        result.mass_charge_ratio = self.mass_charge_ratio
        result.scan_peak_index = self.scan_peak_index
        return result

    def clone(self):
        return DPeak(self)

    def __getstate__(self):
        cdef dict d
        d = dict()
        d['neutral_mass'] = self.neutral_mass
        d['id'] = self.id
        d['charge'] = self.charge
        d['intensity'] = self.intensity
        d['rank'] = self.rank
        d['mass_charge_ratio'] = self.mass_charge_ratio
        d['peak_relations'] = self.peak_relations
        return d

    def __setstate__(self, dict d):
        self.neutral_mass = d['neutral_mass']
        self.id = d['id']
        self.charge = d['charge']
        self.intensity = d['intensity']
        self.rank = d['rank']
        self.mass_charge_ratio = d['mass_charge_ratio']
        self.peak_relations = d['peak_relations']

    def __reduce__(self):
        return DPeak, (None,), self.__getstate__()

    def __hash__(self):
        return hash((self.neutral_mass, self.intensity, self.charge))

    def __richmp__(self, other, int op):
        cdef bint res = True
        if op == 2:
            res &= self.neutral_mass == other.neutral_mass
            res &= self.intensity == other.intensity
            res &= self.charge == other.charge
            return res
        elif op == 3:
            return not self == other


cdef class MatchedSpectrum(object):

    def __init__(self, gsm=None):
        if gsm is not None:
            self.peak_match_map = {
                pid: [FragmentMatch(
                    observed_mass=frag_dict['observed_mass'], intensity=frag_dict['intensity'],
                    key=frag_dict['key'], peak_index=frag_dict['peak_id']) for frag_dict in fragment_list if frag_dict['intensity'] >= 200]
                for pid, fragment_list in gsm.peak_match_map.items()
            }
            self.peak_list = list(gsm)
            self.scan_time = gsm.scan_time
            self.peaks_explained = gsm.peaks_explained
            self.peaks_unexplained = gsm.peaks_unexplained
            self.id = gsm.id
            self.glycopeptide_sequence = str(gsm.glycopeptide_sequence)

            total_mass = gsm.glycopeptide_match.calculated_mass
            site_count = gsm.glycopeptide_match.count_glycosylation_sites
            glycan_mass = gsm.glycopeptide_match.glycan_composition.mass() - (site_count * WATER_MASS)
            peptide_mass = total_mass - glycan_mass

            self.peptide_mass = peptide_mass
            self.glycan_mass = glycan_mass
            self.peptide_mass_rank = peptide_mass_rank(peptide_mass)
            self.glycan_peptide_ratio = glycan_peptide_ratio(glycan_mass, peptide_mass)            

    def reindex_peak_matches(self):
        peaks = sorted(self.peak_list, key=lambda x: x.neutral_mass)
        assigned = set()
        for original_index, matches in self.peak_match_map.items():
            if len(matches) == 0:
                continue
            match = matches[0]
            smallest_diff = float('inf')
            smallest_diff_peak = None
            for peak in peaks:
                if peak.id in assigned:
                    continue
                diff = abs(peak.neutral_mass - match.observed_mass)
                if diff < smallest_diff:
                    smallest_diff = diff
                    smallest_diff_peak = peak
                elif diff > smallest_diff:
                    smallest_diff_peak.id = original_index
                    assigned.add(smallest_diff_peak.id)
                    break

    cpdef DPeak get_peak(self, long id):
        cdef:
            DPeak temp
            size_t i
        for i in range(len(self.peak_list)):
            temp = <DPeak>PyList_GET_ITEM(self.peak_list, i)
            if temp.id == id:
                return temp
        raise KeyError(id)

    def __iter__(self):
        cdef:
            Py_ssize_t i
            DPeak o
        for i in range(PyList_GET_SIZE(self.peak_list)):
            o = <DPeak>PyList_GET_ITEM(self.peak_list, i)
            yield o

    cpdef set peak_explained_by(self, object peak_index):
        cdef:
            set explained
            list matches
            FragmentMatch match
            PyObject* temp
            Py_ssize_t i

        explained = set()
        temp = PyDict_GetItem(self.peak_match_map, peak_index)
        if temp == NULL:
            return explained
        matches = <list>temp
        for i in range(PyList_GET_SIZE(matches)):
            match = <FragmentMatch>PyList_GET_ITEM(matches, i)
            explained.add(match.ion_type)

        return explained

    def matches_for(self, key):
        for peak_index, matches in self.peak_match_map.items():
            for match in matches:
                if match.key == key:
                    yield match

    def __getstate__(self):
        d = {}
        d['peak_match_map'] = self.peak_match_map
        d['peak_list'] = self.peak_list
        d['scan_time'] = self.scan_time
        d['peaks_explained'] = self.peaks_explained
        d['peaks_unexplained'] = self.peaks_unexplained
        d['id'] = self.id
        d['glycopeptide_sequence'] = self.glycopeptide_sequence
        d['peptide_mass'] = self.peptide_mass
        d['glycan_mass'] = self.glycan_mass
        d['glycan_peptide_ratio'] = self.glycan_peptide_ratio
        d['peptide_mass_rank'] = self.peptide_mass_rank
        return d

    def __setstate__(self, d):
        self.peak_match_map = d['peak_match_map']
        self.peak_list = d['peak_list']
        self.scan_time = d['scan_time']
        self.peaks_explained = d['peaks_explained']
        self.peaks_unexplained = d['peaks_unexplained']
        self.id = d['id']
        self.glycopeptide_sequence = d['glycopeptide_sequence']
        self.peptide_mass = d['peptide_mass']
        self.glycan_mass = d['glycan_mass']
        self.glycan_peptide_ratio = d['glycan_peptide_ratio']
        self.peptide_mass_rank = d['peptide_mass_rank']

    def __reduce__(self):
        return MatchedSpectrum, (None,), self.__getstate__()

    def __repr__(self):
        temp = "<MatchedSpectrum %s @ %d %d|%d>" % (
            self.glycopeptide_sequence, self.scan_time, self.peaks_explained,
            self.peaks_unexplained)
        return temp


cdef class FragmentMatch(object):
    def __init__(self, observed_mass, intensity, key, peak_index, ion_type=None, **kwargs):
        if ion_type is None:
            ion_type = interpolate_fragment_ion_type(key)
        self.observed_mass = observed_mass
        self.intensity = intensity
        self.key = key
        self.peak_index = peak_index
        self.ion_type = ion_type

    def __repr__(self):
        return "<FragmentMatch {} {} {}>".format(self.observed_mass, self.intensity, self.key)

    def __reduce__(self):
        return FragmentMatch, (self.observed_mass, self.intensity, self.key, self.peak_index, self.ion_type)

    def _asdict(self):
        d = {
            "observed_mass": self.observed_mass,
            "intensity": self.intensity,
            "key": self.key,
            "peak_index": self.peak_index,
            "ion_type": self.ion_type            
        }
        return d


cpdef DPeak DPeak_from_values(cls, float neutral_mass):
    cdef DPeak peak
    peak = DPeak()
    peak.neutral_mass = neutral_mass
    return peak


# cdef class to Struct unwrapper


cdef MSFeatureStructArray* unwrap_feature_functions(list features):
    cdef:
        MSFeatureStructArray* ms_features
        MSFeatureStruct cfeature
        MassOffsetFeature pfeature
        size_t i, j
    j = PyList_GET_SIZE(features)
    ms_features = <MSFeatureStructArray*>malloc(sizeof(MSFeatureStructArray))
    ms_features.features = <MSFeatureStruct*>malloc(sizeof(MSFeatureStruct) * j)
    ms_features.size = j
    for i in range(j):
        pfeature = <MassOffsetFeature>PyList_GET_ITEM(features, i)
        cfeature = ms_features.features[i]
        cfeature.offset = pfeature.offset
        cfeature.intensity_ratio = pfeature.intensity_ratio
        cfeature.from_charge = pfeature.from_charge
        cfeature.to_charge = pfeature.to_charge
        cfeature.tolerance = pfeature.tolerance
        cfeature.name = PyString_AsString(pfeature.name)
        cfeature.feature_type = PyString_AsString(pfeature.feature_type)
        cfeature.min_peak_rank = pfeature.min_peak_rank
        cfeature.max_peak_rank = pfeature.max_peak_rank
        cfeature.ion_type_matches = new_ion_type_double_map_from_dict(pfeature.ion_type_matches)
        cfeature.ion_type_totals = new_ion_type_double_map_from_dict(pfeature.ion_type_totals)
        cfeature.glycan_peptide_ratio = pfeature.glycan_peptide_ratio
        cfeature.peptide_mass_rank = pfeature.peptide_mass_rank
        cfeature.fixed = pfeature.fixed
        ms_features.features[i] = cfeature
    return ms_features

cdef PeakStructArray* unwrap_peak_list(list py_peaks):
    cdef:
        PeakStructArray* peaks
        PeakStruct cpeak
        DPeak dpeak
        size_t i, j

    j = PyList_GET_SIZE(py_peaks)
    peaks = <PeakStructArray*>malloc(sizeof(PeakStructArray))
    intensity_rank(py_peaks)
    peaks.peaks = <PeakStruct*>malloc(sizeof(PeakStruct) * j)
    peaks.size = j
    for i in range(j):
        dpeak = <DPeak>py_peaks[i]
        cpeak.neutral_mass = dpeak.neutral_mass
        cpeak.id = dpeak.id
        cpeak.charge = dpeak.charge
        cpeak.intensity = dpeak.intensity
        cpeak.rank = dpeak.rank
        cpeak.mass_charge_ratio = dpeak.mass_charge_ratio
        cpeak.scan_peak_index = dpeak.scan_peak_index
        peaks.peaks[i] = cpeak
    return peaks

cdef MatchedSpectrumStruct* unwrap_matched_spectrum(MatchedSpectrum ms):
    cdef:
        FragmentMatchStructArray* frag_matches
        FragmentMatchStruct* current_match
        MatchedSpectrumStruct* ms_struct
        FragmentMatch fm
        size_t i, j, total
        list matches_list, peak_match_list
        dict frag_dict

    ms_struct = <MatchedSpectrumStruct*>malloc(sizeof(MatchedSpectrumStruct))
    ms_struct.peak_list = unwrap_peak_list(ms.peak_list)
    ms_struct.scan_time = ms.scan_time
    ms_struct.peaks_explained = ms.peaks_explained
    ms_struct.peaks_unexplained = ms.peaks_unexplained
    ms_struct.id = ms.id
    ms_struct.glycopeptide_sequence = PyString_AsString(ms.glycopeptide_sequence)

    ms_struct.peptide_mass = ms.peptide_mass
    ms_struct.glycan_mass = ms.glycan_mass
    ms_struct.peptide_mass_rank = ms.peptide_mass_rank
    ms_struct.glycan_peptide_ratio = ms.glycan_peptide_ratio

    total = 0
    matches_list = PyDict_Values(ms.peak_match_map)
    for i in range(PyList_GET_SIZE(matches_list)):
        peak_match_list = <list>PyList_GET_ITEM(matches_list, i)
        for j in range(PyList_GET_SIZE(peak_match_list)):
            total += 1

    frag_matches = <FragmentMatchStructArray*>malloc(sizeof(FragmentMatchStructArray))
    frag_matches.size = (total)
    frag_matches.matches = <FragmentMatchStruct*>malloc(sizeof(FragmentMatchStruct) * total)

    total = 0
    for i in range(PyList_GET_SIZE(matches_list)):
        peak_match_list = <list>PyList_GET_ITEM(matches_list, i)
        for j in range(PyList_GET_SIZE(peak_match_list)):
            current_match = &frag_matches.matches[total]
            fm = <FragmentMatch>PyList_GET_ITEM(peak_match_list, j)
            current_match.observed_mass = fm.observed_mass
            current_match.intensity = fm.intensity
            current_match.key = PyString_AsString(fm.key)
            current_match.ion_type = PyString_AsString(fm.ion_type)
            current_match.peak_index = fm.peak_index
            total += 1
    ms_struct.peak_match_list = frag_matches
    return ms_struct

cdef FragmentMatchStruct* unwrap_fragment_match(FragmentMatch fm):
    cdef FragmentMatchStruct* result = <FragmentMatchStruct*>malloc(sizeof(FragmentMatchStruct))
    result.observed_mass = fm.observed_mass
    result.intensity = fm.intensity
    result.key = PyString_AsString(fm.key)
    result.ion_type = PyString_AsString(fm.ion_type)
    result.peak_index = fm.peak_index
    return result


# struct to cdef class wrapper
cdef DPeak wrap_peak(PeakStruct* peak):
    cdef DPeak dpeak = DPeak()
    dpeak.neutral_mass = peak.neutral_mass
    dpeak.charge = peak.charge
    dpeak.intensity = peak.intensity
    dpeak.rank = peak.rank
    dpeak.id = peak.id
    dpeak.scan_peak_index = peak.scan_peak_index
    return dpeak


cdef MassOffsetFeature wrap_feature(MSFeatureStruct* feature):
    cdef MassOffsetFeature pfeature
    pfeature = MassOffsetFeature(
        offset=feature.offset,
        tolerance=feature.tolerance,
        name=feature.name,
        intensity_ratio=feature.intensity_ratio,
        from_charge=feature.from_charge,
        to_charge=feature.to_charge,
        feature_type=feature.feature_type,
        min_peak_rank=feature.min_peak_rank,
        max_peak_rank=feature.max_peak_rank,
        glycan_peptide_ratio=feature.glycan_peptide_ratio,
        peptide_mass_rank=feature.peptide_mass_rank,
        fixed=feature.fixed)
    pfeature.ion_type_matches = dict_from_ion_type_double_map(feature.ion_type_matches)
    pfeature.ion_type_totals = dict_from_ion_type_double_map(feature.ion_type_totals)
    return pfeature


cdef MatchedSpectrum wrap_matched_spectrum_struct(MatchedSpectrumStruct* ms):
    cdef:
        MatchedSpectrum result
        PeakStruct peak
        DPeak dpeak
        FragmentMatchStruct frag_match
        dict frag_matches
        size_t i, j, total
        list matches_list, peak_match_list
        FragmentMatch frag_match_obj

    frag_matches = dict()

    result = MatchedSpectrum()
    result.glycopeptide_sequence = PyString_FromString(ms.glycopeptide_sequence)
    result.peak_list = [None] * ms.peak_list.size
    result.scan_time = ms.scan_time
    result.peaks_explained = ms.peaks_explained
    result.peaks_unexplained = ms.peaks_unexplained
    result.peptide_mass = ms.peptide_mass
    result.glycan_mass = ms.glycan_mass
    result.glycan_peptide_ratio = ms.glycan_peptide_ratio
    result.peptide_mass_rank = ms.peptide_mass_rank
    result.id = ms.id

    for i in range(ms.peak_list.size):
        peak = ms.peak_list.peaks[i]
        dpeak = wrap_peak(&peak)
        result.peak_list[i] = dpeak

    for i in range(ms.peak_match_list.size):
        frag_match = ms.peak_match_list.matches[i]
        frag_match_obj = FragmentMatch(
            frag_match.observed_mass, frag_match.intensity,
            PyString_FromString(frag_match.key), frag_match.peak_index,
            PyString_FromString(frag_match.ion_type))
        matches_list = frag_matches.setdefault(frag_match.peak_index, [])
        matches_list.append(frag_match_obj)

    result.peak_match_map = frag_matches

    return result


# 


cdef FragmentMatchStructArray* matched_spectrum_struct_peak_explained_by(MatchedSpectrumStruct* ms, long peak_index) nogil:
    cdef:
        size_t i, j
        FragmentMatchStruct current
        FragmentMatchStructArray* results

    results = <FragmentMatchStructArray*>malloc(sizeof(FragmentMatchStructArray))
    results.size = 10
    results.matches = <FragmentMatchStruct*>malloc(sizeof(FragmentMatchStruct) * 10)
    j = 0

    for i in range(ms.peak_match_list.size):
        current = ms.peak_match_list.matches[i]
        if current.peak_index == peak_index:
            results.matches[j] = current
            j += 1
            if j == 10:
                break

    results.size = j
    return results


# Sort PeakStructArray


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


# IonType Mappings


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


cdef IonTypeIndex* ION_TYPE_INDEX
ION_TYPE_INDEX = new_ion_type_index(["b", "y", "stub_ion", "oxonium", "c", "z", "noise", "other"], 8)


cdef double sum_ion_double_map(IonTypeDoubleMap* mapper) nogil:
    cdef:
        double total
        size_t i
    total = 0.
    for i in range(mapper.index_ref.size):
        total += ion_type_double_get(mapper, i)
    return total


# Struct Free Functions


cdef void free_fragment_match_struct_array(FragmentMatchStructArray* matches) nogil:
    free(matches.matches)
    free(matches)


cdef void free_peak_struct_array(PeakStructArray* peaks) nogil:
    free(peaks.peaks)
    free(peaks)


cdef void free_matched_spectrum_struct(MatchedSpectrumStruct* ms) nogil:
    free_peak_struct_array(ms.peak_list)
    free_fragment_match_struct_array(ms.peak_match_list)


cdef void free_ion_type_index(IonTypeIndex* mapper) nogil:
    free(mapper.indices)
    free(mapper.names)
    free(mapper)


cdef void free_ion_type_double_map(IonTypeDoubleMap* mapper) nogil:
    free(mapper.values)
    free(mapper)


cdef void free_ms_feature_struct(MSFeatureStruct* feature) nogil:
    free(feature.ion_type_matches)
    free(feature.ion_type_totals)
    free(feature)


cdef void free_ms_feature_struct_array(MSFeatureStructArray* features) nogil:
    cdef:
        size_t i
    for i in range(features.size):
        free_ms_feature_struct(&features.features[i])
    free(features.features)
    free(features)

# Python Wrappers


def pintensity_rank(list peak_list, float minimum_intensity=100.):
    '''
    Python-accessible wrapper for `intensity_rank`

    See Also
    --------
    intensity_rank
    '''
    intensity_rank(peak_list, minimum_intensity)


def pintensity_ratio_function(DPeak peak1, DPeak peak2):
    '''
    Python-accessible wrapper for `intensity_ratio_function`

    See Also
    --------
    intensity_ratio_function
    '''
    return intensity_ratio_function(peak1, peak2)
