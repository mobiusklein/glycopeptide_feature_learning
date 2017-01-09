from cpython.sequence cimport PySequence_ITEM
from cpython.string cimport PyString_AsString, PyString_FromString
from cpython.ref cimport PyObject
from cpython.dict cimport PyDict_GetItem, PyDict_SetItem
from cpython.list cimport PyList_GET_ITEM, PyList_Append, PyList_GET_SIZE
from cpython.object cimport PyObject_CallFunctionObjArgs

cimport cython
cimport cython.parallel
from cython.parallel cimport parallel, prange
from cython.parallel import prange, threadid
from openmp cimport omp_get_num_threads, omp_get_thread_num, omp_in_parallel
from libc.string cimport strcmp
from libc.stdlib cimport malloc, free, realloc
from libc cimport *
cdef extern from * nogil:
    int printf   (const char *template, ...)

from collections import Counter, namedtuple



from glycresoft_sqlalchemy.utils.ccommon_math cimport (
    DPeak, MassOffsetFeature,
    ppm_error, intensity_rank,
    intensity_ratio_function, search_spectrum, unwrap_matched_spectrum,
    MatchedSpectrum, MatchedSpectrumStruct, MatchedSpectrumStructArray,
    wrap_matched_spectrum_struct,
    matched_spectrum_struct_peak_explained_by,
    _intensity_rank, _search_spectrum, _openmp_search_spectrum,
    _intensity_ratio_function, _precursor_context,
    FragmentMatchStructArray, FragmentMatchStruct, FragmentMatch,
    PeakStructArray,
    unwrap_feature_functions, wrap_feature, wrap_peak,
    MSFeatureStructArray,
    free_fragment_match_struct_array, free_ion_type_double_map,
    IonTypeDoubleMap, new_ion_type_double_map,
    ion_type_index, ion_type_name, ION_TYPE_INDEX,
    ion_type_double_set, ion_type_double_get,
    ion_type_double_inc, ion_type_double_inc_name, NOISE,
    dict_from_ion_type_double_map,
    print_ion_type_index
    )


cdef str PNOISE = PyString_FromString(NOISE)


cdef class PeakRelation(object):

    def __init__(self, DPeak from_peak, DPeak to_peak, MassOffsetFeature feature, int intensity_ratio, str kind=None,
                 MatchedSpectrum annotation=None):
        self.from_peak = from_peak
        self.to_peak = to_peak
        self.from_charge = from_peak.charge
        self.to_charge = to_peak.charge
        self.feature = feature
        self.intensity_ratio = intensity_ratio
        self.kind = kind
        self.annotation = annotation

    def __repr__(self):
        template = "<PeakRelation {s.from_peak.neutral_mass}({s.from_charge}) ->" +\
            " {s.to_peak.neutral_mass}({s.to_charge}) by {s.feature.name} on {s.kind}>"
        return template.format(s=self)

    def __getstate__(self):
        d = {}
        d["from_peak"] = self.from_peak
        d["to_peak"] = self.to_peak
        d["from_charge"] = self.from_charge
        d["to_charge"] = self.to_charge
        d["feature"] = self.feature
        d["intensity_ratio"] = self.intensity_ratio
        d["kind"] = self.kind
        d['annotation'] = self.annotation
        return d

    def __setstate__(self, d):
        self.from_peak = d["from_peak"]
        self.to_peak = d["to_peak"]
        self.from_charge = d["from_charge"]
        self.to_charge = d["to_charge"]
        self.feature = d["feature"]
        self.intensity_ratio = d["intensity_ratio"]
        self.kind = d["kind"]
        self.annotation = d['annotation']

    def __reduce__(self):
        return PeakRelation, (self.from_peak, self.to_peak, self.feature,
                              self.intensity_ratio, self.kind, self.annotation)


FeatureSpecialization = namedtuple("FeatureSpecialization", [
    "glycan_peptide_ratio", "peptide_mass_rank", "base_peak_rank",
    "from_charge", "to_charge", "intensity_ratio", "ion_type"])


cdef class FittedFeature(object):
    def __init__(self, feature, relations=None):
        if relations is None:
            relations = []
        self.feature = feature
        self.relations = relations

    property distribution:
        def __get__(self):
            return self.feature.ion_type_matches

    def __repr__(self):
        temp = "<FittedFeature ({feature}) {count_relations}>\n{dist}"
        return temp.format(
            feature=self.feature, count_relations=len(self.relations),
            dist=', '.join("%s=%0.4f" % (k, self.posterior(k)) for k in self.distribution if self.posterior(k) > 0.0))

    def __reduce__(self):
        return FittedFeature, (self.feature, self.relations)

    def __iter__(self):
        return iter(self.feature.ion_type_matches)

    def probability(self, ion_type):
        count = self.feature.ion_type_matches.get(ion_type, 0.0)
        total = sum(self.feature.ion_type_matches.values())
        return count / total

    def posterior(self, ion_type):
        feature = self.feature
        on_kind_matches = feature.ion_type_matches[ion_type]
        on_kind = feature.ion_type_totals[ion_type]

        off_kind_matches = sum(feature.ion_type_matches.values()) - on_kind_matches
        off_kind = sum(feature.ion_type_totals.values()) - on_kind

        if on_kind == 0:
            on_kind = 1

        a = on_kind_matches / on_kind
        b = off_kind_matches / off_kind

        return (a) / (a + b)

    def partitions(self, minimum_count=10):
        counter = Counter()
        for rel in self:
            fs = FeatureSpecialization(
                glycan_peptide_ratio=rel.annotation.glycan_peptide_ratio,
                peptide_mass_rank=rel.annotation.peptide_mass_rank,
                base_peak_rank=rel.from_peak.rank,
                from_charge=rel.from_charge,
                to_charge=rel.to_charge,
                intensity_ratio=rel.intensity_ratio,
                ion_type=rel.kind
                )
            counter[fs] += 1

        counter = {k: v for k, v in counter.items() if v >= minimum_count}
        return counter

    def specialize(self, minimum_count=10):
        counter = self.partitions(minimum_count)
        parameters = set()
        specialized_features = []
        for params, _ in counter.items():
            if params.ion_type != PNOISE:
                parameters.add(params)

        for params in parameters:
            feature = MassOffsetFeature(
                offset=self.feature.offset,
                tolerance=self.feature.tolerance,
                name=self.feature.name,
                intensity_ratio=params.intensity_ratio,
                from_charge=params.from_charge,
                to_charge=params.to_charge,
                feature_type=self.feature.feature_type,
                min_peak_rank=params.base_peak_rank,
                max_peak_rank=params.base_peak_rank,
                peptide_mass_rank=params.peptide_mass_rank,
                glycan_peptide_ratio=params.glycan_peptide_ratio,
                fixed=self.feature.fixed)

            specialized_features.append(feature)
        return specialized_features

    def peak_relations(self, include_noise=True):
        for spectrum_match, peak_relations in self.relations:
            for pr in peak_relations:
                if not include_noise and pr.kind == PNOISE:
                    continue
                yield pr

    def __iter__(self):
        return self.peak_relations()

    def __call__(self, *args, **kwargs):
        return self.feature(*args, **kwargs)

    def as_records(self):
        for key in self.distribution:
            if self.feature.ion_type_totals[key] == 0:
                continue
            yield {
                "feature": self.feature.name,
                "kind": key,
                "probability": self.probability(key),
                "posterior": self.posterior(key),
                "intensity_ratio": self.feature.intensity_ratio,
                "from_charge": self.feature.from_charge,
                "to_charge": self.feature.to_charge,
                "min_peak_rank": self.feature.min_peak_rank,
                "max_peak_rank": self.feature.max_peak_rank,
                "glycan_peptide_ratio": self.feature.glycan_peptide_ratio,
                "peptide_mass_rank": self.feature.peptide_mass_rank,
                "match_count": self.feature.ion_type_matches[key],
                "total_count": self.feature.ion_type_totals[key]
            }



cdef object PeakRelation__new__ = PeakRelation.__new__


cdef inline PeakRelation make_peak_relation(DPeak from_peak, DPeak to_peak, MassOffsetFeature feature, int intensity_ratio, str kind=None):
    cdef PeakRelation pr
    pr = <PeakRelation>PyObject_CallFunctionObjArgs(PeakRelation__new__, <PyObject*>PeakRelation, NULL)
    pr.from_peak = from_peak
    pr.to_peak = to_peak
    pr.from_charge = from_peak.charge
    pr.to_charge = to_peak.charge
    pr.feature = feature
    pr.intensity_ratio = intensity_ratio
    pr.kind = kind
    return pr


cpdef dict search_features_on_spectrum(DPeak peak, list peak_list, list features, str kind=None):
    cdef:
        dict peak_match_relations = {}
        list match_list
        PyObject* temp
        Py_ssize_t peak_ind, feature_ind
        DPeak query_peak
        MassOffsetFeature feature
        PeakRelation pr

    for i in range(PyList_GET_SIZE(peak_list)):
        query_peak = <DPeak>PyList_GET_ITEM(peak_list, i)
        for j in range(PyList_GET_SIZE(features)):
            feature = <MassOffsetFeature>PyList_GET_ITEM(features, j)
            if feature.test(peak, query_peak):
                temp = PyDict_GetItem(peak_match_relations, feature)
                if temp == NULL:
                    match_list = []
                else:
                    match_list = <list>temp
                pr = make_peak_relation(peak, query_peak, feature, intensity_ratio_function(peak, query_peak), kind)
                match_list.append(pr)
                if temp == NULL:
                    PyDict_SetItem(peak_match_relations, feature, match_list)
    return peak_match_relations


cpdef FittedFeature feature_function_estimator(list gsms, MassOffsetFeature feature_function, str kind='b'):
    cdef:
        DPeak peak
        DPeak match
        int rank
        list peak_relations
        list related
        list peaks, _peaks
        list matches
        Py_ssize_t i, j, k, m, n
        PeakRelation pr
        set peak_explained_by
        list peak_explained_by_list, match_explained_by_list
        dict db_search_match
        str ion_type, match_ion_type
        MatchedSpectrum gsm
        # Used for fixed feature types
        bint on_type_match

    peak_relations = []
    for i in range(PyList_GET_SIZE(gsms)):
        gsm = <MatchedSpectrum>PyList_GET_ITEM(gsms, i)
        if not feature_function._precursor_context(gsm):
            continue
        peaks = gsm.peak_list
        intensity_rank(peaks)
        _peaks = []
        # peaks = [p for p in peaks if p.rank > 0]
        for j in range(PyList_GET_SIZE(peaks)):
            peak = <DPeak>PyList_GET_ITEM(peaks, j)
            rank = peak.rank
            if rank > 0:
                PyList_Append(_peaks, peak)
        peaks = _peaks
        related = []
        for j in range(PyList_GET_SIZE(peaks)):
            peak = <DPeak>PyList_GET_ITEM(peaks, j)
            peak_explained_by = gsm.peak_explained_by(peak.scan_peak_index)
            peak_explained_by_list = list(peak_explained_by)

            matches = search_spectrum(peak, peaks, feature_function)
            for k in range(PyList_GET_SIZE(matches)):
                match = <DPeak>PyList_GET_ITEM(matches, k)
                pr = make_peak_relation(peak, match, feature_function, intensity_ratio_function(peak, match))
                related.append(pr)
                # Add check for fixed features and agreement between annotations between peak and match

                if PyList_GET_SIZE(peak_explained_by_list) == 0:
                    feature_function.ion_type_increment(PNOISE)
                else:
                    match_explained_by_list = list(gsm.peak_explained_by(match.scan_peak_index))

                    for m in range(PyList_GET_SIZE(peak_explained_by_list)):
                        ion_type = <str>PyList_GET_ITEM(peak_explained_by_list, m)
                        if feature_function.fixed:
                            on_type_match = False
                            for n in range(PyList_GET_SIZE(match_explained_by_list)):
                                match_ion_type = <str>PyList_GET_ITEM(match_explained_by_list, n)
                                if ion_type == match_ion_type:
                                    on_type_match = True
                                    break
                            if not on_type_match:
                                ion_type = PNOISE

                        feature_function.ion_type_increment(ion_type)

            # Add check for fixed features and agreement between annotations between peak and match
            if PyList_GET_SIZE(peak_explained_by_list) == 0:
                feature_function.ion_type_increment(PNOISE, "totals")
            for k in range(PyList_GET_SIZE(peak_explained_by_list)):
                ion_type = <str>PyList_GET_ITEM(peak_explained_by_list, k)
                feature_function.ion_type_increment(ion_type, "totals")
                 

        if PyList_GET_SIZE(related) > 0:
            peak_relations.append((gsm, related))


    return FittedFeature(feature_function, peak_relations)


cdef FittedFeatureStruct* _feature_function_estimator(MatchedSpectrumStructArray* gsms, MSFeatureStruct* feature, char* kind, bint filter_peak_ranks) nogil:
    cdef:
        long relation_count, peak_count
        char* ion_type,
        char* matched_ion_type
        bint is_on_kind
        size_t i, j, k, m, relations_counter, features_found_counter, base_relations_size, n
        PeakStruct peak
        PeakStruct match
        PeakRelationStructArray* related
        RelationSpectrumPairArray* peak_relations
        PeakStructArray* peaks
        PeakStructArray* matches
        MatchedSpectrumStruct* gsm
        FragmentMatchStructArray* peak_explained_by_list
        FragmentMatchStructArray* match_explained_by_list

        FragmentMatchStruct fragment_match
        FittedFeatureStruct* result

    features_found_counter = 0

    peak_relations = <RelationSpectrumPairArray*>malloc(sizeof(RelationSpectrumPairArray))
    peak_relations.pairs = <RelationSpectrumPair*>malloc(sizeof(RelationSpectrumPair) * gsms.size)
    peak_relations.size = gsms.size

    if filter_peak_ranks:
        _preprocess_peak_lists(gsms)
    for i in range(gsms.size):
        gsm = &gsms.matches[i]
        if not _precursor_context(feature, gsm):
            continue
        peaks = gsm.peak_list

        # Pre-allocate space for a handful of relations assuming that there won't
        # be many per spectrum
        base_relations_size = 3
        related = <PeakRelationStructArray*>malloc(sizeof(PeakRelationStructArray))
        related.relations = <PeakRelationStruct*>malloc(sizeof(PeakRelationStruct) * base_relations_size)
        related.size = base_relations_size
        relations_counter = 0

        # Search all peaks against all other peaks using the given feature
        for j in range(peaks.size):
            peak = peaks.peaks[j]

            # Determine if peak has been identified as on-kind or off-kind
            peak_explained_by_list = matched_spectrum_struct_peak_explained_by(gsm, peak.scan_peak_index)
            
            # Perform search
            matches = _openmp_search_spectrum(&peak, peaks, feature)

            # For each match, construct a PeakRelationStruct and save it to `related`
            for k in range(matches.size):
                match = matches.peaks[k]
                # Make sure not to overflow `related.relations`. Should be rare because of pre-allocation
                # Add check for fixed features and agreement between annotations between peak and match

                if feature.fixed:
                    match_explained_by_list = matched_spectrum_struct_peak_explained_by(gsm, match.scan_peak_index)

                if peak_explained_by_list.size == 0:
                    if relations_counter == related.size:
                        resize_peak_relation_array(related)

                    ion_type_double_inc_name(feature.ion_type_matches, NOISE)
                    insert_peak_relation_struct(related, relations_counter, peak, match, feature,
                                                _intensity_ratio_function(&peak, &match), NOISE, gsm)
                    relations_counter += 1
                else:
                    for m in range(peak_explained_by_list.size):
                        if relations_counter == related.size:
                            resize_peak_relation_array(related)

                        fragment_match = peak_explained_by_list.matches[m]
                        ion_type = fragment_match.ion_type

                        # This value will be passed through to the PeakRelation, where its life time should
                        # be independent, so it is translated through the ION_TYPE_INDEX global variable which
                        # lives for the duration of the program
                        ion_type = ion_type_name(ION_TYPE_INDEX, ion_type_index(ION_TYPE_INDEX, ion_type))

                        if feature.fixed:
                            is_on_kind = False
                            for n in range(match_explained_by_list.size):
                                matched_ion_type = match_explained_by_list.matches[n].ion_type
                                matched_ion_type = ion_type_name(ION_TYPE_INDEX, ion_type_index(ION_TYPE_INDEX, matched_ion_type))
                                if strcmp(matched_ion_type, ion_type) == 0:
                                    is_on_kind = True
                                    break
                            if not is_on_kind:
                                ion_type = NOISE



                        ion_type_double_inc_name(
                            feature.ion_type_matches,
                            ion_type)

                        insert_peak_relation_struct(related, relations_counter, peak, match, feature,
                                                    _intensity_ratio_function(&peak, &match),
                                                    ion_type, gsm)
                        relations_counter += 1

                if feature.fixed:
                    free_fragment_match_struct_array(match_explained_by_list)


            if peak_explained_by_list.size == 0:
                ion_type_double_inc_name(feature.ion_type_totals, NOISE)
            else:
                for k in range(peak_explained_by_list.size):
                    fragment_match = peak_explained_by_list.matches[k]
                    ion_type = fragment_match.ion_type
                    ion_type_double_inc_name(
                        feature.ion_type_totals,
                        ion_type)
            free_fragment_match_struct_array(peak_explained_by_list)
            free(matches.peaks)
            free(matches)
        related.relations = <PeakRelationStruct*>realloc(related.relations, sizeof(PeakRelationStruct) * relations_counter)
        related.size = relations_counter

            # Save results or free un-needed memory
        if relations_counter > 0:
            if features_found_counter == peak_relations.size:
                peak_relations.pairs = <RelationSpectrumPair*>realloc(peak_relations.pairs, sizeof(RelationSpectrumPair) * peak_relations.size * 2)
                peak_relations.size = peak_relations.size * 2
            peak_relations.pairs[features_found_counter].matched_spectrum = gsm
            peak_relations.pairs[features_found_counter].relations = related
            features_found_counter += 1
        else:
            free_peak_relations_array(related)

    peak_relations.pairs = <RelationSpectrumPair*>realloc(peak_relations.pairs, sizeof(RelationSpectrumPair) * features_found_counter)
    peak_relations.size = features_found_counter

    result = <FittedFeatureStruct*>malloc(sizeof(FittedFeatureStruct))
    result.feature = feature
    result.relation_pairs = peak_relations
    return result



cdef void preprocess_peak_lists(list gsms):
    cdef:
        DPeak peak
        DPeak match
        int rank
        list peaks, _peaks
        Py_ssize_t i, j, k
        MatchedSpectrum gsm

    for i in range(PyList_GET_SIZE(gsms)):
        gsm = <MatchedSpectrum>PyList_GET_ITEM(gsms, i)
        peaks = gsm.peak_list
        intensity_rank(peaks)
        _peaks = []
        for j in range(PyList_GET_SIZE(peaks)):
            peak = <DPeak>PyList_GET_ITEM(peaks, j)
            rank = peak.rank
            if rank > 0:
                PyList_Append(_peaks, peak)
        gsm.peaks = _peaks


cdef void _preprocess_peak_lists(MatchedSpectrumStructArray* gsms) nogil:
    cdef:
        MatchedSpectrumStruct* gsm
        size_t i, j, peak_count
        PeakStructArray* peaks
        PeakStruct* _peaks
        PeakStruct stack_peak

    for i in range(gsms.size):
        gsm = &gsms.matches[i]
        peaks = gsm.peak_list
        _intensity_rank(peaks)
        peak_count = 0
        _peaks = <PeakStruct*>malloc(sizeof(PeakStruct) * peaks.size)
        for j in range(peaks.size):
            stack_peak = peaks.peaks[j]
            if stack_peak.rank > 0:
                _peaks[peak_count] = stack_peak
                peak_count += 1
        free(peaks.peaks)
        peaks.peaks = _peaks
        peaks.size = peak_count


def fit_features(list gsms, list features, str kind, int n_threads=8):
    cdef:
        size_t i
        long n_p, i_p
        MatchedSpectrumStructArray* spectra
        MSFeatureStruct cfeature
        MSFeatureStructArray* cfeatures
        char* ckind
        FittedFeatureStruct* out
        FittedFeatureStruct ff
        double start, end
    spectra = <MatchedSpectrumStructArray*>malloc(sizeof(MatchedSpectrumStructArray))
    spectra.matches = <MatchedSpectrumStruct*>malloc(sizeof(MatchedSpectrumStruct) * len(gsms))
    spectra.size = len(gsms)
    from time import time
    matched_spectrum_map = {gsm.id: gsm for gsm in gsms}
    for i in range(spectra.size):
        spectra.matches[i] = unwrap_matched_spectrum(gsms[i])[0]

    cfeatures = unwrap_feature_functions(features)
    print cfeatures.size
    ckind = PyString_AsString(kind)
    print ckind
    start = time()
    n_p = cfeatures.size
    out = <FittedFeatureStruct*>malloc(sizeof(FittedFeatureStruct) * n_p)
    _preprocess_peak_lists(spectra)

    print "begin parallel"
    with nogil, parallel(num_threads=n_threads):

        for i_p in prange(n_p, schedule='guided'):
            # printf("%d Thread id %d of %d -- %d\n", n_p, omp_get_thread_num(), omp_get_num_threads(), omp_in_parallel())
            out[i_p] = _feature_function_estimator(spectra, &cfeatures.features[i_p], ckind, False)[0]

    end = time() - start
    print end
    copy = list()
    for i in range(n_p):
        copy.append(wrap_fitted_feature(&out[i], matched_spectrum_map))
        print(copy[-1])
        free_fitted_feature(&out[i])
    free(out)
    print end
    return copy


cdef PeakRelation wrap_peak_relation(PeakRelationStruct* peak_relation, MassOffsetFeature feature, dict spectrum_match_map):
    cdef PeakRelation py_peak_relation
    py_peak_relation = PeakRelation(
        wrap_peak(&(peak_relation.from_peak)),
        wrap_peak(&(peak_relation.to_peak)),
        feature,
        peak_relation.intensity_ratio,
        peak_relation.kind,
        spectrum_match_map.get(peak_relation.annotation.id))
    return py_peak_relation

cdef tuple wrap_spectrum_relation_pair(RelationSpectrumPair* pair, MassOffsetFeature feature, dict spectrum_match_map):
    cdef list relations = []
    cdef size_t i
    cdef PeakRelation pr

    for i in range(pair.relations.size):
        pr = wrap_peak_relation(&pair.relations.relations[i], feature, spectrum_match_map)
        relations.append(pr)
    return (spectrum_match_map.get(pair.matched_spectrum.id), relations)

cdef FittedFeature wrap_fitted_feature(FittedFeatureStruct* feature, dict spectrum_match_map):
    cdef:
        size_t i, j
        MassOffsetFeature py_feature
        PeakRelation py_pr
        PeakRelationStruct pr
        FittedFeature fitted_feature
        tuple relations
        list pairs = []
    py_feature = wrap_feature(feature.feature)
    for i in range(feature.relation_pairs.size):
        relations = wrap_spectrum_relation_pair(&feature.relation_pairs.pairs[i], py_feature, spectrum_match_map)
        pairs.append(relations)
    fitted_feature = FittedFeature(py_feature, pairs)
    return fitted_feature


cdef void resize_peak_relation_array(PeakRelationStructArray* related) nogil:
    related.relations = <PeakRelationStruct*>realloc(related.relations, sizeof(PeakRelationStruct) * related.size * 2)
    related.size = related.size * 2


cdef inline void insert_peak_relation_struct(PeakRelationStructArray* related, size_t index, PeakStruct peak, PeakStruct match,
                                             MSFeatureStruct* feature, int intensity_ratio, char* kind,
                                             MatchedSpectrumStruct* annotation) nogil:
    related.relations[index].from_peak = peak
    related.relations[index].to_peak = match
    related.relations[index].feature = feature
    related.relations[index].intensity_ratio = intensity_ratio
    related.relations[index].kind = kind
    related.relations[index].annotation = annotation


cdef void free_peak_relations_array(PeakRelationStructArray* relations) nogil:
    free(relations.relations)
    free(relations)


cdef void free_relation_spectrum_pair(RelationSpectrumPair* pair) nogil:
    free_peak_relations_array(pair.relations)


cdef void free_relation_spectrum_pairs_array(RelationSpectrumPairArray* pairs_array) nogil:
    cdef size_t i
    for i in range(pairs_array.size):
        free_peak_relations_array(pairs_array.pairs[i].relations)
    free(pairs_array.pairs)
    free(pairs_array)

cdef void free_fitted_feature(FittedFeatureStruct* fitted_feature) nogil:
    free_relation_spectrum_pairs_array(fitted_feature.relation_pairs)


cdef void peak_score_distribution(PeakAnnotationStruct* peak) nogil:
    cdef:
        IonTypeDoubleMap* distribution

    distribution = new_ion_type_double_map()

