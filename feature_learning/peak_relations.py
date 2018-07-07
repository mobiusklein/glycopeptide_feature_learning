import numpy as np
from itertools import chain

from collections import Counter, defaultdict, namedtuple
from collections import Mapping

from glycopeptidepy.utils import collectiontools
from glycopeptidepy.structure.sequence import (
    PeptideSequence)
from glycopeptidepy.structure.fragment import (PeptideFragment, IonSeries)
from glycopeptidepy.structure.sequence_composition import (
    AminoAcidSequenceBuildingBlock, AminoAcidResidue, Modification)

from .common import (
    OUT_OF_RANGE_INT, ppm_error,
    intensity_ratio_function, intensity_rank)


from ms_deisotope import DeconvolutedPeakSet

from glycan_profiling.structure.fragment_match_map import PeakFragmentPair


# Lacking a reasonable definition of the "space between fragmentation sites"
SMALLEST_UNIT = 1000 * 2e-5
NOISE = "noise"


chain_iterable = chain.from_iterable


def _delta_series(sequence, mass_shift=0.0):
    sequence = PeptideSequence(str(sequence))

    for frags in sequence.get_fragments(IonSeries.b):
        out = []
        for frag in frags:
            o = PeptideFragment(
                str(mass_shift), frag.position, frag.modification_dict,
                frag.bare_mass - frag.series.mass_shift + mass_shift, None,
                frag.flanking_amino_acids, frag.glycosylation,
                frag.chemical_shift, frag.composition)
            out.append(o)
            yield out


def delta_finder(gsms, delta):
    total_sites = 0
    total_explained = 0

    total_sparsity = 0
    total_unexplained = 0
    for gsm in gsms:
        fragments = list((_delta_series(gsm.structure, delta)))
        n_frag_sites = len(fragments)
        peak_list = list(gsm)
        match_count = 0
        for fragment_site in fragments:
            for fragment in fragment_site:
                for peak in gsm.deconvoluted_peak_set.all_peaks_for(fragment.mass):
                    match_count += 1

        total_explained += match_count
        total_sites += n_frag_sites
        total_unexplained += len(peak_list) - match_count
        total_sparsity += mass_accuracy_sparsity(gsm.structure, IonSeries.b) - n_frag_sites

    return total_explained / float(total_sites), total_unexplained / float(total_sparsity)


# BEGIN REFERENCE Estimation
# Simple one parameter estimator functions for learning the basic alpha, beta and p parameters


def offset_frequency(gsms, series=IonSeries.b):
    total_sites = 0
    total_explained = 0
    for gsm in gsms:
        n_frag_sites = count_fragmentation_sites(gsm.glycopeptide_sequence, series)
        series_explained = sum([1 for i in chain_iterable(gsm.peak_match_map.values()) if i['key'][0] == series])
        total_sites += n_frag_sites
        total_explained += series_explained
    return (total_explained) / float(total_sites)


def unknown_peak_rate(gsms, series=IonSeries.b):
    total_sparsity = 0
    total_unexplained = 0

    for gsm in gsms:
        sequence = gsm.glycopeptide_sequence
        n_frag_sites = count_fragmentation_sites(sequence, series)
        series_explained = sum([1 for i in chain_iterable(gsm.peak_match_map.values()) if i['key'][0] == series])
        peaks_unexplained = gsm.peaks_unexplained + (gsm.peaks_explained - series_explained)
        total_unexplained += peaks_unexplained
        total_sparsity += mass_accuracy_sparsity(sequence, series) - n_frag_sites

    return total_unexplained / float(total_sparsity)


def count_fragmentation_sites(sequence, series=IonSeries.b):
    sequence = PeptideSequence(str(sequence))
    fragmentation_sites = len(collectiontools.flatten(sequence.get_fragments(series)))
    return fragmentation_sites


def prior_fragment_probability(gsms, series=IonSeries.b):
    hits = 0
    for gsm in gsms:
        sequence = PeptideSequence(gsm.glycopeptide_sequence)
        random_mass = np.random.uniform(56., sequence.mass)
        for fragment in collectiontools.flatten(sequence.get_fragments(series)):
            if abs(ppm_error(fragment.mass, random_mass)) <= 2e-5:
                hits += 1
            # elif fragment.mass - (random_mass + 230.) > 0:
            #     break
    return hits / float(len(gsms))

# END REFERENCE Estimation


def estimate_fragment_sparsity(sequence, series=IonSeries.b):
    return sequence.mass / SMALLEST_UNIT


def mass_accuracy_sparsity(sequence, series=IonSeries.b, tolerance=2e-5):
    mass_space, fragment_space = mass_dimensions(sequence, series, tolerance)
    return mass_space - fragment_space


def mass_dimensions(sequence, series=IonSeries.b, tolerance=2e-5):
    mass_space = sequence.mass
    fragment_space = 0.
    for frag in chain_iterable(sequence.get_fragments(series)):
        radius = tolerance * frag.mass
        fragment_space += 2 * radius
    return mass_space, fragment_space


def estimate_offset_parameters(gpsms, series=IonSeries.b, mass_accuracy=2e-5, charge=None, prematched=False):
    total_sites = 0
    total_explained = 0
    total_sparsity = 0
    total_unexplained = 0
    mass_space_area = 0
    fragment_space_area = 0

    i = 0.
    for gpsm in gpsms:
        sequence = gpsm.structure
        fragments = [b for a in sequence.get_fragments(series) for b in a]

        n_frag_sites = len(fragments)

        if not prematched:
            match = gpsm.match(error_tolerance=mass_accuracy)
        else:
            match = gpsm.matcher

        if charge is None:
            series_explained = sum(
                [1 for d in match.solution_map.by_fragment if d.series == series])
        else:
            series_explained = 0
            for peak, fragment in match.solution_map:
                if peak.charge == charge and fragment.series == series:
                    series_explained += 1

        total_sites += n_frag_sites
        total_explained += series_explained

        if charge is None:
            peaks_unexplained = len(gpsm.deconvoluted_peak_set) - series_explained
        else:
            peaks_unexplained = len([p for p in gpsm.deconvoluted_peak_set if p.charge == charge]) - series_explained

        total_unexplained += peaks_unexplained
        total_sparsity += estimate_fragment_sparsity(sequence, series) - n_frag_sites

        s, f = mass_dimensions(sequence, series=series, tolerance=mass_accuracy)
        mass_space_area += s
        fragment_space_area += f

        i += 1

    alpha = total_explained / float(total_sites)
    beta = total_unexplained / float(total_sparsity)
    prior_probability_of_match = fragment_space_area / mass_space_area
    return alpha, beta, prior_probability_of_match


def probability_of_peak_explained(offset_frequency, unknown_peak_rate, prior_probability_of_match):
    a = (prior_probability_of_match * offset_frequency)
    b = (1 - prior_probability_of_match) * unknown_peak_rate
    return a / (a + b)


def fragmentation_probability(peak, probability_of_fragment, features):
    probability_of_noise = 1 - probability_of_fragment
    numer = 1
    denom = 1
    if len(features) == 0:
        return probability_of_fragment
    for feature in features:
        match = probability_of_fragment * feature.on_series
        total = match + (probability_of_noise * feature.off_series)
        numer *= match
        denom *= total
    return numer / denom


def _load_feature_from_json(cls, d):
    feature_type = d['feature_type']
    if feature_type == LinkFeature.feature_type:
        return LinkFeature.from_json(d)
    else:
        return MassOffsetFeature.from_json(d)


class FeatureBase(object):
    def __init__(self, tolerance=2e-5, name=None, intensity_ratio=OUT_OF_RANGE_INT,
                 from_charge=OUT_OF_RANGE_INT, to_charge=OUT_OF_RANGE_INT, feature_type='',
                 terminal=''):
        self.name = name
        self.tolerance = tolerance
        self.intensity_ratio = intensity_ratio
        self.from_charge = from_charge
        self.to_charge = to_charge
        self.feature_type = feature_type
        self.terminal = terminal

    def __eq__(self, other):
        v = self.intensity_ratio == other.intensity_ratio
        if not v:
            return v
        v = self.from_charge == other.from_charge
        if not v:
            return v
        v = self.to_charge == other.to_charge
        if not v:
            return v
        v = self.feature_type == other.feature_type
        if not v:
            return v
        # v = self.terminal == other.terminal
        # if not v:
        #     return v
        return True

    def __ne__(self, other):
        return not (self == other)

    def find_matches(self, peak, peak_list, structure=None):
        raise NotImplementedError()

    def is_valid_match(self, from_peak, to_peak, solution_map, structure=None):
        return to_peak in solution_map.by_peak

    def specialize(self, from_charge, to_charge, intensity_ratio):
        raise NotImplementedError()

    def unspecialize(self):
        raise NotImplementedError()

    def __call__(self, peak1, peak2, structure=None):
        raise NotImplementedError()

    def to_json(self):
        d = {}
        d['name'] = self.name
        d['tolerance'] = self.tolerance
        d['intensity_ratio'] = self.intensity_ratio
        d['from_charge'] = self.from_charge
        d['to_charge'] = self.to_charge
        d['feature_type'] = self.feature_type
        d['terminal'] = self.terminal
        return d

    from_json = classmethod(_load_feature_from_json)


try:
    _FeatureBase = FeatureBase
    from feature_learning._c.peak_relations import FeatureBase as CFeatureBase

    class FeatureBase(CFeatureBase):
        from_json = classmethod(_load_feature_from_json)

except ImportError:
    pass


class MassOffsetFeature(FeatureBase):
    def __init__(self, offset, tolerance=2e-5, name=None, intensity_ratio=OUT_OF_RANGE_INT,
                 from_charge=OUT_OF_RANGE_INT, to_charge=OUT_OF_RANGE_INT, feature_type='',
                 terminal=''):
        if name is None:
            name = "F:" + str(offset)

        super(MassOffsetFeature, self).__init__(
            tolerance, name, intensity_ratio, from_charge, to_charge, feature_type,
            terminal)

        self.offset = offset
        self._hash = hash((self.offset, self.intensity_ratio, self.from_charge,
                           self.to_charge))

    def __eq__(self, other):
        v = np.isclose(self.offset, other.offset)
        if not v:
            return v
        return super(MassOffsetFeature, self).__eq__(other)

    def __hash__(self):
        return self._hash

    def test(self, peak1, peak2):
        if (self.intensity_ratio == OUT_OF_RANGE_INT or
            intensity_ratio_function(peak1, peak2) == self.intensity_ratio) and\
           ((self.from_charge == OUT_OF_RANGE_INT and self.to_charge == OUT_OF_RANGE_INT) or
                (self.from_charge == peak1.charge and self.to_charge == peak2.charge)):

            return abs((peak1.neutral_mass + self.offset - peak2.neutral_mass) / peak2.neutral_mass) <= self.tolerance
        return False

    def __call__(self, peak1, peak2, structure=None):
        return self.test(peak1, peak2)

    def find_matches(self, peak, peak_list, structure=None):
        matches = []
        for peak2 in peak_list.all_peaks_for(peak.neutral_mass + self.offset, self.tolerance):
            if self(peak, peak2) and peak is not peak2:
                matches.append(peak2)
        return matches

    def specialize(self, from_charge, to_charge, intensity_ratio):
        return self.__class__(
            self.offset, self.tolerance, self.name, intensity_ratio,
            from_charge, to_charge, self.feature_type, self.terminal)

    def unspecialize(self):
        return self.__class__(
            self.offset, self.tolerance, self.name, OUT_OF_RANGE_INT,
            OUT_OF_RANGE_INT, OUT_OF_RANGE_INT, self.feature_type, self.terminal)

    def _get_display_fields(self):
        fields = {}
        # fields['feature_type'] = self.feature_type
        fields['offset'] = self.offset
        if self.from_charge != OUT_OF_RANGE_INT:
            fields["from_charge"] = self.from_charge
        if self.to_charge != OUT_OF_RANGE_INT:
            fields['to_charge'] = self.to_charge
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
        return terms

    def __repr__(self):
        terms = self._get_display_fields()
        return "{}(name={!r}, {})".format(
            self.__class__.__name__, self.name, ", ".join(terms))

    def to_json(self):
        d = super(MassOffsetFeature, self).to_json()
        d['offset'] = self.offset
        return d

    @classmethod
    def from_json(cls, d):
        inst = cls(
            d['offset'], d['tolerance'], d['name'], d['intensity_ratio'],
            d['from_charge'], d['to_charge'], d['feature_type'], d['terminal'])
        return inst


try:
    _MassOffsetFeature = MassOffsetFeature
    from feature_learning._c.peak_relations import MassOffsetFeature
except ImportError:
    pass


class LinkFeature(MassOffsetFeature):
    feature_type = 'link'

    def __init__(self, amino_acid, tolerance=2e-5, name=None, intensity_ratio=OUT_OF_RANGE_INT,
                 from_charge=OUT_OF_RANGE_INT, to_charge=OUT_OF_RANGE_INT, feature_type=None,
                 terminal=''):
        if feature_type is None:
            feature_type = self.feature_type
        offset = amino_acid.mass
        if name is None:
            name = str(amino_acid)
        super(LinkFeature, self).__init__(
            offset, tolerance, name, intensity_ratio, from_charge, to_charge, feature_type)
        self.amino_acid = amino_acid

    def specialize(self, from_charge, to_charge, intensity_ratio):
        return self.__class__(
            self.amino_acid, self.tolerance, self.name, intensity_ratio, from_charge,
            to_charge, self.feature_type, self.terminal)

    def unspecialize(self):
        return self.__class__(
            self.amino_acid, self.tolerance, self.name, OUT_OF_RANGE_INT, OUT_OF_RANGE_INT,
            OUT_OF_RANGE_INT, self.feature_type, self.terminal)

    def is_valid_match(self, from_peak, to_peak, solution_map, structure=None):
        is_peak_expected = to_peak in solution_map.by_peak
        if not is_peak_expected:
            return False
        matched_fragments = solution_map.by_peak[from_peak]
        validated_aa = False
        for frag in matched_fragments:
            try:
                flanking_amino_acids = frag.flanking_amino_acids
            except AttributeError:
                continue
            try:
                residue = self.amino_acid.residue
            except AttributeError:
                residue = self.amino_acid
            if residue in flanking_amino_acids:
                validated_aa = True
                break
        return validated_aa

    def to_json(self):
        d = super(LinkFeature, self).to_json()
        try:
            d['amino_acid_residue'] = self.amino_acid.residue.symbol
            d['amino_acid_modification'] = [m.name for m in self.amino_acid.modifications]
        except AttributeError:
            d['amino_acid_residue'] = self.amino_acid.symbol
            d['amino_acid_modification'] = []
        return d

    @classmethod
    def from_json(cls, d):
        res = d['amino_acid_residue']
        mods = d['amino_acid_modification']
        res = AminoAcidResidue(res)
        mods = [Modification(m) for m in mods]
        amino_acid = AminoAcidSequenceBuildingBlock(res, mods)
        inst = cls(
            amino_acid, d['tolerance'], d['name'], d['intensity_ratio'],
            d['from_charge'], d['to_charge'], d['terminal'])
        return inst


class PeakRelation(object):
    __slots__ = ["from_peak", "to_peak", "feature",
                 "intensity_ratio", "series", "from_charge",
                 "to_charge"]

    def __init__(self, from_peak, to_peak, feature, intensity_ratio=None, series=None):
        if intensity_ratio is None:
            intensity_ratio = intensity_ratio_function(from_peak, to_peak)
        self.from_peak = from_peak
        self.to_peak = to_peak
        self.feature = feature
        self.intensity_ratio = intensity_ratio
        self.from_charge = from_peak.charge
        self.to_charge = to_peak.charge
        self.series = series or NOISE

    def __reduce__(self):
        return self.__class__, (self.from_peak, self.to_peak, self.feature, self.intensity_ratio, self.series)

    def __repr__(self):
        template = "<PeakRelation {s.from_peak.neutral_mass}({s.from_charge}) ->" +\
            " {s.to_peak.neutral_mass}({s.to_charge}) by {s.feature.name} on {s.series}>"
        return template.format(s=self)

    def peak_key(self):
        if self.from_peak.index.neutral_mass < self.to_peak.index.neutral_mass:
            return self.from_peak, self.to_peak
        else:
            return self.to_peak, self.from_peak


try:
    _PeakRelation = PeakRelation
    from feature_learning._c.peak_relations import PeakRelation
except ImportError:
    pass


def feature_function_estimator(gpsms, feature_function, series=IonSeries.b, tolerance=2e-5, preranked=True,
                               track_relations=True, verbose=False):
    total_on_series_satisfied = 0.
    total_off_series_satisfied = 0.
    total_on_series = 0.
    total_off_series = 0.
    peak_relations = []
    for i_scan, gpsm in enumerate(gpsms):
        if verbose and i_scan % 1000 == 0:
            print(i_scan, gpsm)
        peaks = gpsm.deconvoluted_peak_set
        try:
            if gpsm.matcher is None:
                gpsm.match(error_tolerance=tolerance)
        except Exception:
            continue
        if not preranked:
            intensity_rank(peaks)
            peaks = DeconvolutedPeakSet([p for p in peaks if p.rank > 0])
            peaks.reindex()
            gpsm.annotations['ranked_peaks'] = peaks
        else:
            try:
                peaks = gpsm.annotations['ranked_peaks']
            except KeyError:
                intensity_rank(peaks)
                peaks = DeconvolutedPeakSet([p for p in peaks if p.rank > 0])
                peaks.reindex()
                gpsm.annotations['ranked_peaks'] = peaks

        related = []
        solution_map = gpsm.solution_map
        structure = gpsm.structure
        for peak in peaks:
            is_on_series = bool([k for k in solution_map.by_peak[peak] if k.get_series() == series])
            matches = feature_function.find_matches(peak, peaks, structure)
            for match in matches:
                if peak is match:
                    continue
                if track_relations:
                    pr = PeakRelation(peak, match, feature_function, intensity_ratio_function(peak, match))
                    related.append(pr)
                is_match_expected = feature_function.is_valid_match(peak, match, solution_map, structure)
                if is_on_series and is_match_expected:
                    total_on_series_satisfied += 1
                    if track_relations:
                        pr.series = series
                else:
                    total_off_series_satisfied += 1
                    if track_relations:
                        pr.series = NOISE
            if is_on_series:
                total_on_series += 1
            else:
                total_off_series += 1
        if len(related) > 0 and track_relations:
            peak_relations.append((gpsm, related))

    total_on_series_satisfied_normalized = total_on_series_satisfied / max(total_on_series, 1)
    total_off_series_satisfied_normalized = total_off_series_satisfied / max(total_off_series, 1)

    return FittedFeature(feature_function, series, total_on_series_satisfied_normalized,
                         total_off_series_satisfied_normalized, peak_relations,
                         on_count=total_on_series_satisfied,
                         off_count=total_off_series_satisfied)


class FittedFeature(object):
    def __init__(self, feature, series, on_series, off_series, relations=None, on_count=0, off_count=0):
        if relations is None:
            relations = []
        self.feature = feature

        # forward these attributes directly rather than using a property
        # to avoid overhead
        self.from_charge = feature.from_charge
        self.to_charge = feature.to_charge

        self.series = series
        # mu
        self.on_series = on_series
        # v
        self.off_series = off_series

        # tracking purposes only
        self.on_count = on_count
        self.off_count = off_count
        self.relations = relations

    @property
    def _total_on_series(self):
        if self.on_series == 0:
            return 1
        return self.on_count / self.on_series

    @property
    def _total_off_series(self):
        if self.off_series == 0:
            return 1
        return self.off_count / self.off_series

    @property
    def name(self):
        return self.feature.name

    @property
    def observations(self):
        count = (self.on_count + self.off_count)
        if count > 0:
            return count
        count = len(list(self.peak_relations()))
        if count > 0:
            return count
        return 0

    def __hash__(self):
        return hash((self.feature, self.series))

    def __eq__(self, other):
        v = self.feature == other.feature and self.series == other.series
        if not v:
            return v
        v = np.isclose(self.on_series, other.on_series) and np.isclose(self.off_series, other.off_series)
        if not v:
            return v
        return True

    def __repr__(self):
        temp = "<FittedFeature {feature.name}, {terms} u:{on_series} v:{off_series} @ {series} {count_relations}>"
        return temp.format(
            feature=self.feature,
            terms=', '.join(map(str, self.feature._get_display_fields())),
            on_series=self.on_series, off_series=self.off_series,
            series=self.series, count_relations=self.on_count + self.off_count)

    def charge_relations(self):
        counter = Counter()
        for rel in self.peak_relations(False):
            counter[rel.from_charge, rel.to_charge] += 1
        return counter

    def intensity_ratio(self):
        counter = Counter()
        for rel in self.peak_relations(False):
            counter[intensity_ratio_function(rel.from_peak, rel.to_peak)] += 1
        return counter

    def charge_intensity_ratio(self):
        counter = Counter()
        for rel in self.peak_relations(False):
            counter[(rel.from_charge, rel.to_charge), intensity_ratio_function(rel.from_peak, rel.to_peak)] += 1
        return counter

    def peak_relations(self, include_noise=True):
        for spectrum_match, peak_relations in self.relations:
            for pr in peak_relations:
                if not include_noise and pr.series == NOISE:
                    continue
                yield pr

    def partitions(self, minimum_count=10):
        counter = Counter()
        on_counter = Counter()
        off_counter = Counter()
        for rel in self.peak_relations(True):
            fs = FeatureSpecialization(
                from_charge=rel.from_charge,
                to_charge=rel.to_charge,
                intensity_ratio=rel.intensity_ratio,
            )
            counter[fs] += 1
            if rel.series == self.series:
                on_counter[fs] += 1
            else:
                off_counter[fs] += 1

        counter = {k: (v, on_counter[k], off_counter[k]) for k, v in counter.items() if v >= minimum_count}
        return counter

    def specialize(self, minimum_count=10):
        counter = self.partitions(minimum_count)
        total_off_series = self._total_off_series
        total_on_series = self._total_on_series
        specialized_features = []
        for params, counts in counter.items():
            feature = self.feature.specialize(
                intensity_ratio=params.intensity_ratio,
                from_charge=params.from_charge,
                to_charge=params.to_charge)
            total, on, off = counts
            fit = FittedFeature(
                feature, self.series, on / total_on_series, off / total_off_series,
                None, on, off)
            specialized_features.append(fit)
        return specialized_features

    def find_matches(self, peak, peak_list, structure=None):
        return self.feature.find_matches(peak, peak_list, structure)

    def is_valid_match(self, from_peak, to_peak, solution_map, structure=None):
        return self.feature.is_valid_match(from_peak, to_peak, solution_map, structure)

    def __call__(self, peak1, peak2, structure=None):
        return self.feature(peak1, peak2, structure)

    def _feature_probability(self, p=0.5):
        return (p * self.on_series) / (
            (p * self.on_series) + ((1 - p) * self.off_series))

    def pack(self):
        self.relations = []
        return self

    def to_json(self):
        d = {}
        d['on_series'] = self.on_series
        d['on_count'] = self.on_count
        d['off_series'] = self.off_series
        d['off_count'] = self.off_count
        d['series'] = self.series.name
        d['feature'] = self.feature.to_json()
        return d

    @classmethod
    def from_json(cls, d):
        feature = FeatureBase.from_json(d['feature'])
        inst = cls(
            feature, IonSeries.get(d['series']), d["on_series"],
            d["off_series"], relations=None, on_count=d['on_count'],
            off_count=d['off_count']
        )
        return inst


FeatureSpecialization = namedtuple("FeatureSpecialization", ["from_charge", "to_charge", "intensity_ratio"])


def train_offest_features(gpsms, error_tolerance=2e-5, prematched=True, max_charge=3):
    if not prematched:
        for gpsm in gpsms:
            gpsm.match(error_tolerance=error_tolerance)

    offset_b = {}
    offset_y = {}
    offset_stub = {}

    for i in range(1, max_charge + 2):
        offset_b[i] = estimate_offset_parameters(
            gpsms, series=IonSeries.b, charge=i, prematched=1)
        offset_y[i] = estimate_offset_parameters(
            gpsms, series=IonSeries.y, charge=i, prematched=1)
        offset_stub[i] = estimate_offset_parameters(
            gpsms, series=IonSeries.stub_glycopeptide, charge=i, prematched=1)
    return offset_b, offset_y, offset_stub


def train_feature_function(gpsms, feature, preranked=True, error_tolerance=2e-5, specialize_filter=lambda x: True):
    feature_fits = {}

    if not preranked:
        for gpsm in gpsms:
            peaks = gpsm.deconvoluted_peak_set
            try:
                if gpsm.matcher is None:
                    gpsm.match(error_tolerance=error_tolerance)
            except Exception:
                continue
            if not preranked:
                intensity_rank(peaks)
                peaks = DeconvolutedPeakSet([p for p in peaks if p.rank > 0])
                peaks.reindex()
                gpsm.annotations['ranked_peaks'] = peaks

    for series in [IonSeries.b, IonSeries.y, IonSeries.stub_glycopeptide]:
        fit = feature_function_estimator(gpsms, feature, series=series, preranked=True, track_relations=True)
        specializations = filter(specialize_filter, fit.specialize())
        fit.pack()
        feature_fits[series] = specializations
    return feature_fits


class FragmentationFeature(object):
    def __init__(self, feature, fits, series):
        self.feature = feature
        self.series = series
        self.fits = dict(fits)

    def find_matches(self, peak, peak_list, structure=None):
        matches = self.feature.find_matches(peak, peak_list, structure)
        pairs = []
        for match in matches:
            rel = PeakRelation(peak, match, self.feature, series=self.series)
            try:
                rel.feature = self.fits[rel.intensity_ratio, rel.from_charge, rel.to_charge]
                pairs.append(rel)
            except KeyError:
                continue
        return pairs

    def __len__(self):
        return len(self.fits)

    def __repr__(self):
        template = "{self.__class__.__name__}(name={self.feature.name}, size={size}, series={self.series})"
        return template.format(self=self, size=len(self))

    @classmethod
    def generalize_fits(cls, fitted_features, series):
        feature_to_fits = defaultdict(dict)
        for fit in fitted_features:
            if fit.on_series < 1e-4:
                continue
            feature = fit.feature.unspecialize()
            feature_to_fits[feature][
                fit.feature.intensity_ratio, fit.feature.from_charge, fit.feature.to_charge] = fit
        result = []
        for feature, table in feature_to_fits.items():
            result.append(cls(feature, table, series))
        return result


class FragmentationModel(object):
    def __init__(self, series, features=None, on_frequency=None, off_frequency=None,
                 prior_probability_of_match=None, error_tolerance=2e-5):
        if features is None:
            features = []
        self.series = series
        self.features = features
        self.feature_table = FragmentationFeature.generalize_fits(features, self.series)
        self.error_tolerance = error_tolerance
        # alpha
        self.on_frequency = on_frequency
        # beta
        self.off_frequency = off_frequency
        # p
        self.prior_probability_of_match = prior_probability_of_match
        # gamma
        self.offset_probability = None
        if None not in (self.on_frequency, self.off_frequency, self.prior_probability_of_match):
            self.offset_probability = self._compute_offset_probability()

    def find_matches(self, scan, solution_map, structure):
        matches_to_features = defaultdict(list)
        deconvoluted_peak_set = scan.deconvoluted_peak_set
        for peak_fragment in solution_map:
            peak = peak_fragment.peak
            fragment = peak_fragment.fragment
            if fragment.get_series() != self.series:
                continue
            for feature in self.feature_table:
                rels = feature.find_matches(peak, deconvoluted_peak_set, structure)
                for rel in rels:
                    if feature.feature.is_valid_match(rel.from_peak, rel.to_peak, solution_map, structure):
                        matches_to_features[rel.from_peak].append(rel)
                        matches_to_features[rel.to_peak].append(rel)
        return matches_to_features

    def _compute_offset_probability(self):
        p_is_fragment = self.on_frequency * self.prior_probability_of_match
        p_is_not_fragment = (1 - self.prior_probability_of_match) * self.off_frequency
        return p_is_fragment / (p_is_fragment + p_is_not_fragment)

    def fit_offset(self, gpsms, prematched=True):
        if not prematched:
            for gpsm in gpsms:
                gpsm.match(error_tolerance=self.error_tolerance)
        on, off, prior = estimate_offset_parameters(gpsms, series=self.series, prematched=True)
        self.on_frequency = on
        self.off_frequency = off
        self.prior_probability_of_match = prior
        self.offset_probability = self._compute_offset_probability()

    def fit_feature(self, gpsms, feature, preranked=True, specialize_filter=lambda x: True):
        if not preranked:
            for gpsm in gpsms:
                peaks = gpsm.deconvoluted_peak_set
                try:
                    if gpsm.matcher is None:
                        gpsm.match(error_tolerance=self.error_tolerance)
                except Exception:
                    continue
                if not preranked:
                    intensity_rank(peaks)
                    peaks = DeconvolutedPeakSet([p for p in peaks if p.rank > 0])
                    peaks.reindex()
                    gpsm.annotations['ranked_peaks'] = peaks
        fit = feature_function_estimator(gpsms, feature, series=self.series, preranked=True, track_relations=True)
        specializations = filter(specialize_filter, fit.specialize())
        fit.pack()
        return specializations

    def _score_peak(self, peak, matched_features, solution_map, structure):
        gamma = self.offset_probability
        a = 1.0
        b = 1.0
        grouped_features = defaultdict(list)
        for relation in matched_features:
            grouped_features[relation.peak_key()].append(relation)
        for relations in grouped_features.values():
            relation = max(relations, key=lambda x: x.feature._feature_probability(gamma))
            feature = relation.feature
            if feature.on_series == 0:
                continue
            a *= feature.on_series
            b *= feature.off_series
        return (gamma * a) / ((gamma * a) + ((1 - gamma) * b))

    def _probability_of_fragment(self, matched_features):
        if matched_features is None:
            matched_features = []
        gamma = self.offset_probability
        if len(matched_features) == 0:
            return gamma
        a = 1
        b = 1
        for feature in matched_features:
            a *= feature.on_series
            b *= feature.off_series
        return (gamma * a) / ((gamma * a) + ((1 - gamma) * b))

    def score(self, scan, solution_map, structure):
        match_to_features = self.find_matches(scan, solution_map, structure)
        fragment_probabilities = {}
        for peak_pair in solution_map:
            if peak_pair.fragment.get_series() != self.series:
                continue
            features = match_to_features[peak_pair.peak]
            fragment_probabilities[peak_pair] = self._score_peak(
                peak_pair.peak, features, solution_map, structure)
        return fragment_probabilities

    def __repr__(self):
        template = ("{self.__class__.__name__}({self.series}, "
                    "{self.offset_probability:g}|{self.prior_probability_of_match})")
        return template.format(self=self)

    def __eq__(self, other):
        if self.series != other.series:
            return False
        if not np.isclose(self.error_tolerance, other.error_tolerance):
            return False
        # alpha
        if not np.isclose(self.on_frequency, other.on_frequency):
            return False
        # beta
        if not np.isclose(self.off_frequency, other.off_frequency):
            return False
        # p
        if not np.isclose(self.prior_probability_of_match, other.prior_probability_of_match):
            return False
        # gamma
        if not np.isclose(self.offset_probability, other.offset_probability):
            return False
        if self.features != other.features:
            return False
        return True

    def to_json(self):
        d = {}
        d['series'] = self.series.name
        d['error_tolerance'] = self.error_tolerance
        d['on_frequency'] = self.on_frequency
        d['off_frequency'] = self.off_frequency
        d['prior_probability_of_match'] = self.prior_probability_of_match
        d['features'] = [f.to_json() for f in self.features]
        return d

    @classmethod
    def from_json(cls, d):
        series = IonSeries.get(d['series'])
        features = [FittedFeature.from_json(f) for f in d['features']]
        inst = cls(
            series=series, features=features, on_frequency=d['on_frequency'],
            off_frequency=d['off_frequency'], prior_probability_of_match=d['prior_probability_of_match'],
            error_tolerance=d['error_tolerance']
        )
        return inst


def feature_divergence(models, feature):
    B = {}
    C = {}

    for model in models:
        B[model] = model.on_frequency * model.prior_probability_of_match
        C[model] = model.offset_probability * model.get_fitted_feature(feature).on_series
    B['noise'] = 1 - sum(B.values())
    C['noise'] = 1 - sum(C.values())

    divergence = 0
    for series in B:
        o = B[series]
        t = C[series]
        if t == 0:
            t = 1e-6
        if o == 0:
            o = 1e-6
        divergence += o * (np.log(o) - np.log(t))
        assert not np.isinf(divergence)
    return divergence


class FragmentationModelCollection(object):
    def __init__(self, models=None):
        if models is None:
            models = {}
        if not isinstance(models, Mapping):
            models = {m.series: m for m in models}
        self.models = models

    def __getitem__(self, k):
        return self.models[k]

    def __setitem__(self, k, v):
        self.models[k] = v

    def __iter__(self):
        return iter(self.models)

    def __contains__(self, k):
        return k in self.models

    def add(self, model):
        self[model.series] = model

    def find_matches(self, scan, solution_map, structure):
        match_to_features = defaultdict(list)
        deconvoluted_peak_set = scan.deconvoluted_peak_set
        for peak_fragment in solution_map:
            peak = peak_fragment.peak
            fragment = peak_fragment.fragment
            try:
                model = self.models[fragment.get_series()]
            except KeyError:
                continue
            for feature in model.feature_table:
                rels = feature.find_matches(peak, deconvoluted_peak_set, structure)
                for rel in rels:
                    for rel in rels:
                        if feature.feature.is_valid_match(rel.from_peak, rel.to_peak, solution_map, structure):
                            match_to_features[rel.from_peak].append(rel)
                            match_to_features[rel.to_peak].append(rel)
        return match_to_features

    def score(self, scan, solution_map, structure):
        match_to_features = self.find_matches(scan, solution_map, structure)
        fragment_probabilities = {}
        for pair in solution_map:
            features = match_to_features[pair.peak]
            try:
                model = self.models[pair.fragment.get_series()]
            except KeyError:
                continue
            features = match_to_features[pair.peak]
            fragment_probabilities[pair] = model._score_peak(
                pair.peak, features, solution_map, structure)
        return fragment_probabilities

    def __repr__(self):
        template = "{self.__class__.__name__}({self.models})"
        return template.format(self=self)

    def __eq__(self, other):
        if self.models == other.models:
            return True

    def __ne__(self, other):
        return not (self == other)

    def to_json(self):
        d = {}
        for series, model in self.models.items():
            d[str(series)] = model.to_json()
        return d

    @classmethod
    def from_json(cls, d):
        models = {}
        for series, model in d.items():
            models[IonSeries(series)] = FragmentationModel.from_json(model)
        return cls(models)
