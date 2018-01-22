# methods derived from:
# https://github.com/PNNL-Comp-Mass-Spec/Informed-Proteomics/tree/master/InformedProteomics.Scoring/LikelihoodScoring
import itertools

from collections import defaultdict, namedtuple
from functools import total_ordering

import numpy as np
from scipy.ndimage import gaussian_filter1d

from glycopeptidepy.structure.fragment import IonSeries

from .histogram import Histogram


@total_ordering
class Probability(object):
    def __init__(self, label, found=0, total=0):
        self.label = label
        self.found = float(found)
        self.total = float(total)

    def __add__(self, other):
        return Probability(self.label, self.found + other.found,
                           self.total + other.total)

    def __repr__(self):
        return "Probability(%s, %d, %d)" % (self.label, self.found, self.total)

    @property
    def probability(self):
        return self.found / self.total

    def __lt__(self, other):
        return self.probability < other.probability

    def __eq__(self, other):
        return abs(self.probability - other.probability) < 1e-6


class ProbabilityTable(dict):
    def __missing__(self, key):
        value = Probability(key, 0, 0)
        self[key] = value
        return value

    def observe(self, key, value):
        p = self[key]
        p.total += 1
        p.found += bool(value)


class IonType(namedtuple("IonType", ("series", "charge", "glycosylated", "neutral_loss"))):
    @classmethod
    def from_peak_and_fragment(cls, peak, fragment):
        return cls(fragment.series, peak.charge,
                   fragment.is_glycosylated,
                   fragment.neutral_loss)

    def is_member(self, fragment):
        value = (self.series == fragment.series) and (
            self.glycosylated == fragment.is_glycosylated) and (
            self.neutral_loss == fragment.neutral_loss)
        return value

    def __repr__(self):
        text = "IonType(%s, %d, %r, %r)" % (
            self.series.name, self.charge, self.glycosylated, self.neutral_loss)
        return text

    @classmethod
    def standard_ion_types(cls):
        ion_types = []
        args = ([IonSeries.b, IonSeries.y], [1, 2, 3, 4], [False, True])
        for series, charge, is_glycosylated in itertools.product(*args):
            ion_types.append(cls(series, charge, is_glycosylated, None))
        for i in range(1, 6):
            ion_types.append(cls(IonSeries.stub_glycopeptide, i, False, None))

        return ion_types


standard_ion_types = IonType.standard_ion_types()


class IonSeriesModelBase(object):
    def _get_fragments(self, sequence, ion_type):
        fragments = list(sequence.get_fragments(ion_type.series))
        nested = isinstance(fragments[0], (list, tuple))
        if nested:
            for position in fragments:
                for fragment in position:
                    if not ion_type.is_member(fragment):
                        continue
                    yield fragment
        else:
            for fragment in fragments:
                if not ion_type.is_member(fragment):
                    continue
                yield fragment


class PeakRankModel(IonSeriesModelBase):
    def __init__(self, ion_types, error_tolerance=2e-5, max_ranks=1000):
        self.error_tolerance = error_tolerance
        self.max_ranks = max_ranks
        self.ion_types = ion_types
        self.rank_table = dict()
        self.totals = np.zeros(max_ranks + 1)

        for ion_type in self.ion_types:
            self.rank_table[ion_type] = np.zeros(max_ranks + 1)

    def __getitem__(self, key):
        return self.rank_table[key]

    @staticmethod
    def rank_peaks(peak_set, peak_mask=None):
        if peak_mask is None:
            peak_mask = set()
        ordered_peaks = sorted(peak_set, key=lambda x: x.intensity, reverse=True)
        last_rank = 1
        out = []
        for peak in ordered_peaks:
            if peak in peak_mask:
                continue
            peak.rank = last_rank
            last_rank += 1
            out.append(peak)
        return out

    def add(self, spectrum_match):
        peak_set = spectrum_match.deconvoluted_peak_set
        sequence = spectrum_match.structure
        ranks = self.rank_peaks(peak_set)
        for i, peak in enumerate(ranks):
            index = i
            if index >= self.max_ranks:
                index = self.max_ranks - 1
            self.totals[index] += 1
            self.totals[self.max_ranks] += 1

        for ion_type in self.ion_types:
            for fragment in self._get_fragments(sequence, ion_type):
                for peak in peak_set.all_peaks_for(fragment.mass, self.error_tolerance):
                    if peak.charge != ion_type.charge:
                        continue
                    index = self.rank_index(peak.rank)
                    self.rank_table[ion_type][index] += 1

    def rank_index(self, rank):
        if rank < 1:
            rank_index = self.max_ranks + 1
        elif rank > self.max_ranks:
            rank_index = self.max_ranks
        else:
            rank_index = rank
        return rank_index - 1

    def rank_probability(self, rank, ion_type):
        index = self.rank_index(rank)
        count = self.rank_table[ion_type][index]
        total = self.totals[index]
        return Probability(ion_type, count, total)

    def smooth(self, sigma=2):
        for channel in self.rank_table.keys():
            self.rank_table[channel] = gaussian_filter1d(self.rank_table[channel], sigma)
        self.totals = gaussian_filter1d(self.totals, sigma)


class IonMatchModel(IonSeriesModelBase):
    def __init__(self, ion_types=None, error_tolerance=2e-5, intensity_threshold=0, combine_charges=False):
        self.frequencies = ProbabilityTable()
        self.ion_types = set(ion_types or [])
        self.error_tolerance = error_tolerance
        self.intensity_threshold = intensity_threshold
        self.charges = [it.charge for it in self.ion_types]

    def __getitem__(self, key):
        return self.frequencies[key]

    def add(self, spectrum_match):
        peak_set = spectrum_match.deconvoluted_peak_set
        sequence = spectrum_match.structure
        for ion_type in self.ion_types:
            total = 0
            found = 0
            for fragment in self._get_fragments(sequence, ion_type):
                total += 1
                hit = 0
                for peak in peak_set.all_peaks_for(fragment.mass, self.error_tolerance):
                    if peak.intensity < self.intensity_threshold:
                        continue
                    elif peak.charge != ion_type.charge:
                        continue
                    else:
                        hit = 1
                found += hit
            self.frequencies[ion_type] += Probability(ion_type, found, total)

    def update(self, spectrum_matches):
        for spectrum_match in spectrum_matches:
            self.add(spectrum_match)

    def probability(self, ion_type):
        return self[ion_type].probability


class SpectrumMatchModel(object):
    def __init__(self, bin_size=50, ion_types=None, error_tolerance=2e-5, max_ranks=1000, *args, **kwargs):
        if ion_types is None:
            ion_types = list(standard_ion_types)
        self.spectrum_matches = list()

        self.ion_type_frequencies = defaultdict(list)
        self.rank_tables = defaultdict(list)
        self.decoy_rank_tables = defaultdict(list)

        self.observed_charge_states = set()
        self.mass_intervals = defaultdict(Histogram)
        self.mass_bins = defaultdict(int)
        self.bin_size = bin_size
        self.ion_types = ion_types
        self.error_tolerance = error_tolerance
        self.max_ranks = max_ranks

    def add(self, spectrum_match):
        self.spectrum_matches.append(spectrum_match)
        self.mass_intervals[spectrum_match.precursor_information.charge].add(
            spectrum_match.structure.total_mass)
        self.observed_charge_states.add(
            spectrum_match.precursor_information.charge)

    def balance(self):
        for charge in self.observed_charge_states:
            n_obs_for_charge = len(self.mass_intervals[charge])
            if n_obs_for_charge < self.bin_size:
                continue
            self.mass_bins[charge] = n_obs_for_charge / self.bin_size
            self.mass_intervals[charge].equalize(self.mass_bins[charge])
            bin_count = len(self.mass_intervals[charge].edges)
            for i in range(bin_count):
                self.ion_type_frequencies[charge].append(
                    IonMatchModel(self.ion_types, self.error_tolerance))
                self.rank_tables[charge].append(
                    PeakRankModel(self.ion_types, self.error_tolerance))
                self.decoy_rank_tables[charge].append(
                    PeakRankModel(self.ion_types, self.error_tolerance, self.max_ranks))

    def compute_match(self, spectrum_match, charge, mass_index, decoy=False):

        if decoy:
            rank_table = self.decoy_rank_tables[charge][mass_index]
            rank_table.add(spectrum_match)
            return
        rank_table = self.rank_tables[charge][mass_index]
        rank_table.add(spectrum_match)
        ion_table = self.ion_type_frequencies[charge][mass_index]
        ion_table.add(spectrum_match)

    def handle_match(self, spectrum_match):
        charge = spectrum_match.precursor_information.charge
        if charge not in self.rank_tables:
            return
        neutral_mass = spectrum_match.precursor_information.neutral_mass
        mass_index = self.mass_intervals[charge].bin_index(neutral_mass)
        self.compute_match(spectrum_match, charge, mass_index, False)
        decoy = spectrum_match.decoy()
        self.compute_match(decoy, charge, mass_index, True)

    def process(self):
        for spectrum_match in self.spectrum_matches:
            self.handle_match(spectrum_match)

    def rank_score(self, ion_type, rank, charge, mass):
        target_probability, decoy_probability = self._rank_score(ion_type, rank, charge, mass)
        return np.log(
            target_probability.found + 1 / target_probability.total + 1) - np.log(
            decoy_probability.found + 1 / decoy_probability.total + 1)

    def _rank_score(self, ion_type, rank, charge, mass):
        mass_index = self.mass_intervals[charge].bin_index(mass)
        rank_table = self.rank_tables[charge][mass_index]
        target_probability = rank_table.rank_probability(rank, ion_type)
        decoy_table = self.decoy_rank_tables[charge][mass_index]
        decoy_probability = decoy_table.rank_probability(rank, ion_type)
        return target_probability, decoy_probability

    def score(self, ion_type, rank, charge, mass):
        rank_score = self.rank_score(ion_type, rank, charge, mass)
        mass_index = self.mass_intervals[charge].bin_index(mass)
        prob = self.ion_type_frequencies[charge][mass_index].probability(ion_type)
        return -np.log(1 - prob) + rank_score
