import math

from glycan_profiling.tandem.spectrum_matcher_base import SpectrumMatcherBase
from glycan_profiling.tandem.glycopeptide.scoring.simple_score import SimpleCoverageScorer
from glycan_profiling.tandem.glycopeptide.scoring.coverage_weighted_binomial import (
    FragmentMatchMap)

from .peak_relations import probability_of_peak_explained, fragmentation_probability


class FragmentFrequencyScorer(SpectrumMatcherBase):
    def __init__(self, scan, sequence, models=None):
        if models is None:
            models = dict()
        super(FragmentFrequencyScorer, self).__init__(scan, sequence)
        self._sanitized_spectrum = set(self.spectrum)
        self.models = models

    def match(self, error_tolerance=2e-5, *args, **kwargs):
        solution_map = FragmentMatchMap()
        spectrum = self.spectrum
        n_theoretical = 0

        for frag in self.target.glycan_fragments(
                all_series=False, allow_ambiguous=False,
                include_large_glycan_fragments=False,
                maximum_fragment_size=4):
            peak = spectrum.has_peak(frag.mass, error_tolerance)
            if peak:
                solution_map.add(peak, frag)
                try:
                    self._sanitized_spectrum.remove(peak)
                except KeyError:
                    continue

        n_glycosylated_b_ions = 0
        for frags in self.target.get_fragments('b'):
            glycosylated_position = False
            n_theoretical += 1
            for frag in frags:
                glycosylated_position |= frag.is_glycosylated
                for peak in spectrum.all_peaks_for(frag.mass, error_tolerance):
                    solution_map.add(peak, frag)
            if glycosylated_position:
                n_glycosylated_b_ions += 1

        n_glycosylated_y_ions = 0
        for frags in self.target.get_fragments('y'):
            glycosylated_position = False
            n_theoretical += 1
            for frag in frags:
                glycosylated_position |= frag.is_glycosylated
                for peak in spectrum.all_peaks_for(frag.mass, error_tolerance):
                    solution_map.add(peak, frag)
            if glycosylated_position:
                n_glycosylated_y_ions += 1

        for frag in self.target.stub_fragments(extended=True):
            for peak in spectrum.all_peaks_for(frag.mass, error_tolerance):
                solution_map.add(peak, frag)

        self.n_theoretical = n_theoretical
        self.glycosylated_b_ion_count = n_glycosylated_b_ions
        self.glycosylated_y_ion_count = n_glycosylated_y_ions
        self.solution_map = solution_map
        return solution_map

    def locate_model(self, peak, fragment, charge_specific=True):
        if charge_specific:
            return self.models[(fragment.series, peak.charge)]
        else:
            return self.models[(fragment.series, None)]

    def _probability_of(self, peak, fragment, charge_specific=True):
        offset_frequency, unknown_peak_rate, prior_fragment_probability = self.locate_model(
            peak, fragment, charge_specific)
        probability_of_explained = probability_of_peak_explained(
            offset_frequency, unknown_peak_rate, prior_fragment_probability)
        probability_of_fragment = fragmentation_probability(None, probability_of_explained, [])
        return probability_of_fragment

    def calculate_score(self, models=None, charge_specific=True, *args, **kwargs):
        if models is not None:
            self.models.update(models)
        running_score = 0.0
        for peak, fragment in self.solution_map:
            if fragment.series == 'oxonium_ion':
                continue
            prob = self._probability_of(peak, fragment, charge_specific)
            running_score += math.log10(prob)
        self._score = -running_score
        return self._score


class CoverageWeightedFragmentFrequencyScorer(FragmentFrequencyScorer, SimpleCoverageScorer):
    def __init__(self, scan, sequence, models=None):
        if models is None:
            models = dict()
        super(FragmentFrequencyScorer, self).__init__(scan, sequence)
        self._sanitized_spectrum = set(self.spectrum)
        self.models = models
        self.glycosylated_b_ion_count = 0
        self.glycosylated_y_ion_count = 0

    def calculate_score(self, models=None, *args, **kwargs):
        freq_score = FragmentFrequencyScorer.calculate_score(self, models=models)
        coverage_score = SimpleCoverageScorer.calculate_score(self)
        self._score = freq_score * coverage_score
        return self._score
