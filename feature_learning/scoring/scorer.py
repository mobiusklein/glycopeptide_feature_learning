import math

import numpy as np

from glycan_profiling.tandem.glycopeptide.scoring.coverage_weighted_binomial import (
    FragmentMatchMap, accuracy_bias)
from glycan_profiling.tandem.glycopeptide.scoring.base import ChemicalShift
from glycan_profiling.tandem.glycopeptide.scoring.simple_score import SimpleCoverageScorer
from glycan_profiling.tandem.glycopeptide.scoring.binomial_score import BinomialSpectrumMatcher
from glycan_profiling.tandem.glycopeptide.scoring.glycan_signature_ions import GlycanCompositionSignatureMatcher

from feature_learning.multinomial_regression import (
    PearsonResidualCDF, least_squares_scale_coefficient)

from feature_learning.utils import distcorr

from .predicate import PredicateTreeBase


class MultinomialRegressionScorer(SimpleCoverageScorer, BinomialSpectrumMatcher, GlycanCompositionSignatureMatcher):
    accuracy_bias = accuracy_bias

    def __init__(self, scan, sequence, mass_shift=None, model_fit=None, partition=None):
        super(MultinomialRegressionScorer, self).__init__(scan, sequence, mass_shift)
        self.structure = self.target
        self.model_fit = model_fit
        self.partition = partition
        self.error_tolerance = None

    def match(self, error_tolerance=2e-5, *args, **kwargs):
        self.error_tolerance = error_tolerance
        GlycanCompositionSignatureMatcher.match(self, error_tolerance=error_tolerance)

        solution_map = FragmentMatchMap()
        spectrum = self.spectrum
        # this does not include stub glycopeptides
        n_theoretical = 0
        backbone_mass_series = []
        neutral_losses = tuple(kwargs.pop("neutral_losses", []))

        masked_peaks = set()
        for frag in self.target.glycan_fragments(
                all_series=False, allow_ambiguous=False,
                include_large_glycan_fragments=False,
                maximum_fragment_size=4):
            peak = spectrum.has_peak(frag.mass, error_tolerance)
            if peak:
                solution_map.add(peak, frag)
                masked_peaks.add(peak.index.neutral_mass)
                # try:
                #     self._sanitized_spectrum.remove(peak)
                # except KeyError:
                #     continue
        if self.mass_shift.tandem_mass != 0:
            chemical_shift = ChemicalShift(
                self.mass_shift.name, self.mass_shift.tandem_composition)
        else:
            chemical_shift = None
        for frag in self.target.stub_fragments(extended=True):
            for peak in spectrum.all_peaks_for(frag.mass, error_tolerance):
                # should we be masking these? peptides which have amino acids which are
                # approximately the same mass as a monosaccharide unit at ther terminus
                # can produce cases where a stub ion and a backbone fragment match the
                # same peak.
                #
                masked_peaks.add(peak.index.neutral_mass)
                solution_map.add(peak, frag)

            # If the precursor match was caused by a mass shift, that mass shift may
            # be associated with stub fragments.
            if chemical_shift is not None:
                shifted_mass = frag.mass + self.mass_shift.tandem_mass
                for peak in spectrum.all_peaks_for(shifted_mass, error_tolerance):
                    masked_peaks.add(peak.index.neutral_mass)
                    shifted_frag = frag.clone()
                    shifted_frag.chemical_shift = chemical_shift
                    shifted_frag.name += "+ %s" % (self.mass_shift.name,)
                    solution_map.add(peak, shifted_frag)

        n_glycosylated_b_ions = 0
        for frags in self.target.get_fragments('b', neutral_losses):
            glycosylated_position = False
            n_theoretical += 1
            for frag in frags:
                backbone_mass_series.append(frag)
                glycosylated_position |= frag.is_glycosylated
                for peak in spectrum.all_peaks_for(frag.mass, error_tolerance):
                    if peak.index.neutral_mass in masked_peaks:
                        continue
                    solution_map.add(peak, frag)
            if glycosylated_position:
                n_glycosylated_b_ions += 1

        n_glycosylated_y_ions = 0
        for frags in self.target.get_fragments('y', neutral_losses):
            glycosylated_position = False
            n_theoretical += 1
            for frag in frags:
                backbone_mass_series.append(frag)
                glycosylated_position |= frag.is_glycosylated
                for peak in spectrum.all_peaks_for(frag.mass, error_tolerance):
                    if peak.index.neutral_mass in masked_peaks:
                        continue
                    solution_map.add(peak, frag)
            if glycosylated_position:
                n_glycosylated_y_ions += 1

        self.n_theoretical = n_theoretical
        self.glycosylated_b_ion_count = n_glycosylated_b_ions
        self.glycosylated_y_ion_count = n_glycosylated_y_ions
        self.solution_map = solution_map
        self._backbone_mass_series = backbone_mass_series
        return solution_map

    def _calculate_pearson_residual_score(self, use_reliability=True, base_reliability=0.5):
        c, intens, t, yhat = self._get_predicted_intensities()
        if self.model_fit.reliability_model is None or not use_reliability:
            reliability = np.ones_like(yhat)
        else:
            reliability = self._get_reliabilities(c, base_reliability=base_reliability)
        # standardize intensity
        intens = intens / t
        # remove the unassigned signal term
        intens = intens[:-1]
        yhat = yhat[:-1]

        delta = (intens - yhat) ** 2
        mask = intens > yhat
        # reduce penalty for exceeding predicted intensity
        delta[mask] = delta[mask] / 2.
        denom = yhat * (1 - yhat)
        denom *= (reliability[:-1])  # divide by the reliability
        pearson_residual_score = delta / denom
        model_score = -np.log10(PearsonResidualCDF(pearson_residual_score) + 1e-6).sum()
        return model_score

    def _get_intensity_observed_expected(self, use_reliability=False, base_reliability=0.5):
        c, intens, t, yhat = self._get_predicted_intensities()
        p = (intens / t)[:-1]
        yhat = yhat[:-1]
        if len(p) == 1:
            return 0., 0.
        if use_reliability:
            reliability = self._get_reliabilities(c, base_reliability=base_reliability)[:-1]
            p = t * p / np.sqrt(t * reliability * p * (1 - p))
            yhat = t * yhat / np.sqrt(t * reliability * yhat * (1 - yhat))
        return p, yhat

    def _calculate_correlation_coef(self, use_reliability=False, base_reliability=0.5):
        p, yhat = self._get_intensity_observed_expected(use_reliability, base_reliability)
        return np.corrcoef(p, yhat)[0, 1]

    def _calculate_correlation_distance(self, use_reliability=False, base_reliability=0.5):
        p, yhat = self._get_intensity_observed_expected(use_reliability, base_reliability)
        return distcorr(p, yhat)

    def _transform_correlation(self, use_reliability=False, base_reliability=0.5):
        r = self._calculate_correlation_coef(use_reliability=use_reliability, base_reliability=base_reliability)
        if np.isnan(r):
            r = -0.5
        c = (r + 1) / 2.
        return c

    def _transform_correlation_distance(self, use_reliability=False, base_reliability=0.5):
        r = self._calculate_correlation_distance(use_reliability=use_reliability, base_reliability=base_reliability)
        if np.isnan(r):
            r = 0
        c = (r + 1) / 2.
        return c

    def _get_predicted_peaks(self, scaled=True):
        c, intens, t, yhat = self.model_fit._get_predicted_intensities(self)
        mz = [ci.peak.mz for ci in c if ci.peak_pair]
        intensities = yhat[:-1] * (least_squares_scale_coefficient(yhat[:-1], intens[:-1]) if scaled else 1.0)
        return zip(mz, intensities)

    def _get_predicted_intensities(self):
        c, intens, t, yhat = self.model_fit._get_predicted_intensities(self)
        return c, intens, t, yhat

    def _get_reliabilities(self, fragment_match_features, base_reliability=0.5):
        reliability = self.model_fit._calculate_reliability(
            self, fragment_match_features, base_reliability=base_reliability)
        return reliability

    def glycan_score(self, use_reliability=True, base_reliability=0.5):
        c, intens, t, yhat = self._get_predicted_intensities()
        if self.model_fit.reliability_model is None or not use_reliability:
            reliability = np.ones_like(yhat)
        else:
            reliability = self._get_reliabilities(c, base_reliability=base_reliability)
        stubs = []
        intens /= t
        for i in range(len(c)):
            if c[i].series == 'stub_glycopeptide':
                stubs.append((c[i], intens[i], yhat[i], reliability[i]))
        if not stubs:
            return 0
        c, intens, yhat, reliability = zip(*stubs)
        intens = np.array(intens)
        yhat = np.array(yhat)
        corr = (np.corrcoef(intens, yhat)[0, 1])
        if np.isnan(corr):
            corr = -0.5
        corr = (1.0 + corr) / 2.0
        reliability = np.array(reliability)
        delta = (intens - yhat) ** 2
        mask = intens > yhat
        delta[mask] = delta[mask] / 2.
        denom = yhat * (1 - yhat) * reliability
        stub_component = -np.log10(PearsonResidualCDF(delta / denom) + 1e-6).sum()
        oxonium_component = self._signature_ion_score(self.error_tolerance)
        glycan_score = (stub_component) * corr + oxonium_component
        return max(glycan_score, 0)

    def peptide_score(self, use_reliability=True, base_reliability=0.5):
        c, intens, t, yhat = self._get_predicted_intensities()
        if self.model_fit.reliability_model is None or not use_reliability:
            reliability = np.ones_like(yhat)
        else:
            reliability = self._get_reliabilities(c, base_reliability=base_reliability)

        backbones = []
        intens /= t
        for i in range(len(c)):
            if c[i].series != 'stub_glycopeptide':
                backbones.append((c[i], intens[i], yhat[i], reliability[i]))
        if not backbones:
            return 0
        c, intens, yhat, reliability = zip(*backbones)
        intens = np.array(intens)
        yhat = np.array(yhat)
        corr = (np.corrcoef(intens, yhat)[0, 1])
        if np.isnan(corr):
            corr = -0.5
        corr = (1.0 + corr) / 2.0
        reliability = np.array(reliability)
        delta = (intens - yhat) ** 2
        mask = intens > yhat
        delta[mask] = delta[mask] / 2.
        denom = yhat * (1 - yhat) * reliability
        peptide_score = -np.log10(PearsonResidualCDF(delta / denom) + 1e-6).sum()
        coverage_score = self._coverage_score()
        return peptide_score * coverage_score * corr

    def calculate_score(self, error_tolerance=2e-5, backbone_weight=None,
                        glycosylated_weight=None, stub_weight=None,
                        use_reliability=True, base_reliability=0.5,
                        weighting=None, *args, **kwargs):
        assert self.model_fit is not None
        model_score = self._calculate_pearson_residual_score(
            use_reliability=use_reliability,
            base_reliability=base_reliability)
        intensity = -math.log10(self._intensity_component_binomial())
        fragments_matched = -math.log10(self._fragment_matched_binomial())
        coverage_score = self._coverage_score(backbone_weight, glycosylated_weight, stub_weight)
        offset = self.determine_precursor_offset()
        mass_accuracy = -10 * math.log10(
            1 - self.accuracy_bias.score(self.precursor_mass_accuracy(offset)))
        signature_component = GlycanCompositionSignatureMatcher.calculate_score(self)
        self._score = ((intensity + fragments_matched + model_score) * coverage_score
                       ) + mass_accuracy + signature_component
        if weighting is None:
            pass
        elif weighting == 'correlation':
            self._score *= self._transform_correlation(False)
        elif weighting == 'normalized_correlation':
            self._score *= self._transform_correlation(True, base_reliability=base_reliability)
        elif weighting == 'correlation_distance':
            self._score *= self._transform_correlation_distance(False)
        elif weighting == 'normalized_correlation_distance':
            self._score *= self._transform_correlation_distance(True, base_reliability=base_reliability)
        return self._score


class ShortPeptideMultinomialRegressionScorer(MultinomialRegressionScorer):
    stub_weight = 0.65


class MultinomialRegressionMixtureScorer(MultinomialRegressionScorer):

    def __init__(self, scan, sequence, mass_shift=None, model_fits=None, partition=None, power=4):
        assert len(model_fits) > 0
        super(MultinomialRegressionMixtureScorer, self).__init__(
            scan, sequence, mass_shift, model_fit=model_fits[0], partition=partition)
        self.model_fits = list(model_fits)
        self.power = power
        self.feature_cache = dict()
        self.mixture_coefficients = None

    def _calculate_mixture_coefficients(self, scan):
        if len(self.model_fits) == 1:
            return np.array([1.])
        ps = np.array([
            (1. / m.pearson_residual_score(scan)) ** self.power
            for m in self.model_fits
        ])
        total = ps.sum()
        return ps / total

    def _calculate_pearson_residual_score(self, use_reliability=True, base_reliability=0.5):
        scores = []
        for model_fit in self.model_fits:
            self.model_fit = model_fit
            score = super(
                MultinomialRegressionMixtureScorer, self)._calculate_pearson_residual_score(
                    use_reliability=use_reliability, base_reliability=base_reliability)
            scores.append(score)
        return np.dot(scores, self.mixture_coefficients)

    def glycan_score(self, use_reliability=True, base_reliability=0.5):
        scores = []
        for model_fit in self.model_fits:
            self.model_fit = model_fit
            score = super(
                MultinomialRegressionMixtureScorer, self).glycan_score(
                    use_reliability=use_reliability, base_reliability=base_reliability)
            scores.append(score)
        return np.dot(scores, self.mixture_coefficients)

    def peptide_score(self, use_reliability=True, base_reliability=0.5):
        scores = []
        for model_fit in self.model_fits:
            self.model_fit = model_fit
            score = super(
                MultinomialRegressionMixtureScorer, self).peptide_score(
                    use_reliability=use_reliability, base_reliability=base_reliability)
            scores.append(score)
        return np.dot(scores, self.mixture_coefficients)

    def _calculate_correlation_coef(self, use_reliability=False, base_reliability=0.5):
        scores = []
        for model_fit in self.model_fits:
            self.model_fit = model_fit
            score = super(
                MultinomialRegressionMixtureScorer, self)._calculate_correlation_coef(
                    use_reliability=use_reliability, base_reliability=base_reliability)
            scores.append(score)
        return np.array(scores)

    def _calculate_correlation_distance(self, use_reliability=False, base_reliability=0.5):
        scores = []
        for model_fit in self.model_fits:
            self.model_fit = model_fit
            score = super(
                MultinomialRegressionMixtureScorer, self)._calculate_correlation_distance(
                    use_reliability=use_reliability, base_reliability=base_reliability)
            scores.append(score)
        return np.array(scores)

    def _transform_correlation(self, use_reliability=False, base_reliability=0.5):
        r = self._calculate_correlation_coef(
            use_reliability=use_reliability, base_reliability=base_reliability)
        r[np.isnan(r)] = -0.5
        c = (r + 1) / 2.
        return np.dot(c, self.mixture_coefficients)

    def _transform_correlation_distance(self, use_reliability=False, base_reliability=0.5):
        r = self._calculate_correlation_distance(
            use_reliability=use_reliability, base_reliability=base_reliability)
        r[np.isnan(r)] = -0.5
        c = (r + 1) / 2.
        return np.dot(c, self.mixture_coefficients)

    def calculate_score(self, error_tolerance=2e-5, backbone_weight=None,
                        glycosylated_weight=None, stub_weight=None,
                        use_reliability=True, base_reliability=0.5,
                        weighting=None, *args, **kwargs):
        self.mixture_coefficients = self._calculate_mixture_coefficients(self.scan)
        return super(MultinomialRegressionMixtureScorer, self).calculate_score(
            error_tolerance=error_tolerance, backbone_weight=backbone_weight,
            glycosylated_weight=glycosylated_weight, stub_weight=stub_weight,
            use_reliability=use_reliability, base_reliability=base_reliability,
            weighting=weighting, *args, **kwargs)


class ShortPeptideMultinomialRegressionMixtureScorer(MultinomialRegressionMixtureScorer):
    stub_weight = 0.65


class PredicateTree(PredicateTreeBase):
    _scorer_type = MultinomialRegressionMixtureScorer
    _short_peptide_scorer_type = ShortPeptideMultinomialRegressionMixtureScorer


PartitionTree = PredicateTree
