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


class CachedModelPrediction(object):
    __slots__ = ("fragment_specifications", "experimental_intensities", "total_signal",
                 "feature_vectors", "reliability_vector")

    def __init__(self, fragment_specifications, experimental_intensities, total_signal, feature_vectors):
        self.fragment_specifications = fragment_specifications
        self.experimental_intensities = experimental_intensities
        self.total_signal = total_signal
        self.feature_vectors = feature_vectors
        self.reliability_vector = None

    def __iter__(self):
        yield self.fragment_specifications
        yield self.experimental_intensities
        yield self.total_signal
        yield self.feature_vectors

    def __reduce__(self):
        return self.__class__, tuple(self)


class MultinomialRegressionScorer(SimpleCoverageScorer, BinomialSpectrumMatcher, GlycanCompositionSignatureMatcher):
    accuracy_bias = accuracy_bias

    def __init__(self, scan, sequence, mass_shift=None, model_fit=None, partition=None):
        super(MultinomialRegressionScorer, self).__init__(scan, sequence, mass_shift)
        self.structure = self.target
        self.model_fit = model_fit
        self.partition = partition
        self.error_tolerance = None
        self._cached_model = None
        self._cached_transform = None

    def __reduce__(self):
        return self.__class__, (self.scan, self.sequence, self.mass_shift, self.model_fit, self.partition)

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

    def _calculate_pearson_residuals(self, use_reliability=True, base_reliability=0.5):
        r"""Calculate the raw Pearson residuals of the Multinomial model

        .. math::
            \frac{y - \hat{y}}{\hat{y} * (1 - \hat{y}) * r}

        Parameters
        ----------
        use_reliability : bool, optional
            Whether or not to use the fragment reliabilities to adjust the weight of
            each matched peak
        base_reliability : float, optional
            The lowest reliability a peak may have, compressing the range of contributions
            from the model based on the experimental evidence

        Returns
        -------
        np.ndarray
            The Pearson residuals
        """
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
        pearson_residuals = delta / denom
        return pearson_residuals

    def _calculate_pearson_residual_score(self, use_reliability=True, base_reliability=0.5):
        """Compute a model-based score by summing the Pearson residuals after transforming
        through them an Empirically measured CDF and followed by a -log10 transform.

        Parameters
        ----------
        use_reliability : bool, optional
            Whether or not to use the fragment reliabilities to adjust the weight of
            each matched peak
        base_reliability : float, optional
            The lowest reliability a peak may have, compressing the range of contributions
            from the model based on the experimental evidence


        Returns
        -------
        float:
            The model score
        """
        pearson_residuals = self._calculate_pearson_residuals(use_reliability, base_reliability)
        model_score = -np.log10(PearsonResidualCDF(pearson_residuals) + 1e-6).sum()
        if np.isnan(model_score):
            model_score = 0.0
        c, intens, t, yhat = self._get_predicted_intensities()
        reliability = self._get_reliabilities(c, base_reliability=base_reliability)[:-1]
        intensity_component = np.log10(intens[:-1]).dot(reliability + 1.0)
        stub_component = self._get_stub_component(
            c[:-1], use_reliability=use_reliability, base_reliability=base_reliability)
        return model_score + intensity_component + stub_component

    def _get_intensity_observed_expected(self, use_reliability=False, base_reliability=0.5):
        """Get the vector of matched experimental intensities and their predicted intensities.

        If the reliability model is used, the peak intensities are recalibrated according
        to a multinomial variance model with the reliabilities as weights.

        Parameters
        ----------
        use_reliability : bool, optional
            Whether or not to use the fragment reliabilities to adjust the weight of
            each matched peak
        base_reliability : float, optional
            The lowest reliability a peak may have, compressing the range of contributions
            from the model based on the experimental evidence

        Returns
        -------
        np.ndarray:
            Matched experimental peak intensities
        np.ndarray:
            Predicted peak intensities
        """
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
            r = -0.5
        c = (r + 1) / 2.
        return c

    def _get_predicted_peaks(self, scaled=True):
        c, intens, t, yhat = self.model_fit._get_predicted_intensities(self)
        mz = [ci.peak.mz for ci in c if ci.peak_pair]
        intensities = yhat[:-1] * (least_squares_scale_coefficient(yhat[:-1], intens[:-1]) if scaled else 1.0)
        return zip(mz, intensities)

    def _transform_matched_peaks(self):
        if self._is_model_cached(self.model_fit):
            return self._get_cached_model_transform(self.model_fit)
        result = self._transform_matched_peaks_uncached()
        self._cache_model_transform(self.model_fit, result)
        return result

    def _get_predicted_intensities(self):
        c, intens, t, X = self._transform_matched_peaks()
        yhat = np.exp(X.dot(self.model_fit.coef))
        yhat = yhat / (1 + yhat.sum())
        return c, intens, t, yhat

    def _cache_model_transform(self, model_fit, transform):
        self._cached_model = model_fit
        self._cached_transform = CachedModelPrediction(*transform)

    def _is_model_cached(self, model_fit):
        return self.model_fit is self._cached_model

    def _get_cached_model_transform(self, model_fit):
        return self._cached_transform

    def _clear_cache(self):
        self._cached_model = None
        self._cached_transform = None

    def _transform_matched_peaks_uncached(self):
        model_fit = self.model_fit
        c, intens, t = model_fit.model_type.build_fragment_intensity_matches(self)
        X = model_fit.model_type.encode_classification(c)
        return (c, intens, t, X)

    def _get_reliabilities(self, fragment_match_features, base_reliability=0.5):
        is_model_cached = False
        cached_data = None
        # Does not actually populate the cache if the model hasn't had its result
        # cached in the first place.
        if self._is_model_cached(self.model_fit):
            is_model_cached = True
            cached_data = self._get_cached_model_transform(self.model_fit)
            if cached_data.reliability_vector is not None:
                return cached_data.reliability_vector

        reliability = self.model_fit._calculate_reliability(
            self, fragment_match_features, base_reliability=base_reliability)
        if is_model_cached:
            if cached_data is not None:
                cached_data.reliability_vector = reliability
        return reliability

    def _get_stub_component(self, fragments, use_reliability=True, base_reliability=0.5, filtered=False):
        if self.model_fit.reliability_model is None or not use_reliability:
            reliability = np.ones(len(fragments))
        else:
            reliability = self._get_reliabilities(fragments, base_reliability=base_reliability)
        if not filtered:
            stubs = []
            for i in range(len(fragments)):
                if fragments[i].fragment.series == 'stub_glycopeptide':
                    stubs.append((fragments[i], reliability[i]))
            if not stubs:
                return 0
            fragments, reliability = zip(*stubs)
            return self._glycan_coverage(fragments, reliability)
        else:
            if len(fragments) == 0:
                return 0
            return self._glycan_coverage(fragments, reliability)

    def _glycan_coverage(self, fragments, reliability):
        theoretical_set = list(self.target.stub_fragments(extended=True))
        core_fragments = set()
        for frag in theoretical_set:
            if not frag.is_extended:
                core_fragments.add(frag.name)
        core_matches = []
        extended_matches = []
        for ci, rel in zip(fragments, reliability):
            if ci.fragment.name in core_fragments:
                core_matches.append(1.0 + rel)
            else:
                extended_matches.append(1.0 + rel)
        core_coverage = (sum(core_matches) ** 2) / len(core_fragments)
        extended_coverage = (
            sum(extended_matches) + sum(core_matches)) / (
                sum(self.target.glycan_composition.values()))
        coverage = core_coverage * extended_coverage
        if np.isnan(coverage):
            coverage = 0.0
        return coverage

    def glycan_score(self, use_reliability=True, base_reliability=0.5):
        c, intens, t, yhat = self._get_predicted_intensities()
        if self.model_fit.reliability_model is None or not use_reliability:
            reliability = np.ones_like(yhat)
        else:
            reliability = self._get_reliabilities(c, base_reliability=base_reliability)
        stubs = []
        intens = intens / t
        for i in range(len(c)):
            if c[i].series == 'stub_glycopeptide':
                stubs.append((c[i], intens[i], yhat[i], reliability[i]))
        if not stubs:
            return 0
        c, intens, yhat, reliability = zip(*stubs)
        intens = np.array(intens)
        yhat = np.array(yhat)
        reliability = np.array(reliability)
        if use_reliability:
            corr = (np.corrcoef(t * intens / np.sqrt(t * reliability * intens * (1 - intens)),
                                t * yhat / np.sqrt(t * reliability * yhat * (1 - yhat)))[0, 1])
        else:
            corr = np.corrcoef(intens, yhat)[0, 1]
        if np.isnan(corr):
            corr = -0.5
        corr = (1.0 + corr) / 2.0
        corr_score = corr * 10.0
        delta = (intens - yhat) ** 2
        mask = intens > yhat
        delta[mask] = delta[mask] / 2.
        denom = yhat * (1 - yhat) * reliability
        stub_component = -np.log10(PearsonResidualCDF(delta / denom) + 1e-6).sum()
        if np.isnan(stub_component):
            stub_component = 0
        oxonium_component = self._signature_ion_score(self.error_tolerance)
        # Unlike peptide coverage, the glycan composition coverage operates as a bias towards
        # selecting matches which contain more reliable glycan Y ions, but not to act as a scaling
        # factor because the set of all possible fragments for the glycan composition is a much larger
        # superset of the possible fragments of glycan structures because of recurring patterns
        # not reflected in the glycan composition.
        coverage = self._glycan_coverage(c, reliability)
        glycan_score = (np.log10(intens * t).dot(reliability + 1) + stub_component + coverage
                        # ) * corr + oxonium_component
                        ) + corr_score + oxonium_component
        return max(glycan_score, 0)

    def peptide_score(self, use_reliability=True, base_reliability=0.5):
        c, intens, t, yhat = self._get_predicted_intensities()
        if self.model_fit.reliability_model is None or not use_reliability:
            reliability = np.ones_like(yhat)
        else:
            reliability = self._get_reliabilities(c, base_reliability=base_reliability)

        backbones = []
        intens = intens / t
        for i in range(len(c)):
            if c[i].series in ('b', 'y'):
                backbones.append((c[i], intens[i], yhat[i], reliability[i]))
        if not backbones:
            return 0
        c, intens, yhat, reliability = zip(*backbones)
        intens = np.array(intens)
        yhat = np.array(yhat)
        reliability = np.array(reliability)
        # peptide reliability is usually less powerful, so it does not benefit
        # us to use the normalized correlation coefficient here
        corr = np.corrcoef(intens, yhat)[0, 1]
        if np.isnan(corr):
            corr = -0.5
        # peptide fragment correlation is weaker than the overall correlation.
        corr = (3.0 + corr) / 4.0
        corr_score = corr * 10.0
        reliability = np.array(reliability)
        delta = (intens - yhat) ** 2
        mask = intens > yhat
        delta[mask] = delta[mask] / 2.
        denom = yhat * (1 - yhat) * reliability
        peptide_score = -np.log10(PearsonResidualCDF(delta / denom) + 1e-6).sum()
        # peptide backbone coverage without separate term for glycosylation site parsimony
        b_ions, y_ions = self._compute_coverage_vectors()[:2]
        coverage_score = ((b_ions + y_ions[::-1])).sum() / float(self.n_theoretical)
        peptide_score = (np.log10(intens * t).dot(reliability + 1).sum() + peptide_score)
        # peptide_score *= corr
        peptide_score += corr_score
        peptide_score *= coverage_score
        return peptide_score

    def calculate_score(self, error_tolerance=2e-5, backbone_weight=None,
                        glycosylated_weight=None, stub_weight=None,
                        use_reliability=True, base_reliability=0.5,
                        weighting=None, *args, **kwargs):
        intensity = -math.log10(self._intensity_component_binomial())
        # fragments_matched = -math.log10(self._fragment_matched_binomial())
        fragments_matched = 0.0
        coverage_score = self._coverage_score(backbone_weight, glycosylated_weight, stub_weight)
        offset = self.determine_precursor_offset()
        mass_accuracy = -10 * math.log10(
            1 - self.accuracy_bias.score(self.precursor_mass_accuracy(offset)))
        signature_component = GlycanCompositionSignatureMatcher.calculate_score(self)
        model_score = self._calculate_pearson_residual_score(
            use_reliability=use_reliability,
            base_reliability=base_reliability)
        self._score = (intensity + fragments_matched + model_score)
        if weighting is None:
            pass
        elif weighting == 'correlation':
            # self._score *= self._transform_correlation(False)
            self._score += self._transform_correlation(False) * 10.0
        elif weighting == 'normalized_correlation':
            # self._score *= self._transform_correlation(True, base_reliability=base_reliability)
            self._score += self._transform_correlation(True, base_reliability=base_reliability) * 10.0
        elif weighting in ('correlation_distance', 'distance_correlation'):
            # self._score *= self._transform_correlation_distance(False)
            self._score += self._transform_correlation_distance(False) * 10.0
        elif weighting in ('normalized_correlation_distance', 'normalized_distance_correlation'):
            # self._score *= self._transform_correlation_distance(True, base_reliability=base_reliability)
            self._score += self._transform_correlation_distance(True, base_reliability=base_reliability) * 10.0
        else:
            raise ValueError("Unrecognized Weighting Scheme %s" % (weighting,))
        self._score *= coverage_score
        self._score += (mass_accuracy + signature_component)
        return self._score


class ShortPeptideMultinomialRegressionScorer(MultinomialRegressionScorer):
    stub_weight = 0.65


class MultinomialRegressionMixtureScorer(MultinomialRegressionScorer):

    def __init__(self, scan, sequence, mass_shift=None, model_fits=None, partition=None, power=4):
        super(MultinomialRegressionMixtureScorer, self).__init__(
            scan, sequence, mass_shift, model_fit=model_fits[0], partition=partition)
        self.model_fits = list(model_fits)
        self.power = power
        self._feature_cache = dict()
        self.mixture_coefficients = None

    def __reduce__(self):
        return self.__class__, (self.scan, self.sequence, self.mass_shift, self.model_fits, self.partition, self.power)

    def _iter_model_fits(self):
        for model_fit in self.model_fits:
            self.model_fit = model_fit
            yield model_fit
        self.model_fit = self.model_fits[0]

    def _cache_model_transform(self, model_fit, transform):
        self._feature_cache[model_fit] = CachedModelPrediction(*transform)

    def _is_model_cached(self, model_fit):
        return model_fit in self._feature_cache

    def _get_cached_model_transform(self, model_fit):
        return self._feature_cache[model_fit]

    def _clear_cache(self):
        self._feature_cache.clear()

    def _calculate_mixture_coefficients(self):
        if len(self.model_fits) == 1:
            return np.array([1.])
        ps = np.empty(len(self.model_fits))
        for i, model_fit in enumerate(self._iter_model_fits()):
            ps[i] = (1. / self._calculate_pearson_residuals().sum()) ** self.power
        total = ps.sum() + 1e-6 * ps.shape[0]
        return ps / total

    def _calculate_pearson_residual_score(self, use_reliability=True, base_reliability=0.5):
        scores = []
        for model_fit in self._iter_model_fits():
            score = super(
                MultinomialRegressionMixtureScorer, self)._calculate_pearson_residual_score(
                    use_reliability=use_reliability, base_reliability=base_reliability)
            scores.append(score)
        return np.dot(scores, self.mixture_coefficients)

    def glycan_score(self, use_reliability=True, base_reliability=0.5):
        scores = []
        for model_fit in self._iter_model_fits():
            score = super(
                MultinomialRegressionMixtureScorer, self).glycan_score(
                    use_reliability=use_reliability, base_reliability=base_reliability)
            scores.append(score)
        return np.dot(scores, self.mixture_coefficients)

    def peptide_score(self, use_reliability=True, base_reliability=0.5):
        scores = []
        for model_fit in self._iter_model_fits():
            score = super(
                MultinomialRegressionMixtureScorer, self).peptide_score(
                    use_reliability=use_reliability, base_reliability=base_reliability)
            scores.append(score)
        return np.dot(scores, self.mixture_coefficients)

    def _calculate_correlation_coef(self, use_reliability=False, base_reliability=0.5):
        scores = []
        for model_fit in self._iter_model_fits():
            score = super(
                MultinomialRegressionMixtureScorer, self)._calculate_correlation_coef(
                    use_reliability=use_reliability, base_reliability=base_reliability)
            scores.append(score)
        return np.array(scores)

    def _calculate_correlation_distance(self, use_reliability=False, base_reliability=0.5):
        scores = []
        for model_fit in self._iter_model_fits():
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
        self.mixture_coefficients = self._calculate_mixture_coefficients()
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
