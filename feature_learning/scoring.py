import math

from collections import defaultdict, deque

import numpy as np

from glycan_profiling.tandem.glycopeptide.scoring.simple_score import SimpleCoverageScorer
from glycan_profiling.tandem.glycopeptide.scoring.coverage_weighted_binomial import (
    FragmentMatchMap, accuracy_bias)

from glycan_profiling.tandem.glycopeptide.scoring.binomial_score import (
    binomial_fragments_matched,
    BinomialSpectrumMatcher
)

from glycan_profiling.tandem.glycopeptide.scoring.base import GlycopeptideSpectrumMatcherBase, ChemicalShift
from glycan_profiling.tandem.glycopeptide.scoring.glycan_signature_ions import GlycanCompositionSignatureMatcher
from glycan_profiling.tandem.spectrum_match import Unmodified

from ms_deisotope.data_source import ChargeNotProvided

from .multinomial_regression import (MultinomialRegressionFit,
                                     PearsonResidualCDF,
                                     least_squares_scale_coefficient)
from .partitions import classify_proton_mobility, partition_cell_spec
from .utils import distcorr


INF = float('inf')


class MultinomialRegressionScorer(SimpleCoverageScorer, BinomialSpectrumMatcher, GlycanCompositionSignatureMatcher):
    accuracy_bias = accuracy_bias

    def __init__(self, scan, sequence, mass_shift=None, multinomial_model=None, partition=None):
        super(MultinomialRegressionScorer, self).__init__(scan, sequence, mass_shift)
        self.structure = self.target
        self.model_fit = multinomial_model
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

    def _calculate_pearson_residual_score(self, use_reliability=True, base_reliability=0.5, pearson_bias=1.0):
        pearson_residual_score = self.model_fit._calculate_pearson_residuals(
            self,
            use_reliability=use_reliability,
            base_reliability=base_reliability)
        model_score = -np.log10(PearsonResidualCDF(pearson_residual_score / pearson_bias) + 1e-6).sum()
        return model_score

    def _get_intensity_observed_expected(self, normalized=False, base_reliability=0.5):
        c, intens, t, yhat = self.model_fit._get_predicted_intensities(self)
        p = (intens / t)[:-1]
        yhat = yhat[:-1]
        if len(p) == 1:
            return 0., 0.
        if normalized:
            reliability = self.model_fit._calculate_reliability(
                self, c, base_reliability=base_reliability)[:-1]
            p = t * p / np.sqrt(t * reliability * p * (1 - p))
            yhat = t * yhat / np.sqrt(t * reliability * yhat * (1 - yhat))
        return p, yhat

    def _calculate_correlation_coef(self, normalized=False, base_reliability=0.5):
        p, yhat = self._get_intensity_observed_expected(normalized, base_reliability)
        return np.corrcoef(p, yhat)[0, 1]

    def _calculate_correlation_distance(self, normalized=False, base_reliability=0.5):
        p, yhat = self._get_intensity_observed_expected(normalized, base_reliability)
        return distcorr(p, yhat)

    def _transform_correlation(self, normalized=False, base_reliability=0.5):
        r = self._calculate_correlation_coef(normalized=normalized, base_reliability=base_reliability)
        if np.isnan(r):
            r = -0.5
        c = (r + 1) / 2.
        return c

    def _transform_correlation_distance(self, normalized=False, base_reliability=0.5):
        r = self._calculate_correlation_distance(normalized=normalized, base_reliability=base_reliability)
        if np.isnan(r):
            r = 0
        c = (r + 1) / 2.
        return c

    def _get_predicted_peaks(self, scaled=True):
        c, intens, t, yhat = self.model_fit._get_predicted_intensities(self)
        mz = [ci.peak.mz for ci in c if ci.peak_pair]
        intensities = yhat[:-1] * (least_squares_scale_coefficient(yhat[:-1], intens[:-1]) if scaled else 1.0)
        return zip(mz, intensities)

    def glycan_score(self, use_reliability=True, base_reliability=0.5, pearson_bias=1.0):
        c, intens, t, yhat = self.model_fit._get_predicted_intensities(self)
        if self.model_fit.reliability_model is None or not use_reliability:
            reliability = np.ones_like(yhat)
        else:
            reliability = self.model_fit._calculate_reliability(
                self, c, base_reliability=base_reliability)

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

    def calculate_score(self, error_tolerance=2e-5, backbone_weight=None,
                        glycosylated_weight=None, stub_weight=None,
                        use_reliability=True, base_reliability=0.5,
                        pearson_bias=1.0, weighting=None, *args, **kwargs):
        assert self.model_fit is not None
        model_score = self._calculate_pearson_residual_score(
            use_reliability=use_reliability,
            base_reliability=base_reliability,
            pearson_bias=pearson_bias)
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


class ModelBindingScorer(GlycopeptideSpectrumMatcherBase):
    def __init__(self, tp, *args, **kwargs):
        self.tp = tp
        self.args = args
        self.kwargs = kwargs

    def __repr__(self):
        return "ModelBindingScorer(%s)" % (repr(self.tp),)

    def __eq__(self, other):
        return (self.tp == other.tp) and (self.args == other.args) and (self.kwargs == other.kwargs)

    def __call__(self, scan, target, *args, **kwargs):
        mass_shift = kwargs.pop("mass_shift", Unmodified)
        kwargs.update(self.kwargs)
        args = self.args + args
        return self.tp(scan, target, mass_shift=mass_shift, *args, **kwargs)

    def evaluate(self, scan, target, *args, **kwargs):
        mass_shift = kwargs.pop("mass_shift", Unmodified)
        inst = self.tp(scan, target, mass_shift=mass_shift, *self.args, **self.kwargs)
        inst.match(*args, **kwargs)
        inst.calculate_score(*args, **kwargs)
        return inst

    def __reduce__(self):
        return self.__class__, (self.tp, self.args, self.kwargs)


class DummyScorer(GlycopeptideSpectrumMatcherBase):
    def __init__(self, *args, **kwargs):
        raise TypeError("DummyScorer should not be instantiated!")


class PredicateBase(object):
    def __init__(self, root):
        self.root = root

    def value_for(self, scan, structure, *args, **kwargs):
        raise NotImplementedError()

    def query(self, point, *args, **kwargs):
        raise NotImplementedError()

    def find_nearest(self, point, *args, **kwargs):
        raise NotImplementedError()

    def get(self, scan, structure, *args, **kwargs):
        value = self.value_for(scan, structure, *args, **kwargs)
        try:
            result = self.query(value, *args, **kwargs)
            if result is None:
                result = self.find_nearest(value, *args, **kwargs)
        except (ValueError, KeyError):
            result = self.find_nearest(value, *args, **kwargs)
        return result

    def __call__(self, scan, structure, *args, **kwargs):
        return self.get(scan, structure, *args, **kwargs)


class IntervalPredicate(PredicateBase):

    def query(self, point, *args, **kwargs):
        for key, branch in self.root.items():
            if key[0] <= point <= key[1]:
                return branch
        return None

    def find_nearest(self, point, *args, **kwargs):
        best_key = None
        best_distance = float('inf')
        for key, branch in self.root.items():
            centroid = (key[0] + key[1]) / 2.
            distance = math.sqrt((centroid - point) ** 2)
            if distance < best_distance:
                best_distance = distance
                best_key = key
        return self.root[best_key]


class PeptideLengthPredicate(IntervalPredicate):
    def value_for(self, scan, structure, *args, **kwargs):
        peptide_size = len(structure)
        return peptide_size


class GlycanSizePredicate(IntervalPredicate):
    def value_for(self, scan, structure, *args, **kwargs):
        glycan_size = sum(structure.glycan_composition.values())
        return glycan_size


class MappingPredicate(PredicateBase):
    def query(self, point, *args, **kwargs):
        try:
            return self.root[point]
        except KeyError:
            return None

    def _distance(self, x, y):
        return x - y

    def find_nearest(self, point, *args, **kwargs):
        best_key = None
        best_distance = float('inf')
        for key, branch in self.root.items():
            distance = math.sqrt(self._distance(key, point) ** 2)
            if distance < best_distance:
                best_distance = distance
                best_key = key
        return self.root[best_key]


class ChargeStatePredicate(MappingPredicate):
    def value_for(self, scan, structure, *args, **kwargs):
        charge = scan.precursor_information.charge
        return charge

    def find_nearest(self, point, *args, **kwargs):
        try:
            return super(ChargeStatePredicate, self).find_nearest(point, *args, **kwargs)
        except TypeError:
            if point == ChargeNotProvided:
                keys = sorted(self.root.keys())
                n = len(keys)
                if n > 0:
                    return self.root[keys[int(n / 2)]]
                raise


class ProtonMobilityPredicate(MappingPredicate):

    def _distance(self, x, y):
        enum = {'mobile': 0, 'partial': 1, 'immobile': 2}
        return enum[x] - enum[y]

    def value_for(self, scan, structure, *args, **kwargs):
        return classify_proton_mobility(scan, structure)


class GlycanTypeCountPredicate(PredicateBase):
    def value_for(self, scan, structure, *args, **kwargs):
        return structure.glycosylation_manager

    def query(self, point, *args, **kwargs):
        glycosylation_manager = point
        for key, branch in self.root.items():
            count = glycosylation_manager.count_glycosylation_type(key)
            if count != 0:
                try:
                    return branch[count]
                except KeyError:
                    raise ValueError("Could Not Find Leaf")
        return None

    def find_nearest(self, point, *args, **kwargs):
        best_key = None
        best_distance = float('inf')
        glycosylation_manager = point
        for key, branch in self.root.items():
            count = glycosylation_manager.count_glycosylation_type(key)
            if count != 0:
                for cnt, value in branch.items():
                    distance = math.sqrt((count - cnt) ** 2)
                    if distance < best_distance:
                        best_distance = distance
                        best_key = (key, cnt)
        return self.root[best_key[0]][best_key[1]]


class PartitionTree(DummyScorer):

    _scorer_type = MultinomialRegressionScorer
    _short_peptide_scorer_type = ShortPeptideMultinomialRegressionScorer

    def __init__(self, root):
        self.root = root
        self.size = 5

    def get_model_for(self, scan, structure, *args, **kwargs):
        i = 0
        layer = self.root
        while i < self.size:
            if i == 0:
                predicate = PeptideLengthPredicate(layer)
                layer = predicate(scan, structure, *args, **kwargs)
                i += 1
            if i == 1:
                predicate = GlycanSizePredicate(layer)
                layer = predicate(scan, structure, *args, **kwargs)
                i += 1
            if i == 2:
                predicate = ChargeStatePredicate(layer)
                layer = predicate(scan, structure, *args, **kwargs)
                i += 1
            if i == 3:
                predicate = ProtonMobilityPredicate(layer)
                layer = predicate(scan, structure, *args, **kwargs)
                i += 1
            if i == 4:
                predicate = GlycanTypeCountPredicate(layer)
                layer = predicate(scan, structure, *args, **kwargs)
                i += 1
                return layer
            else:
                raise ValueError("Could Not Find Leaf %d" % i)
        raise ValueError("Could Not Find Leaf %d" % i)

    def evaluate(self, scan, target, *args, **kwargs):
        model = self.get_model_for(scan, target, *args, **kwargs)
        return model.evaluate(scan, target, *args, **kwargs)

    def __call__(self, scan, target, *args, **kwargs):
        model = self.get_model_for(scan, target, *args, **kwargs)
        return model(scan, target, *args, **kwargs)

    @classmethod
    def build_tree(cls, key_tuples, i, n, solution_map):
        aggregate = defaultdict(list)
        for key in key_tuples:
            aggregate[key[i]].append(key)
        if i < n:
            result = dict()
            for k, vs in aggregate.items():
                result[k] = cls.build_tree(vs, i + 1, n, solution_map)
            return result
        else:
            result = dict()
            for k, vs in aggregate.items():
                if len(vs) > 1:
                    raise ValueError("Multiple specifications at a leaf node")
                result[k] = solution_map[vs[0]]
            return result

    @classmethod
    def from_json(cls, d):
        arranged_data = dict()
        for spec_d, model_d in d:
            model = MultinomialRegressionFit.from_json(model_d)
            spec = partition_cell_spec.from_json(spec_d)
            scorer_type = cls._scorer_type_for_spec(spec)
            arranged_data[spec] = cls._bind_model_scorer(scorer_type, model, spec)
        root = cls.build_tree(arranged_data, 0, 5, arranged_data)
        return cls(root)

    @classmethod
    def _scorer_type_for_spec(cls, spec):
        if spec.peptide_length_range[1] <= 10:
            scorer_type = cls._short_peptide_scorer_type
        else:
            scorer_type = cls._scorer_type
        return scorer_type

    def __iter__(self):
        work = deque()
        work.extend(self.root.values())
        while work:
            item = work.popleft()
            if isinstance(item, dict):
                work.extend(item.values())
            else:
                yield item

    def __len__(self):
        return len(list(iter(self)))

    @classmethod
    def _bind_model_scorer(cls, scorer_type, model, partition=None):
        return ModelBindingScorer(
            scorer_type, multinomial_model=model, partition=partition)

    def __reduce__(self):
        return self.__class__, (self.root,)

    def __repr__(self):
        return "PartitionTree(%d)" % (len(self.root),)
