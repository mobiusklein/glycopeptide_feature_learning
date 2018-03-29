import math

from collections import defaultdict

import numpy as np

from glycan_profiling.tandem.glycopeptide.scoring.simple_score import SimpleCoverageScorer
from glycan_profiling.tandem.glycopeptide.scoring.coverage_weighted_binomial import (
    FragmentMatchMap)

from glycan_profiling.tandem.glycopeptide.scoring.base import GlycopeptideSpectrumMatcherBase, ChemicalShift
from glycan_profiling.tandem.glycopeptide.scoring.glycan_signature_ions import GlycanCompositionSignatureMatcher
from glycan_profiling.tandem.glycopeptide.scoring.coverage_weighted_binomial import accuracy_bias
from glycan_profiling.tandem.spectrum_match import ModelTreeNode, Unmodified


from .peak_relations import probability_of_peak_explained, fragmentation_probability
from .multinomial_regression import PEARSON_BIAS, MultinomialRegressionFit
from .partitions import classify_proton_mobility, partition_cell_spec


class MultinomialRegressionScorer(SimpleCoverageScorer, GlycanCompositionSignatureMatcher):
    accuracy_bias = accuracy_bias

    def __init__(self, scan, sequence, mass_shift=None, multinomial_model=None):
        super(MultinomialRegressionScorer, self).__init__(scan, sequence, mass_shift)
        self.structure = self.target
        self.model_fit = multinomial_model
        self._init_signature_matcher()

    def match(self, error_tolerance=2e-5, *args, **kwargs):
        GlycanCompositionSignatureMatcher.match(self, error_tolerance=error_tolerance)

        solution_map = FragmentMatchMap()
        spectrum = self.spectrum
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

    def calculate_score(self, error_tolerance=2e-5, backbone_weight=None,
                        glycosylated_weight=None, stub_weight=None,
                        use_reliability=True, base_reliability=0.5,
                        pearson_bias=1.0,
                        *args, **kwargs):
        assert self.model_fit is not None

        model_score = self.model_fit.compound_score(
            self, use_reliability=use_reliability,
            base_reliability=base_reliability,
            pearson_bias=pearson_bias)
        coverage_score = self._coverage_score(backbone_weight, glycosylated_weight, stub_weight)
        offset = self.determine_precursor_offset()
        mass_accuracy = -10 * math.log10(
            1 - self.accuracy_bias.score(self.precursor_mass_accuracy(offset)))
        signature_component = GlycanCompositionSignatureMatcher.calculate_score(self)
        self._score = (model_score * coverage_score) + mass_accuracy + signature_component
        return self._score


class ShortPeptideMultinomialRegressionScorer(MultinomialRegressionScorer):
    stub_weight = 0.75


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
    def __init__(self, root):
        self.root = root
        self.size = 5

    def get_model_for(self, scan, structure, *args, **kwargs):
        i = 0
        layer = self.root
        # glycan_size = sum(structure.glycan_composition.values())
        # peptide_size = len(structure)
        # precursor_charge = scan.precursor_information.charge
        # mobility = classify_proton_mobility(scan, structure)
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

            # for key, branch in layer.items():
            #     if i == 0:
            #         if key[0] <= peptide_size <= key[1]:
            #             layer = branch
            #             i += 1
            #             break
            #     if i == 1:
            #         if key[0] <= glycan_size <= key[1]:
            #             layer = branch
            #             i += 1
            #             break
            #     if i == 2:
            #         if key == precursor_charge:
            #             layer = branch
            #             i += 1
            #             break
            #     if i == 3:
            #         if key == mobility:
            #             layer = branch
            #             i += 1
            #             break
            #     elif i == 4:
            #         count = structure.glycosylation_manager.count_glycosylation_type(key)
            #         if count != 0:
            #             try:
            #                 return branch[count]
            #             except KeyError:
            #                 raise ValueError("Could Not Find Leaf")
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
            spec = partition_cell_spec.from_json(spec_d)
            model = MultinomialRegressionFit.from_json(model_d)
            if spec.peptide_length_range[1] <= 10:
                scorer_type = ShortPeptideMultinomialRegressionScorer
            else:
                scorer_type = MultinomialRegressionScorer
            arranged_data[spec] = ModelBindingScorer(scorer_type, multinomial_model=model)
        root = cls.build_tree(arranged_data, 0, 5, arranged_data)
        return cls(root)

    def __reduce__(self):
        return self.__class__, (self.root,)

    def __repr__(self):
        return "PartitionTree(%d)" % (len(self.root),)
