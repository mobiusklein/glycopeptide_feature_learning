import math
from collections import defaultdict, deque

from ms_deisotope.data_source import ChargeNotProvided

from feature_learning.partitions import classify_proton_mobility, partition_cell_spec
from feature_learning.multinomial_regression import MultinomialRegressionFit


from .base import (DummyScorer, ModelBindingScorer)


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


class PredicateTreeBase(DummyScorer):

    # _scorer_type = MultinomialRegressionScorer
    # _short_peptide_scorer_type = ShortPeptideMultinomialRegressionScorer
    _scorer_type = None
    _short_peptide_scorer_type = None

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
        arranged_data = defaultdict(list)
        for spec_d, model_d in d:
            model = MultinomialRegressionFit.from_json(model_d)
            spec = partition_cell_spec.from_json(spec_d)
            # scorer_type = cls._scorer_type_for_spec(spec)
            # arranged_data[spec] = cls._bind_model_scorer(scorer_type, model, spec)
            arranged_data[spec].append(model)
        for spec, models in arranged_data.items():
            scorer_type = cls._scorer_type_for_spec(spec)
            arranged_data[spec] = cls._bind_model_scorer(scorer_type, models, spec)
        arranged_data = dict(arranged_data)
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
    def _bind_model_scorer(cls, scorer_type, models, partition=None):
        return ModelBindingScorer(scorer_type, model_fits=models, partition=partition)

    def __reduce__(self):
        return self.__class__, (self.root,)

    def __repr__(self):
        return "%s(%d)" % (self.__class__.__name__, len(self.root),)
