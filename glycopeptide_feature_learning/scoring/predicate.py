import math
import json
import gzip
import io

try:
    import cPickle as pickle
except ImportError:
    import pickle

from collections import defaultdict, deque, OrderedDict

from ms_deisotope.data_source import ChargeNotProvided

from glycopeptide_feature_learning.partitions import classify_proton_mobility, partition_cell_spec
from glycopeptide_feature_learning.multinomial_regression import MultinomialRegressionFit


from .base import (DummyScorer, ModelBindingScorer)


class PredicateBase(object):
    """A base class for defining a model tree layer based upon some
    property of a query of scan and glycopeptide

    Attributes
    ----------
    root: :class:`dict`
    """
    def __init__(self, root):
        self.root = root

    def value_for(self, scan, structure, *args, **kwargs):
        """Obtain the value for this predicate from the query

        Parameters
        ----------
        scan : :class:`~.ProcessedScan`
            The processed mass spectrum to analyze
        structure : :class:`~.PeptideSequence`
            The structure to map against the spectrum.

        Returns
        -------
        object
        """
        raise NotImplementedError()

    def query(self, point, *args, **kwargs):
        """Find the appropriate branch or leaf to continue the search in

        Parameters
        ----------
        point : object
            A value of the appropriate type returned by :meth:`value_for`

        Returns
        -------
        object
        """
        raise NotImplementedError()

    def find_nearest(self, point, *args, **kwargs):
        """Find the nearest appropriate branch of leaf to continue the search
        in.

        Only used if :meth:`query` cannot find an appropriate match.

        Parameters
        ----------
        point : object
            A value of the appropriate type returned by :meth:`value_for`

        Returns
        -------
        object
        """
        raise NotImplementedError()

    def get(self, scan, structure, *args, **kwargs):
        """Find the next model tree layer (branch) or model
        specification (leaf) in the tree that fits the query
        parameters.

        Parameters
        ----------
        scan : :class:`~.ProcessedScan`
            The processed mass spectrum to analyze
        structure : :class:`~.PeptideSequence`
            The structure to map against the spectrum.

        Returns
        -------
        object
        """
        value = self.value_for(scan, structure, *args, **kwargs)
        try:
            result = self.query(value, *args, **kwargs)
            if result is None:
                result = self.find_nearest(value, *args, **kwargs)
                # print(
                #     "find_nearest: %s, %s -> %s -> %s" % (scan.id, structure, value, result))
        except (ValueError, KeyError) as _err:
            result = self.find_nearest(value, *args, **kwargs)
            # print("find_nearest: %s, %s -> %s -> %s (err = %r)" %
            #       (scan.id, structure, value, result, _err))
        return result

    def __call__(self, scan, structure, *args, **kwargs):
        return self.get(scan, structure, *args, **kwargs)


class IntervalPredicate(PredicateBase):
    """A predicate layer which selects its matches by
    interval inclusion
    """
    def query(self, point, *args, **kwargs):
        for key, branch in self.root.items():
            if key[0] <= point <= key[1]:
                return branch
        return None

    def find_nearest(self, point, *args, **kwargs):
        best_key = None
        best_distance = float('inf')
        for key, _ in self.root.items():
            centroid = (key[0] + key[1]) / 2.
            distance = abs((centroid - point))
            if distance < best_distance:
                best_distance = distance
                best_key = key
        return self.root[best_key]


class PeptideLengthPredicate(IntervalPredicate):
    """An :class:`IntervalPredicate` whose point value
    is the length of a peptide
    """
    def value_for(self, scan, structure, *args, **kwargs):
        peptide_size = len(structure)
        return peptide_size


class GlycanSizePredicate(IntervalPredicate):
    """An :class:`IntervalPredicate` whose point value is the
    overall size of a glycan composition aggregate.
    """
    def value_for(self, scan, structure, *args, **kwargs):
        glycan_size = sum(structure.glycan_composition.values())
        return glycan_size


class MappingPredicate(PredicateBase):
    """A predicate layer which selects its matches by
    :class:`~.Mapping` lookup.
    """
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
        for key, _ in self.root.items():
            distance = abs(self._distance(key, point))
            if distance < best_distance:
                best_distance = distance
                best_key = key
        return self.root[best_key]


class ChargeStatePredicate(MappingPredicate):
    """A :class:`MappingPredicate` whose point value is the charge
    state of the query scan's precursor ion.
    """
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
                if n % 2:
                    n -= 1
                if n > 0:
                    return self.root[keys[int(n // 2)]]
                raise


class ProtonMobilityPredicate(MappingPredicate):
    """A :class:`MappingPredicate` whose point value is the proton mobility
    class of the query scan and structure
    """
    def _distance(self, x, y):
        enum = {'mobile': 0, 'partial': 1, 'immobile': 2}
        return enum[x] - enum[y]

    def value_for(self, scan, structure, *args, **kwargs):
        return classify_proton_mobility(scan, structure)


class GlycanTypeCountPredicate(PredicateBase):
    """A :class:`PredicateBase` which selects based upon the type and number
    of the glycans attached to the query peptide.
    """
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
                for cnt, _ in branch.items():
                    distance = abs(count - cnt)
                    if distance < best_distance:
                        best_distance = distance
                        best_key = (key, cnt)
        # we didn't find a match strictly by glycan type and count, so instead
        # use the first glycan type with the best count, though a type match with the
        # wrong count would be acceptable.
        if best_key is None:
            count = len(glycosylation_manager)
            for key, branch in self.root.items():
                for cnt, _ in branch.items():
                    distance = math.sqrt((count - cnt) ** 2)
                    if key not in point.values():
                        distance += 1
                    if distance < best_distance:
                        best_distance = distance
                        best_key = (key, cnt)
        return self.root[best_key[0]][best_key[1]]


def decompressing_reconstructor(cls, data):
    if isinstance(data, (str, bytes)):
        buff = io.BytesIO(data)
        data = pickle.load(gzip.GzipFile(fileobj=buff))
    return cls(data)


def compressing_reducer(self):
    data = self.root
    buff = io.BytesIO()
    writer = gzip.GzipFile(fileobj=buff, mode='wb')
    pickle.dump(data, writer, 2)
    writer.flush()
    data = buff.getvalue()
    return decompressing_reconstructor, (self.__class__, data, )


class PredicateTreeBase(DummyScorer):
    """A base class for predicate tree based model determination.
    """

    _scorer_type = None
    _short_peptide_scorer_type = None

    def __init__(self, root): # pylint: disable=super-init-not-called
        self.root = root
        self.size = 5

    def get_model_for(self, scan, structure, *args, **kwargs):
        """Locate the appropriate model for the query scan and glycopeptide

        Parameters
        ----------
        scan : :class:`~.ProcessedScan`
            The query scan
        structure : :class:`~.PeptideSequence`
            The query peptide

        Returns
        -------
        object
        """
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
            result = OrderedDict()
            for k, vs in sorted(aggregate.items(), key=lambda x: x[0]):
                result[k] = cls.build_tree(vs, i + 1, n, solution_map)
            return result
        else:
            result = OrderedDict()
            for k, vs in sorted(aggregate.items(), key=lambda x: x[0]):
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
            arranged_data[spec].append(model)
        for spec, models in arranged_data.items():
            scorer_type = cls._scorer_type_for_spec(spec)
            arranged_data[spec] = cls._bind_model_scorer(scorer_type, models, spec)
        arranged_data = dict(arranged_data)
        root = cls.build_tree(arranged_data, 0, 5, arranged_data)
        return cls(root)

    def to_json(self):
        d_list = []
        for node in self:
            partition_cell_spec_inst = node.kwargs.get('partition')
            for model_fit in node.kwargs.get('model_fits', []):
                d_list.append((partition_cell_spec_inst.to_json(), model_fit.to_json(False)))
        return d_list

    @classmethod
    def from_file(cls, path):
        if not hasattr(path, 'read'):
            fh = open(path, 'rt')
        else:
            fh = path
        data = json.load(fh)
        inst = cls.from_json(data)
        return inst

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
        return compressing_reducer(self)

    def __repr__(self):
        return "%s(%d)" % (self.__class__.__name__, len(self.root),)

    def __eq__(self, other):
        try:
            my_root = self.root
            other_root = other.root
        except AttributeError:
            return False
        return my_root == other_root
