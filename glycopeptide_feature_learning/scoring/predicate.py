import math
import json
import gzip
import io

try:
    import cPickle as pickle
except ImportError:
    import pickle

from collections import defaultdict, deque, OrderedDict, namedtuple

from ms_deisotope.data_source import ChargeNotProvided

from glycopeptide_feature_learning.partitions import classify_proton_mobility, partition_cell_spec, count_labile_monosaccharides
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

    def value_for(self, scan, structure):
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

    def query(self, point):
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

    def find_nearest(self, point):
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

    def get(self, scan, structure):
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
        value = self.value_for(scan, structure)
        try:
            result = self.query(value)
            if result is None:
                result = self.find_nearest(value)
                # print(
                #     "find_nearest: %s, %s -> %s -> %s" % (scan.id, structure, value, result))
        except (ValueError, KeyError) as _err:
            result = self.find_nearest(value)
            # print("find_nearest: %s, %s -> %s -> %s (err = %r)" %
            #       (scan.id, structure, value, result, _err))
        return result

    def __call__(self, scan, structure):
        return self.get(scan, structure)


class IntervalPredicate(PredicateBase):
    """A predicate layer which selects its matches by
    interval inclusion
    """
    def query(self, point):
        # traverses earlier intervals first which may overlap edges of later
        # intervals.
        for key, branch in self.root.items():
            if key[0] <= point <= key[1]:
                return branch
        return None

    def find_nearest(self, point):
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
    def value_for(self, scan, structure):
        peptide_size = len(structure)
        return peptide_size


class GlycanSizePredicate(IntervalPredicate):
    """An :class:`IntervalPredicate` whose point value is the
    overall size of a glycan composition aggregate.
    """

    def __init__(self, root, omit_labile=False):
        super(GlycanSizePredicate, self).__init__(root)
        self.omit_labile = omit_labile

    def value_for(self, scan, structure):
        glycan_size = structure.total_glycosylation_size
        if self.omit_labile:
            glycan_size -= count_labile_monosaccharides(structure.glycan_composition)
        return glycan_size


class MappingPredicate(PredicateBase):
    """A predicate layer which selects its matches by
    :class:`~.Mapping` lookup.
    """
    def query(self, point):
        try:
            return self.root[point]
        except (KeyError, TypeError):
            return None

    def _distance(self, x, y):
        return x - y

    def find_nearest(self, point):
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
    def value_for(self, scan, structure):
        charge = scan.precursor_information.charge
        return charge

    def find_nearest(self, point):
        try:
            return super(ChargeStatePredicate, self).find_nearest(point)
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

    def value_for(self, scan, structure):
        return classify_proton_mobility(scan, structure)


class GlycanTypeCountPredicate(PredicateBase):
    """A :class:`PredicateBase` which selects based upon the type and number
    of the glycans attached to the query peptide.
    """
    def value_for(self, scan, structure):
        return structure.glycosylation_manager

    def query(self, point):
        glycosylation_manager = point
        for key, branch in self.root.items():
            count = glycosylation_manager.count_glycosylation_type(key)
            if count != 0:
                try:
                    return branch[count]
                except KeyError:
                    raise ValueError("Could Not Find Leaf")
        return None

    def find_nearest(self, point):
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


class SialylatedPredicate(MappingPredicate):
    def value_for(self, scan, structure):
        return count_labile_monosaccharides(structure.glycan_composition)

    def query(self, point):
        try:
            return self.root[point]
        except KeyError:
            return self.root[None]


def decompressing_reconstructor(cls, data, kwargs=None):
    if kwargs is None:
        kwargs = {}
    if isinstance(data, (str, bytes)):
        buff = io.BytesIO(data)
        data = pickle.load(gzip.GzipFile(fileobj=buff))
    return cls(data, **kwargs)


def compressing_reducer(self):
    data = self.root
    buff = io.BytesIO()
    writer = gzip.GzipFile(fileobj=buff, mode='wb')
    pickle.dump(data, writer, 2)
    writer.flush()
    data = buff.getvalue()
    return decompressing_reconstructor, (self.__class__, data, {"size": self.size, "omit_labile": self.omit_labile})


class PredicateFilterBase(object):

    predicate_sequence = [
        PeptideLengthPredicate,
        GlycanSizePredicate,
        ChargeStatePredicate,
        ProtonMobilityPredicate,
        GlycanTypeCountPredicate,
        SialylatedPredicate
    ]

    def __init__(self, root, size=None, omit_labile=False):
        self.root = root
        self.size = size
        self.omit_labile = omit_labile

        if self.size is None:
            self.size = self._guess_size()

    def _guess_size(self):
        n = 0
        work = deque([(self.root, 0)])
        while work:
            item, current_depth = work.popleft()
            if current_depth > n:
                n = current_depth
            if isinstance(item, dict):
                for val in item.values():
                    work.append((val, current_depth + 1))
        return n

    def get_model_for(self, scan, structure):
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
                layer = predicate(scan, structure)
                i += 1
            elif i == 1:
                predicate = GlycanSizePredicate(
                    layer, omit_labile=self.omit_labile)
                layer = predicate(scan, structure)
                i += 1
            elif i == 2:
                predicate = ChargeStatePredicate(layer)
                layer = predicate(scan, structure)
                i += 1
            elif i == 3:
                predicate = ProtonMobilityPredicate(layer)
                layer = predicate(scan, structure)
                i += 1
            elif i == 4:
                predicate = GlycanTypeCountPredicate(layer)
                layer = predicate(scan, structure)
                i += 1
            elif i == 5:
                if not isinstance(layer, dict):
                    return layer
                predicate = SialylatedPredicate(layer)
                layer = predicate(scan, structure)
                i += 1
            else:
                if not isinstance(layer, dict):
                    return layer
                raise ValueError("Could Not Find Leaf %d" % i)
        if not isinstance(layer, dict):
            return layer
        raise ValueError("Could Not Find Leaf %d" % i)

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

    def __repr__(self):
        return "%s(%d)" % (self.__class__.__name__, len(self),)

    def __eq__(self, other):
        try:
            my_root = self.root
            other_root = other.root
        except AttributeError:
            return False
        return my_root == other_root


spec_members_pair = namedtuple("spec_members_pair", ('spec', 'members'))


class PredicateFilter(PredicateFilterBase):

    @classmethod
    def build_tree(cls, key_tuples, i, n, node_map):
        '''A recursive function to reconstruct sub-trees from a flattened
        tree.

        Parameters
        ----------
        key_tuples : :class:`Iterable` of :class:`~.partition_cell_spec`
            The set of model bin specifications
        i : :class:`int`
            The index of the field of the specification to order by
        n : :class:`int`
            The total number of indices in a specification
        node_map : :class:`defaultdict` of :class:`list`
            A mapping between :class:`~.partition_cell_spec` and a value type, e.g. :class:`~.MultinomialRegressionFit`

        Returns
        -------
        :class:`OrderedDict`

        '''
        aggregate = defaultdict(list)
        for key in key_tuples:
            aggregate[key[i]].append(key)
        if i < n:
            result = OrderedDict()
            for k, vs in sorted(aggregate.items(), key=lambda x: x[0]):
                result[k] = cls.build_tree(vs, i + 1, n, node_map)
            return result
        else:
            result = OrderedDict()
            for k, vs in sorted(aggregate.items(), key=lambda x: x[0]):
                if len(vs) > 1:
                    raise ValueError("Multiple specifications at a leaf node")
                result[k] = node_map[vs[0]]
            return result

    @classmethod
    def from_spec_list(cls, specs, **kwargs):
        arranged_data = dict()
        n = None
        for spec in specs:
            # Ensure that all model specifications have the same number of dimensions for
            # the tree reconstruction to be consistent
            if n is None:
                n = len(spec) - 1
            else:
                assert n == (len(spec) - 1)
            arranged_data[spec] = spec_members_pair(spec, [])
        arranged_data = dict(arranged_data)
        root = cls.build_tree(arranged_data, 0, n, arranged_data)
        return cls(root, n, **kwargs)

    def __getitem__(self, scan_structure):
        return self.get_model_for(*scan_structure)

    def build_reverse_mapping(self):
        acc = dict()
        for spec, value in self:
            acc[spec] = value
        return acc


class PredicateTreeBase(PredicateFilterBase, DummyScorer):
    """A base class for predicate tree based model determination.
    """

    _scorer_type = None
    _short_peptide_scorer_type = None

    def __init__(self, root, size=None, omit_labile=False): # pylint: disable=super-init-not-called
        self.root = root
        self.size = size
        self.omit_labile = omit_labile

        if self.size is None:
            self.size = self._guess_size()

    def evaluate(self, scan, target, *args, **kwargs):
        model = self.get_model_for(scan, target)
        return model.evaluate(scan, target, *args, **kwargs)

    def __call__(self, scan, target, *args, **kwargs):
        model = self.get_model_for(scan, target)
        return model(scan, target, *args, **kwargs)

    @classmethod
    def build_tree(cls, key_tuples, i, n, node_map):
        '''A recursive function to reconstruct sub-trees from a flattened
        tree.

        Parameters
        ----------
        key_tuples : :class:`Iterable` of :class:`~.partition_cell_spec`
            The set of model bin specifications
        i : :class:`int`
            The index of the field of the specification to order by
        n : :class:`int`
            The total number of indices in a specification
        node_map : :class:`defaultdict` of :class:`list`
            A mapping between :class:`~.partition_cell_spec` and :class:`~.MultinomialRegressionFit`

        Returns
        -------
        :class:`OrderedDict`

        '''
        aggregate = defaultdict(list)
        for key in key_tuples:
            aggregate[key[i]].append(key)
        if i < n:
            result = OrderedDict()
            for k, vs in sorted(aggregate.items(), key=lambda x: x[0]):
                result[k] = cls.build_tree(vs, i + 1, n, node_map)
            return result
        else:
            result = OrderedDict()
            for k, vs in sorted(aggregate.items(), key=lambda x: x[0]):
                if len(vs) > 1:
                    raise ValueError("Multiple specifications at a leaf node")
                result[k] = node_map[vs[0]]
            return result

    @classmethod
    def from_json(cls, d):
        arranged_data = defaultdict(list)
        n = None
        # Check whether the payload is just a raw list of (spec, model) pairs or a wrapper
        # with extra metadata
        if isinstance(d, dict):
            meta = d['metadata']
            d = d['models']
            omit_labile = meta.get('omit_labile', False)
        else:
            meta = dict()
            omit_labile = False
        for spec_d, model_d in d:
            model = MultinomialRegressionFit.from_json(model_d)
            spec = partition_cell_spec.from_json(spec_d)
            # Ensure that all model specifications have the same number of dimensions for
            # the tree reconstruction to be consistent
            if n is None:
                n = len(spec) - 1
            else:
                assert n == (len(spec) - 1)
            arranged_data[spec].append(model)
        for spec, models in arranged_data.items():
            scorer_type = cls._scorer_type_for_spec(spec)
            arranged_data[spec] = cls._bind_model_scorer(scorer_type, models, spec)
        arranged_data = dict(arranged_data)
        root = cls.build_tree(arranged_data, 0, n, arranged_data)
        return cls(root, n, omit_labile)

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

    @classmethod
    def _bind_model_scorer(cls, scorer_type, models, partition=None):
        return ModelBindingScorer(scorer_type, model_fits=models, partition=partition)

    def __reduce__(self):
        return compressing_reducer(self)
