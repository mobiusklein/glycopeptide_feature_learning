import itertools
import logging
from collections import namedtuple, defaultdict, OrderedDict

import numpy as np
from ms_deisotope.data_source import ProcessedScan

from glycopeptidepy.structure.glycan import GlycosylationType
from glypy.utils import make_struct

from glycan_profiling.tandem.glycopeptide.core_search import approximate_internal_size_of_glycan, FrozenMonosaccharideResidue

from .amino_acid_classification import proton_mobility


logger = logging.getLogger(
    "glycopeptide_feature_learning.partitions")
logger.addHandler(logging.NullHandler())



def classify_proton_mobility(scan, structure):
    try:
        k = structure.proton_mobility
    except AttributeError:
        k = proton_mobility(structure)
        # Try to abuse non-strict attributes for caching.
        try:
            structure.proton_mobility = k
        except AttributeError:
            pass
    charge = scan.precursor_information.charge
    if k < charge:
        return 'mobile'
    elif k == charge:
        return 'partial'
    else:
        return 'immobile'


_NEUAC = FrozenMonosaccharideResidue.from_iupac_lite("NeuAc")
_NEUGC = FrozenMonosaccharideResidue.from_iupac_lite("NeuGc")


def count_labile_monosaccharides(glycan_composition):
    k = glycan_composition._getitem_fast(_NEUAC)
    k += glycan_composition._getitem_fast(_NEUGC)
    return k


_partition_cell_spec = namedtuple("partition_cell_spec", ("peptide_length_range",
                                                          "glycan_size_range",
                                                          "charge",
                                                          "proton_mobility",
                                                          "glycan_type",
                                                          "glycan_count",
                                                        #   "sialylated"
                                                          ))


class partition_cell_spec(_partition_cell_spec):
    __slots__ = ()

    def __new__(cls, peptide_length_range, glycan_size_range, charge,
                proton_mobility, glycan_type, glycan_count, sialylated=None):
        self = super(partition_cell_spec, cls).__new__(
            cls, peptide_length_range, glycan_size_range, charge,
            proton_mobility, glycan_type, glycan_count,
            # sialylated
            )
        return self

    def test(self, gpsm, omit_labile=False):
        structure = gpsm.structure
        if structure.glycosylation_manager.count_glycosylation_type(self.glycan_type) != self.glycan_count:
            return False
        glycan_size = glycan_size = structure.total_glycosylation_size
        if omit_labile:
            glycan_size -= count_labile_monosaccharides(structure.glycan_composition)
        peptide_size = len(structure)
        if peptide_size < self.peptide_length_range[0] or peptide_size > self.peptide_length_range[1]:
            return False
        if glycan_size < self.glycan_size_range[0] or glycan_size > self.glycan_size_range[1]:
            return False
        if classify_proton_mobility(gpsm, structure) != self.proton_mobility:
            return False
        if gpsm.precursor_information.charge != self.charge:
            return False
        # if bool(count_labile_monosaccharides(structure.glycan_composition)) != self.sialylated:
        #     return False
        return True

    def test_peptide_size(self, scan, structure, *args, **kwargs):
        peptide_size = len(structure)
        invalid = (peptide_size < self.peptide_length_range[0] or
                   peptide_size > self.peptide_length_range[1])
        return not invalid

    def test_glycan_size(self, scan, structure, omit_labile=False, *args, **kwargs):
        if omit_labile:
            glycan_size = approximate_internal_size_of_glycan(
                structure.glycan_composition)
        else:
            glycan_size = sum(structure.glycan_composition.values())
        invalid = (glycan_size < self.glycan_size_range[0] or
                   glycan_size > self.glycan_size_range[1])
        return not invalid

    def test_proton_mobility(self, scan, structure, *args, **kwargs):
        pm = classify_proton_mobility(scan, structure)
        return self.proton_mobility == pm

    def test_charge(self, scan, structure, *args, **kwargs):
        return scan.precursor_information.charge == self.charge

    def test_glycan_count(self, scan, structure, *args, **kwargs):
        count = structure.structure.glycosylation_manager.count_glycosylation_type(self.glycan_type)
        return count == self.glycan_count

    def compact(self, sep=':'):
        return sep.join(map(str, self))

    def to_json(self):
        d = {}
        d['peptide_length_range'] = self.peptide_length_range
        d['glycan_size_range'] = self.glycan_size_range
        d['charge'] = self.charge
        d['proton_mobility'] = self.proton_mobility
        d['glycan_type'] = str(getattr(self.glycan_type, "name", self.glycan_type))
        d['glycan_count'] = self.glycan_count
        # d['sialylated'] = self.sialylated
        return d

    @classmethod
    def from_json(cls, d):
        d['glycan_type'] = GlycosylationType[d['glycan_type']]
        d['peptide_length_range'] = tuple(d['peptide_length_range'])
        d['glycan_size_range'] = tuple(d['glycan_size_range'])
        return cls(**d)


k = 5
peptide_backbone_length_ranges = [(a, a + k) for a in range(0, 50, k)]
glycan_size_ranges = [(a, a + 4) for a in range(1, 20, 4)]
precursor_charges = (2, 3, 4, 5, 6)
proton_mobilities = ('mobile', 'partial', 'immobile')
glycosylation_types = tuple(GlycosylationType[i] for i in range(1, 4))
glycosylation_counts = (1, 2,)
sialylated = (False, True)


def build_partition_rules_from_bins(peptide_backbone_length_ranges=peptide_backbone_length_ranges, glycan_size_ranges=glycan_size_ranges,
                                    precursor_charges=precursor_charges, proton_mobilities=proton_mobilities, glycosylation_types=glycosylation_types,
                                    glycosylation_counts=glycosylation_counts):
    dimensions = itertools.product(
        peptide_backbone_length_ranges,
        glycan_size_ranges,
        precursor_charges,
        proton_mobilities,
        glycosylation_types,
        glycosylation_counts,
    )
    return [partition_cell_spec(*x) for x in dimensions]



class partition_cell(make_struct("partition_cell", ("subset", "fit", "spec"))):
    def __len__(self):
        return len(self.subset)


def init_cell(subset=None, fit=None, spec=None):
    return partition_cell(subset or [], fit or {}, spec)


def adjacent_specs(spec, charge=1, glycan_count=True):
    adjacent = []
    charges = [spec.charge]
    if charge:
        min_charge = min(precursor_charges)
        max_charge = max(precursor_charges)
        current_charge = spec.charge
        for i in range(1, charge + 1):
            if current_charge - i > min_charge:
                adjacent.append(spec._replace(charge=current_charge - i))
                charges.append(current_charge - i)
            if current_charge + i < max_charge:
                adjacent.append(spec._replace(charge=current_charge + i))
                charges.append(current_charge + i)
    # for adj in list(adjacent):
    #     adjacent.append(adj._replace(sialylated=not adj.sialylated))
    # adjacent.append(spec._replace(sialylated=not spec.sialylated))
    return adjacent


class PartitionMap(OrderedDict):

    def adjacent(self, spec, charge=True, glycan_count=True):
        cells = [self.get(spec)]
        for other_spec in adjacent_specs(spec, charge=charge):
            cells.append(self.get(other_spec))
        matches = []
        for cell in cells:
            if cell is None:
                continue
            matches.extend(cell.subset)
        return matches

    def sort(self):
        items = sorted(self.items(), key=lambda x: x[0])
        self.clear()
        for key, value in items:
            self[key] = value
        return self


def partition_observations(gpsms, exclusive=True, partition_specifications=None, omit_labile=False):
    # Consider re-organizing to move PredicateFilter to partitions
    from glycopeptide_feature_learning.scoring.predicate import PredicateFilter
    if partition_specifications is None:
        partition_specifications = build_partition_rules_from_bins()
    partition_map = PartitionMap()
    forward_map = PredicateFilter.from_spec_list(
        partition_specifications, omit_labile=omit_labile)
    for i, gpsm in enumerate(gpsms):
        if i % 1000 == 0 and i:
            logger.info("Partitioned %d GPSMs" % (i, ))
        pair = forward_map[gpsm, gpsm.target]
        # Ensure that the GPSM actually belongs to the partition spec and isn't a nearest
        # neighbor match
        if pair.spec.test(gpsm, omit_labile=omit_labile):
            pair.members.append(gpsm)
        else:
            logger.info("%s @ %s does not have a matching partition" %
                        (gpsm.target, gpsm.precursor_information))
    reverse_map = forward_map.build_reverse_mapping()
    for spec in partition_specifications:
        subset = reverse_map[spec]
        n = len(subset)
        if n > 0:
            partition_map[spec] = init_cell(subset, {}, spec)
    return partition_map



def make_shuffler(seed=None):
    if seed is None:
        return np.random.shuffle
    return np.random.RandomState(int(seed)).shuffle


def _identity(x):
    return x


class KFoldSplitter(object):
    def __init__(self, n_splits, shuffler=None):
        if shuffler is None:
            shuffler = _identity
        self.n_splits = n_splits
        self.shuffler = shuffler

    def _indices(self, data):
        n_samples = len(data)
        indices = np.arange(n_samples)
        n_splits = self.n_splits
        fold_sizes = (n_samples // n_splits) * np.ones(n_splits, dtype=np.int)
        fold_sizes[:n_samples % n_splits] += 1
        current = 0
        for fold_size in fold_sizes:
            start, stop = current, current + fold_size
            yield indices[start:stop]
            current = stop

    def _mask(self, data):
        n = len(data)
        for test_index in self._indices(data):
            mask = np.zeros(n, dtype=np.bool)
            mask[test_index] = True
            yield mask

    def split(self, data):
        n_samples = len(data)
        indices = np.arange(n_samples)
        self.shuffler(data)
        for test_index in self._mask(data):
            train_index = indices[np.logical_not(test_index)]
            test_index = indices[test_index]
            train_group = [data[i] for i in train_index]
            test_group = [data[i] for i in test_index]
            yield train_group, test_group


def group_by_structure(gpsms):
    holders = defaultdict(list)
    for gpsm in gpsms:
        holders[gpsm.structure].append(gpsm)
    return holders


def split_groups(groups, splitter=None):
    if splitter is None:
        splitter = KFoldSplitter(3)
    combinables = []
    singletons = set()
    for k, v in groups.items():
        if len(v) == 1:
            singletons.add(k)
        else:
            combinables.append((splitter.split(v), v))
    v = [groups[k][0] for k in singletons]
    combinables.append((splitter.split(v), v))
    return combinables


def crossvalidation_sets(gpsms, kfolds=3, shuffler=None, stratified=True):
    '''
    Create k stratified cross-validation sets, stratified
    by glycopeptide identity.
    '''
    splitter = KFoldSplitter(kfolds, shuffler)
    if not stratified:
        return list(splitter.split(gpsms))
    holders = group_by_structure(gpsms)
    combinables = split_groups(holders, splitter)

    splits = [(list(), list()) for i in range(kfolds)]
    for combinable, v in combinables:
        for i, pair in enumerate(combinable):
            assert not isinstance(pair[0], ProcessedScan)
            assert not isinstance(pair[1], ProcessedScan)
            try:
                if isinstance(pair[0][0], np.ndarray):
                    pair = [np.hstack(pair[0]), np.hstack(pair[1])]

                splits[i][0].extend(pair[0])
                splits[i][1].extend(pair[1])
            except (IndexError, ValueError):
                continue
    return splits
