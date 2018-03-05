import itertools

from collections import namedtuple, defaultdict, OrderedDict

import numpy as np

from glycopeptidepy.structure.glycan import GlycosylationType
from glypy.utils import make_struct

from .amino_acid_classification import proton_mobility


def classify_proton_mobility(gpsm):
    k = proton_mobility(gpsm.structure)
    charge = gpsm.precursor_information.charge
    if k < charge:
        return 'mobile'
    elif k == charge:
        return 'partial'
    else:
        return 'immobile'


partition_cell_spec = namedtuple("partition_cell_spec", ("peptide_length_range",
                                                         "glycan_size_range",
                                                         "charge",
                                                         "proton_mobility",
                                                         "glycan_type",
                                                         "glycan_count"))


class partition_cell_spec(partition_cell_spec):

    def test(self, gpsm):
        glycan_size = sum(gpsm.structure.glycan_composition.values())
        if len(gpsm.structure) > self.peptide_length_range[1] or len(gpsm.structure) < self.peptide_length_range[0]:
            return False
        if glycan_size > self.glycan_size_range[1] or glycan_size < self.glycan_size_range[0]:
            return False
        if classify_proton_mobility(gpsm) != self.proton_mobility:
            return False
        if gpsm.precursor_information.charge != self.charge:
            return False
        if gpsm.structure.glycosylation_manager.count_glycosylation_type(self.glycan_type) != self.glycan_count:
            return False
        return True


peptide_backbone_length_ranges = [(a, a + 5) for a in range(0, 50, 5)]
glycan_size_ranges = [(a, a + 4) for a in range(1, 20, 4)]
precursor_charges = (2, 3, 4, 5, 6)
proton_mobilities = ('mobile', 'partial', 'immobile')
# glycosylation_type = tuple(GlycosylationType[i] for i in range(1, 4))
glycosylation_type = (GlycosylationType.n_linked,)
glycosylation_count = (1, 2,)


partition_by = map(lambda x: partition_cell_spec(*x), itertools.product(
    peptide_backbone_length_ranges, glycan_size_ranges, precursor_charges, proton_mobilities, glycosylation_type,
    glycosylation_count))


partition_cell = make_struct("partition_cell", ("subset", "fit", "spec"))


def init_cell(subset=None, fit=None, spec=None):
    return partition_cell(subset or [], fit, spec)


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


def partition_observations(gpsms, exclusive=True):
    partition_map = PartitionMap()
    j = 0
    cnt = 0
    interval = 250
    for spec in partition_by:
        rest = []
        subset = []
        j += 1
        if j % interval == 0:
            print(spec)
        for i in range(len(gpsms)):
            gpsm = gpsms[i]
            if not spec.test(gpsm):
                rest.append(gpsm)
                continue
            subset.append(gpsm)
            cnt += 1
        n = len(subset)
        if j % interval == 0:
            print(cnt)
        if n > 0:
            partition_map[spec] = init_cell(subset, {}, spec)
        if exclusive:
            gpsms = rest
    return partition_map


class KFoldSplitter(object):
    def __init__(self, n_splits):
        self.n_splits = n_splits

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
        data = np.array(data)
        for test_index in self._mask(data):
            train_index = indices[np.logical_not(test_index)]
            test_index = indices[test_index]
            yield data[train_index], data[test_index]


def crossvalidation_sets(gpsms, kfolds=3):
    '''
    Create k stratified cross-validation sets, stratified
    by glycopeptide identity.
    '''
    holders = defaultdict(list)
    for gpsm in gpsms:
        holders[gpsm.structure].append(gpsm)
    splitter = KFoldSplitter(kfolds)
    singletons = set()
    combinables = []
    for k, v in holders.items():
        if len(v) == 1:
            singletons.add(k)
        else:
            combinables.append((splitter.split(v), v))
    v = [holders[k] for k in singletons]
    combinables.append((splitter.split(v), v))

    splits = [(list(), list()) for i in range(kfolds)]
    for combinable, v in combinables:
        for i, pair in enumerate(combinable):
            try:
                if isinstance(pair[0][0], np.ndarray):
                    pair = [np.hstack(pair[0]), np.hstack(pair[1])]
                splits[i][0].extend(pair[0])
                splits[i][1].extend(pair[1])
            except (IndexError, ValueError):
                continue
    return splits
