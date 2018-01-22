import itertools

from collections import namedtuple, defaultdict, OrderedDict
from glycopeptidepy.structure.glycan import GlycosylationType
from glypy.utils import make_struct


partition_cell_spec = namedtuple("partition_cell_spec", ("peptide_length_range",
                                                         "glycan_size_range",
                                                         "charge",
                                                         "glycan_type",
                                                         "glycan_count"))


class partition_cell_spec(partition_cell_spec):

    def test(self, gpsm):
        glycan_size = sum(gpsm.structure.glycan_composition.values())
        if len(gpsm.structure) > self.peptide_length_range[1] or len(gpsm.structure) < self.peptide_length_range[0]:
            return False
        if glycan_size > self.glycan_size_range[1] or glycan_size < self.glycan_size_range[0]:
            return False
        if gpsm.precursor_information.charge != self.charge:
            return False
        if gpsm.structure.glycosylation_manager.count_glycosylation_type(self.glycan_type) != self.glycan_count:
            return False
        return True


peptide_backbone_length_ranges = ((0, 10), (10, 20), (20, 30), (30, 40))
glycan_size_ranges = ((0, 5), (5, 9), (9, 13), (13, 20))
precursor_charges = (2, 3, 4, 5, 6)
glycosylation_type = tuple(GlycosylationType[i] for i in range(1, 4))
glycosylation_count = (1, 2,)


partition_by = map(lambda x: partition_cell_spec(*x), itertools.product(
    peptide_backbone_length_ranges, glycan_size_ranges, precursor_charges, glycosylation_type,
    glycosylation_count))


partition_cell = make_struct("partition_cell", ("breaks", "matched", "totals", "subset", "fit", "spec"))


def init_cell(breaks=None, matched=None, totals=None, subset=None, fit=None, spec=None):
    return partition_cell(breaks or [], matched or [], totals or [], subset or [], None, None)
