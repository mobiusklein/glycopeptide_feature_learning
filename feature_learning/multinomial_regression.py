import six
import base64
import io

from collections import namedtuple, defaultdict, OrderedDict

import numpy as np
from scipy.stats import norm
from scipy.linalg import solve_triangular, cho_solve

from glypy.utils import Enum
from glypy.structure.glycan_composition import FrozenMonosaccharideResidue
from glycopeptidepy.structure.fragment import IonSeries

from glycan_profiling.structure.fragment_match_map import PeakFragmentPair
from ms_deisotope import DeconvolutedPeak

from .amino_acid_classification import (
    AminoAcidClassification, classify_amide_bond_frank, classify_residue_frank)
from .approximation import PearsonResidualCDF
from .peak_relations import FragmentationModelCollection


array_dtype = np.dtype("<d")

PEARSON_BIAS = 3.3


class FragmentSeriesClassification(Enum):
    b = 0
    y = 1
    stub_glycopeptide = 2
    unassigned = 3


# the number of ion series to consider is one less than the total number of series
# because unassigned is a special case which no matched peaks will receive
FragmentSeriesClassification_max = max(FragmentSeriesClassification, key=lambda x: x[1].value)[1].value - 1

# the number of backbone ion series to consider is two less because the stub_glycopeptide
# series is not a backbone fragmentation series
BackboneFragmentSeriesClassification_max = FragmentSeriesClassification_max - 1


class FragmentTypeClassification(AminoAcidClassification):
    pass


try:
    from feature_learning._c.model_types import (
        FragmentSeriesClassification,
        FragmentTypeClassification
    )
except ImportError:
    pass


FragmentTypeClassification_max = max(FragmentTypeClassification, key=lambda x: x[1].value)[1].value

# consider fragments with up to 2 monosaccharides attached to a backbone fragment
BackboneFragment_max_glycosylation_size = 2
# consider fragments of up to charge 4+
FragmentCharge_max = 4
# consider up to 10 monosaccharides of glycan still attached to a stub ion
StubFragment_max_glycosylation_size = 10

_FragmentType = namedtuple(
    "FragmentType", [
        "nterm", "cterm", "series", "glycosylated", "charge", "peak_pair", "sequence"])


class _FragmentType(_FragmentType):
    _is_backbone = None
    _is_assigned = None
    _is_stub_glycopeptide = None

    @property
    def fragment(self):
        return self.peak_pair.fragment

    @property
    def peak(self):
        return self.peak_pair.peak

    def is_assigned(self):
        if self._is_assigned is None:
            self._is_assigned = self.series != FragmentSeriesClassification.unassigned
        return self._is_assigned

    def is_backbone(self):
        if self._is_backbone is None:
            self._is_backbone = (
                self.series != FragmentSeriesClassification.stub_glycopeptide) and self.is_assigned()
        return self._is_backbone

    def is_stub_glycopeptide(self):
        if self._is_stub_glycopeptide is None:
            self._is_stub_glycopeptide = (
                self.series == FragmentSeriesClassification.stub_glycopeptide)
        return self._is_stub_glycopeptide

    def _allocate_feature_array(self):
        return np.zeros(self.feature_count, dtype=np.uint8)

    def build_feature_vector(self, X, offset):
        k_ftypes = (FragmentTypeClassification_max + 1)
        k_series = (FragmentSeriesClassification_max + 1)
        k_unassigned = 1
        k_charge = FragmentCharge_max + 1
        k_charge_series = k_charge * k_series

        k_glycosylated = BackboneFragment_max_glycosylation_size + 1

        k = (
            (k_ftypes * 2) + k_series + k_unassigned + k_charge + k_charge_series +
            k_glycosylated)

        X = np.zeros(k, dtype=np.uint8)
        offset = 0

        if self.nterm is not None:
            X[self.nterm.value] = 1
        offset += k_ftypes

        if self.cterm is not None:
            X[offset + self.cterm.value] = 1
        offset += k_ftypes

        if self.is_assigned():
            X[offset + self.series.value] = 1
        offset += k_series

        # tarck the unassigned placeholder observation separately
        X[offset] = int(not self.is_assigned())
        offset += k_unassigned

        # use charge - 1 because there is no 0 charge
        if self.is_backbone():
            X[offset + (self.charge - 1)] = 1
        offset += k_charge

        if self.is_assigned():
            index = (self.series.value * k_charge) + (self.charge - 1)
            X[offset + index] = 1
        offset += k_charge_series

        # non-stub ion glycosylation
        if self.is_backbone():
            X[offset + int(self.peak_pair.fragment.glycosylation_size)] = 1
        offset += k_glycosylated

        return X

    def as_feature_vector(self):
        X = self._allocate_feature_array()
        offset = 0
        self.build_feature_vector(X, offset)
        return X


def get_nterm_index_from_fragment(fragment, structure):
    size = len(structure)
    direction = fragment.series.direction
    if direction < 0:
        index = size + (fragment.series.direction * fragment.position + fragment.series.direction)
    else:
        index = fragment.position - 1
    return index


def get_cterm_index_from_fragment(fragment, structure):
    size = len(structure)
    direction = fragment.series.direction
    if direction < 0:
        index = size + (fragment.series.direction * fragment.position)
    else:
        index = fragment.position
    return index


class FragmentTypeMeta(type):
    type_cache = dict()

    def __new__(mcs, name, parents, attrs):
        new_type = type.__new__(mcs, name, parents, attrs)
        new_type._feature_count = None
        mcs.type_cache[name] = new_type
        return new_type

    @property
    def feature_count(self):
        if self._feature_count is None:
            self._feature_count = len(self.feature_names())
        return self._feature_count

    def get_model_by_name(self, name):
        return self.type_cache[name]


try:
    from feature_learning._c.model_types import _FragmentType
except ImportError:
    pass


@six.add_metaclass(FragmentTypeMeta)
class FragmentType(_FragmentType):
    def as_feature_dict(self):
        return OrderedDict(zip(self.feature_names(), self.as_feature_vector()))

    @classmethod
    def feature_names(cls):
        names = []
        for label, tp in sorted(FragmentTypeClassification, key=lambda x: x[1].value):
            if tp.value is None:
                continue
            names.append("n-term %s" % label)
        for label, tp in sorted(FragmentTypeClassification, key=lambda x: x[1].value):
            if tp.value is None:
                continue
            names.append("c-term %s" % label)

        for label, tp in sorted(FragmentSeriesClassification, key=lambda x: x[1].value):
            if tp.value is None or label == "unassigned":
                continue
            names.append("series %s" % label)
        names.append("unassigned")
        for i in range(FragmentCharge_max + 1):
            i += 1
            names.append("charge %d" % i)
        for label, tp in sorted(FragmentSeriesClassification, key=lambda x: x[1].value):
            for i in range(FragmentCharge_max + 1):
                if tp.value is None or label == 'unassigned':
                    continue
                names.append("series %s:charge %d" % (label, i + 1))

        for i in range(BackboneFragment_max_glycosylation_size + 1):
            names.append("is_glycosylated %r" % (i))
        return names

    def __str__(self):
        return '(%s, %s, %s, %r, %r)' % (
            self[0].name if self[0] else '',
            self[1].name if self[1] else '',
            self[2].name, self[3], self[4])

    @classmethod
    def from_peak_peptide_fragment_pair(cls, peak_fragment_pair, structure):
        peak, fragment = peak_fragment_pair
        nterm, cterm = classify_amide_bond_frank(*fragment.flanking_amino_acids)
        glycosylation = fragment.is_glycosylated
        inst = cls(
            nterm, cterm, FragmentSeriesClassification[fragment.series],
            glycosylation, min(peak.charge, FragmentCharge_max + 1),
            peak_fragment_pair, structure)
        return inst

    @classmethod
    def build_fragment_intensity_matches(cls, gpsm):
        fragment_classification = []
        intensities = []
        matched_total = 0
        total = sum(p.intensity for p in gpsm.deconvoluted_peak_set)
        counted = set()
        if gpsm.solution_map is None:
            gpsm.match()
        for peak_fragment_pair in gpsm.solution_map:
            peak, fragment = peak_fragment_pair
            if peak not in counted:
                matched_total += peak.intensity
                counted.add(peak)
            if fragment.series == 'oxonium_ion':
                continue

            intensities.append(peak.intensity)
            if fragment.series == 'stub_glycopeptide':
                fragment_classification.append(
                    cls(
                        None, None, FragmentSeriesClassification.stub_glycopeptide,
                        min(fragment.glycosylation_size, StubFragment_max_glycosylation_size),
                        min(peak.charge, FragmentCharge_max + 1), peak_fragment_pair, gpsm.structure))
                continue
            inst = cls.from_peak_peptide_fragment_pair(peak_fragment_pair, gpsm.structure)
            fragment_classification.append(inst)

        unassigned = total - matched_total
        ft = cls(None, None, FragmentSeriesClassification.unassigned, 0, 0, None, None)
        fragment_classification.append(ft)
        intensities.append(unassigned)
        return fragment_classification, np.array(intensities), total

    @classmethod
    def encode_classification(cls, classification):
        X = []
        for _, row in enumerate(classification):
            X.append(row.as_feature_vector())
        return np.vstack(X)

    @classmethod
    def fit_regression(cls, gpsms, reliability_model=None, base_reliability=0., **kwargs):
        breaks = []
        matched = []
        totals = []
        reliabilities = []
        for gpsm in gpsms:
            c, y, t = cls.build_fragment_intensity_matches(gpsm)
            x = cls.encode_classification(c)
            breaks.append(x)
            matched.append(y)
            totals.append(t)
            if reliability_model is None:
                reliability = np.ones_like(y)
            else:
                reliability = np.zeros_like(y)
                remaining_reliability = 1 - base_reliability
                reliability_score_map = reliability_model.score(gpsm, gpsm.solution_map, gpsm.structure)
                for i, frag_spec in enumerate(c):
                    peak_pair = frag_spec.peak_pair
                    if peak_pair is None:
                        reliability[i] = 1.0
                    else:
                        reliability[i] = (remaining_reliability * reliability_score_map[peak_pair]
                                          ) + base_reliability

            reliabilities.append(reliability)
        try:
            fit = multinomial_fit(breaks, matched, totals, reliabilities=reliabilities, **kwargs)
        except np.linalg.LinAlgError:
            fit = multinomial_fit(breaks, matched, totals, reliabilities=None, **kwargs)
        return MultinomialRegressionFit(model_type=cls, reliability_model=reliability_model, **fit)

    @classmethod
    def generate_all_products(cls, solution_map, structure, charge_max=FragmentCharge_max):
        elements = []
        for frags in structure.get_fragments(IonSeries.b):
            for frag in frags:
                peaks = {p.charge: p for p in solution_map.peaks_for(frag)}
                for charge in range(1, charge_max + 2):
                    try:
                        peak = peaks[charge]
                    except KeyError:
                        peak = DeconvolutedPeak(frag.mass, 1, charge, 1, -1, 0, 0)
                    pfp = PeakFragmentPair(peak, frag)
                    elements.append(cls.from_peak_peptide_fragment_pair(pfp, structure))
        for frags in structure.get_fragments(IonSeries.y):
            for frag in frags:
                peaks = {p.charge: p for p in solution_map.peaks_for(frag)}
                for charge in range(1, charge_max + 2):
                    try:
                        peak = peaks[charge]
                    except KeyError:
                        peak = DeconvolutedPeak(frag.mass, 1, charge, 1, -1, 0, 0)
                    pfp = PeakFragmentPair(peak, frag)
                    elements.append(cls.from_peak_peptide_fragment_pair(pfp, structure))

        # stubs may be duplicated
        seen = set()
        for frag in structure.stub_fragments(True):
            if frag.name in seen:
                continue
            seen.add(frag.name)
            peaks = {p.charge: p for p in solution_map.peaks_for(frag)}
            for charge in range(1, charge_max + 2):
                try:
                    peak = peaks[charge]
                except KeyError:
                    peak = DeconvolutedPeak(frag.mass, 1, charge, 1, -1, 0, 0)
                pfp = PeakFragmentPair(peak, frag)
                inst = cls(
                    None, None, FragmentSeriesClassification.stub_glycopeptide,
                    min(frag.glycosylation_size, StubFragment_max_glycosylation_size),
                    min(peak.charge, FragmentCharge_max + 1), pfp, structure)
                assert inst not in elements
                elements.append(inst)
        elements.append(cls(None, None, FragmentSeriesClassification.unassigned, 0, 0, None, None))
        return elements


try:
    from feature_learning._c.model_types import (
        encode_classification,
        build_fragment_intensity_matches,
        from_peak_peptide_fragment_pair
    )
    FragmentType.encode_classification = classmethod(encode_classification)
    FragmentType.from_peak_peptide_fragment_pair = classmethod(
        from_peak_peptide_fragment_pair)
    FragmentType.build_fragment_intensity_matches = classmethod(
        build_fragment_intensity_matches)
except ImportError:
    pass


class ProlineSpecializingModel(FragmentType):
    def specialize_proline(self, X, offset):
        k_charge_cterm_pro = (FragmentCharge_max + 1)
        k_series_cterm_pro = (BackboneFragmentSeriesClassification_max + 1)
        k_glycosylated_proline = BackboneFragment_max_glycosylation_size + 1

        if self.cterm == FragmentTypeClassification.pro:
            index = (self.charge - 1)
            X[offset + index] = 1
        offset += k_charge_cterm_pro

        if self.cterm == FragmentTypeClassification.pro:
            X[offset + self.series.value] = 1
        offset += k_series_cterm_pro

        if self.cterm == FragmentTypeClassification.pro:
            X[offset + int(self.peak_pair.fragment.glycosylation_size)] = 1
        offset += k_glycosylated_proline
        return X, offset

    def build_feature_vector(self, X, offset):
        X, offset = super(ProlineSpecializingModel, self).build_feature_vector(X, offset)
        X, offset = self.specialize_proline(X, offset)
        return X, offset

    @classmethod
    def feature_names(cls):
        names = super(ProlineSpecializingModel, cls).feature_names()
        for i in range(FragmentCharge_max + 1):
            names.append("charge %d:c-term pro" % (i + 1))
        for label, tp in sorted(FragmentSeriesClassification, key=lambda x: x[1].value):
            if tp.value is None or label in ("unassigned", "stub_glycopeptide"):
                continue
            names.append("series:c-term pro %s" % label)
        for i in range(BackboneFragment_max_glycosylation_size + 1):
            names.append("is_glycosylated:c-term pro %r" % (i))
        return names


try:
    from feature_learning._c.model_types import specialize_proline
    ProlineSpecializingModel.specialize_proline = specialize_proline
except ImportError as err:
    print(err)


class StubGlycopeptideCompositionModel(ProlineSpecializingModel):

    def encode_stub_information(self, X, offset):
        k_glycosylated_stubs = StubFragment_max_glycosylation_size + 1
        k_sequence_composition_stubs = FragmentTypeClassification_max + 1

        if self.is_stub_glycopeptide():
            X[offset + int(self.glycosylated)] = 1
        offset += k_glycosylated_stubs
        if self.is_stub_glycopeptide():
            ctr = classify_sequence_by_residues(self.sequence)
            for tp, c in ctr:
                X[offset + tp.value] = c
        offset += k_sequence_composition_stubs
        return X, offset

    def build_feature_vector(self, X, offset):
        X, offset = super(StubGlycopeptideCompositionModel, self).build_feature_vector(X, offset)
        X, offset = self.encode_stub_information(X, offset)
        return X, offset

    @classmethod
    def feature_names(self):
        names = super(StubGlycopeptideCompositionModel, self).feature_names()
        for i in range(StubFragment_max_glycosylation_size + 1):
            names.append("stub glycopeptide:is_glycosylated %r" % (i))
        for label, tp in sorted(FragmentTypeClassification, key=lambda x: x[1].value):
            if tp.value is None:
                continue
            names.append("stub glycopeptide:composition %s" % (label,))
        return names


try:
    from feature_learning._c.model_types import encode_stub_information
    StubGlycopeptideCompositionModel.encode_stub_information = encode_stub_information
except ImportError as err:
    print(err)


_FUC = FrozenMonosaccharideResidue.from_iupac_lite("Fuc")


class StubGlycopeptideFucosylationModel(StubGlycopeptideCompositionModel):
    def encode_stub_fucosylation(self, X, offset):
        k = 2
        if self.is_stub_glycopeptide():
            i = int(self.peak_pair.fragment.glycosylation._getitem_fast(_FUC) > 0)
            X[offset + i] = 1
        offset += k
        return X, offset

    def build_feature_vector(self, X, offset):
        X, offset = super(StubGlycopeptideFucosylationModel, self).build_feature_vector(X, offset)
        X, offset = self.encode_stub_fucosylation(X, offset)
        return X, offset

    @classmethod
    def feature_names(self):
        names = super(StubGlycopeptideFucosylationModel, self).feature_names()
        for i in range(2):
            names.append("stub glycopeptide:is_fucosylated %r" % (i))
        return names


try:
    from feature_learning._c.model_types import encode_stub_fucosylation
    StubGlycopeptideFucosylationModel.encode_stub_fucosylation = encode_stub_fucosylation
except ImportError as err:
    print(err)


class NeighboringAminoAcidsModel(StubGlycopeptideFucosylationModel):
    bond_offset_depth = 1

    def get_nterm_neighbor(self, offset=1):
        index = get_nterm_index_from_fragment(self.fragment, self.sequence)
        index -= offset
        if index < 0:
            return None
        else:
            residue = self.sequence[index][0]
            return classify_residue_frank(residue)

    def get_cterm_neighbor(self, offset=1):
        index = get_cterm_index_from_fragment(self.fragment, self.sequence)
        index += offset
        if index > len(self.sequence) - 1:
            return None
        else:
            residue = self.sequence[index][0]
            return classify_residue_frank(residue)

    def encode_neighboring_residues(self, X, offset):
        k_ftypes = (FragmentTypeClassification_max + 1)

        if self.is_backbone():
            for i in range(1, self.bond_offset_depth + 1):
                nterm = self.get_nterm_neighbor(i)
                if nterm is not None:
                    X[offset + nterm.value] = 1
                offset += k_ftypes
            for i in range(1, self.bond_offset_depth + 1):
                cterm = self.get_cterm_neighbor(i)
                if cterm is not None:
                    X[offset + cterm.value] = 1
                offset += k_ftypes
        return X, offset

    def build_feature_vector(self, X, offset):
        X, offset = super(NeighboringAminoAcidsModel, self).build_feature_vector(X, offset)
        X, offset = self.encode_neighboring_residues(X, offset)
        return X, offset

    @classmethod
    def feature_names(cls):
        names = super(NeighboringAminoAcidsModel, cls).feature_names()
        for i in range(1, cls.bond_offset_depth + 1):
            for label, tp in sorted(FragmentTypeClassification, key=lambda x: x[1].value):
                if tp.value is None:
                    continue
                names.append("n-term - %d %s" % (i, label))
        for i in range(1, cls.bond_offset_depth + 1):
            for label, tp in sorted(FragmentTypeClassification, key=lambda x: x[1].value):
                if tp.value is None:
                    continue
                names.append("c-term + %d %s" % (i, label))
        return names


try:
    from feature_learning._c.model_types import (
        encode_neighboring_residues, get_nterm_neighbor, get_cterm_neighbor,
        get_nterm_index_from_fragment, get_cterm_index_from_fragment
    )
    NeighboringAminoAcidsModel.encode_neighboring_residues = encode_neighboring_residues
    NeighboringAminoAcidsModel.get_nterm_neighbor = get_nterm_neighbor
    NeighboringAminoAcidsModel.get_cterm_neighbor = get_cterm_neighbor
except ImportError as err:
    print(err)


class NeighboringAminoAcidsModelDepth2(NeighboringAminoAcidsModel):
    bond_offset_depth = 2


class NeighboringAminoAcidsModelDepth3(NeighboringAminoAcidsModel):
    bond_offset_depth = 3


class NeighboringAminoAcidsModelDepth4(NeighboringAminoAcidsModel):
    bond_offset_depth = 4


class CleavageSiteCenterDistanceModel(NeighboringAminoAcidsModelDepth2):
    max_cleavage_site_distance_from_center = 10

    def get_cleavage_site_distance_from_center(self):
        index = get_cterm_index_from_fragment(self.fragment, self.sequence)
        seq_size = len(self.sequence)
        center = (seq_size / 2)
        return abs(center - index)

    def encode_cleavage_site_distance_from_center(self, X, offset):
        k_distance = self.max_cleavage_site_distance_from_center + 1
        k_series = BackboneFragmentSeriesClassification_max + 1

        if self.is_backbone():
            distance = self.get_cleavage_site_distance_from_center()
            distance = min(distance, self.max_cleavage_site_distance_from_center)
            series_offset = self.series.value * k_distance
            X[offset + series_offset + distance] = 1
        offset += (k_distance * k_series)
        return X, offset

    def build_feature_vector(self, X, offset):
        X, offset = super(CleavageSiteCenterDistanceModel, self).build_feature_vector(X, offset)
        X, offset = self.encode_cleavage_site_distance_from_center(X, offset)
        return X, offset

    @classmethod
    def feature_names(cls):
        names = super(CleavageSiteCenterDistanceModel, cls).feature_names()
        for label, tp in sorted(FragmentSeriesClassification, key=lambda x: x[1].value):
            if tp.value is None or label in ("unassigned", "stub_glycopeptide"):
                continue
            for i in range(cls.max_cleavage_site_distance_from_center + 1):
                names.append("cleavage site distance %d:series %r" % (i, label))
        return names


try:
    from feature_learning._c.model_types import (
        encode_cleavage_site_distance_from_center, get_cleavage_site_distance_from_center)
    CleavageSiteCenterDistanceModel.encode_cleavage_site_distance_from_center =\
        encode_cleavage_site_distance_from_center
    CleavageSiteCenterDistanceModel.get_cleavage_site_distance_from_center =\
        get_cleavage_site_distance_from_center
except ImportError as err:
    print(err)


class StubChargeModel(CleavageSiteCenterDistanceModel):

    def encode_stub_charge(self, X, offset):
        k_glycosylated_stubs = (StubFragment_max_glycosylation_size * 2) + 1
        k_stub_charges = FragmentCharge_max + 1
        k_glycosylated_stubs_x_charge = (k_glycosylated_stubs * k_stub_charges)

        if self.is_stub_glycopeptide():
            loss_size = sum(self.sequence.glycan_composition.values()) - int(self.glycosylated)
            if loss_size >= k_glycosylated_stubs:
                loss_size = k_glycosylated_stubs - 1
            # d = k_glycosylated_stubs * (self.charge - 1) + int(self.glycosylated)
            d = k_glycosylated_stubs * (self.charge - 1) + loss_size
            X[offset + d] = 1
        offset += k_glycosylated_stubs_x_charge
        return X, offset

    @classmethod
    def feature_names(self):
        names = super(StubChargeModel, self).feature_names()
        k_glycosylated_stubs = (StubFragment_max_glycosylation_size * 2) + 1
        k_stub_charges = FragmentCharge_max + 1
        for i in range(k_stub_charges):
            for j in range(k_glycosylated_stubs):
                names.append("stub glycopeptide:charge %d:glycan loss %d" % (i + 1, j))
        return names

    def build_feature_vector(self, X, offset):
        # X, offset = super(StubChargeModel, self).build_feature_vector(X, offset)
        # X, offset = self.encode_stub_charge(X, offset)

        # Directly invoke feature vector construction because super() costs too much
        # in a tight loop
        X, offset = FragmentType.build_feature_vector(self, X, offset)
        X, offset = ProlineSpecializingModel.specialize_proline(self, X, offset)
        X, offset = StubGlycopeptideCompositionModel.encode_stub_information(self, X, offset)
        X, offset = StubGlycopeptideFucosylationModel.encode_stub_fucosylation(self, X, offset)
        X, offset = NeighboringAminoAcidsModelDepth2.encode_neighboring_residues(self, X, offset)
        X, offset = CleavageSiteCenterDistanceModel.encode_cleavage_site_distance_from_center(self, X, offset)
        X, offset = StubChargeModel.encode_stub_charge(self, X, offset)
        return X, offset


try:
    from feature_learning._c.model_types import (
        encode_stub_charge,
        StubChargeModel_build_feature_vector)
    StubChargeModel.encode_stub_charge = encode_stub_charge
    StubChargeModel.build_feature_vector = StubChargeModel_build_feature_vector
except ImportError as err:
    print(err)


class AmideBondCrossproductModel(StubGlycopeptideCompositionModel):
    def specialize_fragmentation_site(self):
        k_series_ftypes = ((FragmentTypeClassification_max + 1) ** 2) * (
            BackboneFragmentSeriesClassification_max + 1)
        k = k_series_ftypes

        X = np.zeros(k, dtype=np.uint8)
        offset = 0

        if self.is_backbone():
            index_series = self.series.value
            index_nterm = self.nterm.value
            index_cterm = self.cterm.value
            # the cterm-th slot in the nterm-th section
            index_ftypes = index_cterm + ((FragmentTypeClassification_max + 1) * index_nterm)
            index = index_ftypes + ((FragmentTypeClassification_max + 1) ** 2) * index_series
            X[offset + index] = 1

        offset += k_series_ftypes

        return X

    def as_feature_vector(self):
        X = super(AmideBondCrossproductModel, self).as_feature_vector()
        return np.concatenate((X, self.specialize_fragmentation_site()))

    @classmethod
    def feature_names(self):
        names = super(AmideBondCrossproductModel, self).feature_names()
        for i in range(BackboneFragmentSeriesClassification_max + 1):
            for j in range(FragmentTypeClassification_max + 1):
                for k in range(FragmentTypeClassification_max + 1):
                    series = FragmentSeriesClassification[i].name
                    n_term = FragmentTypeClassification[j].name
                    c_term = FragmentTypeClassification[k].name
                    names.append("series %s:n-term %s:c-term %s" % (series, n_term, c_term))
        return names


def classify_sequence_by_residues(sequence):
    ctr = defaultdict(int)
    for res, _ in sequence:
        ctr[classify_residue_frank(res)] += 1
    return sorted(ctr.items())


try:
    from feature_learning._c.model_types import classify_sequence_by_residues
except ImportError as err:
    print(err)

build_fragment_intensity_matches = FragmentType.build_fragment_intensity_matches


encode_classification = FragmentType.encode_classification


def logit(x):
    return np.log(x / (1 - x))


def invlogit(x):
    return 1 / (1 + np.exp(-x))


def multinomial_control(epsilon=1e-8, maxit=25, nsamples=1000, trace=False):
    return dict(epsilon=epsilon, maxit=maxit, nsamples=nsamples, trace=trace)


def deviance(y, mu, wt, reliabilities, unobserved_reliability=1.0):
    # wt would be the total signal * reliabilities
    ys = np.array(list(map(np.sum, y)))
    ms = np.array(list(map(np.sum, mu)))
    # this is the leftover count after accounting for signal matched by
    # the fragment. Requires a reliability measure as well, use 0 until we have a better idea?
    # when we use reliability, there is an extra wt (not the weight vector) that would be used
    # here to reflect the weight leftover from the unmatched signal (total signal - matched signal)
    # which used to be based upon the total signal in the spectrum.
    dc = np.where(wt == ys, 0, wt * (1 - ys) * np.log((1 - ys) / (1 - ms + 1e-6)))
    return np.sum([
        # this inner sum is the squared residuals
        a.sum() + (2 * dc[i]) for i, a in enumerate(deviance_residuals(y, mu, wt, reliabilities))])


def deviance_residuals(y, mu, wt, reliability=None):
    # returns the squared residual. The sign is lost? The sign can be regained
    # from (yi - mu[i])
    residuals = []
    if reliability is None:
        reliability = np.ones_like(y)
    for i, yi in enumerate(y):
        # this is similar to the unit deviance of the Poisson distribution
        # "sub-residual", contributing to the total being the actual residual,
        # but these are not residuals themselves, the sum is, and that must be positive
        ym = np.where(yi == 0, 0, yi * np.log(yi / mu[i]) * reliability[i])
        residuals.append(2 * wt[i] * ym)
    return residuals


def cho2inv(C, lower=False):
    """Compute the inverse of a matrix from its Cholesky decomposition

    Parameters
    ----------
    C : np.ndarray
        The Cholesky decomposition of a matrix
    lower : bool, optional
        Whether this is an upper or lower (default) Cholesky decomposition

    Returns
    -------
    nd.array
        The inverse of the matrix whose Cholesky decomposition was ``C``
    """
    return cho_solve((C, lower), np.identity(C.shape[0]))


def multinomial_fit(x, y, weights, reliabilities=None, dispersion=1, adjust_dispersion=True, prior_coef=None, prior_disp=None, **control):
    """Fit a multinomial generalized linear model to bond-type by intensity observations
    of glycopeptide fragmentation.

    Parameters
    ----------
    x : list
        list of fragment type matrices
    y : list
        list of observed intensities
    weights : list
        list of total intensities
    dispersion : int, optional
        The dispersion of the model
    adjust_dispersion : bool, optional
        Whether or not to adjust the dispersion according to
        the prior dispersion parameters
    prior_coef : dict, optional
        Description
    prior_disp : dict, optional
        Description
    **control
        Description

    Returns
    -------
    dict
    """
    control = multinomial_control(**control)

    # make a copy of y and convert each observation
    # into an ndarray.
    y = list(map(np.array, y))
    if reliabilities is None:
        reliabilities = list(map(np.ones_like, y))
    n = weights
    lengths = np.array(list(map(len, y)))
    nvars = x[0].shape[1]
    if prior_coef is None:
        prior_coef = {"mean": np.zeros(nvars), "precision": np.zeros(nvars)}
    prior_coef = {k: np.array(v) for k, v in prior_coef.items()}
    if prior_disp is None:
        prior_disp = {"df": 0, "scale": 0}
    S_inv0 = prior_coef['precision']
    beta0 = prior_coef['mean']
    beta0 = S_inv0 * beta0
    nu = prior_disp['df']
    nu_tau2 = nu * prior_disp['scale']

    beta = None

    phi = dispersion
    mu = [0 for _ in y]
    eta = [0 for _ in y]

    # intialize parameters
    for i in range(len(y)):
        # ensure no value is 0 by adding 0.5
        mu[i] = (y[i] + 0.5) / (1 + n[i] + 0.5 * lengths[i])
        if np.sum(mu[i]) > 1.0:
            raise ValueError('mu sums to greater than 1')
        # put on the scale of mu
        y[i] = y[i] / n[i]
        # link function
        eta[i] = np.log(mu[i]) + np.log(1 + np.exp(logit(np.sum(mu[i]))))
        assert not np.any(np.isnan(eta[i]))

    dev = deviance(y, mu, n, reliabilities)
    iter_ = 0
    for iter_ in range(control['maxit']):
        if control['trace']:
            print("Iteration %d" % (iter_,))
        z = phi * beta0
        H = phi * np.diag(S_inv0)
        if control['trace']:
            assert not np.any(np.isnan(H))
        for i in range(len(y)):
            reliability = reliabilities[i]
            # Variance of Y_i, multinomial, e.g. covariance matrix
            # Here an additional dimension is introduced to coerce mu[i] into
            # a matrix to match the behavior of tcrossprod
            # pre-multiply with diagonal matrix of the reliabilities (instead of identity)?
            W = reliability * np.diag(mu[i]) - mu[i][:, None].dot(mu[i][:, None].T)
            # Since both mu and y are on the same scale it is convenient to multiply both by the total
            # Here, x[i] is transposed to match the behavior of crossprod
            # Working Residual analog
            z += n[i] * x[i].T.dot(y[i] - mu[i] + W.dot(eta[i]))
            # Sum of covariances, close to the Hessian (log-likelihood)
            H += n[i] * x[i].T.dot(W.dot(x[i]))
            if control['trace']:
                assert not np.any(np.isnan(H))

        H += np.identity(H.shape[0])
        # H = CtC, H_inv = C_inv * Ct_inv
        # get the upper triangular Cholesky decomposition, s.t. C.T.dot(C) == H
        # np.linalg.cholesky returns lower by default, so it must be transposed
        C = np.linalg.cholesky(H).T
        # Solve for updated coefficients. Use back substitution algorithm.
        beta = solve_triangular(C, solve_triangular(C, z, trans="T"))
        # if control['trace']:
        #     assert np.all(np.abs(beta) < 1e3)

        for i in range(len(y)):
            # linear predictor
            eta[i] = x[i].dot(beta)
            # canonical poisson inverse link
            if control['trace']:
                mu_i_temp = np.exp(eta[i])
                assert not np.any(np.isinf(mu_i_temp))
            mu[i] = np.exp(eta[i])
            # Apply a normalizing constraint for multinomial
            # inverse link to expected value from linear predictor
            mu[i] = mu[i] / (1 + mu[i].sum())

        dev_new = deviance(y, mu, n, reliabilities)
        if adjust_dispersion:
            phi = (nu_tau2 + dev_new) / (nu + np.sum(lengths))
        if control['trace']:
            print("[%d] deviance = %f" % (iter_, dev_new))
        rel_error = np.abs((dev_new - dev) / dev)
        # converged?
        if (not np.isinf(dev)) and (rel_error < control["epsilon"]):
            break
        if np.isinf(dev_new):
            print("Infinite Deviance")
            break
        dev = dev_new
    return dict(
        coef=beta, scaled_y=y, mu=mu, reliabilities=reliabilities,
        dispersion=dispersion, weights=np.array(n),
        covariance_unscaled=cho2inv(C, False), iterations=iter_, deviance=dev,
        H=H, C=C)


class MultinomialRegressionFit(object):
    def __init__(self, coef, scaled_y, mu, reliabilities, dispersion, weights, covariance_unscaled,
                 deviance, H, model_type=FragmentType, reliability_model=None, **info):
        self.coef = coef
        self.scaled_y = scaled_y
        self.mu = mu
        self.weights = weights
        self.reliabilities = reliabilities
        self.dispersion = dispersion
        self.covariance_unscaled = covariance_unscaled
        self.deviance = deviance
        self.H = H
        self.info = info
        self.model_type = model_type
        self.reliability_model = reliability_model

    def copy(self):
        return self.__class__(
            self.coef, self.scaled_y, self.mu, self.reliabilities, self.dispersion,
            self.weights, self.covariance_unscaled, self.deviance, self.H, self.model_type,
            self.reliability_model, **self.info)

    @property
    def hessian(self):
        return self.H

    def estimate_dispersion(self):
        return deviance(self.scaled_y, self.mu, self.weights, self.reliabilities
                        ) / (len(self.scaled_y) - len(self.coef))

    def predict(self, x):
        yhat = np.exp(x.dot(self.coef))
        yhat /= (1 + yhat.sum())
        return yhat

    def parameter_intervals(self):
        return np.sqrt(np.diag(self.covariance_unscaled)) * np.sqrt((self.estimate_dispersion()))

    def residuals(self, gpsms, normalized=False):
        y = []
        total = []
        mu = []
        reliabilities = []
        for gpsm in gpsms:
            c, intens, t = self.model_type.build_fragment_intensity_matches(gpsm)
            X = self.model_type.encode_classification(c)
            yhat = self.predict(X)
            y.append(intens / t)
            mu.append(yhat)
            total.append(t)
            if self.reliability_model is None:
                reliabilities.append(np.ones_like(yhat))
            else:
                reliability = np.zeros_like(yhat)
                reliability_score_map = self.reliability_model.score(gpsm, gpsm.solution_map, gpsm.structure)
                for i, frag_spec in enumerate(c):
                    peak_pair = frag_spec.peak_pair
                    if peak_pair is None:
                        reliability[i] = 1.0
                    else:
                        reliability[i] = reliability_score_map[peak_pair]
                reliabilities.append(reliability)
        ys = np.array(list(map(np.sum, y)))
        ms = np.array(list(map(np.sum, mu)))
        sign = (ys - ms) / np.abs(ys - ms)
        wt = np.array(total)
        # this is the leftover count after accounting for signal matched by
        # the fragment
        dc = np.where(wt == ys, 0, wt * (1 - ys) * np.log((1 - ys) / (1 - ms + 1e-6)))
        as_ = deviance_residuals(y, mu, wt, reliabilities)
        return sign * np.sqrt(np.array([
            # this inner sum is the squared residuals
            a.sum() + (2 * dc[i]) for i, a in enumerate(as_)]))

    def deviance_residuals(self, gpsm):
        c, intens, t = self.model_type.build_fragment_intensity_matches(gpsm)
        X = self.model_type.encode_classification(c)
        yhat = self.predict(X)
        if self.reliability_model is None:
            reliability = np.ones_like(yhat)
        else:
            reliability = np.zeros_like(yhat)
            reliability_score_map = self.reliability_model.score(gpsm, gpsm.solution_map, gpsm.structure)
            for i, frag_spec in enumerate(c):
                peak_pair = frag_spec.peak_pair
                if peak_pair is None:
                    reliability[i] = 1.0
                else:
                    reliability[i] = reliability_score_map[peak_pair]
        return deviance_residuals([intens / t], [yhat], [t], [reliability])[0]

    def _calculate_reliability(self, gpsm, fragment_specs, base_reliability=0.0):
        n = len(fragment_specs)
        if self.reliability_model is None:
            reliability = np.ones(n)
        else:
            reliability = np.zeros(n)
            remaining_reliability = 1 - base_reliability
            reliability_score_map = self.reliability_model.score(gpsm, gpsm.solution_map, gpsm.structure)
            for i, frag_spec in enumerate(fragment_specs):
                try:
                    peak_pair = frag_spec.peak_pair
                except AttributeError:
                    peak_pair = frag_spec
                if peak_pair is None:
                    reliability[i] = 1.0
                else:
                    reliability[i] = (remaining_reliability * reliability_score_map[peak_pair]
                                      ) + base_reliability

        return reliability

    def test_goodness_of_fit(self, gpsm):
        c, intens, t = self.model_type.build_fragment_intensity_matches(gpsm)
        X = self.model_type.encode_classification(c)
        yhat = self.predict(X)

        # standardize intensity
        yhat *= 100
        intens = intens / t * 100

        dc = (100 - intens.sum()) * np.log((100 - intens.sum()) / (100 - yhat.sum() + 1e-6))

        intens = (intens)
        theor = (yhat)

        ratio = np.log(intens / theor)
        G = intens.dot(ratio) + dc
        return G

    def _calculate_pearson_residuals(self, gpsm, use_reliability=True, base_reliability=0.0):
        c, intens, t, yhat = self._get_predicted_intensities(gpsm)

        if self.reliability_model is None or not use_reliability:
            reliability = np.ones_like(yhat)
        else:
            reliability = self._calculate_reliability(
                gpsm, c, base_reliability=base_reliability)

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
        return delta / denom

    def pearson_residual_score(self, gpsm, use_reliability=True, base_reliability=0.0):
        residuals = self._calculate_pearson_residuals(gpsm, use_reliability, base_reliability)
        return residuals.sum()

    def calculate_correlation(self, gpsm, use_reliability=True, base_reliability=0.0):
        c, intens, t, yhat = self._get_predicted_intensities(gpsm)
        p = (intens / t)[:-1]
        yhat = yhat[:-1]
        if use_reliability and self.reliability_model is not None:
            reliability = self._calculate_reliability(
                gpsm, c, base_reliability=base_reliability)[:-1]
            p = t * p / np.sqrt(t * reliability * p * (1 - p))
            yhat = t * yhat / np.sqrt(t * reliability * yhat * (1 - yhat))
        return np.corrcoef(p, yhat)[0, 1]

    def _get_predicted_intensities(self, gpsm, all_fragments=False):
        c, intens, t = self.model_type.build_fragment_intensity_matches(gpsm)
        X = self.model_type.encode_classification(c)
        yhat_unnormed = np.exp(X.dot(self.coef))
        if not all_fragments:
            # this branch is identical to :meth:`predict`
            # predict normalizes by the sum, which can make removing a low quality peak
            # increasing the score by virtue of making the remaining yhat values larger
            yhat = yhat_unnormed / (1 + yhat_unnormed.sum())
        else:
            # this branch attempts to correct for inconsistencies caused
            # by predict when a peak match is omitted, increasing the value
            # of the peaks matched by reducing the size of the sum in the normalizing
            # expression (1 + yhat.sum())
            c_all = self.model_type.generate_all_products(gpsm.solution_map, gpsm.structure)
            X_all = self.model_type.encode_classification(c_all)
            yhat_all = np.exp(X_all.dot(self.coef))
            #
            yhat = yhat_unnormed / (1 + yhat_all.sum())
        return c, intens, t, yhat

    def compound_score(self, gpsm, use_reliability=True, base_reliability=0.0, pearson_bias=PEARSON_BIAS):
        c, intens, t, yhat = self._get_predicted_intensities(gpsm)
        W = np.ones_like(intens)
        # for i, ci in enumerate(c):
        #     W[i] = series_weights[ci.series]
        if self.reliability_model is None or not use_reliability:
            reliability = np.ones_like(yhat)
        else:
            reliability = self._calculate_reliability(
                gpsm, c, base_reliability=base_reliability)
        # standardize intensity
        intens = intens / t
        # remove the unassigned signal term
        W = W[:-1]
        intens = intens[:-1]
        yhat = yhat[:-1]

        delta = (intens - yhat) ** 2
        mask = intens > yhat
        # reduce penalty for exceeding predicted intensity
        delta[mask] = delta[mask] / 2.
        denom = yhat * (1 - yhat)
        denom *= (reliability[:-1])  # divide by the reliability
        pearson_residual_score = delta * (1.0 / denom)
        lg_inverted_pearson_residual_score = -np.log10(
            PearsonResidualCDF(pearson_residual_score / pearson_bias) + 1e-6)

        lg_inverted_pearson_residual_score[lg_inverted_pearson_residual_score < 0] = 1e-6
        if np.any(lg_inverted_pearson_residual_score < 0):
            raise ValueError("lg_inverted_pearson_residual_score is negative")
        signal_utilization = intens * 100
        return (W * signal_utilization * lg_inverted_pearson_residual_score).sum()

    def describe(self):
        table = []
        header = ['Name', 'Value', 'SE', "Z", "p-value"]
        # compute Z-statistic (parameter value / std err) Wald test (approx) p-value = 2 * pnorm(-abs(Z))
        table.append(header)
        for name, value, se in zip(self.model_type.feature_names(), self.coef, self.parameter_intervals()):
            z = value / se
            p = 2 * norm.cdf(-abs(z))
            table.append((name, str(value), str(se), str(z), str(p)))
        columns = zip(*table)
        column_widths = [max(map(len, col)) + 2 for col in columns]
        output_buffer = six.StringIO()
        for row in table:
            for i, col in enumerate(row):
                width = column_widths[i]
                if i != 0:
                    col = col.center(width)
                    col = "|%s" % col
                else:
                    col = col.ljust(width)
                output_buffer.write(col)
            output_buffer.write("\n")
        return output_buffer.getvalue()

    def to_json(self, include_fit_source=True):
        d = {}
        d['coef'] = save_array(self.coef)
        d['covariance_unscaled'] = save_array(self.covariance_unscaled)
        d['hessian'] = save_array(self.H)
        d['mu'] = save_array_list(self.mu) if include_fit_source else []
        d['scaled_y'] = save_array_list(self.scaled_y) if include_fit_source else []
        d['weights'] = save_array(self.weights) if include_fit_source else ""
        d['reliabilities'] = save_array_list(self.reliabilities) if include_fit_source else []
        d['deviance'] = self.deviance
        d['reliability_model'] = self.reliability_model.to_json() if self.reliability_model else None
        d['dispersion'] = self.dispersion
        d['model_type'] = self.model_type.__name__
        return d

    @classmethod
    def from_json(cls, d):
        fit_coef = load_array(d['coef'])
        fit_covariance_unscaled = load_array(d['covariance_unscaled'])
        fit_H = load_array(d['hessian'])
        fit_mu = load_array_list(d['mu'])
        fit_scaled_y = load_array_list(d['scaled_y'])
        fit_weights = load_array(d['weights'])
        fit_reliabilities = load_array_list(d['reliabilities'])
        fit_deviance = d['deviance']
        fit_dispersion = d['dispersion']
        fit_reliability_model = d['reliability_model']
        if fit_reliability_model is not None:
            fit_reliability_model = FragmentationModelCollection.from_json(fit_reliability_model)
        fit_model_type_name = FragmentType.get_model_by_name(d['model_type'])
        fit = cls(
            coef=fit_coef,
            scaled_y=fit_scaled_y,
            mu=fit_mu,
            reliabilities=fit_reliabilities,
            dispersion=fit_dispersion,
            weights=fit_weights,
            covariance_unscaled=fit_covariance_unscaled,
            deviance=fit_deviance,
            H=fit_H,
            model_type=fit_model_type_name,
            reliability_model=fit_reliability_model)
        return fit

    def __eq__(self, other):
        if not np.allclose(self.coef, other.coef):
            return False
        if self.reliability_model != other.reliability_model:
            return False
        if self.model_type != other.model_type:
            return False
        return True

    def __ne__(self, other):
        return not (self == other)

    def __repr__(self):
        template = '{self.__class__.__name__}(<{self.model_type} with {self.coef.size} parameters>)'
        return template.format(self=self)


def save_array(a):
    buf = io.BytesIO()
    np.save(buf, a, False)
    return base64.standard_b64encode(buf.getvalue())


def save_array_list(a_list):
    return [save_array(a) for a in a_list]


def load_array(bytestring):
    if not bytestring:
        return None
    try:
        decoded_string = bytestring.encode("ascii")
    except AttributeError:
        decoded_string = bytestring
    decoded_string = base64.decodestring(decoded_string)
    buf = io.BytesIO(decoded_string)
    array = np.load(buf)
    return array


def load_array_list(bytestring_list):
    return [load_array(a) for a in bytestring_list]


def least_squares_scale_coefficient(x, y):
    return x.dot(y) / x.dot(x)
