from collections import Counter

import glypy
from glycopeptidepy.structure.fragment import IonSeries
from glycopeptidepy.structure.sequence_composition import (
    AminoAcidSequenceBuildingBlock)

from . import peak_relations

amino_acid_blocks = [
    'Q(Gln->pyro-Glu)',
    'T(O-Glycosylation)',
    'S(GAG-Linker)',
    'C(Carbamidomethyl)',
    'M(Oxidation)',
    'A',
    'C',
    'E',
    'D',
    'G',
    'F',
    'I',
    'H',
    'K',
    'M',
    'L',
    'N',
    'Q',
    'P',
    'S',
    'R',
    'T',
    'W',
    'V',
    'Y',
    'S(O-Glycosylation)',
    'N(Deamidated)',
    'N(N-Glycosylation)'
]

amino_acid_blocks = list(map(AminoAcidSequenceBuildingBlock.from_str, amino_acid_blocks))


# TODO: Use this to extract theset of amino acid blocks to learn instead of the
# fixed common list.
def build_building_block_set(gpsms):
    linking_amino_acids = Counter()
    for gpsm in gpsms:
        for res, mods in gpsm.structure:
            linking_amino_acids[(AminoAcidSequenceBuildingBlock(res, mods))] += 1
    return linking_amino_acids


def get_peak_relation_features():
    features = {
        peak_relations.MassOffsetFeature(0.0, name="charge-diff"): lambda x: x.feature.from_charge
        != x.feature.to_charge,
        peak_relations.MassOffsetFeature(
            name="HexNAc", offset=glypy.monosaccharide_residues.HexNAc.mass()
        ): lambda x: True,
    }

    stub_features = {
        peak_relations.MassOffsetFeature(name="Hex", offset=glypy.monosaccharide_residues.Hex.mass()): lambda x: True,
        peak_relations.MassOffsetFeature(0.0, name="charge-diff"): lambda x: x.feature.from_charge
        != x.feature.to_charge,
        peak_relations.MassOffsetFeature(
            name="HexNAc", offset=glypy.monosaccharide_residues.HexNAc.mass()
        ): lambda x: True,
        peak_relations.MassOffsetFeature(name="Fuc", offset=glypy.monosaccharide_residues.Fuc.mass()): lambda x: True,
    }

    link_features = {}
    for link in amino_acid_blocks:
        feat = peak_relations.LinkFeature(link)
        link_features[feat] = lambda x: True

    link_features[peak_relations.ComplementFeature(0, name="Complement")] = lambda x: True
    link_features[
        peak_relations.ComplementFeature(glypy.monosaccharide_residues.HexNAc.mass(), name="Complement plus HexNAc")
    ] = lambda x: True
    link_features[
        peak_relations.ComplementFeature(-glypy.monosaccharide_residues.HexNAc.mass(), name="Complement minus HexNAc")
    ] = lambda x: True
    return features, stub_features, link_features


def fit_fragmentation_model(gpsms, common_features, backbone_features, stub_features):
    from .peak_relations import FragmentationModel, FragmentationModelCollection

    models = FragmentationModelCollection()

    for series in [IonSeries.b, IonSeries.y]:
        fm = FragmentationModel(series)
        fm.fit_offset(gpsms)
        for feature, filt in common_features.items():
            fits = fm.fit_feature(gpsms, feature)
            fm.features.extend(fits)
        for feature, filt in backbone_features.items():
            fits = fm.fit_feature(gpsms, feature)
            fm.features.extend(fits)
        models.add(fm)
    for series in [IonSeries.stub_glycopeptide]:
        fm = FragmentationModel(series)
        fm.fit_offset(gpsms)
        for feature, filt in stub_features.items():
            fits = fm.fit_feature(gpsms, feature)
            fm.features.extend(fits)
        models.add(fm)
    return models
