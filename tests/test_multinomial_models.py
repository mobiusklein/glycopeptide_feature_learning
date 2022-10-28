from unittest import TestCase

import numpy as np

from .common import datafile

from glycopeptide_feature_learning.multinomial_regression import (
    FragmentType, ProlineSpecializingModel, StubGlycopeptideCompositionModel,
    StubGlycopeptideFucosylationModel, NeighboringAminoAcidsModel, NeighboringAminoAcidsModelDepth2,
    CleavageSiteCenterDistanceModel, StubChargeModel, LabileMonosaccharideAwareModel,
    LabileMonosaccharideAwareModelApproximate)

from glycopeptide_feature_learning.data_source import read


class FragmentTypeTest(TestCase):
    test_file = "test1.mgf"

    model_cls = FragmentType

    def _load_gpsm(self):
        mgf_reader = read(datafile(self.test_file))
        return mgf_reader[0].match()

    def test_encode(self):
        gpsm = self._load_gpsm()
        n_features = self.model_cls.feature_count
        model_insts, intensities, total = self.model_cls.build_fragment_intensity_matches(gpsm)
        for m in model_insts:
            X = m._allocate_feature_array()
            _, offset = m.build_feature_vector(X, 0)
            assert offset == n_features


class ProlineSpecializingModelTest(FragmentTypeTest):
    model_cls = ProlineSpecializingModel


class StubGlycopeptideCompositionModelTest(FragmentTypeTest):
    model_cls = StubGlycopeptideCompositionModel


class StubGlycopeptideFucosylationModelTest(FragmentTypeTest):
    model_cls = StubGlycopeptideFucosylationModel


class NeighboringAminoAcidsModelTest(FragmentTypeTest):
    model_cls = NeighboringAminoAcidsModel


class NeighboringAminoAcidsModelDepth2Test(FragmentTypeTest):
    model_cls = NeighboringAminoAcidsModelDepth2


class CleavageSiteCenterDistanceModelTest(FragmentTypeTest):
    model_cls = CleavageSiteCenterDistanceModel


class StubChargeModelTest(FragmentTypeTest):
    model_cls = StubChargeModel


class LabileMonosaccharideAwareModelTest(FragmentTypeTest):
    model_cls = LabileMonosaccharideAwareModel


class LabileMonosaccharideAwareModelApproximateTest(FragmentTypeTest):
    model_cls = LabileMonosaccharideAwareModelApproximate
