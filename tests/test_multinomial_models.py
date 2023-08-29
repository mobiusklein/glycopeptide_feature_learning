import pickle

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
            offset = m.build_feature_vector(X, 0)
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


class FullEncodingMixin(object):
    test_file = "MouseBrain-Z-T-5.mgf.gz"
    ref_file_pattern = "reps_{}.pkl"

    model_cls = FragmentType

    def open_reader(self):
        return read(datafile(self.test_file))

    def load_reps(self):
        with open(datafile(self.ref_file_pattern.format(self.model_cls.__name__)), 'rb') as fh:
            return pickle.load(fh)

    def test_encoding(self):
        reader = self.open_reader()
        reps = self.load_reps()
        assert len(reader) == len(reps)

        for scan, (scan_id, rep) in zip(reader, reps):
            assert scan.id == scan_id
            gpsm = scan.match()
            model_insts, _intensities, _total = self.model_cls.build_fragment_intensity_matches(gpsm)
            X = self.model_cls.encode_classification(
                sorted(model_insts[:-1],
                       key=lambda x: (x.fragment.name, x.peak.index.neutral_mass)))
            assert np.all(X == rep)


class LabileMonosaccharideAwareModelFullEncodingTest(TestCase, FullEncodingMixin):
    model_cls = LabileMonosaccharideAwareModel
