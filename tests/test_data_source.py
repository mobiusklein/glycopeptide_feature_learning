import unittest

from typing import Counter

from ms_deisotope.data_source import get_opener
from glycopeptide_feature_learning.data_source import AnnotatedMGFDeserializer, AnnotatedScan

from .common import datafile


class TestAnnotatedMGF(unittest.TestSuite):
    filename = "MouseBrain-Z-T-5.mgf.gz"

    def open_file(self):
        return AnnotatedMGFDeserializer(get_opener(datafile(self.filename)))

    def test_read(self):
        reader = self.open_file()
        assert len(reader) == 8355

    def test_scan(self):
        reader = self.open_file()
        scan: AnnotatedScan = reader[100]

        assert scan.target == "AC(Carbamidomethyl)QFN(N-Glycosylation)R{Hex:3; HexNAc:4}"
        assert scan.annotations['analyzer'] == "orbitrap"
        assert scan.mass_shift.name == "Unmodified"
        assert scan.title == "MouseBrain-Z-T-5.mgf.gz.controllerType=0 controllerNumber=1 scan=3652"

        assert len(scan.deconvoluted_peak_set) == 198
        for peak in scan.deconvoluted_peak_set:
            assert peak.rank == -1

        scan.rank()

        assert len(scan.deconvoluted_peak_set) == 198
        rank_counts = Counter([p.rank for p in scan.deconvoluted_peak_set])
        expected_rank_counts = Counter({
            2: 10,
            -1: 39,
            10: 9,
            5: 10,
            7: 10,
            1: 70,
            8: 10,
            9: 10,
            4: 10,
            6: 10,
            3: 10
        })

        assert rank_counts == expected_rank_counts
