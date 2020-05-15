import struct
from collections import namedtuple

from glypy.utils import Enum

from glycan_profiling.tandem.spectrum_match import SpectrumMatch
from glycan_profiling.tandem.glycopeptide.scoring.coverage_weighted_binomial import (
    CoverageWeightedBinomialScorer)


from glycan_profiling.plotting import spectral_annotation


def match_scan_to_sequence(scan, sequence, mass_accuracy=2e-5):
    return CoverageWeightedBinomialScorer.evaluate(
        scan, sequence, error_tolerance=mass_accuracy)


class SpectrumMatchAnnotator(spectral_annotation.SpectrumMatchAnnotator):
    def __init__(self, spectrum_match, ax=None):
        super(SpectrumMatchAnnotator, self).__init__(spectrum_match, ax)

    def label_peak(self, fragment, peak, fontsize=12, rotation=90, **kw):
        if fragment.series == 'oxonium_ion':
            return
        super(SpectrumMatchAnnotator, self).label_peak(
            fragment, peak, fontsize, rotation, **kw)


class SpectrumMatchClassification(Enum):
    target_peptide_target_glycan = 0
    target_peptide_decoy_glycan = 1
    decoy_peptide_target_glycan = 2
    decoy_peptide_decoy_glycan = 3


_ScoreSet = namedtuple("ScoreSet", ['glycopeptide_score', 'peptide_score', 'glycan_score'])


class ScoreSet(_ScoreSet):
    __slots__ = ()
    packer = struct.Struct("!fff")

    @classmethod
    def from_spectrum_matcher(cls, match):
        return cls(match.score, match.peptide_score(), match.glycan_score())

    def pack(self):
        return self.packer.pack(*self)

    @classmethod
    def unpack(cls, binary):
        return cls(*cls.packer.unpack(binary))


class MultiScoreSpectrumMatch(SpectrumMatch):
    __slots__ = ('score_set', 'match_type')

    def __init__(self, scan, target, score_set, best_match=False, data_bundle=None,
                 q_value=None, id=None, mass_shift=None, match_type=None):
        super(MultiScoreSpectrumMatch, self).__init__(
            scan, target, score_set[0], best_match, data_bundle, q_value, id, mass_shift)
        self.score_set = ScoreSet(*score_set)
        self.match_type = SpectrumMatchClassification[match_type]

    def __reduce__(self):
        return self.__class__, (self.scan, self.target, self.score_set, self.best_match,
                                self.data_bundle, self.q_value, self.id, self.mass_shift,
                                self.match_type.value)

    def pack(self):
        return (self.target.id, self.score_set.pack(), int(self.best_match),
                self.mass_shift.name, self.match_type.value)

    @classmethod
    def from_match_solution(cls, match):
        raise NotImplementedError()
