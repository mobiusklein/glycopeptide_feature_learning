
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
