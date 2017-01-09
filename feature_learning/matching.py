from glycan_profiling.tandem.glycopeptide.scoring.coverage_weighted_binomial import (
    CoverageWeightedBinomialScorer)


def match_scan_to_sequence(scan, sequence, mass_accuracy=2e-5):
    return CoverageWeightedBinomialScorer.evaluate(
        scan, sequence, error_tolerance=mass_accuracy)
