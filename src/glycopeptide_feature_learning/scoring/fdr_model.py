from typing import List
import numpy as np

from glycan_profiling.tandem.target_decoy import svm


class CorrelationPeptideSVMModel(svm.PeptideScoreSVMModel):
    def extract_features(self, psms) -> np.ndarray:
        features = np.zeros((len(psms), 3))
        for i, psm in enumerate(psms):
            features[i, :] = (
                psm.score_set.peptide_score,
                psm.score_set.peptide_coverage,
                ((psm.score_set.peptide_correlation + 1) / 2)
                * psm.score_set.peptide_backbone_count
                * psm.score_set.peptide_coverage,
            )
        return features

    def feature_names(self) -> List[str]:
        return [
            "peptide_score",
            "peptide_coverage",
            "peptide_correlation_score"
        ]