cimport cython

from cpython.object cimport PyObject

from glycan_profiling._c.tandem.spectrum_match cimport ScoreSet


cdef class ModelScoreSet(ScoreSet):
    cdef:
        public float peptide_correlation
        public int peptide_backbone_count
        public float glycan_correlation

    @staticmethod
    cdef ModelScoreSet _create_model_score_set(float glycopeptide_score, float peptide_score, float glycan_score, float glycan_coverage,
                                               float stub_glycopeptide_intensity_utilization, float oxonium_ion_intensity_utilization,
                                               int n_stub_glycopeptide_matches, float peptide_coverage, float total_signal_utilization,
                                               float peptide_correlation, int peptide_backbone_count, float glycan_correlation)