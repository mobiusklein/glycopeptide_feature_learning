cimport cython

from cpython.object cimport PyObject

from glycresoft._c.tandem.spectrum_match cimport ScoreSet


cdef class ModelScoreSet(ScoreSet):
    cdef:
        public float peptide_correlation
        public int peptide_backbone_count
        public float glycan_correlation
        public float peptide_reliability_total
        public float glycan_reliability_total
        public int partition_label

    @staticmethod
    cdef ModelScoreSet _create_model_score_set(float glycopeptide_score, float peptide_score, float glycan_score, float glycan_coverage,
                                               float stub_glycopeptide_intensity_utilization, float oxonium_ion_intensity_utilization,
                                               int n_stub_glycopeptide_matches, float peptide_coverage, float total_signal_utilization,
                                               float peptide_correlation, int peptide_backbone_count, float glycan_correlation,
                                               float peptide_reliability_total, float glycan_reliability_total, int partition_label)