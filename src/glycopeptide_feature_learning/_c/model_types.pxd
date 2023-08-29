from ms_deisotope._c.peak_set cimport DeconvolutedPeak, DeconvolutedPeakSet

from glycan_profiling._c.structure.fragment_match_map cimport (
    FragmentMatchMap, PeakFragmentPair)

from glycopeptidepy._c.structure.base cimport AminoAcidResidueBase, SequencePosition
from glycopeptidepy._c.structure.sequence_methods cimport _PeptideSequenceCore
from glycopeptidepy._c.structure.fragment cimport PeptideFragment, FragmentBase, IonSeriesBase, ChemicalShiftBase

from glypy.utils.enum import Enum
from glypy.utils.cenum cimport EnumValue

import numpy as np
cimport numpy as np

np.import_array()


ctypedef np.uint8_t feature_dtype_t


cpdef int get_nterm_index_from_fragment(PeptideFragment fragment, _PeptideSequenceCore structure)
cpdef int get_cterm_index_from_fragment(PeptideFragment fragment, _PeptideSequenceCore structure)


cdef class _FragmentType(object):

    cdef:
        public EnumValue nterm
        public EnumValue cterm
        public EnumValue series
        public PeakFragmentPair peak_pair
        public int glycosylated
        public int charge
        public _PeptideSequenceCore sequence

        public bint _is_backbone
        public bint _is_assigned
        public bint _is_stub_glycopeptide


    @staticmethod
    cdef _FragmentType _create(type fragment_type, EnumValue nterm, EnumValue cterm, EnumValue series, int glycosylated, int charge, PeakFragmentPair peak_pair, _PeptideSequenceCore sequence)

    cdef DeconvolutedPeak get_peak(self)
    cdef FragmentBase get_fragment(self)

    cpdef bint is_assigned(self)
    cpdef bint is_backbone(self)
    cpdef bint is_stub_glycopeptide(self)

    cdef long get_feature_count(self)
    cpdef np.ndarray[feature_dtype_t, ndim=1, mode='c'] _allocate_feature_array(self)
    cpdef np.ndarray[feature_dtype_t, ndim=1, mode='c'] as_feature_vector(self, dict context=*)
    cpdef Py_ssize_t build_feature_vector(self, feature_dtype_t[::1] X, Py_ssize_t offset, dict context=*)

cpdef np.ndarray[feature_dtype_t, ndim=2, mode='c'] encode_classification(cls, list classification)

cpdef from_peak_peptide_fragment_pair(cls, PeakFragmentPair peak_fragment_pair, _PeptideSequenceCore structure)

cpdef EnumValue get_nterm_neighbor(_FragmentType self, int offset=*)
cpdef EnumValue get_cterm_neighbor(_FragmentType self, int offset=*)

cpdef int get_cleavage_site_distance_from_center(_FragmentType self)

cpdef Py_ssize_t specialize_proline(_FragmentType self, feature_dtype_t[::1] X, Py_ssize_t offset, dict context=*)
cpdef Py_ssize_t encode_stub_information(_FragmentType self, feature_dtype_t[::1] X, Py_ssize_t offset, dict context=*)
cpdef Py_ssize_t encode_stub_fucosylation(_FragmentType self, feature_dtype_t[::1] X, Py_ssize_t offset, dict context=*)
cpdef Py_ssize_t encode_neighboring_residues(_FragmentType self, feature_dtype_t[::1] X, Py_ssize_t offset, dict context=*)
cpdef Py_ssize_t encode_stub_charge(_FragmentType self, feature_dtype_t[::1] X, Py_ssize_t offset, dict context=*)
cpdef Py_ssize_t encode_stub_charge_loss_approximate(_FragmentType self, feature_dtype_t[::1] X, Py_ssize_t offset, dict context=*)
cpdef Py_ssize_t encode_labile_monosaccharides_charge(_FragmentType self, feature_dtype_t[::1] X, Py_ssize_t offset, dict context=*)

cpdef Py_ssize_t StubChargeModel_build_feature_vector(_FragmentType self, feature_dtype_t[::1] X, Py_ssize_t offset, dict context=*)
cpdef Py_ssize_t StubChargeModelApproximate_build_feature_vector(_FragmentType self, feature_dtype_t[::1] X, Py_ssize_t offset, dict context=*)
cpdef Py_ssize_t LabileMonosaccharideAwareModel_build_feature_vector(_FragmentType self, feature_dtype_t[::1] X, Py_ssize_t offset, dict context=*)
cpdef Py_ssize_t LabileMonosaccharideAwareModelApproximate_build_feature_vector(_FragmentType self, feature_dtype_t[::1] X, Py_ssize_t offset, dict context=*)


cpdef list classify_sequence_by_residues(_PeptideSequenceCore sequence)

cpdef build_fragment_intensity_matches(cls, gpsm, bint include_unassigned_sum=*)