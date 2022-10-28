import numpy as np
cimport numpy as np

np.import_array()

from numpy.math cimport isnan

from ms_deisotope._c.peak_set cimport DeconvolutedPeak, DeconvolutedPeakSet

from glycan_profiling._c.structure.fragment_match_map cimport (
    FragmentMatchMap, PeakFragmentPair)

from glycopeptidepy._c.structure.base cimport AminoAcidResidueBase, SequencePosition
from glycopeptidepy._c.structure.sequence_methods cimport _PeptideSequenceCore
from glycopeptidepy._c.structure.fragment cimport (
    PeptideFragment, FragmentBase, IonSeriesBase, ChemicalShiftBase)


from glycopeptide_feature_learning._c.model_types cimport _FragmentType


ctypedef fused scalar_or_array:
    double
    np.ndarray


cdef class BackbonePosition(object):
    cdef:
        public _FragmentType match
        public double intensity
        public double predicted
        public double reliability

    @staticmethod
    cdef BackbonePosition _create(_FragmentType match, double intensity, double predicted, double reliability)


cpdef _calculate_pearson_residuals(self, bint use_reliability=*, double base_reliability=*)


cpdef double classify_ascending_abundance_peptide_Y(spectrum_match)
