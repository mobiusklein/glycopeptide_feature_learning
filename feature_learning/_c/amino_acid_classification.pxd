from glypy.utils.cenum cimport EnumValue
from glycopeptidepy._c.structure.sequence_methods cimport _PeptideSequenceCore


cpdef int proton_mobility(_PeptideSequenceCore sequence)
cpdef EnumValue classify_residue_frank(residue)

