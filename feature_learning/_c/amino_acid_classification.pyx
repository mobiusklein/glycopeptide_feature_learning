# cython: embedsignature=True
cimport cython
from cpython cimport PyTuple_GetItem, PyTuple_Size, PyList_GET_ITEM, PyList_GET_SIZE
from cpython.object cimport PyObject
from cpython.dict cimport PyDict_GetItem
from cpython.int cimport PyInt_AsLong

from glycopeptidepy.structure import residue, fragment

from glycopeptidepy._c.structure.base cimport AminoAcidResidueBase, SequencePosition
from glycopeptidepy._c.structure.sequence_methods cimport _PeptideSequenceCore

from glypy.utils import Enum
from glypy.utils.cenum cimport EnumValue, IntEnumValue, EnumMeta

from glycopeptidepy.utils import memoize

R = residue.Residue

cdef AminoAcidResidueBase Proline = R("P")
cdef AminoAcidResidueBase Serine = R("S")
cdef AminoAcidResidueBase Threonine = R("T")
cdef AminoAcidResidueBase Glycine = R("G")

cdef AminoAcidResidueBase Alanine = R("A")
cdef AminoAcidResidueBase Leucine = R("L")
cdef AminoAcidResidueBase Isoleucine = R("I")
cdef AminoAcidResidueBase Valine = R("V")

cdef AminoAcidResidueBase Asparagine = R("N")

cdef AminoAcidResidueBase Histidine = R("H")
cdef AminoAcidResidueBase Arginine = R("R")
cdef AminoAcidResidueBase Lysine = R("K")

cdef AminoAcidResidueBase AsparticAcid = R("D")
cdef AminoAcidResidueBase GlutamicAcid = R("E")


class AminoAcidClassification(Enum):
    __enum_type__ = IntEnumValue

    pro = 0
    gly = 1
    ser_thr = 2
    leu_iso_val_ala = 3
    asn = 4
    his = 5
    arg_lys = 6
    x = 7


cdef:
    EnumValue AminoAcidClassification_pro = AminoAcidClassification.pro
    EnumValue AminoAcidClassification_gly = AminoAcidClassification.gly
    EnumValue AminoAcidClassification_ser_thr = AminoAcidClassification.ser_thr
    EnumValue AminoAcidClassification_leu_iso_val_ala = AminoAcidClassification.leu_iso_val_ala
    EnumValue AminoAcidClassification_asn = AminoAcidClassification.asn
    EnumValue AminoAcidClassification_his = AminoAcidClassification.his
    EnumValue AminoAcidClassification_arg_lys = AminoAcidClassification.arg_lys
    EnumValue AminoAcidClassification_x = AminoAcidClassification.x


cpdef int proton_mobility(_PeptideSequenceCore sequence):
    cdef:
        int p
        size_t i
        SequencePosition pos
    p = 0
    for i in range(sequence.get_size()):
        pos = sequence.get(i)
        if pos.amino_acid.name in (Histidine, Lysine, Arginine):
            p += 1
    return p


cdef EnumValue _classify_residue_frank(AminoAcidResidueBase residue_):
    if residue_ == Proline:
        return AminoAcidClassification.pro
    elif residue_ == Glycine:
        return AminoAcidClassification.gly
    elif residue_ in (Serine, Threonine):
        return AminoAcidClassification.ser_thr
    elif residue_ in (Leucine, Isoleucine, Valine, Alanine):
        return AminoAcidClassification.leu_iso_val_ala
    elif residue_ == Asparagine:
        return AminoAcidClassification.asn
    elif residue_ == Histidine:
        return AminoAcidClassification.his
    elif residue_ in (Arginine, Lysine):
        return AminoAcidClassification.arg_lys
    else:
        return AminoAcidClassification.x


cdef dict build_frank_residue_classifier():
    cdef:
        dict result
        set residues
        AminoAcidResidueBase res
    result = {}
    residues = residue.get_all_residues()

    for res in residues:
        result[res] = _classify_residue_frank(res)
    return result


cdef dict frank_residue_classifier = build_frank_residue_classifier()


cpdef EnumValue classify_residue_frank(residue):
    cdef:
        PyObject* result

    result = PyDict_GetItem(frank_residue_classifier, residue)
    if result == NULL:
        return AminoAcidClassification_x
    else:
        return <EnumValue>result


@memoize.memoize(10000)
def classify_amide_bond_frank(AminoAcidResidueBase n_term, AminoAcidResidueBase c_term):
    if n_term == Proline:
        return AminoAcidClassification_pro, AminoAcidClassification_x
    elif c_term == Proline:
        return AminoAcidClassification_x, AminoAcidClassification_pro

    elif n_term == Glycine:
        return AminoAcidClassification_gly, AminoAcidClassification_x
    elif c_term == Glycine:
        return AminoAcidClassification_x, AminoAcidClassification_gly

    elif n_term in (Serine, Threonine):
        return AminoAcidClassification_ser_thr, AminoAcidClassification_x
    elif c_term in (Serine, Threonine):
        return AminoAcidClassification_x, AminoAcidClassification_ser_thr

    elif n_term in (Leucine, Isoleucine, Valine, Alanine):
        return AminoAcidClassification_leu_iso_val_ala, AminoAcidClassification_x
    elif c_term in (Leucine, Isoleucine, Valine, Alanine):
        return AminoAcidClassification_x, AminoAcidClassification_leu_iso_val_ala

    elif n_term == Asparagine:
        return AminoAcidClassification_asn, AminoAcidClassification_x
    elif c_term == Asparagine:
        return AminoAcidClassification_x, AminoAcidClassification_asn

    elif n_term == Histidine:
        return AminoAcidClassification_his, AminoAcidClassification_x
    elif c_term == Histidine:
        return AminoAcidClassification_x, AminoAcidClassification_his

    return AminoAcidClassification_x, AminoAcidClassification_x
