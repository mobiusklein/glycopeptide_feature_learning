from glycopeptidepy.structure import residue, fragment
from glypy.utils import Enum

R = residue.Residue

Proline = R("P")
Serine = R("S")
Threonine = R("T")
Glycine = R("G")

Alanine = R("A")
Leucine = R("L")
Isoleucine = R("I")
Valine = R("V")

Asparagine = R("N")

Histidine = R("H")
Arginine = R("R")
Lysine = R("K")

AsparticAcid = R("D")
GlutamicAcid = R("E")


class AminoAcidClassification(Enum):
    pro = 0
    gly = 1
    ser_thr = 2
    leu_iso_val_ala = 3
    asn = 4
    his = 5
    arg_lys = 6
    x = 7


def proton_mobility(sequence):
    p = 0
    for res, mods in sequence:
        if res.name in (Histidine, Lysine, Arginine):
            p += 1
    return p


def classify_residue_frank(residue_):
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


def classify_amide_bond_frank(n_term, c_term):
    if n_term == Proline:
        return AminoAcidClassification.pro, AminoAcidClassification.x
    elif c_term == Proline:
        return AminoAcidClassification.x, AminoAcidClassification.pro

    elif n_term == Glycine:
        return AminoAcidClassification.gly, AminoAcidClassification.x
    elif c_term == Glycine:
        return AminoAcidClassification.x, AminoAcidClassification.gly

    elif n_term in (Serine, Threonine):
        return AminoAcidClassification.ser_thr, AminoAcidClassification.x
    elif c_term in (Serine, Threonine):
        return AminoAcidClassification.x, AminoAcidClassification.ser_thr

    elif n_term in (Leucine, Isoleucine, Valine, Alanine):
        return AminoAcidClassification.leu_iso_val_ala, AminoAcidClassification.x
    elif c_term in (Leucine, Isoleucine, Valine, Alanine):
        return AminoAcidClassification.x, AminoAcidClassification.leu_iso_val_ala

    elif n_term == Asparagine:
        return AminoAcidClassification.asn, AminoAcidClassification.x
    elif c_term == Asparagine:
        return AminoAcidClassification.x, AminoAcidClassification.asn

    elif n_term == Histidine:
        return AminoAcidClassification.his, AminoAcidClassification.x
    elif c_term == Histidine:
        return AminoAcidClassification.x, AminoAcidClassification.his

    return AminoAcidClassification.x, AminoAcidClassification.x
