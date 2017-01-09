from glycopeptidepy.structure import residue, fragment

R = residue.Residue

Proline = R("P")
Serine = R("S")
Threonine = R("T")
Glycine = R("G")

Leucine = R("L")
Isoleucine = R("I")
Valine = R("V")

Asparagine = R("N")

Histidine = R("H")


def classify_residue_frank(residue_):
    if residue_ == Proline:
        return "pro"
    elif residue_ == Glycine:
        return "gly"
    elif residue_ in (Serine, Threonine):
        return "ser/thr"
    elif residue_ in (Leucine, Isoleucine, Valine):
        return "leu/iso/val"
    elif residue_ == Asparagine:
        return "asn"
    elif residue_ == Histidine:
        return "his"
    else:
        return "X"


def classify_amide_bond_frank(n_term, c_term):
    if n_term == Proline:
        return "pro", "X"
    elif c_term == Proline:
        return "X", "pro"

    elif n_term == Glycine:
        return "glycine", "X"
    elif c_term == Glycine:
        return "X", "glycine"

    elif n_term in (Serine, Threonine):
        return "ser/thr", "X"
    elif c_term in (Serine, Threonine):
        return "X", "ser/thr"

    elif n_term in (Leucine, Isoleucine, Valine):
        return "leu/iso/val", "X"
    elif c_term in (Leucine, Isoleucine, Valine):
        return "X", "leu/iso/val"

    elif n_term == Asparagine:
        return "asn", "X"
    elif c_term == Asparagine:
        return "X", "asn"

    elif n_term == Histidine:
        return "his", "X"
    elif c_term == Histidine:
        return "X", "his"

    return "X", "X"
