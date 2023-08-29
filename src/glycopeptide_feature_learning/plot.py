import numpy as np

import ms_deisotope

from glycan_profiling.plotting.spectral_annotation import (
    plt,
    normalize_scan,
    MirrorSpectrumAnnotatorFacet
)


def make_theoretical_spectrum(model_match):
    c, _ey, _, y = model_match._get_predicted_peaks()
    c = c[:-1]
    y = y[:-1].copy()
    y /= y.max()
    y *= max(c, key=lambda x: x.peak.intensity).peak.intensity
    x = [ci.peak.mz for ci in c]

    peaks = []
    for mz, inten, frag in zip(x, y, c):
        peak = ms_deisotope.DeconvolutedPeak(
            ms_deisotope.neutral_mass(mz, frag.charge),
            inten, frag.charge, inten, 0, 0
        )
        peaks.append(peak)
    peak_set = ms_deisotope.DeconvolutedPeakSet(peaks)
    peak_set.reindex()
    scan = model_match.scan.copy()
    scan.deconvoluted_peak_set = peak_set
    return scan


def mirror_spectra(psm_a, psm_b, ax=None, usemathtext: bool=True, **kwargs):
    art = MirrorSpectrumAnnotatorFacet(psm_a, ax=ax, usemathtext=usemathtext)
    art.draw(**kwargs)
    reflect = MirrorSpectrumAnnotatorFacet(
        psm_b, ax=art.ax, usemathtext=usemathtext)
    reflect.draw(mirror=True, **kwargs)
    reflect.ax.set_ylim(-1100, 1600)
    for lab in reflect.peak_labels:
        lab.remove()
    ax = art.ax
    ax.set_yticks(np.arange(-1000, 1000 + 250, 250))
    use_tex = plt.rcParams['text.usetex']
    ax.set_yticklabels(map(lambda x: str(x) + ('%' if not use_tex else '\%'),
                           list(range(100, -25, -25)) + list(range(25, 125, 25))))
    art.sequence_logo = art.add_logo_plot(xrel=0.15, width=0.65, yrel=0.85, draw_glycan=True, lw=0.1)
    return art.ax, art, reflect


def mirror_predicted_spectrum(model_tree, model_match, **kwargs):
    m1 = model_tree.evaluate(normalize_scan(
        model_match.scan), model_match.target)
    m2 = model_tree.evaluate(
        normalize_scan(
            make_theoretical_spectrum(model_match),
            factor=model_match.scan.base_peak().intensity / 1000,
        ),
        model_match.target,
    )
    ax, f1, f2 = mirror_spectra(m1, m2, **kwargs)

    tot_corr = model_match.total_correlation()
    pep_corr = model_match.peptide_correlation()
    gly_corr = model_match.glycan_correlation()
    use_tex = plt.rcParams['text.usetex']
    corr_text = (
        fr'$\rho_{{total}}={tot_corr:0.2f}$\n$\rho_{{peptide}}={pep_corr:0.2f}$\n$\rho_{{glycan}}={gly_corr:0.2f}$'
    )
    if use_tex:
        corr_text = corr_text.replace(r'\n', r'\\').replace(
            "=", "=&").replace("$", '')
        corr_text = r"\begin{eqnarray*}" + corr_text + r'\end{eqnarray*}'
    else:
        corr_text = corr_text.replace(r'\n', '\n')
    ax.figure.text(.15, 0.27,
                   corr_text,
                   ha="left", va='center')
    xlim = ax.get_xlim()
    ax.plot(xlim, [0, 0], color='black', lw=0.5)
    ax.set_xlim(*xlim)
    return ax, f1, f2
