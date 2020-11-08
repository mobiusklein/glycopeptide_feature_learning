import numpy as np

import ms_deisotope
from ms_peak_picker.plot import peaklist_to_vector, _normalize_ylabels

from glycan_profiling.plotting import sequence_fragment_logo
from glycan_profiling.plotting.spectral_annotation import TidySpectrumMatchAnnotator, font_options, default_ion_series_to_color


def normalize_scan(scan, factor=None):
    scan = scan.copy()
    scan.deconvoluted_peak_set = scan.deconvoluted_peak_set.__class__(
        p.clone() for p in scan.deconvoluted_peak_set)
    bp = scan.base_peak().clone()
    if factor is None:
        factor = bp.intensity / 100
    for peak in scan:
        peak.intensity /= factor
    return scan


class MirrorSpectrumAnnotator(TidySpectrumMatchAnnotator):

    def draw_all_peaks(self, color='black', alpha=0.5, **kwargs):
        ms_deisotope.plot.draw_peaklist(
            self.spectrum_match.deconvoluted_peak_set,
            alpha=0.3, color='grey', ax=self.ax, lw=0.75, **kwargs)

    def base_peak_factor(self):
        return self.spectrum_match.scan.base_peak().intensity / 100

    def label_peak(self, fragment, peak, fontsize=12, rotation=90, mirror=False, **kw):
        label = "%s" % fragment.name
        if fragment.series == 'oxonium_ion':
            return ''
        if peak.charge > 1:
            label += "$^{%d}$" % peak.charge
        y = peak.intensity
        upshift = 2
        sign = 1 if y > 0 else -1
        y = min(y + sign * upshift, self.upper * 0.9)

        kw.setdefault("clip_on", self.clip_labels)
        clip_on = kw['clip_on']

        text = self.ax.text(
            peak.mz, y, label, rotation=rotation, va='bottom',
            ha='center', fontsize=fontsize, fontproperties=font_options,
            clip_on=clip_on)
        self.peak_labels.append(text)
        return text

    def add_logo_plot(self, xrel=0.15, yrel=0.8, width=0.67, height=0.13, **kwargs):
        figure = self.ax.figure
        iax = figure.add_axes([xrel, yrel, width, height])
        logo = sequence_fragment_logo.glycopeptide_match_logo(
            self.spectrum_match, ax=iax, **kwargs)
        return logo


def mirror_predicted_spectrum(model, scan, target, *args, **kwargs):
    target = target
    scan = normalize_scan(scan)
    case = model.evaluate(scan, target, **kwargs)

    c, ey, _, y = case.model_fit._get_predicted_intensities(case)
    c = c[:-1]
    y = y[:-1].copy()
    y /= y.max()
    y *= max(c, key=lambda x: x.peak.intensity).peak.intensity
    x = [ci.peak.mz for ci in c]
    series = [ci.fragment.series for ci in c]

    art = MirrorSpectrumAnnotator(case)
    art.draw(fontsize=kwargs.get("font_size", 7))

    x, y, series = zip(*sorted(zip(x, y, series), key=lambda f: f[0]))

    for i in range(len(x)):
        art.ax.plot(*peaklist_to_vector([[x[i], -y[i]]]),
                    color=default_ion_series_to_color[series[i]])

    art.ax.text(150, -80, r'$\rho=%0.2f$' %
                case._mixture_apply(case._calculate_correlation_coef))
    art.ax.figure.set_figwidth(10)

    art.ax.set_ylim(-105, kwargs.get("upper_bound", 195))
    art.ax.set_yticks(np.arange(-100, 120, 20))
    _normalize_ylabels(art.ax)
    art.ax.spines['left'].set_visible(False)
    art.ax.spines['bottom'].set_visible(False)
    art.ax.vlines(100, -100, 100, lw=0.5)
    art.ax.set_xlim(100, 2050)
    art.ax.set_ylabel('Relative Intensity                   ')
    art.add_logo_plot(xrel=0.12, width=0.27, yrel=0.65, draw_glycan=True)
    return art