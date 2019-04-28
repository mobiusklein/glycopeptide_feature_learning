import os

from ms_deisotope import DeconvolutedPeak, DeconvolutedPeakSet, neutral_mass
from ms_deisotope.data_source import ProcessedScan, ActivationInformation
from ms_deisotope.output.mgf import ProcessedMGFDeserializer, pymgf

from glycan_profiling.structure import FragmentCachingGlycopeptide
from glycan_profiling.tandem.glycopeptide.scoring import LogIntensityScorer

from glycopeptidepy.algorithm import reverse_preserve_sequon
from glycopeptidepy.utils import memoize
from glypy.utils import opener

from .common import intensity_rank
from .matching import SpectrumMatchAnnotator


def _parse_charge(z, list_only=False, **kwargs):
    '''Pyteomics _parse_charge is very general-purpose, and
    can't be sped up, so we monkey-patch it here.'''
    try:
        if not list_only:
            return int(z.replace('+', ''))
        else:
            return map(_parse_charge, z.split(" "))
    except Exception:
        if '-' in z:
            return int(z.replace("-", '')) * -1
        else:
            raise


pymgf.aux._parse_charge = _parse_charge


class RankedPeak(DeconvolutedPeak):
    def __init__(self, neutral_mass, intensity, charge, signal_to_noise, index,
                 rank=-1):
        DeconvolutedPeak.__init__(
            self, neutral_mass, intensity, charge, signal_to_noise, index, 0)
        self.rank = rank

    def __reduce__(self):
        return self.__class__, (self.neutral_mass, self.intensity, self.charge,
                                self.signal_to_noise, self.index, self.rank)

    def clone(self):
        return self.__class__(self.neutral_mass, self.intensity, self.charge,
                              self.signal_to_noise, self.index, self.rank)

    def __repr__(self):
        return "RankedPeak(%0.2f, %0.2f, %d, %0.2f, %s, %d)" % (
            self.neutral_mass, self.intensity, self.charge,
            self.signal_to_noise, self.index, self.rank)


@memoize.memoize(4000)
def parse_sequence(glycopeptide):
    return FragmentCachingGlycopeptide(glycopeptide)


class AnnotatedScan(ProcessedScan):
    _structure = None
    # if matcher is populated, then pickling will fail due to recursive
    # sharing of the peak set
    matcher = None

    def __reduce__(self):
        return self.__class__, (self.id, self.title, self.precursor_information,
                                self.ms_level, self.scan_time, self.index, self.peak_set,
                                self.deconvoluted_peak_set, self.polarity, self.activation,
                                self.acquisition_information, self.isolation_window,
                                self.instrument_configuration, self.product_scans,
                                self.annotations)

    @property
    def structure(self):
        if self._structure is None:
            self._structure = parse_sequence(self.annotations['structure'])
        return self._structure

    def match(self, **kwargs):
        self.matcher = LogIntensityScorer.evaluate(
            self, self.structure, **kwargs)
        return self.matcher

    @property
    def solution_map(self):
        try:
            return self.matcher.solution_map
        except AttributeError:
            return None

    @property
    def mass_shift(self):
        return self.annotations.get('mass_shift', "Unmodified")

    def plot(self, ax=None):
        art = SpectrumMatchAnnotator(self.match(), ax=ax)
        art.draw()
        return art

    def rank(self, cache=True):
        if 'ranked_peaks' not in self.annotations or not cache:
            peaks = self.deconvoluted_peak_set
            intensity_rank(peaks)
            peaks = DeconvolutedPeakSet([p for p in peaks if p.rank > 0])
            peaks.reindex()
            if cache:
                self.annotations['ranked_peaks'] = peaks
            return peaks
        return self.annotations['ranked_peaks']


def build_deconvoluted_peak_set_from_arrays(mz_array, intensity_array, charge_array):
    peaks = []
    for i in range(len(mz_array)):
        peak = RankedPeak(
            neutral_mass(mz_array[i], charge_array[i]), intensity_array[i], charge_array[i],
            intensity_array[i], i)
        peaks.append(peak)
    peak_set = DeconvolutedPeakSet(peaks)
    peak_set.reindex()
    return peak_set


class AnnotatedMGFDeserializer(ProcessedMGFDeserializer):
    def _build_peaks(self, scan):
        mz_array = scan['m/z array']
        intensity_array = scan["intensity array"]
        charge_array = scan['charge array']
        peak_set = build_deconvoluted_peak_set_from_arrays(mz_array, intensity_array, charge_array)
        intensity_rank(peak_set)
        return peak_set

    def _activation(self, scan):
        method = scan.get('annotations', {}).get('activation_method')
        if method is None or method.startswith("unknown"):
            method = 'hcd'
        return ActivationInformation(
            method,
            scan.get('annotations', {}).get('activation_energy'))

    def _scan_title(self, scan):
        title = super(AnnotatedMGFDeserializer, self)._scan_title(scan)
        try:
            fname = os.path.basename(self.source_file)
        except Exception:
            fname = os.path.basename(self.source_file.name)
        return "%s.%s" % (fname, title)

    def _make_scan(self, scan):
        scan = super(AnnotatedMGFDeserializer, self)._make_scan(scan)
        precursor_info = scan.precursor_information
        scan.annotations.pop("is_hcd", None)
        scan.annotations.pop("is_exd", None)
        return AnnotatedScan(
            scan.id, scan.title, precursor_info,
            scan.ms_level, scan.scan_time, scan.index,
            scan.peak_set.pack() if scan.peak_set is not None else None,
            scan.deconvoluted_peak_set,
            scan.polarity,
            scan.activation,
            scan.acquisition_information,
            scan.isolation_window,
            scan.instrument_configuration,
            scan.product_scans,
            scan.annotations)


def read(path):
    return AnnotatedMGFDeserializer(opener(path))


try:
    has_c = True
    _RankedPeak = RankedPeak
    _build_deconvoluted_peak_set_from_arrays = build_deconvoluted_peak_set_from_arrays
    from feature_learning._c.data_source import RankedPeak, build_deconvoluted_peak_set_from_arrays
except ImportError:
    has_c = False
