import os

from ms_deisotope import DeconvolutedPeak, DeconvolutedPeakSet, neutral_mass
from ms_deisotope.data_source import ProcessedScan, ActivationInformation
from ms_deisotope.output.mgf import ProcessedMGFDeserializer

from glycan_profiling.structure import FragmentCachingGlycopeptide
from glycan_profiling.tandem.glycopeptide.scoring import CoverageWeightedBinomialScorer

from glycopeptidepy.algorithm import reverse_preserve_sequon
from glycopeptidepy.utils import memoize
from glypy.utils import opener

from .common import intensity_rank
from .matching import SpectrumMatchAnnotator


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
    matcher = None

    @property
    def structure(self):
        if self._structure is None:
            self._structure = parse_sequence(self.annotations['structure'])
        return self._structure

    def match(self, **kwargs):
        self.matcher = CoverageWeightedBinomialScorer.evaluate(self, self.structure, **kwargs)
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

    def decoy(self, method='reverse'):
        copy = self.clone()
        copy._structure = reverse_preserve_sequon(copy.structure)
        return copy

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


class AnnotatedMGFDeserializer(ProcessedMGFDeserializer):
    def _build_peaks(self, scan):
        mz_array = scan['m/z array']
        intensity_array = scan["intensity array"]
        charge_array = scan['charge array']
        peaks = []
        for i in range(len(mz_array)):
            peak = RankedPeak(
                neutral_mass(mz_array[i], charge_array[i]), intensity_array[i], charge_array[i],
                intensity_array[i], i)
            peaks.append(peak)
        peak_set = DeconvolutedPeakSet(peaks)
        peak_set.reindex()
        intensity_rank(peak_set)
        return peak_set

    def _activation(self, scan):
        return ActivationInformation(
            scan.get('annotations', {}).get('activation_method'),
            scan.get('annotations', {}).get('activation_energy'))

    def _scan_title(self, scan):
        title = super(AnnotatedMGFDeserializer, self)._scan_title(scan)
        try:
            fname = os.path.basename(self.source_file)
        except Exception:
            fname = ''
        return "%s.%s" % (fname, title)

    def _make_scan(self, scan):
        scan = super(AnnotatedMGFDeserializer, self)._make_scan(scan)
        precursor_info = scan.precursor_information
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
