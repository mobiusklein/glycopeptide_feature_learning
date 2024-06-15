import os
from typing import List, DefaultDict, Set, Tuple

from ms_deisotope import DeconvolutedPeak, DeconvolutedPeakSet, neutral_mass
from ms_deisotope.data_source import ProcessedScan, ActivationInformation
from ms_deisotope.output.mgf import ProcessedMGFDeserializer

from glycresoft.structure import FragmentCachingGlycopeptide, DecoyFragmentCachingGlycopeptide
from glycresoft import symbolic_expression
from glycresoft.tandem.glycopeptide.scoring import LogIntensityScorer
from glycresoft.chromatogram_tree.mass_shift import (
    MassShift, mass_shift_index, MassShiftCollection, Unmodified,
    Ammonium, Sodium, Potassium)

from glycopeptidepy.algorithm import reverse_sequence
from glycopeptidepy.utils import memoize
from glypy.utils import opener

from .common import intensity_rank
from .matching import SpectrumMatchAnnotator


mass_shifts = MassShiftCollection([Unmodified, Ammonium, Sodium, Potassium])


def _typecast(value: str):
    if not isinstance(value, str):
        return value
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    lowered = value.lower()
    if lowered == 'true':
        return True
    elif lowered == 'false':
        return False
    elif lowered == 'null':
        return None
    else:
        return value



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
    _structure: FragmentCachingGlycopeptide = None
    # if matcher is populated, then pickling will fail due to recursive
    # sharing of the peak set
    matcher: LogIntensityScorer = None

    def __reduce__(self):
        return self.__class__, (self.id, self.title, self.precursor_information,
                                self.ms_level, self.scan_time, self.index, self.peak_set,
                                self.deconvoluted_peak_set, self.polarity, self.activation,
                                self.acquisition_information, self.isolation_window,
                                self.instrument_configuration, self.product_scans,
                                self.annotations)

    def decoy(self, peptide=True, glycan=True):
        if not (peptide | glycan):
            raise ValueError("Must specify which dimension to make into a decoy")
        gp = self.structure
        if peptide:
            gp = reverse_sequence(gp, peptide_type=FragmentCachingGlycopeptide)
        if glycan:
            gp = DecoyFragmentCachingGlycopeptide.from_target(gp)
        dup = self.copy()
        dup._structure = gp
        return dup

    @property
    def structure(self) -> FragmentCachingGlycopeptide:
        if self._structure is None:
            self._structure = parse_sequence(self.annotations['structure'])
        return self._structure

    # Alias
    @property
    def target(self):
        return self.structure

    def match(self, **kwargs):
        self.matcher = LogIntensityScorer.evaluate(
            self, self.structure, mass_shift=self.mass_shift, **kwargs)
        return self.matcher

    @property
    def solution_map(self):
        try:
            return self.matcher.solution_map
        except AttributeError:
            return None

    @property
    def mass_shift(self) -> MassShift:
        mass_shift_name = self.annotations.get('mass_shift', "Unmodified")
        if not isinstance(mass_shift_name, str):
            mass_shift_name = mass_shift_name.encode('utf8')
        try:
            mass_shift = mass_shifts[mass_shift_name]
            return mass_shift
        except Exception:
            composition = mass_shift_index.get(mass_shift_name)
            if composition is None:
                expr = symbolic_expression.parse_expression(mass_shift_name)
                ctx = symbolic_expression.SymbolContext(
                    {k: MassShift(k, v) for k, v in mass_shift_index.items()})
                if ctx.partially_defined(expr):
                    mass_shift = ctx[expr]
                else:
                    import warnings
                    warnings.warn("Unknown mass shift %r" % (mass_shift_name, ))
                    composition = mass_shift_index['Unmodified'].copy()
                    mass_shift = MassShift(mass_shift_name, composition)
            else:
                mass_shift = MassShift(mass_shift_name, composition)
            mass_shifts.append(mass_shift)
            return mass_shift

    def plot(self, ax=None):
        art = SpectrumMatchAnnotator(self.match(), ax=ax)
        art.draw()
        return art

    def rank(self, cache=True) -> DeconvolutedPeakSet:
        if 'ranked_peaks' not in self.annotations or not cache:
            peaks = self.deconvoluted_peak_set
            intensity_rank(peaks)
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
    _cached_basename: str = None

    def _build_peaks(self, scan):
        mz_array = scan['m/z array']
        intensity_array = scan["intensity array"]
        charge_array = scan['charge array']
        peak_set = build_deconvoluted_peak_set_from_arrays(mz_array, intensity_array, charge_array)
        # intensity_rank(peak_set)
        return peak_set

    def _activation(self, scan):
        method = scan.get('annotations', {}).get('activation_method')
        if method is None or method.startswith("unknown"):
            method = 'hcd'
        return ActivationInformation(
            method,
            scan.get('annotations', {}).get('activation_energy'))

    def _scan_index(self, scan):
        """Returns the base 0 offset from the start
        of the data file in number of scans to reach
        this scan.

        If the original format does not natively include
        an index value, this value may be computed from
        the byte offset index.

        Parameters
        ----------
        scan : Mapping
            The underlying scan information storage,
            usually a `dict`

        Returns
        -------
        int
        """
        try:
            return self._title_to_index[super(AnnotatedMGFDeserializer, self)._scan_title(scan)]
        except KeyError:
            try:
                return self._title_to_index[super(AnnotatedMGFDeserializer, self)._scan_title(scan) + '.']
            except KeyError:
                return -1
        return -1

    def _scan_title(self, scan):
        title = super(AnnotatedMGFDeserializer, self)._scan_title(scan)
        if self._cached_basename is None:
            try:
                fname = os.path.basename(self.source_file)
            except Exception:
                fname = os.path.basename(self.source_file.name)
            self._cached_basename = fname
        else:
            fname = self._cached_basename
        return "%s.%s" % (fname, title)

    def _annotations(self, scan):
        annotations = super()._annotations(scan)
        return {k: _typecast(v) for k, v in annotations.items()}

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
    return AnnotatedMGFDeserializer(opener(path, 'rb'))


try:
    has_c = True
    _RankedPeak = RankedPeak
    _build_deconvoluted_peak_set_from_arrays = build_deconvoluted_peak_set_from_arrays
    from glycopeptide_feature_learning._c.data_source import RankedPeak, build_deconvoluted_peak_set_from_arrays
except ImportError:
    has_c = False


def describe_training_observations(annotated_spectra: List[AnnotatedScan]) -> Tuple[DefaultDict[str, Set[str]],
                                                                                    DefaultDict[str, Set[str]],
                                                                                    DefaultDict[str, Set[str]]]:
    by_structure = DefaultDict(set)
    by_backbone = DefaultDict(set)
    by_glycan = DefaultDict(set)
    by_precursor = DefaultDict(set)

    for spectrum in annotated_spectra:
        by_structure[str(spectrum.structure)].add(spectrum.title)
        by_backbone[str(spectrum.structure.clone().deglycosylate())].add(
            spectrum.title)
        by_glycan[str(spectrum.structure.glycan_composition)
                  ].add(spectrum.title)
        by_precursor[str(spectrum.structure), spectrum.precursor_information.charge, spectrum.mass_shift.name].add(
            spectrum.title)

    return by_structure, by_backbone, by_glycan, by_precursor
