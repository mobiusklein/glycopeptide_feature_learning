from __future__ import print_function
from math import fabs
import operator

get_intensity = operator.attrgetter("intensity")


def log(message, *args):
    print(message % args)


def ppm_error(x, y):
    return (x - y) / y


def tol_ppm_error(x, y, tolerance):
    err = (x - y) / y
    if fabs(err) <= tolerance:
        return err
    else:
        return None


def mass_offset_match(mass, peak, offset=0., tolerance=2e-5):
    return abs(ppm_error(mass + offset, peak.neutral_mass)) <= tolerance


class DPeak(object):
    def __init__(self, peak):
        self.neutral_mass = peak.neutral_mass
        self.id = peak.id
        self.charge = peak.charge
        self.intensity = peak.intensity
        self.rank = 0
        self.peak_relations = []

    def clone(self):
        return DPeak(self)


OUT_OF_RANGE_INT = -999


def search_spectrum(peak, peak_list, feature, tolerance=2e-5):
    matches = []
    for peak in peak_list:
        if feature(peak, peak):
            matches.append(peak)
    return matches


def intensity_ratio_function(peak1, peak2):
    ratio = peak1.intensity / float(peak2.intensity)
    if ratio >= 5:
        return -4
    elif 2.5 <= ratio < 5:
        return -3
    elif 1.7 <= ratio < 2.5:
        return -2
    elif 1.3 <= ratio < 1.7:
        return -1
    elif 1.0 <= ratio < 1.3:
        return 0
    elif 0.8 <= ratio < 1.0:
        return 1
    elif 0.6 <= ratio < 0.8:
        return 2
    elif 0.4 <= ratio < 0.6:
        return 3
    elif 0.2 <= ratio < 0.4:
        return 4
    elif 0. <= ratio < 0.2:
        return 5


def intensity_rank(peak_list, minimum_intensity=100.):
    peak_list = sorted(peak_list, key=get_intensity, reverse=True)
    i = 0
    rank = 10
    tailing = 6
    for p in peak_list:
        if p.intensity < minimum_intensity:
            p.rank = 0
            continue
        i += 1
        if i == 10 and rank != 0:
            if rank == 1:
                if tailing != 0:
                    i = 0
                    tailing -= 1
                else:
                    i = 0
                    rank -= 1
            else:
                i = 0
                rank -= 1
        if rank == 0:
            break
        p.rank = rank


def peptide_mass_rank(mass):
    unit = 121.6
    ratio = (mass / unit)
    if ratio < 9:
        return 1
    elif ratio < 13:
        return 2
    elif ratio < 17:
        return 3
    elif ratio < 20:
        return 4
    else:
        return 5


def glycan_peptide_ratio(glycan_mass, peptide_mass):
    ratio = (glycan_mass / peptide_mass)
    if ratio < 0.4:
        return 0
    elif 0.4 <= ratio < 0.8:
        return 1
    elif 0.8 <= ratio < 1.2:
        return 2
    elif 1.2 <= ratio < 1.6:
        return 3
    elif 1.6 <= ratio < 2.0:
        return 4
    elif ratio >= 2.0:
        return 5


class MatchedSpectrum(object):

    def __init__(self, gsm):
        self.peak_match_map = gsm.peak_match_map
        self.peak_list = list(gsm)
        self.scan_time = gsm.scan_time
        self.peaks_explained = gsm.peaks_explained
        self.peaks_unexplained = gsm.peaks_unexplained
        self.id = gsm.id
        self.glycopeptide_sequence = gsm.glycopeptide_sequence

    def __iter__(self):
        for o in self.peak_list:
            yield o

    def peak_explained_by(self, peak_id):
        explained = set()
        try:
            matches = self.peak_match_map[peak_id]
            for match in matches:
                explained.add(match["key"])
        except KeyError:
            pass
        return explained

    def __repr__(self):
        temp = "<MatchedSpectrum %s @ %d %d|%d>" % (
            self.glycopeptide_sequence, self.scan_time, self.peaks_explained,
            self.peaks_unexplained)
        return temp

    def reindex_peak_matches(self):
        peaks = sorted(self.peak_list, key=lambda x: x.neutral_mass)
        assigned = set()
        for original_index, matches in self.peak_match_map.items():
            if len(matches) == 0:
                continue
            match = matches[0]
            smallest_diff = float('inf')
            smallest_diff_peak = None
            for peak in peaks:
                if peak.id in assigned:
                    continue
                diff = abs(peak.neutral_mass - match.observed_mass)
                if diff < smallest_diff:
                    smallest_diff = diff
                    smallest_diff_peak = peak
                elif diff > smallest_diff:
                    smallest_diff_peak.id = original_index
                    assigned.add(smallest_diff_peak.id)
                    break


class solution(object):
    def __init__(self, peak, ppm_error):
        self.peak = peak
        self.ppm_error = ppm_error

    def __repr__(self):
        return "solution(%r, %e)" % (self.peak, self.ppm_error)


def binary_search(mass, peak_list, tolerance, low, high, solutions):
    if (high - low) < 3:
        for i in range(low, high + 1):
            error = ppm_error(peak_list[i], mass)
            if abs(error) <= tolerance:
                solutions.append(solution(peak_list[i], error))
        return solutions
    else:
        mid = int((low + high) / 2)
        if abs(ppm_error(peak_list[mid], mass)) <= tolerance:
            solutions.append(peak_list[mid])
            center = mid - 1
            while(True):
                error = ppm_error(peak_list[center], mass)
                if abs(error) <= tolerance:
                    solutions.append(solution(peak_list[center], error))
                    center -= 1
                else:
                    break
            center = mid + 1
            while(True):
                error = ppm_error(peak_list[center], mass)
                if abs(error) <= tolerance:
                    solutions.append(solution(peak_list[center], error))
                    center += 1
                else:
                    break
            return solutions
        elif peak_list[mid] > mass:
            return binary_search(mass, peak_list, tolerance, low, mid, solutions)
        elif peak_list[mid] < mass:
            return binary_search(mass, peak_list, tolerance, mid, high, solutions)
        else:
            raise ValueError("No Solutions?")


def do_binary_search(mass, peak_list, tolerance=2e-5):
    low = 0
    high = len(peak_list) - 1
    return binary_search(mass, peak_list, tolerance=tolerance, low=low, high=high, solutions=[])
