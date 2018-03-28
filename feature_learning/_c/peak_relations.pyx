cimport cython

import numpy as np
cimport numpy as np

np.import_array()

from ms_deisotope._c.peak_set cimport DeconvolutedPeak, DeconvolutedPeakSet


cdef int OUT_OF_RANGE_INT = 999


@cython.cdivision(True)
cdef int intensity_ratio_function(DeconvolutedPeak peak1, DeconvolutedPeak peak2):
    cdef double ratio
    ratio = peak1.intensity / (peak2.intensity)
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


cdef class FeatureBase(object):
    cdef:
        public str name
        public double tolerance
        public double intensity_ratio
        public int from_charge
        public int to_charge
        public object feature_type
        public object terminal

    cpdef list find_matches(self, DeconvolutedPeak peak, DeconvolutedPeakSet peak_list, object structure=None):
        raise NotImplementedError()

    def __init__(self, tolerance=2e-5, name=None, intensity_ratio=OUT_OF_RANGE_INT,
                 from_charge=OUT_OF_RANGE_INT, to_charge=OUT_OF_RANGE_INT, feature_type='',
                 terminal=''):
        self.name = str(name)
        self.tolerance = tolerance
        self.intensity_ratio = intensity_ratio
        self.from_charge = from_charge
        self.to_charge = to_charge
        self.feature_type = feature_type
        self.terminal = terminal

    def __eq__(self, other):
        v = self.intensity_ratio == other.intensity_ratio
        if not v:
            return v
        v = self.from_charge == other.from_charge
        if not v:
            return v
        v = self.to_charge == other.to_charge
        if not v:
            return v
        v = self.feature_type == other.feature_type
        if not v:
            return v
        # v = self.terminal == other.terminal
        # if not v:
        #     return v
        return True

    def __ne__(self, other):
        return not (self == other)

    def is_valid_match(self, from_peak, to_peak, solution_map, structure=None):
        return to_peak in solution_map.by_peak

    def specialize(self, from_charge, to_charge, intensity_ratio):
        raise NotImplementedError()

    def __call__(self, peak1, peak2, structure=None):
        raise NotImplementedError()

    def to_json(self):
        d = {}
        d['name'] = self.name
        d['tolerance'] = self.tolerance
        d['intensity_ratio'] = self.intensity_ratio
        d['from_charge'] = self.from_charge
        d['to_charge'] = self.to_charge
        d['feature_type'] = self.feature_type
        d['terminal'] = self.terminal
        return d

    @classmethod
    def from_json(cls, d):
        # feature_type = d['feature_type']
        # if feature_type == LinkFeature.feature_type:
        #     return LinkFeature.from_json(d)
        # else:
        #     return MassOffsetFeature.from_json(d)
        raise NotImplementedError()


cdef class MassOffsetFeature(FeatureBase):
    
    cdef:
        public double offset
        public long _hash


    @cython.cdivision(True)
    cpdef bint test(self, DeconvolutedPeak peak1, DeconvolutedPeak peak2):
        if (self.intensity_ratio == OUT_OF_RANGE_INT or
            intensity_ratio_function(peak1, peak2) == self.intensity_ratio) and\
           ((self.from_charge == OUT_OF_RANGE_INT and self.to_charge == OUT_OF_RANGE_INT) or
                (self.from_charge == peak1.charge and self.to_charge == peak2.charge)):

            return abs((peak1.neutral_mass + self.offset - peak2.neutral_mass) / peak2.neutral_mass) <= self.tolerance
        return False

    cpdef list find_matches(self, DeconvolutedPeak peak, DeconvolutedPeakSet peak_list, object structure=None):
        cdef:
            list matches
            tuple peaks_in_range
            size_t i, n
        matches = []
        peaks_in_range = peak_list.all_peaks_for(peak.neutral_mass + self.offset, self.tolerance)
        n = len(peaks_in_range)
        for i in range(n):
            peak2 = <DeconvolutedPeak>(peaks_in_range[i])
            if peak is not peak2 and self.test(peak, peak2):
                matches.append(peak2)
        return matches

    def __init__(self, offset, tolerance=2e-5, name=None, intensity_ratio=OUT_OF_RANGE_INT,
                 from_charge=OUT_OF_RANGE_INT, to_charge=OUT_OF_RANGE_INT, feature_type='',
                 terminal=''):
        if name is None:
            name = "F:" + str(offset)

        super(MassOffsetFeature, self).__init__(
            tolerance, name, intensity_ratio, from_charge, to_charge, feature_type,
            terminal)

        self.offset = offset
        self._hash = hash((self.offset, self.intensity_ratio, self.from_charge,
                           self.to_charge))

    def __eq__(self, other):
        v = np.isclose(self.offset, other.offset)
        if not v:
            return v
        return super(MassOffsetFeature, self).__eq__(other)

    def __hash__(self):
        return self._hash

    def __call__(self, peak1, peak2, structure=None):
        return self.test(peak1, peak2)

    def specialize(self, from_charge, to_charge, intensity_ratio):
        return self.__class__(
            self.offset, self.tolerance, self.name, intensity_ratio,
            from_charge, to_charge, self.feature_type, self.terminal)

    def _get_display_fields(self):
        fields = {}
        # fields['feature_type'] = self.feature_type
        fields['offset'] = self.offset
        if self.from_charge != OUT_OF_RANGE_INT:
            fields["from_charge"] = self.from_charge
        if self.to_charge != OUT_OF_RANGE_INT:
            fields['to_charge'] = self.to_charge
        if self.intensity_ratio != OUT_OF_RANGE_INT:
            fields["intensity_ratio"] = self.intensity_ratio
        terms = []
        for k, v in fields.items():
            if isinstance(v, int):
                terms.append("%s=%d" % (k, v))
            elif isinstance(v, float):
                terms.append("%s=%0.4f" % (k, v))
            else:
                terms.append("%s=%r" % (k, v))
        return terms

    def __repr__(self):
        terms = self._get_display_fields()
        return "{}(name={!r}, {})".format(
            self.__class__.__name__, self.name, ", ".join(terms))

    def to_json(self):
        d = super(MassOffsetFeature, self).to_json()
        d['offset'] = self.offset
        return d

    @classmethod
    def from_json(cls, d):
        inst = cls(
            d['offset'], d['tolerance'], d['name'], d['intensity_ratio'],
            d['from_charge'], d['to_charge'], d['feature_type'], d['terminal'])
        return inst