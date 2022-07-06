cimport cython

from cpython.object cimport PyObject
from cpython.dict cimport PyDict_GetItem, PyDict_SetItem

from glycan_profiling._c.tandem.spectrum_match cimport ScoreSet


DEF NUM_SCORES = 11


cdef class ModelScoreSet(ScoreSet):
    def __init__(self, glycopeptide_score=0., peptide_score=0., glycan_score=0., glycan_coverage=0.,
                 stub_glycopeptide_intensity_utilization=0., oxonium_ion_intensity_utilization=0.,
                 n_stub_glycopeptide_matches=0, peptide_coverage=0.0, peptide_correlation=0.0, peptide_backbone_count=0,
                 glycan_correlation=0.0):
        self.glycopeptide_score = glycopeptide_score
        self.peptide_score = peptide_score
        self.glycan_score = glycan_score
        self.glycan_coverage = glycan_coverage

        self.stub_glycopeptide_intensity_utilization = stub_glycopeptide_intensity_utilization
        self.oxonium_ion_intensity_utilization = oxonium_ion_intensity_utilization
        self.n_stub_glycopeptide_matches = n_stub_glycopeptide_matches

        self.peptide_coverage = peptide_coverage

        self.peptide_correlation = peptide_correlation
        self.peptide_backbone_count = peptide_backbone_count
        self.glycan_correlation = glycan_correlation

    cpdef bytearray pack(self):
        cdef:
            float[NUM_SCORES] data
        data[0] = self.glycopeptide_score
        data[1] = self.peptide_score
        data[2] = self.glycan_score
        data[3] = self.glycan_coverage
        data[4] = self.stub_glycopeptide_intensity_utilization
        data[5] = self.oxonium_ion_intensity_utilization
        data[6] = <int>self.n_stub_glycopeptide_matches
        data[7] = self.peptide_coverage
        data[8] = self.peptide_correlation
        data[9] = <int>self.peptide_backbone_count
        data[10] = self.glycan_correlation
        return ((<char*>data)[:sizeof(float) * NUM_SCORES])

    @staticmethod
    def unpack(bytearray data):
        cdef:
            float* buff
            char* temp
            int n_stub_glycopeptide_matches
            float stub_utilization
            float oxonium_utilization
            float peptide_coverage
            float peptide_correlation
            int peptide_backbone_count
            float glycan_correlation
        temp = data
        buff = <float*>(temp)
        stub_utilization = buff[4]
        oxonium_utilization = buff[5]
        n_stub_glycopeptide_matches = <int>buff[6]
        peptide_coverage = buff[7]
        peptide_correlation = buff[8]
        peptide_backbone_count = <int>buff[9]
        glycan_correlation = buff[10]
        return ModelScoreSet._create_model_score_set(
            buff[0], buff[1], buff[2], buff[3], stub_utilization,
            oxonium_utilization, n_stub_glycopeptide_matches, peptide_coverage,
            peptide_correlation, peptide_backbone_count, glycan_correlation)

    @staticmethod
    cdef ModelScoreSet _create_model_score_set(float glycopeptide_score, float peptide_score, float glycan_score, float glycan_coverage,
                                               float stub_glycopeptide_intensity_utilization, float oxonium_ion_intensity_utilization,
                                               int n_stub_glycopeptide_matches, float peptide_coverage, float peptide_correlation,
                                               int peptide_backbone_count, float glycan_correlation):
        cdef:
            ModelScoreSet self
        self = ModelScoreSet.__new__(ModelScoreSet)
        self.glycopeptide_score = glycopeptide_score
        self.peptide_score = peptide_score
        self.glycan_score = glycan_score
        self.glycan_coverage = glycan_coverage
        self.stub_glycopeptide_intensity_utilization = stub_glycopeptide_intensity_utilization
        self.oxonium_ion_intensity_utilization = oxonium_ion_intensity_utilization
        self.n_stub_glycopeptide_matches = n_stub_glycopeptide_matches
        self.peptide_coverage = peptide_coverage
        self.peptide_correlation = peptide_correlation
        self.peptide_backbone_count = peptide_backbone_count
        self.glycan_correlation = glycan_correlation
        return self

    def __repr__(self):
        template = (
            "{self.__class__.__name__}({self.glycopeptide_score}, {self.peptide_score},"
            " {self.glycan_score}, {self.glycan_coverage}, {self.stub_glycopeptide_intensity_utilization},"
            " {self.oxonium_ion_intensity_utilization}, {self.n_stub_glycopeptide_matches}, {self.peptide_coverage},"
            " {self.peptide_correlation}, {self.peptide_backbone_count}, {self.glycan_correlation}"
            ")")
        return template.format(self=self)

    def __len__(self):
        return 7

    def __getitem__(self, int i):
        if i == 0:
            return self.glycopeptide_score
        elif i == 1:
            return self.peptide_score
        elif i == 2:
            return self.glycan_score
        elif i == 3:
            return self.glycan_coverage
        elif i == 4:
            return self.stub_glycopeptide_intensity_utilization
        elif i == 5:
            return self.oxonium_ion_intensity_utilization
        elif i == 6:
            return self.n_stub_glycopeptide_matches
        elif i == 7:
            return self.peptide_coverage
        elif i == 8:
            return self.peptide_correlation
        elif i == 9:
            return self.peptide_backbone_count
        elif i == 10:
            return self.glycan_correlation
        else:
            raise IndexError(i)

    def __iter__(self):
        yield self.glycopeptide_score
        yield self.peptide_score
        yield self.glycan_score
        yield self.glycan_coverage
        yield self.stub_glycopeptide_intensity_utilization
        yield self.oxonium_ion_intensity_utilization
        yield self.n_stub_glycopeptide_matches
        yield self.peptide_coverage
        yield self.peptide_correlation
        yield self.peptide_backbone_count
        yield self.glycan_correlation

    def __reduce__(self):
        return self.__class__, (self.glycopeptide_score, self.peptide_score,
                                self.glycan_score, self.glycan_coverage, self.stub_glycopeptide_intensity_utilization,
                                self.oxonium_ion_intensity_utilization, self.n_stub_glycopeptide_matches,
                                self.peptide_coverage, self.peptide_correlation, self.peptide_backbone_count,
                                self.glycan_correlation)

    @classmethod
    def from_spectrum_matcher(cls, match):
        stub_utilization, stub_count, peptide_count = match.count_peptide_Y_ion_utilization()
        oxonium_utilization = match.oxonium_ion_utilization()

        return cls(match.score,
            match.peptide_score(),
            match.glycan_score(),
            match.glycan_coverage(),
            stub_utilization,
            oxonium_utilization,
            stub_count,
            match.peptide_coverage(),
            match.peptide_correlation(),
            peptide_count,
            match.glycan_correlation()
        )

    @classmethod
    def field_names(cls):
        cdef list field_names = super(ModelScoreSet, cls).field_names()
        field_names.extend([
            "peptide_correlation",
            "peptide_backbone_count",
            "glycan_correlation",
        ])
        return field_names

    cpdef list values(self):
        cdef list values = super(ModelScoreSet, self).values()
        values.append(self.peptide_correlation)
        values.append(self.peptide_backbone_count)
        values.append(self.glycan_correlation)
        return values