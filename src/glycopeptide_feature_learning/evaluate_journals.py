try:
    import cPickle as pickle
except ImportError:
    import pickle

import ms_deisotope
from ms_deisotope.output import ProcessedMzMLDeserializer

from glycresoft.task import TaskBase
from glycresoft import serialize
from glycresoft.structure import FragmentCachingGlycopeptide, DecoyFragmentCachingGlycopeptide
from glycresoft.tandem.glycopeptide.dynamic_generation.journal import (
    JournalSetLoader, JournalFileWriter, JournalFileReader, SolutionSetGrouper)


class JournalEvaluator(serialize.DatabaseBoundOperation):
    def __init__(self, database_connection, sample_loader, model, analysis_id=1):
        super(JournalEvaluator, self).__init__(database_connection)
        self.model = model
        self.sample_loader = sample_loader
        self.analysis_id = analysis_id

    def iterjournals(self):
        analysis = self.session.query(serialize.Analysis).get(self.analysis_id)
        file_list = analysis.files
        mass_shifts = {m.name: m for m in analysis.parameters['mass_shifts']}
        for journal_file in file_list:
            reader = JournalFileReader(
                journal_file.open(), mass_shift_map=mass_shifts, scan_loader=self.sample_loader)
            for gpsm in reader:
                yield gpsm

    def load_groups(self):
        analysis = self.session.query(serialize.Analysis).get(self.analysis_id)
        file_list = analysis.files
        mass_shifts = {m.name: m for m in analysis.parameters['mass_shifts']}
        loader = JournalSetLoader(
            [f.open() for f in file_list], self.sample_loader, mass_shift_map=mass_shifts)
        return loader.load()
