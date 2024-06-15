import pickle

from ms_deisotope.data_source import get_opener

from glycresoft.tandem.spectrum_match import MultiScoreSpectrumMatch

from glycopeptide_feature_learning.data_source import AnnotatedMGFDeserializer
from glycopeptide_feature_learning.scoring.base import ModelScoreSet

from .common import datafile

def test_spectrum_match():
    mgf_file = datafile("MouseBrain-Z-T-5.mgf.gz")
    reference_model_file = datafile("reference_compiled.pkl")

    with open(reference_model_file, 'rb') as fh:
        model_tree = pickle.load(fh)

    scan_reader = AnnotatedMGFDeserializer(get_opener(mgf_file))

    scan = scan_reader[0]
    match = model_tree.evaluate(scan, scan.target)
    sm = MultiScoreSpectrumMatch.from_match_solution(match)

    assert isinstance(sm, MultiScoreSpectrumMatch)
    assert isinstance(sm.score_set, ModelScoreSet)

    score_set = sm.score_set
    assert score_set.peptide_backbone_count == 4
    assert score_set.n_stub_glycopeptide_matches == 4
    assert score_set.peptide_coverage == 0.5
    assert score_set.partition_label == 0

    scan = scan_reader[3000]
    match = model_tree.evaluate(scan, scan.target)
    sm = MultiScoreSpectrumMatch.from_match_solution(match)

    assert isinstance(sm, MultiScoreSpectrumMatch)
    assert isinstance(sm.score_set, ModelScoreSet)

    score_set = sm.score_set
    assert score_set.peptide_backbone_count == 30
    assert score_set.n_stub_glycopeptide_matches == 13
    assert score_set.partition_label == 10

    assert score_set.to_dict() == ModelScoreSet.unpack(score_set.pack()).to_dict()