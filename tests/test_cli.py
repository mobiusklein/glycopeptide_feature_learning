import os
import json
import pickle

import pytest

import numpy as np

from click.testing import CliRunner

from glycopeptide_feature_learning import tool
from glycopeptide_feature_learning.multinomial_regression import MultinomialRegressionFit
from glycopeptide_feature_learning.partitions import partition_cell_spec, SplitModelFit
from glycopeptide_feature_learning.scoring.scorer import NoGlycosylatedPeptidePartitionedPredicateTree

from .common import datafile


@pytest.mark.slow
def test_fit_model():
    training_data = datafile("MouseBrain-Z-T-5.mgf.gz")
    reference_model_data = datafile("reference_fit.json.gz")

    runner = CliRunner()
    with runner.isolated_filesystem():
        result = runner.invoke(tool.cli, [
            "fit-model",
            "-P",
            "-t", "20",
            "-o", "model.json",
            "-b",
            "-M", "LabileMonosaccharideAwareModel",
            training_data,
        ])
        assert result.exit_code == 0
        with open("model.json", 'rt') as fh:
            model_fit_state = json.load(fh)
        meta = model_fit_state['metadata']
        model_fits = model_fit_state['models']
        omit_labile = meta.get('omit_labile', False)
        assert omit_labile
        assert meta['fit_partitioned']
        assert meta['fit_info']['spectrum_count'] == 5523
        assert len(model_fits) == 46

        with tool.get_opener(reference_model_data, 'rt') as fh:
            reference_fit_state = json.load(fh)
        assert reference_fit_state['metadata'] == meta

        submodel_parts = {
            partition_cell_spec.from_json(x[0]): SplitModelFit.from_json(x[1])
            for x in model_fits
        }


        expected_submodel_parts = {
            partition_cell_spec.from_json(x[0]): SplitModelFit.from_json(x[1])
            for x in reference_fit_state['models']

        }

        assert set(submodel_parts) == set(expected_submodel_parts)

        for key, submodel in submodel_parts.items():
            expected_submodel = expected_submodel_parts[key]
            assert submodel == expected_submodel


def test_compile_model():
    reference_model_data = datafile("reference_fit.json.gz")

    runner = CliRunner()
    with runner.isolated_filesystem():
        result = runner.invoke(tool.cli, [
            "compile-model",
            "-m", "no-glycosylated-partitioned-glycan",
            reference_model_data,
            "compiled-model.pkl"
        ])
        assert result.exit_code == 0

        with open("compiled-model.pkl", 'rb') as fh:
            model_tree = pickle.load(fh)

        assert isinstance(
            model_tree,
            NoGlycosylatedPeptidePartitionedPredicateTree)

        with open(datafile("reference_compiled.pkl"), 'rb') as fh:
            expected_tree = pickle.load(fh)

        assert isinstance(
            expected_tree,
            NoGlycosylatedPeptidePartitionedPredicateTree)

        assert model_tree == expected_tree


def test_correlation():
    training_data = datafile("MouseBrain-Z-T-5.mgf.gz")
    reference_model = datafile("reference_compiled.pkl")

    runner = CliRunner()

    with runner.isolated_filesystem():

        result = runner.invoke(tool.cli, [
            "calculate-correlation",
            training_data,
            "metrics.pkl",
            reference_model,
        ])

        assert result.exit_code == 0

        with open('metrics.pkl', 'rb') as fh:
            metrics = pickle.load(fh)

        with open(datafile("reference_metrics.pkl"), 'rb') as fh:
            expected_metrics = pickle.load(fh)

        assert metrics.keys() == expected_metrics.keys()

        for metric_name, values in metrics.items():
            if metric_name not in expected_metrics:
                continue
            expected_values = expected_metrics[metric_name]
            if values.dtype.kind != 'f':
                assert np.all(values == expected_values), f"{metric_name} does not match"
            else:
                assert np.allclose(
                    values,
                    expected_values,
                    equal_nan=True), f"{metric_name} does not match"
