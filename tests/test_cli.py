import os
import json

import pytest

from click.testing import CliRunner

from glycopeptide_feature_learning import tool
from glycopeptide_feature_learning.multinomial_regression import MultinomialRegressionFit
from glycopeptide_feature_learning.partitions import partition_cell_spec, SplitModelFit

from .common import datafile


@pytest.mark.slow
def test_fit_model():
    training_data = datafile("MouseBrain-Z-T-5.mgf.gz")
    reference_model_data = datafile("reference_fit.json")

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

        with open(reference_model_data, 'rt') as fh:
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
