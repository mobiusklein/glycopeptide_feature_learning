import os
import sys
import glob
import json
import logging
import warnings
try:
    import cPickle as pickle
except ImportError:
    import pickle

try:
    from collections import Iterable
except ImportError:
    from collections.abc import Iterable

import click

import numpy as np

from glycopeptide_feature_learning import (
    data_source, peak_relations,
    common_features, partitions,
    multinomial_regression)

from glycopeptide_feature_learning.scoring import (
    PartialSplitScorerTree, SplitScorerTree)

import glypy
from glycopeptidepy.structure.fragment import IonSeries


def get_training_data(paths, blacklist_path=None, threshold=50.0):
    training_files = []
    for path in paths:
        training_files.extend(glob.glob(path))
    training_instances = []
    if blacklist_path is not None:
        with open(blacklist_path) as fh:
            blacklist = {line.strip() for line in fh}
    else:
        blacklist = set()

    seen = set()
    n_files = len(training_files)
    progbar = click.progressbar(
        training_files, length=n_files, show_eta=True, label='Loading Training Data',
        item_show_func=lambda x: os.path.basename(x) if x is not None else '')
    with progbar:
        for train_file in progbar:
            reader = data_source.AnnotatedMGFDeserializer(open(train_file, 'rb'))
            if progbar.is_hidden:
                click.echo("Reading %s" % (os.path.basename(train_file),))
            for instance in reader:
                if instance.annotations['ms2_score'] < threshold:
                    continue
                if instance.mass_shift.name not in ("Unmodified", "Ammonium"):
                    warnings.warn("Skipping mass shift %r" % (instance.mass_shift))
                    continue
                key = (instance.title, str(instance.structure))
                if key in seen:
                    continue
                if instance.title in blacklist:
                    continue
                seen.add(key)
                training_instances.append(instance)
    return training_instances


def match_spectra(matches, error_tolerance):
    progbar = click.progressbar(
        enumerate(matches), length=len(matches), show_eta=True, label='Matching Peaks',
        item_show_func=lambda x: "%d Spectra Matched" % (x[0],) if x is not None else '')
    with progbar:
        for i, match in progbar:
            match.match(error_tolerance=error_tolerance)
            if progbar.is_hidden and i % 1000 == 0 and i != 0:
                click.echo("%d Spectra Matched" % (i,))
    return matches


def partition_training_data(training_instances):
    partition_map = partitions.partition_observations(training_instances)
    return partition_map


def save_partitions(partition_map, output_directory):
    from glycan_profiling.output.text_format import AnnotatedMGFSerializer

    try:
        os.makedirs(output_directory)
    except OSError:
        pass

    for partition, cell in partition_map.items():
        click.echo("%s - %d" % (partition, len(cell.subset)))
        fields = partition.to_json()
        fields = sorted(fields.items())

        def format_field(field):
            if isinstance(field, basestring):
                return field
            elif isinstance(field, Iterable):
                return '-'.join(map(str, field))
            else:
                return str(field)

        fname = '_'.join(["%s_%s" % (k, format_field(v)) for k, v in fields])
        path = os.path.join(output_directory, fname + '.mgf')
        with AnnotatedMGFSerializer(open(path, 'wb')) as writer:
            for k, v in fields:
                writer.add_global_parameter(k, str(v))
            writer.add_global_parameter("total spectra", str(len(cell.subset)))
            for scan in cell.subset:
                writer.save(scan)


def get_peak_relation_features():
    features = {
        peak_relations.MassOffsetFeature(
            0.0, name='charge-diff'): lambda x: x.feature.from_charge != x.feature.to_charge,
        peak_relations.MassOffsetFeature(
            name='HexNAc', offset=glypy.monosaccharide_residues.HexNAc.mass()): lambda x: True,
    }

    stub_features = {
        peak_relations.MassOffsetFeature(
            name='Hex', offset=glypy.monosaccharide_residues.Hex.mass()): lambda x: True,
        peak_relations.MassOffsetFeature(
            0.0, name='charge-diff'): lambda x: x.feature.from_charge != x.feature.to_charge,
        peak_relations.MassOffsetFeature(
            name='HexNAc', offset=glypy.monosaccharide_residues.HexNAc.mass()): lambda x: True,
        peak_relations.MassOffsetFeature(
            name='Fuc', offset=glypy.monosaccharide_residues.Fuc.mass()): lambda x: True,
    }

    link_features = {}
    for link in common_features.amino_acid_blocks:
        feat = peak_relations.LinkFeature(link)
        link_features[feat] = lambda x: True
    return features, stub_features, link_features


def fit_peak_relation_features(partition_map):
    features, stub_features, link_features = get_peak_relation_features()
    group_to_fit = {}
    cell_sequence = [
        (spec, cell, partition_map.adjacent(spec, 10))
        for spec, cell in partition_map.items()]
    progressbar = click.progressbar(
        cell_sequence, label="Fitting Peak Relationships",
        width=15, show_percent=True, show_eta=False,
        item_show_func=lambda x: "%s" % (x[0].compact(' '),) if x is not None else '')
    with progressbar:
        for spec, cell, subset in progressbar:
            key = frozenset([gpsm.title for gpsm in subset])
            if key in group_to_fit:
                cell.fit = group_to_fit[key]
                continue
            if progressbar.is_hidden:
                click.echo("Fitting Peak Relationships for %s with %d observations" % (spec, len(subset)))
            # NOTE: The feature filters are not used, but are not necessary with the current
            # feature fitting algorithm. Future implementations using more feature classifications
            # might require them.
            for series in [IonSeries.b, IonSeries.y, ]:
                fm = peak_relations.FragmentationModel(series)
                fm.fit_offset(subset)
                for feature, _filt in features.items():
                    fits = fm.fit_feature(subset, feature)
                    fm.features.extend(fits)
                for feature, _filt in link_features.items():
                    fits = fm.fit_feature(subset, feature)
                    fm.features.extend(fits)
                cell.fit[series] = fm
            for series in [IonSeries.stub_glycopeptide]:
                fm = peak_relations.FragmentationModel(series)
                fm.fit_offset(subset)
                for feature, _filt in stub_features.items():
                    fits = fm.fit_feature(subset, feature)
                    fm.features.extend(fits)
                cell.fit[series] = fm
            group_to_fit[key] = cell.fit
    return group_to_fit


def fit_regression_model(partition_map, regression_model=None, use_mixture=True, **kwargs):
    if regression_model is None:
        regression_model = multinomial_regression.StubChargeModel
    model_fits = []
    for spec, cell in partition_map.items():
        click.echo("Fitting peak intensity model for %s with %d observations" % (spec, len(cell.subset)))
        _, fits = _fit_model_inner(spec, cell, regression_model, use_mixture=use_mixture, **kwargs)
        if fits:
            click.echo("Total Deviance: %f" % fits[0].deviance)
        for fit in fits:
            model_fits.append((spec, fit))
    return model_fits


def task_fn(args):
    spec, cell, regression_model = args
    return _fit_model_inner(spec, cell, regression_model)


def _fit_model_inner(spec, cell, regression_model, use_mixture=True, **kwargs):
    fm = peak_relations.FragmentationModelCollection(cell.fit)
    try:
        fit = regression_model.fit_regression(
            cell.subset, reliability_model=fm, base_reliability=0.5, **kwargs)
        if np.isinf(fit.estimate_dispersion()):
            click.echo("Infinite dispersion, refitting without per-fragment weights")
            fit = regression_model.fit_regression(
                cell.subset, reliability_model=None, **kwargs)
    except ValueError as ex:
        click.echo("%r, refitting without per-fragment weights" % (ex, ))
        try:
            fit = regression_model.fit_regression(
                cell.subset, reliability_model=None)
        except Exception as err:
            click.echo("Failed to fit model with error: %r" % (err, ))
            return (spec, [])
    fit.reliability_model = fm
    fits = [fit]
    if use_mixture:
        mismatches = []
        for case in cell.subset:
            r = fit.calculate_correlation(case)
            if r < 0.5:
                mismatches.append(case)
        if mismatches:
            try:
                click.echo("Fitting Mismatch Model with %d cases" % len(mismatches))
                try:
                    mismatch_fit = regression_model.fit_regression(
                        mismatches, reliability_model=fm, base_reliability=0.5, **kwargs)
                    if np.isinf(mismatch_fit.estimate_dispersion()):
                        mismatch_fit = regression_model.fit_regression(
                            mismatches, reliability_model=None, **kwargs)
                except ValueError:
                    mismatch_fit = regression_model.fit_regression(
                        mismatches, reliability_model=None, **kwargs)
                mismatch_fit.reliability_model = fm
                fits.append(mismatch_fit)
            except Exception as err:
                click.echo("Failed to fit mismatch model with error: %r" % (err, ))
    return (spec, fits)


@click.command('fit-glycopeptide-regression-model', short_help="Fit glycopeptide fragmentation model")
@click.argument('paths', metavar='PATH', type=click.Path(exists=True, dir_okay=False), nargs=-1)
@click.option('-t', '--threshold', type=float, default=50.0)
@click.option('--blacklist-path', type=click.Path(exists=True, dir_okay=False), default=None)
@click.option('-o', '--output-path', type=click.Path())
@click.option('-m', '--error-tolerance', type=float, default=2e-5)
@click.option("--debug", is_flag=True, default=False)
def main(paths, threshold=50.0, output_path=None, blacklist_path=None, error_tolerance=2e-5, debug=False):
    if debug:
        logger = logging.getLogger()
        logger.setLevel("DEBUG")
        logger.addHandler(logging.StreamHandler(sys.stdout))
    click.echo("Loading data from %s" % (', '.join(paths)))
    training_instances = get_training_data(paths, blacklist_path, threshold)
    match_spectra(training_instances, error_tolerance)
    click.echo("Partitioning %d instances" % (len(training_instances), ))
    partition_map = partition_training_data(training_instances)
    click.echo("Fitting Peak Relation Features")
    fit_peak_relation_features(partition_map)
    click.echo("Fitting Peak Intensity Regression")
    model_fits = fit_regression_model(partition_map, trace=debug)
    click.echo("Writing Models To %s" % (output_path,))
    export = []
    for spec, fit in model_fits:
        export.append((spec.to_json(), fit.to_json(False)))
    with open(output_path, 'wt') as fh:
        json.dump(export, fh, sort_keys=1, indent=2)


@click.command("partition-glycopeptide-training-data", short_help="Pre-separate training data along partitions")
@click.option('-t', '--threshold', type=float, default=0.0)
@click.argument('paths', metavar='PATH', type=click.Path(exists=True, dir_okay=False), nargs=-1)
@click.argument('outdir', metavar='OUTDIR', type=click.Path(dir_okay=True, file_okay=False), nargs=1)
def partition_glycopeptide_training_data(paths, outdir, threshold=50.0, output_path=None, blacklist_path=None, error_tolerance=2e-5):
    click.echo("Loading data from %s" % (', '.join(paths)))
    training_instances = get_training_data(paths, blacklist_path, threshold)
    click.echo("Partitioning %d instances" % (len(training_instances), ))
    partition_map = partition_training_data(training_instances)
    save_partitions(partition_map, outdir)


@click.command("strip-model", short_help="Strip out extra arrays from serialized model JSON")
@click.argument("inpath", type=click.Path(exists=True, dir_okay=False))
@click.argument("outpath", type=click.Path(dir_okay=False, writable=True))
def strip_model_arrays(inpath, outpath):
    model_tree = PartialSplitScorerTree.from_file(inpath)
    d = model_tree.to_json()
    with click.open_file(outpath, "wt") as fh:
        json.dump(d, fh)


@click.command("compile-model", short_help="Compile a model into a Python-loadable file.")
@click.argument("inpath", type=click.Path(exists=True, dir_okay=False))
@click.argument("outpath", type=click.Path(dir_okay=False, writable=True))
@click.option("-m", "--model-type", type=click.Choice(["partial-peptide", "full"]), default='partial-peptide')
def compile_model(inpath, outpath, model_type="partial-peptide"):
    model_cls = {
        "partial-peptide": PartialSplitScorerTree,
        "full": SplitScorerTree
    }[model_type]
    click.echo("Loading Model", err=True)
    model_tree = model_cls.from_file(inpath)
    click.echo("Packing Model", err=True)
    for node in model_tree:
        node.compact()
    click.echo("Saving Model", err=True)
    stream = click.open_file(outpath, 'wb')
    pickle.dump(model_tree, stream, 2)
    stream.close()


CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])


@click.group(context_settings=CONTEXT_SETTINGS)
def cli():
    pass


cli.add_command(main, "fit-model")
cli.add_command(partition_glycopeptide_training_data, "partition-samples")
cli.add_command(strip_model_arrays, "strip-model")
cli.add_command(compile_model, "compile-model")


def info(type, value, tb):
    if hasattr(sys, 'ps1') or not sys.stderr.isatty():
        sys.__excepthook__(type, value, tb)
    else:
        import ipdb, traceback
        traceback.print_exception(type, value, tb)
        ipdb.post_mortem(tb)


sys.excepthook = info


if __name__ == "__main__":
    cli.main()
