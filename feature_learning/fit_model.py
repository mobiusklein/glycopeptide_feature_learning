import os
import glob
import json
import multiprocessing

import click

import numpy as np

from feature_learning import (
    data_source, peak_relations,
    _common_features, partitions,
    multinomial_regression)

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
                if instance.mass_shift not in ("Unmodified", "Ammonium"):
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
            if progbar.is_hidden and (i + 1) % 1000 == 0:
                click.echo("%d Spectra Matched" % (i,))
    return matches


def partition_training_data(training_instances):
    partition_map = partitions.partition_observations(training_instances)
    return partition_map


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
    for link in _common_features.amino_acid_blocks:
        feat = peak_relations.LinkFeature(link)
        link_features[feat] = lambda x: True
    return features, stub_features, link_features


def fit_peak_relation_features(partition_map):
    features, stub_features, link_features = get_peak_relation_features()
    group_to_fit = {}
    for spec, cell in partition_map.items():
        subset = partition_map.adjacent(spec, 10)
        key = frozenset([gpsm.title for gpsm in subset])
        if key in group_to_fit:
            cell.fit = group_to_fit[key]
            continue
        click.echo("%s %d" % (spec, len(subset)))
        for series in [IonSeries.b, IonSeries.y, ]:
            fm = peak_relations.FragmentationModel(series)
            fm.fit_offset(subset)
            for feature, filt in features.items():
                fits = fm.fit_feature(subset, feature)
                fm.features.extend(fits)
            for feature, filt in link_features.items():
                fits = fm.fit_feature(subset, feature)
                fm.features.extend(fits)
            cell.fit[series] = fm
        for series in [IonSeries.stub_glycopeptide]:
            fm = peak_relations.FragmentationModel(series)
            fm.fit_offset(subset)
            for feature, filt in stub_features.items():
                fits = fm.fit_feature(subset, feature)
                fm.features.extend(fits)
            cell.fit[series] = fm
        group_to_fit[key] = cell.fit
    return group_to_fit


def fit_regression_model(partition_map, regression_model=None, use_mixture=True, n_processes=1):
    if regression_model is None:
        regression_model = multinomial_regression.StubChargeModel
    model_fits = []
    if n_processes == 1:
        for spec, cell in partition_map.items():
            click.echo("%s %d" % (spec, len(cell.subset)))
            _, fits = _fit_model_inner(spec, cell, regression_model, use_mixture=use_mixture)
            click.echo(fits[0].deviance)
            for fit in fits:
                model_fits.append((spec, fit))
    else:
        pool = multiprocessing.Pool(n_processes)
        workload = (tuple(kv) + (regression_model,) for kv in partition_map.items())
        for spec, fits in pool.map(task_fn, workload):
            click.echo("%s %d" % (spec, len(fits[0].weights)))
            click.echo(fits[0].deviance)
            for fit in fits:
                model_fits.append((spec, fit))
    return model_fits


def task_fn(args):
    spec, cell, regression_model = args
    return _fit_model_inner(spec, cell, regression_model)


def _fit_model_inner(spec, cell, regression_model, use_mixture=True):
    fm = peak_relations.FragmentationModelCollection(cell.fit)
    try:
        fit = regression_model.fit_regression(
            cell.subset, reliability_model=fm, base_reliability=0.5)
        if np.isinf(fit.estimate_dispersion()):
            fit = regression_model.fit_regression(
                cell.subset, reliability_model=None)
    except ValueError:
        fit = regression_model.fit_regression(
            cell.subset, reliability_model=None)
    fits = [fit]
    if use_mixture:
        mismatches = []
        for case in cell.subset:
            r = fit.calculate_correlation(case)
            if r < 0.5:
                mismatches.append(case)
        if mismatches:
            click.echo("Fitting Mismatch Model with %d cases" % len(mismatches))
            try:
                mismatch_fit = regression_model.fit_regression(
                    mismatches, reliability_model=fm, base_reliability=0.5)
                if np.isinf(mismatch_fit.estimate_dispersion()):
                    mismatch_fit = regression_model.fit_regression(
                        mismatches, reliability_model=None)
            except ValueError:
                mismatch_fit = regression_model.fit_regression(
                    mismatches, reliability_model=None)
            fits.append(mismatch_fit)
    return (spec, fits)


@click.command('fit-glycopeptide-regression-model')
@click.argument('paths', metavar='PATH', type=click.Path(exists=True, dir_okay=False), nargs=-1)
@click.option('-t', '--threshold', type=float, default=50.0)
@click.option('--blacklist-path', type=click.Path(exists=True, dir_okay=False), default=None)
@click.option('-o', '--output-path', type=click.Path())
@click.option('-m', '--error-tolerance', type=float, default=2e-5)
def main(paths, threshold=50.0, output_path=None, blacklist_path=None, error_tolerance=2e-5):
    click.echo("Loading data from %s" % (', '.join(paths)))
    training_instances = get_training_data(paths, blacklist_path, threshold)
    match_spectra(training_instances, error_tolerance)
    click.echo("Partitioning %d instances" % (len(training_instances), ))
    partition_map = partition_training_data(training_instances)
    click.echo("Fitting Peak Relation Features")
    fit_peak_relation_features(partition_map)
    click.echo("Fitting Peak Intensity Regression")
    model_fits = fit_regression_model(partition_map)
    click.echo("Writing Models To %s" % (output_path,))
    export = []
    for spec, fit in model_fits:
        export.append((spec.to_json(), fit.to_json(False)))
    with open(output_path, 'wt') as fh:
        json.dump(export, fh, sort_keys=1, indent=2)


if __name__ == '__main__':
    main()
