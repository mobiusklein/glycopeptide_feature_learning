import glob
import json

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
    for train_file in training_files:
        reader = data_source.AnnotatedMGFDeserializer(open(train_file, 'rb'))
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
        if spec.glycan_type != 'N-Linked':
            continue
        subset = partition_map.adjacent(spec, 10)
        key = frozenset([gpsm.title for gpsm in subset])
        if key in group_to_fit:
            cell.fit = group_to_fit[key]
            continue
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
            for feature, filt in features.items():
                fits = fm.fit_feature(subset, feature)
                fm.features.extend(fits)
            for feature, filt in stub_features.items():
                fits = fm.fit_feature(subset, feature)
                fm.features.extend(fits)
            cell.fit[series] = fm
        group_to_fit[key] = cell.fit
    return group_to_fit


def fit_regression_model(partition_map, regression_model=None):
    if regression_model is None:
        regression_model = multinomial_regression.CleavageSiteCenterDistanceModel
    model_fits = []
    for spec, cell in partition_map.items():
        print(spec, len(cell.subset))
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
        print(fit.deviance)
        model_fits.append((spec, fit))
    return model_fits


@click.command('fit-glycopeptide-regression-model')
@click.argument('paths', metavar='PATH', type=click.Path(exists=True, dir_okay=False), nargs=-1)
@click.option('-t', '--threshold', type=float, default=50.0)
@click.option('--blacklist-path', type=click.Path(exists=True, dir_okay=False), default=None)
@click.option('-o', '--output-path', type=click.Path())
@click.option('-m', '--error-tolerance', type=float, default=2e-5)
def main(paths, threshold=50.0, output_path=None, blacklist_path=None, error_tolerance=2e-5):
    click.echo("Loading data from %s" % (', '.join(paths)))
    training_instances = get_training_data(paths, blacklist_path, threshold)
    click.echo("Matching peaks")
    for i, match in enumerate(training_instances):
        if i % 1000 == 0:
            click.echo("%d matches calculated" % (i, ))
        match.match(error_tolerance=error_tolerance)
    click.echo("Partitioning %d instances" % (len(training_instances), ))
    partition_map = partition_training_data(training_instances)
    click.echo("Fitting Peak Relation Features")
    fit_peak_relation_features(partition_map)
    click.echo("Fitting Peak Intensity Regression")
    model_fits = fit_regression_model(partition_map)
    export = []
    for spec, fit in model_fits:
        export.append((spec.to_json(), fit.to_json(False)))
    with open(output_path, 'wt') as fh:
        json.dump(export, fh, sort_keys=1, indent=2)


if __name__ == '__main__':
    main()
