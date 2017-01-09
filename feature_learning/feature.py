from glycopeptidepy import Composition
from glypy import MonosaccharideResidue, monosaccharides

from .peak_relations import (
    feature_function_estimator,
    FittedFeature, MassOffsetFeature)

from .common import (OUT_OF_RANGE_INT, log, intensity_ratio_function, ppm_error)


def fit_features(training_set, features, kinds=('b', 'y')):
    fitted_features = []
    for feat in features:
        log("Fitting %r", feat)
        for kind in kinds:
            log("for %s", kind)
            u, v, rels = feature_function_estimator(training_set, feat, kind)
            fit_feat = FittedFeature(feat, kind, u, v, rels)
            log("fit: %r", fit_feat)
            fitted_features.append(fit_feat)
    return fitted_features


def specialize_features(fitted_features):
    specialized = []
    for feat in fitted_features:
        for charge_pairs_intensity_ratio, count in feat.charge_intensity_ratio().items():
            if count < 3:
                continue
            charge_pairs, intensity_ratio = charge_pairs_intensity_ratio
            from_charge, to_charge = charge_pairs
            offset = feat.feature.offset
            name = feat.feature.name + " %d->%d / %d" % (from_charge, to_charge, intensity_ratio)
            f = MassOffsetFeature(
                offset=offset, name=name, tolerance=2e-5,
                from_charge=from_charge, to_charge=to_charge,
                intensity_ratio=intensity_ratio)
            specialized.append(f)
    return specialized


shifts = [
    MassOffsetFeature(
        -Composition("NH3").mass, name='Neutral-Loss-Ammonia',
        tolerance=2e-5, feature_type='neutral_loss'),
    MassOffsetFeature(
        -Composition("H2O").mass, name='Neutral-Loss-Water',
        tolerance=2e-5, feature_type='neutral_loss'),
    MassOffsetFeature(
        0.0, name="Charge-Increased Species", from_charge=1,
        to_charge=2, tolerance=2e-5, feature_type='support_relation'),
    MassOffsetFeature(
        0.0, name="Charge-Decreased Species", from_charge=2,
        to_charge=1, tolerance=2e-5, feature_type='support_relation'),
]

for monosacch_name in ['Hex', 'HexNAc', 'Fuc', 'NeuAc']:
    m = MonosaccharideResidue.from_monosaccharide(monosaccharides[monosacch_name])
    shifts.append(
        MassOffsetFeature(
            m.mass(), name=monosacch_name, tolerance=2e-5, feature_type='glycosylation'))
