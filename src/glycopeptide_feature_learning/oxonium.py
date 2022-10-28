from collections import defaultdict

import numpy as np


def normalize_oxonium_ions(match, reference="HexNAc"):
    measures = {
        p.fragment_name: p.peak.intensity for p in match.solution_map if p.fragment.series == 'oxonium_ion'}
    ref = measures[reference]
    measures = {"oxonium_%s" % k: v / ref for k, v in measures.items()}
    for mono, count in match.target.glycan_composition.items():
        measures['composition_%s' % mono] = count
    return defaultdict(float, measures)


class OxoniumIonEncoder(object):
    def __init__(self, max_values, oxonium_ion_names):
        self.boundary_values = sorted({k: int(v) for k, v in max_values.items()}.items())
        self.oxonium_ion_names = sorted(oxonium_ion_names)
        self.total_features = sum([v[1] + 1 for v in self.boundary_values]) + len(oxonium_ion_names)

    def feature_names(self):
        names = []
        for k, v in self.boundary_values:
            for i in range(v + 1):
                names.append(k + ':' + str(i))
        names.extend(self.oxonium_ion_names)
        return names

    def transform(self, features, include_names=False):
        X = np.zeros(self.total_features)
        names = []
        i = 0
        j = 0
        for k, v in self.boundary_values:
            if k in features:
                X[i + int(min(v, features[k]))] = 1
            i += v + 1
        obs = []
        ys = []
        for j, k in enumerate(self.oxonium_ion_names):
            if k in features and features[k] > 0:
                x = X.copy()
                x[j + i] = 1
                if include_names:
                    names.append(k)
                ys.append(features[k])
                obs.append(x)

        X = np.vstack(obs)
        y = np.array(ys)
        if include_names:
            return X, y, names
        return X, y

