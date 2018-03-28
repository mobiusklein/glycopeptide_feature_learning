import operator

get_intensity = operator.attrgetter("intensity")


def ppm_error(x, y):
    return (x - y) / y


OUT_OF_RANGE_INT = -999


def intensity_ratio_function(peak1, peak2):
    ratio = peak1.intensity / float(peak2.intensity)
    if ratio >= 5:
        return -4
    elif 2.5 <= ratio < 5:
        return -3
    elif 1.7 <= ratio < 2.5:
        return -2
    elif 1.3 <= ratio < 1.7:
        return -1
    elif 1.0 <= ratio < 1.3:
        return 0
    elif 0.8 <= ratio < 1.0:
        return 1
    elif 0.6 <= ratio < 0.8:
        return 2
    elif 0.4 <= ratio < 0.6:
        return 3
    elif 0.2 <= ratio < 0.4:
        return 4
    elif 0. <= ratio < 0.2:
        return 5


def intensity_rank(peak_list, minimum_intensity=100.):
    peak_list = sorted(peak_list, key=get_intensity, reverse=True)
    i = 0
    rank = 10
    tailing = 6
    for p in peak_list:
        if p.intensity < minimum_intensity:
            p.rank = 0
            continue
        i += 1
        if i == 10 and rank != 0:
            if rank == 1:
                if tailing != 0:
                    i = 0
                    tailing -= 1
                else:
                    i = 0
                    rank -= 1
            else:
                i = 0
                rank -= 1
        if rank == 0:
            break
        p.rank = rank


try:
    _intensity_ratio_function = intensity_ratio_function
    from ._c.peak_relations import intensity_ratio_function
except ImportError:
    pass
