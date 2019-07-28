from collections import defaultdict

def extract_oxoniums(gpsm):
    result = defaultdict(float, {
        "glycopeptide": str(gpsm.structure),
        "glycan_composition": str(gpsm.structure.glycan_composition),
        "base_peak": gpsm.base_peak().intensity,
        "tic": gpsm.tic(),
        "scan_id": gpsm.scan_id,
        "title": gpsm.title
    })
    match = gpsm.match()
    for fragment, peak in match.solution_map.items():
        if fragment.series != "oxonium_ion" or peak.charge != 1:
            continue
        result[fragment.name] = peak.intensity
    return result
