import numpy as np


def hasDrawnComp(comp):
    return len(comp['value'][0]['value']) > 0


def parse_sersic_comp(comp):
    if not hasDrawnComp(comp):
        return None
    drawing = comp['value'][0]['value'][0]
    majorAxis = max(drawing['rx'], drawing['ry'])
    minorAxis = min(drawing['rx'], drawing['ry'])
    roll = drawing['angle'] \
        * (1 if comp['task'] == 'bar' else -1) \
        + (90 if majorAxis != drawing['rx'] else 0)
    out = {
        'mu': np.array((drawing['x'], drawing['y'])),
        # zooniverse rotation is confusing
        'roll': np.deg2rad(roll),
        'rEff': majorAxis * comp['value'][1]['value'],
        'axRatio': majorAxis / minorAxis,
        'i0': comp['value'][2]['value'],
        'c': 2,
        'n': 1,
    }
    try:
        out['n'] = float(comp['value'][3]['value'])
        out['c'] = float(comp['value'][4]['value'])
    except IndexError:
        pass
    return out


def parse_spiral_comp(comp):
    out = []
    for arm in comp['value'][0]['value']:
        points = np.array([[p['x'], p['y']] for p in arm['points']])
        params = {
            'i0': arm['details'][0]['value'],
            'spread': arm['details'][1]['value'],
            'falloff': comp['value'][1]['value'],
        }
        out.append((points, params))
    return out


def parse_annotation(annotation):
    out = {'disk': None, 'bulge': None, 'bar': None, 'spiral': []}
    for component in annotation:
        if component['task'] == 'spiral':
            out['spiral'] = parse_spiral_comp(component)
        else:
            out[component['task']] = parse_sersic_comp(component)
    return out
