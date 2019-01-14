import numpy as np
from copy import deepcopy


def hasDrawnComp(comp):
    return len(comp['value'][0]['value']) > 0


def parse_sersic_comp(comp, size_diff=1):
    if not hasDrawnComp(comp):
        return None
    drawing = comp['value'][0]['value'][0]
    major_axis_index = 0 if drawing['rx'] > drawing['ry'] else 1
    major_axis = max(drawing['rx'], drawing['ry'])
    minor_axis = min(drawing['rx'], drawing['ry'])
    roll = drawing['angle'] \
        + (90 if major_axis_index == 0 else 0)
    out = {
        'mu': np.array((drawing['x'], drawing['y'])) * size_diff,
        # zooniverse rotation is confusing
        'roll': np.deg2rad(roll),
        'rEff': max(
            1e-5,
            major_axis * float(comp['value'][1]['value']) * size_diff
        ),
        'axRatio': major_axis / minor_axis,
        'i0': float(comp['value'][2]['value']),
        'c': 2,
        'n': 1,
    }
    try:
        out['n'] = float(comp['value'][3]['value'])
        out['c'] = float(comp['value'][4]['value'])
    except IndexError:
        pass
    return out


def parse_bar_comp(comp, **kwargs):
    if not hasDrawnComp(comp):
        return None
    _comp = deepcopy(comp)
    drawing = _comp['value'][0]['value'][0]
    drawing['rx'] = drawing['width']
    drawing['ry'] = drawing['height']
    # get center position of box
    drawing['x'] = drawing['x'] + drawing['width'] / 2
    drawing['y'] = drawing['y'] + drawing['height'] / 2
    drawing['angle'] = -drawing['angle']
    _comp['value'][0]['value'][0] = drawing
    return parse_sersic_comp(_comp, **kwargs)


def parse_spiral_comp(comp, size_diff=1):
    out = []
    for arm in comp['value'][0]['value']:
        points = np.array([[p['x'], p['y']] for p in arm['points']], dtype='float')
        points *= size_diff
        params = {
            'i0': float(arm['details'][0]['value']),
            'spread': float(arm['details'][1]['value']),
            'falloff': max(float(comp['value'][1]['value']), 1E-5),
        }
        out.append((points, params))
    return out


def parse_annotation(annotation, size_diff=1):
    out = {'disk': None, 'bulge': None, 'bar': None, 'spiral': []}
    for component in annotation:
        if len(component['value'][0]['value']) == 0:
            out['task'] = None
        if component['task'] == 'spiral':
            out['spiral'] = parse_spiral_comp(component, size_diff=size_diff)
        elif component['task'] == 'bar':
            out['bar'] = parse_bar_comp(component, size_diff=size_diff)
        else:
            out[component['task']] = parse_sersic_comp(component, size_diff=size_diff)
    return out
