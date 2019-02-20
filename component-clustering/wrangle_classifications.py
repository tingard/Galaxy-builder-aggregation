import copy
import numpy as np
from shapely.geometry import box, Point
from shapely.affinity import rotate, scale


def migrate_slider_to_subtask(task):
    # nb. we ignore the scale slider
    _t = copy.deepcopy(task)
    for i in _t['value'][0]['value']:
        i['details'] = [
            {'task': subtask['task'], 'value': float(subtask['value'])}
            for subtask in task['value'][1:]
        ]
    return _t


def move_to_zero_frame(task):
    _t = copy.deepcopy(task)
    for i in _t['value'][0]['value']:
        i['frame'] = 0
    return _t


def scale_from_slider(task, bar=False):
    _t = copy.deepcopy(task)
    for i in range(len(_t['value'][0]['value'])):
        drawn_obj = copy.deepcopy(task['value'][0]['value'][i])
        scale_value = float(_t['value'][1]['value'])
        if bar:
            a = np.deg2rad(drawn_obj['angle'])
            rotation_matrix = np.array(((np.cos(a), -np.sin(a)), (np.sin(a), np.cos(a))))
            origin = np.array((drawn_obj['x'], drawn_obj['y']))
            # O' = O + 0.5 * (1 - s) * R dot (w, h)
            new_origin = origin + 0.5 * (1 - scale_value) * np.dot(
                rotation_matrix,
                (drawn_obj['width'], drawn_obj['height'])
            )

            drawn_obj['x'], drawn_obj['y'] = new_origin
            drawn_obj['width'] *= float(_t['value'][1]['value'])
            drawn_obj['height'] *= float(_t['value'][1]['value'])
        else:
            drawn_obj['rx'] *= scale_value
            drawn_obj['ry'] *= scale_value
        _t['value'][0]['value'][i] = drawn_obj
    return _t


def remove_from_combo_task(task):
    _t = copy.deepcopy(task['value'][0])
    _t['task'] = task['task']
    return _t


def move_bar_origin_to_centre(task):
    """We want to cluster bars based on centre position, not corner position.
    This could be combined with scale_from_slider for efficiency, but
    separating functionality is more readable
    """
    _t = copy.deepcopy(task)
    for i in range(len(_t['value'])):
        drawn_obj = copy.deepcopy(task['value'][i])
        origin = np.array((drawn_obj['x'], drawn_obj['y']))
        a = np.deg2rad(drawn_obj['angle'])
        rotation_matrix = np.array(((np.cos(a), -np.sin(a)), (np.sin(a), np.cos(a))))
        new_origin = origin + 0.5 * np.dot(
            rotation_matrix,
            (drawn_obj['width'], drawn_obj['height'])
        )
        drawn_obj['x'], drawn_obj['y'] = new_origin
        _t['value'][i] = drawn_obj
    return _t


def convert_shape(task, bar=False):
    p0 = migrate_slider_to_subtask(task)
    p1 = move_to_zero_frame(p0)
    p2 = p1  # scale_from_slider(p1, bar=bar)
    p3 = remove_from_combo_task(p2)
    return p3 if not bar else move_bar_origin_to_centre(p3)


def decenter_bar(x, y, width, height, angle):
    a = np.deg2rad(angle)
    rotation_matrix = np.array(((np.cos(a), -np.sin(a)), (np.sin(a), np.cos(a))))
    original_location = np.array((x, y)) - 0.5 * np.dot(
        rotation_matrix,
        (width, height)
    )
    return original_location


def bar_geom_from_zoo(a):
    b = box(
        a['x'],
        a['y'],
        a['x'] + a['width'],
        a['y'] + a['height']
    )
    return rotate(b, a['angle'])


def ellipse_geom_from_zoo(a):
    ellipse = rotate(
        scale(
            Point(a['x'], a['y']).buffer(1.0),
            xfact=a['rx'],
            yfact=a['ry']
        ),
        -a['angle']
    )
    return ellipse


def jaccard_distance(ob1, ob2):
    if ob1.union(ob2).area <= 0:
        return 1
    return 1 - ob1.intersection(ob2).area / ob1.union(ob2).area


def gen_jaccard_distances(geoms):
    distances = np.zeros((len(geoms), len(geoms)))
    for i in range(len(geoms)):
        for j in range(i, len(geoms)):
            distances[i, j] = jaccard_distance(geoms[i], geoms[j])
    return distances + distances.T


def _flatten_component(component):
    _c = copy.deepcopy(component)
    for subtask in _c['details']:
        _c[subtask['task']] = subtask['value']
    _c.pop('details', None)
    return _c


def sklearn_flatten(component_array):
    return [_flatten_component(c) for c in component_array]
