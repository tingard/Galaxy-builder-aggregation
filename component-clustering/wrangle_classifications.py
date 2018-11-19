import numpy as np
import copy
import galaxy_utilities as gu


def migrate_slider_to_subtask(task):
    # nb. we ignore the scale slider
    _t = copy.deepcopy(task)
    for i in _t['value'][0]['value']:
        i['details'] = [
            { 'task': subtask['task'], 'value': float(subtask['value']) }
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
    for i in _t['value'][0]['value']:
        if bar:
            i['width'] *= float(_t['value'][1]['value'])
            i['height'] *= float(_t['value'][1]['value'])
        else:
            i['rx'] *= float(_t['value'][1]['value'])
            i['ry'] *= float(_t['value'][1]['value'])

    return _t


def remove_from_combo_task(task):
    _t = copy.deepcopy(task['value'][0])
    _t['task'] = task['task']
    return _t


def convert_shape(task, bar=False):
    p0 = migrate_slider_to_subtask(task)
    p1 = move_to_zero_frame(p0)
    p2 = scale_from_slider(p1, bar=bar)
    p3 = remove_from_combo_task(p2)
    return p3


def _flatten_component(component):
    _c = copy.deepcopy(component)
    for subtask in _c['details']:
        _c[subtask['task']] = subtask['value']
    _c.pop('details', None)
    return _c


def sklearn_flatten(component_array):
    return [_flatten_component(c) for c in component_array]


# if __name__ == "__main__":
#     id = 21096878
#     print('Getting galaxy data')
#     gal, angle = gu.get_galaxy_and_angle(id)
#     url = gu.getUrl(id)
#     classifications, pic_array, deprojected_image = gu.get_image(
#         gal, id, angle
#     )
