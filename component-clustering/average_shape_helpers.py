import copy
import numpy as np
from sklearn.cluster import DBSCAN
from shapely.geometry import Point, box
from shapely.affinity import rotate as shapely_rotate
from shapely.affinity import scale as shapely_scale
from shapely.affinity import translate as shapely_translate
import wrangle_classifications as wc


# Helper functions
def reset_task_scale_slider(task):
    _t = copy.deepcopy(task)
    _t['value'][1]['value'] = 1.0
    return _t


def remove_scaling(annotation):
    return [
        reset_task_scale_slider(task)
        for task in annotation
    ]


def make_ellipse(comp):
    return shapely_rotate(
        shapely_scale(
            Point(*comp['mu']).buffer(1.0),
            xfact=comp['rEff'],
            yfact=comp['rEff'] * comp['axRatio']
        ),
        -np.rad2deg(comp['roll'])
    ) if comp is not None else None


def make_box(comp):
    if comp is None or comp['axRatio'] == 0:
        return None
    return shapely_rotate(
        box(
            comp['mu'][0] - comp['rEff'] / 2 / comp['axRatio'],
            comp['mu'][1] - comp['rEff'] / 2,
            comp['mu'][0] + comp['rEff'] / 2 / comp['axRatio'],
            comp['mu'][1] + comp['rEff'] / 2,
        ),
        -np.rad2deg(comp['roll'])
    )


def get_clusters(comps, labels):
    return [comps[labels == i] for i in range(np.max(labels) + 1)]


def largest_cluster_label(labels):
    if not np.any(labels >= 0):
        return np.nan
    return np.argmax([
        len(labels[labels == v]) for v in np.unique(labels[labels >= 0])
    ])


def cluster_comp(comp_geoms, eps, min_samples):
    labels = np.zeros(len(comp_geoms)) - 1
    na_mask = comp_geoms.notna().values
    filtered = comp_geoms.dropna().values
    if len(filtered) == 0:
        return []
    distances = wc.gen_jaccard_distances(filtered)
    clf = DBSCAN(eps=eps, min_samples=min_samples,
                 metric='precomputed')
    clf.fit(distances)
    labels[na_mask] = clf.labels_
    return labels


def sanitize_param_dict(p):
    # rEff > 0
    # 0 < axRatio < 1
    # 0 < roll < np.pi (not 2*pi due to rotational symmetry)
    out = {k: v for k, v in p.items()}
    out['rEff'] = (
        abs(out['rEff'])
        * (abs(p['axRatio']) if abs(p['axRatio']) > 1 else 1)
    )
    out['axRatio'] = min(abs(p['axRatio']), 1 / abs(p['axRatio']))
    out['roll'] = p['roll'] % np.pi
    return out


def get_param_list(d):
    if d is None:
        d = {}
    return [*d.get('mu', (0, 0)), d.get('rEff', 5), d.get('axRatio', 0.7),
            d.get('roll', 0)]


def get_param_dict(p):
    return sanitize_param_dict({
        k: v
        for k, v in zip(
            ('mu', 'rEff', 'axRatio', 'roll'),
            [p[:2].tolist()] + p[2:].tolist()
        )
    })


def ellipse_from_param_list(p):
    return make_ellipse(get_param_dict(p))


def box_from_param_list(p):
    return make_box(get_param_dict(p))


def transform_val(v, npix, petro_theta):
        return (v - npix / 2) * 4 * petro_theta / npix


def transform_shape(shape, npix, petro_theta):
    return shapely_scale(
        shapely_translate(
            shape,
            -npix/2,
            -npix/2,
        ),
        4 * petro_theta / npix, 4 * petro_theta / npix, origin=(0, 0)
    )


def sanitize_comp_for_json(comp):
    if comp is None:
        return None
    elif type(comp['mu']) != list:
        return {**comp, 'mu': comp['mu'].tolist()} if comp is not None else None
    return comp
