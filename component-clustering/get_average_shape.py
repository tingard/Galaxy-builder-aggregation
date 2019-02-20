# %load_ext autoreload
# %autoreload 2
import json
import copy
import numpy as np
from scipy.optimize import minimize
from sklearn.cluster import DBSCAN
import pandas as pd
import matplotlib.pyplot as plt
from shapely.geometry import Point, box
from shapely.affinity import rotate as shapely_rotate
from shapely.affinity import scale as shapely_scale
from shapely.affinity import translate as shapely_translate
# from shapely.affinity import translate as shapely_translate
from descartes import PolygonPatch
from gzbuilderspirals import metric
from gzbuilderspirals.oo import Pipeline
import wrangle_classifications as wc
import lib.galaxy_utilities as gu
import lib.python_model_renderer.parse_annotation as pa


# Clustering parameters
DISK_EPS = 0.2
DISK_MIN_SAMPLES = 5
BULGE_EPS = 0.3
BULGE_MIN_SAMPLES = 5
BAR_EPS = 0.4
BAR_MIN_SAMPLES = 3


# Helper functions
def __reset_task_scale_slider(task):
    _t = copy.deepcopy(task)
    _t['value'][1]['value'] = 1.0
    return _t


def __remove_scaling(annotation):
    return [
        __reset_task_scale_slider(task)
        for task in annotation
    ]


def __make_ellipse(comp):
    return shapely_rotate(
        shapely_scale(
            Point(*comp['mu']).buffer(1.0),
            xfact=comp['rEff'],
            yfact=comp['rEff'] * comp['axRatio']
        ),
        -np.rad2deg(comp['roll'])
    ) if comp is not None else None


def __make_box(comp):
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


def __get_clusters(comps, labels):
    return [comps[labels == i] for i in range(np.max(labels) + 1)]


def __cluster_comp(comp_geoms, eps, min_samples):
    filtered = comp_geoms.dropna().values
    if len(filtered) == 0:
        return []
    distances = wc.gen_jaccard_distances(filtered)
    clf = DBSCAN(eps=eps, min_samples=min_samples,
                 metric='precomputed')
    clf.fit(distances)
    return __get_clusters(filtered, clf.labels_)


def __ellipse_from_param_list(p):
    return __make_ellipse(get_ellipse_param_dict(p))


def __box_from_param_list(p):
    return __make_box(get_ellipse_param_dict(p))


def __transform_val(v, npix, petro_theta):
        return (v - npix / 2) * 4 * petro_theta / npix


def __transform_shape(shape, npix, petro_theta):
    return shapely_scale(
        shapely_translate(
            shape,
            -npix/2,
            -npix/2,
        ),
        4 * petro_theta / npix, 4 * petro_theta / npix, origin=(0, 0)
    )


# Functions to cluster and aggregate components

def get_geoms(model_details):
    """Function to obtain shapely geometries from parsed Zooniverse
    classifications
    """
    disk = __make_ellipse(model_details['disk'])
    bulge = __make_ellipse(model_details['bulge'])
    bar = __make_box(model_details['bar'])
    return disk, bulge, bar


def cluster_components(geoms):
    disk_clusters = __cluster_comp(geoms['disk'], DISK_EPS,
                                   DISK_MIN_SAMPLES)
    bulge_clusters = __cluster_comp(geoms['bulge'], BULGE_EPS,
                                    BULGE_MIN_SAMPLES)
    bar_clusters = __cluster_comp(geoms['bar'], BAR_EPS,
                                  BAR_MIN_SAMPLES)

    return (disk_clusters, bulge_clusters, bar_clusters)


def get_ellipse_param_dict(p):
    return {
        k: v
        for k, v in zip(
            ('mu', 'rEff', 'axRatio', 'roll'),
            [p[:2].tolist()] + p[2:].tolist()
        )
    }


def get_box_param_dict(p):
    return {
        k: v
        for k, v in zip(
            ('mu', 'rEff', 'axRatio', 'roll'),
            [p[:2].tolist()] + p[2:].tolist()
        )
    }


def aggregate_geom(geoms, x0=np.array((256, 256, 5, 0.7, 0)),
                   constructor_func=__ellipse_from_param_list):
    def __distance_func(p):
        p = np.array(p)
        assert len(p) == 5, 'Invalid number of parameters supplied'
        comp = constructor_func(p)
        s = sum(wc.jaccard_distance(comp, other) for other in geoms)
        return s
    return minimize(__distance_func, x0)


def make_model_and_plots(subject_id):
    print('\n🌌 Working on subject id', subject_id)
    gal, angle = gu.get_galaxy_and_angle(subject_id)
    pic_array, _ = gu.get_image(gal, subject_id, angle)

    annotations = gu.classifications[
        gu.classifications['subject_ids'] == subject_id
    ]['annotations'].apply(
        json.loads
    )
    models = annotations.apply(
        __remove_scaling
    ).apply(
        pa.parse_annotation
    )
    spirals = models.apply(lambda d: d.get('spiral', None))
    geoms = pd.DataFrame(
        models.apply(get_geoms).values.tolist(),
        columns=('disk', 'bulge', 'bar')
    )
    geoms['spirals'] = spirals

    disk_clusters, bulge_clusters, bar_clusters = cluster_components(geoms)

    print('Clustered:')
    print('\tDisks:', len(disk_clusters))
    print('\tBulges:', len(bulge_clusters))
    print('\tBars:', len(bar_clusters))

    # Choose best cluster based on number of members
    max_disk_cluster_length = max(len(i) for i in disk_clusters) if len(disk_clusters) > 0 else None
    max_bulge_cluster_length = max(len(i) for i in bulge_clusters) if len(bulge_clusters) > 0 else None
    max_bar_cluster_length = max(len(i) for i in bar_clusters) if len(bar_clusters) > 0 else None
    disk_cluster = (
        None if max_disk_cluster_length is None
        else [i for i in disk_clusters
              if len(i) == max_disk_cluster_length][0]
    )
    bulge_cluster = (
        None if max_bulge_cluster_length is None
        else [i for i in bulge_clusters
              if len(i) == max_bulge_cluster_length][0]
    )
    bar_cluster = (
        None if max_bar_cluster_length is None
        else [i for i in bar_clusters
              if len(i) == max_bar_cluster_length][0]
    )

    print('Fitting aggregate components')
    aggregate_disk = get_ellipse_param_dict(
        aggregate_geom(disk_cluster)['x']
    ) if disk_cluster is not None else None
    aggregate_bulge = get_ellipse_param_dict(
        aggregate_geom(bulge_cluster)['x']
    ) if bulge_cluster is not None else None
    aggregate_bar = get_box_param_dict(
        aggregate_geom(
            bar_cluster,
            constructor_func=__box_from_param_list
        )['x']
    ) if bar_cluster is not None else None

    aggregate_disk_geom = None if aggregate_disk is None else __make_ellipse(aggregate_disk)
    aggregate_bulge_geom = None if aggregate_bulge is None else __make_ellipse(aggregate_bulge)
    aggregate_bar_geom = None if aggregate_bar is None else __make_box(aggregate_bar)

    print('Calculating spiral arms')
    drawn_arms = gu.get_drawn_arms(subject_id, gu.classifications)
    distances = gu.get_distances(subject_id)
    if distances is None:
        print('\t- Calculating distances')
        distances = metric.calculate_distance_matrix(drawn_arms)
        np.save('./lib/distances/subject-{}.npy'.format(subject_id), distances)
    p = Pipeline(drawn_arms, phi=angle, ba=gal['PETRO_BA90'],
                 image_size=pic_array.shape[0], distances=distances)
    arms = [p.get_arm(i, clean_points=True)
            for i in range(max(p.db.labels_) + 1)]
    logsps = [arm.reprojected_log_spiral for arm in arms]

    # save the resulting models
    with open('cluster-output/{}.json'.format(subject_id), 'w') as f:
        json.dump(
            {
                'disk': aggregate_disk,
                'bulge': aggregate_bulge,
                'bar': aggregate_bar,
                'spirals': [a.tolist() for a in logsps],
            },
            f,
        )

    # ------------------------- SECTION: Plotting -------------------------
    print('Plotting...')
    ts = lambda s: __transform_shape(s, pic_array.shape[0], gal['PETRO_THETA'].iloc[0])
    tv = lambda v: __transform_val(v, pic_array.shape[0], gal['PETRO_THETA'].iloc[0])

    fig, ((ax0, ax1), (ax2, ax3)) = plt.subplots(
        ncols=2, nrows=2,
        figsize=(10, 10),
        sharex=True, sharey=True
    )
    imshow_kwargs = {
        'cmap': 'gray',
        'origin': 'lower',
        'extent': [tv(0), tv(pic_array.shape[0])]*2,
    }
    ax0.imshow(pic_array, **imshow_kwargs)
    for comp in geoms['disk'].values:
        if comp is not None:
            ax0.add_patch(
                PolygonPatch(ts(comp), fc='C0', ec='k',
                             alpha=0.2, zorder=3)
            )
    ax1.imshow(pic_array, **imshow_kwargs)
    for comp in geoms['bulge'].values:
        if comp is not None:
            ax1.add_patch(
                PolygonPatch(ts(comp), fc='C1', ec='k',
                             alpha=0.5, zorder=3)
            )
    ax2.imshow(pic_array, **imshow_kwargs)
    for comp in geoms['bar'].values:
        if comp is not None:
            ax2.add_patch(
                PolygonPatch(ts(comp), fc='C2', ec='k',
                             alpha=0.2, zorder=3)
            )
    ax3.imshow(pic_array, **imshow_kwargs)
    for arm in drawn_arms:
        ax3.plot(*tv(arm).T)

    for i, ax in enumerate((ax0, ax1, ax2, ax3)):
        ax.set_xlim(imshow_kwargs['extent'][:2])
        ax.set_ylim(imshow_kwargs['extent'][2:])
        if i % 2 == 0:
            ax.set_ylabel('Arcseconds from center')
        if i > 1:
            ax.set_xlabel('Arcseconds from center')
    fig.subplots_adjust(wspace=0.05, hspace=0.05)
    plt.savefig('drawn_shapes/{}.pdf'.format(subject_id), bbox_inches='tight')
    plt.close()

    fig, ((ax0, ax1), (ax2, ax3)) = plt.subplots(
        ncols=2, nrows=2,
        figsize=(10, 10),
        sharex=True, sharey=True
    )
    ax0.imshow(pic_array, **imshow_kwargs)
    for cluster in disk_clusters:
        for comp in cluster:
            ax0.add_patch(
                PolygonPatch(ts(comp), fc='C0', ec='k', alpha=0.1, zorder=3)
            )
    if aggregate_disk_geom is not None:
        ax0.add_patch(
            PolygonPatch(ts(aggregate_disk_geom), fc='C1', ec='k', alpha=0.5, zorder=3)
        )
    ax1.imshow(pic_array, **imshow_kwargs)
    for cluster in bulge_clusters:
        for comp in cluster:
            ax1.add_patch(
                PolygonPatch(ts(comp), fc='C1', ec='k', alpha=0.1, zorder=3)
            )
    if aggregate_bulge_geom is not None:
        ax1.add_patch(
            PolygonPatch(ts(aggregate_bulge_geom), fc='C2', ec='k', alpha=0.5, zorder=3)
        )
    ax2.imshow(pic_array, **imshow_kwargs)
    for cluster in bar_clusters:
        for comp in cluster:
            ax2.add_patch(
                PolygonPatch(ts(comp), fc='C2', ec='k', alpha=0.1, zorder=3)
            )
    if aggregate_bar_geom is not None:
        ax2.add_patch(
            PolygonPatch(ts(aggregate_bar_geom), fc='C3', ec='k', alpha=0.5, zorder=3)
        )
    ax3.imshow(pic_array, **imshow_kwargs)
    for arm in arms:
        plt.plot(*tv(arm.coords).T, '.', alpha=0.5, markersize=0.5)
    for arm in logsps:
        plt.plot(*tv(arm).T)

    for i, ax in enumerate((ax0, ax1, ax2, ax3)):
        ax.set_xlim(imshow_kwargs['extent'][:2])
        ax.set_ylim(imshow_kwargs['extent'][2:])
        if i % 2 == 0:
            ax.set_ylabel('Arcseconds from center')
        if i > 1:
            ax.set_xlabel('Arcseconds from center')
    fig.subplots_adjust(wspace=0.05, hspace=0.05)
    plt.savefig('clustered_shapes/{}.pdf'.format(subject_id), bbox_inches='tight')
    plt.close()

    fig = plt.figure(figsize=(10, 10))
    ax = plt.gca()
    ax.imshow(pic_array, **imshow_kwargs)
    if aggregate_disk_geom is not None:
        ax.add_patch(
            PolygonPatch(ts(aggregate_disk_geom), fc='C0', ec='k', alpha=0.25, zorder=3)
        )
    if aggregate_bulge_geom is not None:
        ax.add_patch(
            PolygonPatch(ts(aggregate_bulge_geom), fc='C1', ec='k', alpha=0.25, zorder=3)
        )
    if aggregate_bar_geom is not None:
        ax.add_patch(
            PolygonPatch(ts(aggregate_bar_geom), fc='C2', ec='k', alpha=0.25, zorder=3)
        )
    for arm in logsps:
        plt.plot(*tv(arm).T, c='C3')

    ax.set_xlim(imshow_kwargs['extent'][:2])
    ax.set_ylim(imshow_kwargs['extent'][2:])
    ax.set_ylabel('Arcseconds from center')
    ax.set_xlabel('Arcseconds from center')
    plt.savefig('aggregate_model/{}.pdf'.format(subject_id), bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    sid_list = sorted(np.loadtxt('lib/subject-id-list.csv', dtype='u8'))
    for subject_id in sid_list:
        make_model_and_plots(subject_id)
