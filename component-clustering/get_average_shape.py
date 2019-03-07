import os
import json
import numpy as np
from scipy.optimize import minimize
import pandas as pd
import matplotlib.pyplot as plt
from descartes import PolygonPatch
from panoptes_aggregation.reducers.shape_metric import avg_angle
from gzbuilderspirals import metric
from gzbuilderspirals.oo import Pipeline
import wrangle_classifications as wc
import lib.galaxy_utilities as gu
import lib.python_model_renderer.parse_annotation as pa
import average_shape_helpers as ash
from progress.bar import Bar
import warnings
from astropy.utils.exceptions import AstropyWarning
warnings.simplefilter('ignore', category=AstropyWarning)

# Clustering parameters
DISK_EPS = 0.3
DISK_MIN_SAMPLES = 5

BULGE_EPS = 0.3
BULGE_MIN_SAMPLES = 5

BAR_EPS = 0.4
BAR_MIN_SAMPLES = 4


# SECTION: Functions to cluster and aggregate components
def get_geoms(model_details):
    """Function to obtain shapely geometries from parsed Zooniverse
    classifications
    """
    disk = ash.make_ellipse(model_details['disk'])
    bulge = ash.make_ellipse(model_details['bulge'])
    bar = ash.make_box(model_details['bar'])
    return disk, bulge, bar


def cluster_components(geoms):
    disk_labels = ash.cluster_comp(geoms['disk'], DISK_EPS,
                                   DISK_MIN_SAMPLES)
    bulge_labels = ash.cluster_comp(geoms['bulge'], BULGE_EPS,
                                    BULGE_MIN_SAMPLES)
    bar_labels = ash.cluster_comp(geoms['bar'], BAR_EPS,
                                  BAR_MIN_SAMPLES)

    return disk_labels, bulge_labels, bar_labels


def aggregate_geom_jaccard(geoms, x0=np.array((256, 256, 5, 0.7, 0)),
                           constructor_func=ash.ellipse_from_param_list):
    def __distance_func(p):
        p = np.array(p)
        assert len(p) == 5, 'Invalid number of parameters supplied'
        comp = constructor_func(p)
        s = sum(wc.jaccard_distance(comp, other) for other in geoms)
        return s
    # sanitze results rather than imposing bounds to avoid getting stuck in
    # local minima
    return ash.get_param_dict(
        minimize(__distance_func, x0)['x']
    )


def aggregate_comp_mean(comps):
    comps = list(map(ash.sanitize_param_dict, comps))
    out = {'i0': 1, 'n': 1, 'c': 2}
    if len(comps) == 0:
        return out
    keys = list(comps[0].keys())
    for key in keys:
        if key == 'roll':
            clustered_angles = [
                np.rad2deg(i['roll'])
                for i in comps
            ]
            out['roll'] = np.deg2rad(avg_angle(clustered_angles, factor=2))
        else:
            out[key] = np.mean([i.get(key, np.nan) for i in comps], axis=0)
    return out


def get_spiral_arms(subject_id):
    gal, angle = gu.get_galaxy_and_angle(subject_id)
    drawn_arms = gu.get_drawn_arms(subject_id, gu.classifications)
    distances = gu.get_distances(subject_id)
    if distances is None:
        distances = metric.calculate_distance_matrix(drawn_arms)
        np.save('lib/distances/subject-{}.npy'.format(subject_id), distances)
    p = Pipeline(drawn_arms, phi=angle, ba=gal['PETRO_BA90'],
                 image_size=512, distances=distances)
    return [p.get_arm(i, clean_points=True)
            for i in range(max(p.db.labels_) + 1)]


def make_model(subject_id):
    annotations = gu.classifications[
        gu.classifications['subject_ids'] == subject_id
    ]['annotations'].apply(json.loads)
    models = annotations\
        .apply(ash.remove_scaling)\
        .apply(pa.parse_annotation)
    spirals = models.apply(lambda d: d.get('spiral', None))
    geoms = pd.DataFrame(
        models.apply(get_geoms).values.tolist(),
        columns=('disk', 'bulge', 'bar')
    )
    geoms['spirals'] = spirals
    labels = list(map(np.array, cluster_components(geoms)))

    cluster_labels = list(map(ash.largest_cluster_label, labels))
    cluster_masks = [a == b for a, b in zip(labels, cluster_labels)]

    np.save('cluster_masks/{}.npy'.format(subject_id), cluster_masks)

    disk_cluster_geoms = geoms['disk'][cluster_masks[0]]
    bulge_cluster_geoms = geoms['bulge'][cluster_masks[1]]
    bar_cluster_geoms = geoms['bar'][cluster_masks[2]]

    # calculate an aggregate disk
    aggregate_disk = {
        **aggregate_comp_mean(
            models[labels[0] == cluster_labels[0]].apply(
                lambda v: v.get('disk', None)
            ).dropna()
        ),
        **aggregate_geom_jaccard(disk_cluster_geoms.values)
    } if np.any(labels[0] == cluster_labels[0]) else None

    # calculate an aggregate bulge
    aggregate_bulge = {
        **aggregate_comp_mean(
            models[labels[1] == cluster_labels[1]].apply(
                lambda v: v.get('bulge', None)
            ).dropna()
        ),
        **aggregate_geom_jaccard(bulge_cluster_geoms.values)
    } if np.any(labels[1] == cluster_labels[1]) else None

    # calculate an aggregate bar
    aggregate_bar = {
        **aggregate_comp_mean(
            models[labels[2] == cluster_labels[2]].apply(
                lambda v: v.get('bar', None)
            ).dropna()
        ),
        **aggregate_geom_jaccard(
            bar_cluster_geoms.values,
            constructor_func=ash.box_from_param_list
        )
    } if np.any(labels[2] == cluster_labels[2]) else None

    arms = get_spiral_arms(subject_id)
    logsps = [arm.reprojected_log_spiral for arm in arms]

    with open('cluster-output/{}.json'.format(subject_id), 'w') as f:
        json.dump(
            {
                'disk': ash.sanitize_comp_for_json(aggregate_disk),
                'bulge': ash.sanitize_comp_for_json(aggregate_bulge),
                'bar': ash.sanitize_comp_for_json(aggregate_bar),
                'spirals': [a.tolist() for a in logsps],
            },
            f,
        )
    return {
        'disk': aggregate_disk,
        'bulge': aggregate_bulge,
        'bar': aggregate_bar,
        'spirals': logsps,
    }, cluster_masks, arms


def plot_aggregation(subject_id, model=None, cluster_masks=None, arms=None):
    if model is None or cluster_masks is None or arms is None:
        print(model)
        model_path = os.path.join(
            'cluster-output', '{}.json'.format(subject_id)
        )
        masks_path = os.path.join('cluster_masks', '{}.npy'.format(subject_id))
        if not (os.path.exists(model_path) and os.path.exists(masks_path)):
            return
        with open(model_path) as f:
            model = json.load(f)
        with open(masks_path) as f:
            cluster_masks = np.load(f)
        arms = get_spiral_arms(subject_id)

    annotations = gu.classifications[
        gu.classifications['subject_ids'] == subject_id
    ]['annotations'].apply(json.loads)
    models = annotations\
        .apply(ash.remove_scaling)\
        .apply(pa.parse_annotation)
    spirals = models.apply(lambda d: d.get('spiral', None))
    geoms = pd.DataFrame(
        models.apply(get_geoms).values.tolist(),
        columns=('disk', 'bulge', 'bar')
    )
    geoms['spirals'] = spirals

    drawn_arms = gu.get_drawn_arms(subject_id, gu.classifications)

    arms = get_spiral_arms(subject_id)
    logsps = [arm.reprojected_log_spiral for arm in arms]

    disk_cluster_geoms = geoms['disk'][cluster_masks[0]]
    bulge_cluster_geoms = geoms['bulge'][cluster_masks[1]]
    bar_cluster_geoms = geoms['bar'][cluster_masks[2]]

    aggregate_disk_geom = ash.make_ellipse(model['disk'])
    aggregate_bulge_geom = ash.make_ellipse(model['bulge'])
    aggregate_bar_geom = ash.make_box(model['bar'])

    gal, angle = gu.get_galaxy_and_angle(subject_id)
    pic_array, _ = gu.get_image(gal, subject_id, angle)

    def ts(s):
        return ash.transform_shape(s, pic_array.shape[0],
                                   gal['PETRO_THETA'].iloc[0])

    def tv(v):
        return ash.transform_val(v, pic_array.shape[0],
                                 gal['PETRO_THETA'].iloc[0])

    # fig, ((ax0, ax1), (ax2, ax3)) = plt.subplots(
    #     ncols=2, nrows=2,
    #     figsize=(10, 10),
    #     sharex=True, sharey=True
    # )
    # imshow_kwargs = {
    #     'cmap': 'gray',
    #     'origin': 'lower',
    #     'extent': [tv(0), tv(pic_array.shape[0])]*2,
    # }
    # ax0.imshow(pic_array, **imshow_kwargs)
    # for comp in geoms['disk'].values:
    #     if comp is not None:
    #         ax0.add_patch(
    #             PolygonPatch(ts(comp), fc='C0', ec='k',
    #                          alpha=0.2, zorder=3)
    #         )
    # ax1.imshow(pic_array, **imshow_kwargs)
    # for comp in geoms['bulge'].values:
    #     if comp is not None:
    #         ax1.add_patch(
    #             PolygonPatch(ts(comp), fc='C1', ec='k',
    #                          alpha=0.5, zorder=3)
    #         )
    # ax2.imshow(pic_array, **imshow_kwargs)
    # for comp in geoms['bar'].values:
    #     if comp is not None:
    #         ax2.add_patch(
    #             PolygonPatch(ts(comp), fc='C2', ec='k',
    #                          alpha=0.2, zorder=3)
    #         )
    # ax3.imshow(pic_array, **imshow_kwargs)
    # for arm in drawn_arms:
    #     ax3.plot(*tv(arm).T)
    #
    # for i, ax in enumerate((ax0, ax1, ax2, ax3)):
    #     ax.set_xlim(imshow_kwargs['extent'][:2])
    #     ax.set_ylim(imshow_kwargs['extent'][2:])
    #     if i % 2 == 0:
    #         ax.set_ylabel('Arcseconds from center')
    #     if i > 1:
    #         ax.set_xlabel('Arcseconds from center')
    # fig.subplots_adjust(wspace=0.05, hspace=0.05)
    # plt.savefig('drawn_shapes/{}.pdf'.format(subject_id), bbox_inches='tight')
    # plt.close()

    fig, ((ax0, ax1), (ax2, ax3)) = plt.subplots(
        ncols=2, nrows=2,
        figsize=(10, 10),
        sharex=True, sharey=True
    )
    ax0.imshow(pic_array, **imshow_kwargs)
    for comp in disk_cluster_geoms.values:
        ax0.add_patch(
            PolygonPatch(ts(comp), fc='C0', ec='k', alpha=0.1, zorder=3)
        )
    if model['disk'] is not None:
        aggregate_disk_geom = ash.make_ellipse(model['disk'])
        ax0.add_patch(
            PolygonPatch(ts(aggregate_disk_geom), fc='C1', ec='k', alpha=0.5,
                         zorder=3)
        )
    ax1.imshow(pic_array, **imshow_kwargs)
    for comp in bulge_cluster_geoms.values:
        ax1.add_patch(
            PolygonPatch(ts(comp), fc='C1', ec='k', alpha=0.1, zorder=3)
        )
    if aggregate_bulge_geom is not None:
        ax1.add_patch(
            PolygonPatch(ts(aggregate_bulge_geom), fc='C2', ec='k', alpha=0.5,
                         zorder=3)
        )
    ax2.imshow(pic_array, **imshow_kwargs)
    for comp in bar_cluster_geoms.values:
        ax2.add_patch(
            PolygonPatch(ts(comp), fc='C2', ec='k', alpha=0.1, zorder=3)
        )
    if aggregate_bar_geom is not None:
        ax2.add_patch(
            PolygonPatch(ts(aggregate_bar_geom), fc='C3', ec='k', alpha=0.5,
                         zorder=3)
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
    plt.savefig('clustered_shapes/{}.pdf'.format(subject_id),
                bbox_inches='tight')
    plt.close()

    fig = plt.figure(figsize=(10, 10))
    ax = plt.gca()
    ax.imshow(pic_array, **imshow_kwargs)
    if aggregate_disk_geom is not None:
        ax.add_patch(
            PolygonPatch(ts(aggregate_disk_geom), fc='C0', ec='k', alpha=0.25,
                         zorder=3)
        )
    if aggregate_bulge_geom is not None:
        ax.add_patch(
            PolygonPatch(ts(aggregate_bulge_geom), fc='C1', ec='k', alpha=0.25,
                         zorder=3)
        )
    if aggregate_bar_geom is not None:
        ax.add_patch(
            PolygonPatch(ts(aggregate_bar_geom), fc='C2', ec='k', alpha=0.25,
                         zorder=3)
        )
    for arm in logsps:
        plt.plot(*tv(arm).T, c='C3')

    ax.set_xlim(imshow_kwargs['extent'][:2])
    ax.set_ylim(imshow_kwargs['extent'][2:])
    ax.set_ylabel('Arcseconds from center')
    ax.set_xlabel('Arcseconds from center')
    plt.savefig('aggregate_model/{}.pdf'.format(subject_id),
                bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    sid_list = sorted(np.loadtxt('lib/subject-id-list.csv', dtype='u8'))
    to_iter = sid_list
    bar = Bar('Calculating models', max=len(to_iter), suffix='%(percent).1f%% - %(eta)ds')
    for subject_id in to_iter:
        model, cluster_masks, arms = make_model(subject_id)
        plot_aggregation(subject_id, model, cluster_masks, arms)
        bar.next()
    bar.finish()
