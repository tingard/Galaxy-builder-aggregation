# Method copied from cluster-components.ipynb 10/12/18
import numpy as np
import matplotlib.pyplot as plt
import os
import json
from descartes import PolygonPatch
from sklearn.cluster import DBSCAN
from panoptes_aggregation.reducers.shape_metric import avg_angle
from gzbuilderspirals import metric
from gzbuilderspirals.oo import Pipeline
import lib.galaxy_utilities as gu
import wrangle_classifications as wc
import warnings
from astropy.utils.exceptions import AstropyWarning
warnings.simplefilter('ignore', category=AstropyWarning)
from progress.bar import Bar


DISK_EPS = 0.3
DISK_MIN_SAMPLES = 5

BULGE_EPS = 0.3
BULGE_MIN_SAMPLES = 5

BAR_EPS = 0.4
BAR_MIN_SAMPLES = 4

MAX_BAR_AXRATIO = 0.7
MAX_BAR_FRACTION = 0.5
MAX_BULGE_BAR_SIMILARITY = 0.8


def cluster_disks(drawn_disks, eps=DISK_EPS, min_samples=DISK_MIN_SAMPLES):
    if len(drawn_disks) == 0:
        # print('\tNo drawn disks')
        return None, None, None
    disk_geoms = np.array([
        wc.ellipse_geom_from_zoo(d['value'][0]['value'][0])
        for d in drawn_disks
    ])
    disk_distances = wc.gen_jaccard_distances(disk_geoms)
    clf_disk = DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed')
    clf_disk.fit(disk_distances)
    disk_labels = clf_disk.labels_
    # if len(np.unique(disk_labels)) > 2:
    #     print('\tMultiple ({}) disk clusters found'.format(len(np.unique(disk_labels))))
    clustered_annotations = np.array(drawn_disks)[disk_labels == 0]
    clustered_disks = [d['value'][0]['value'][0] for d in clustered_annotations]
    clustered_angles = [
        i['angle'] if i['rx'] > i['ry'] else (i['angle'] + 90)
        for i in clustered_disks
    ]
    clustered_intensities = np.fromiter(
        (d['value'][1]['value'] for d in clustered_annotations),
        dtype=float
    )

    if len(clustered_disks) == 0:
        # print('\tNo disk cluster')
        return disk_geoms, None, None
    mean_disk = {}
    mean_disk['x'] = np.mean([i['x'] for i in clustered_disks])
    mean_disk['y'] = np.mean([i['y'] for i in clustered_disks])
    mean_disk['rx'] = np.mean([max(i['rx'], i['ry']) for i in clustered_disks])
    mean_disk['ry'] = np.mean([min(i['rx'], i['ry']) for i in clustered_disks])
    mean_disk['angle'] = avg_angle(clustered_angles, factor=2)
    mean_disk['i0'] = clustered_intensities.mean()
    mean_disk['n'] = 1
    mean_disk['c'] = 2

    mean_disk_geom = wc.ellipse_geom_from_zoo(mean_disk)
    return disk_geoms, mean_disk, mean_disk_geom


def get_bulge_from_cluster(drawn_bulges, bulge_labels):
    if np.max(bulge_labels) < 0:
        return None, None
    areas, bulges = [], []
    for cluster_label in range(np.max(bulge_labels) + 1):
        clustered_annotations = np.array(drawn_bulges)[
            bulge_labels == cluster_label
        ]
        clustered_bulges = [
            b['value'][0]['value'][0]
            for b in np.array(drawn_bulges)[
                bulge_labels == cluster_label
            ]
        ]
        clustered_angles = [
            i['angle'] if i['rx'] > i['ry'] else (i['angle'] + 90)
            for i in clustered_bulges
        ]
        clustered_intensities = np.fromiter(
            (d['value'][1]['value'] for d in clustered_annotations),
            dtype=float,
        )
        clustered_ns = np.fromiter(
            (d['value'][2]['value'] for d in clustered_annotations),
            dtype=float,
        )
        mean_bulge = {}
        mean_bulge['x'] = np.mean([i['x'] for i in clustered_bulges])
        mean_bulge['y'] = np.mean([i['y'] for i in clustered_bulges])
        mean_bulge['rx'] = np.mean([max(i['rx'], i['ry']) for i in clustered_bulges])
        mean_bulge['ry'] = np.mean([min(i['rx'], i['ry']) for i in clustered_bulges])
        mean_bulge['angle'] = avg_angle(clustered_angles, factor=2)
        mean_bulge['i0'] = clustered_intensities.mean()
        mean_bulge['n'] = clustered_ns.mean()
        mean_bulge['c'] = 2

        mean_bulge_geom = wc.ellipse_geom_from_zoo(mean_bulge)
        areas.append(mean_bulge_geom.area)
        bulges.append((mean_bulge, mean_bulge_geom))
    return bulges.pop(np.argmin(areas))


def cluster_bulges(drawn_bulges, eps=BULGE_EPS, min_samples=BULGE_MIN_SAMPLES):
    if len(drawn_bulges) == 0:
        # print('\tNo drawn bulges')
        return None, None, None
    bulge_geoms = np.array([
        wc.ellipse_geom_from_zoo(b['value'][0]['value'][0])
        for b in drawn_bulges
    ])
    bulge_distances = wc.gen_jaccard_distances(bulge_geoms)
    clf_bulge = DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed')
    clf_bulge.fit(bulge_distances)
    bulge_labels = clf_bulge.labels_
    mean_bulge = {}
    # if len(np.unique(bulge_labels)) > 2:
    #     print('\tMultiple ({}) bulge clusters found'.format(len(np.unique(bulge_labels))))

    mean_bulge, mean_bulge_geom = get_bulge_from_cluster(drawn_bulges, bulge_labels)

    # if mean_bulge is None:
    #     print('\tNo bulge cluster')
    return bulge_geoms, mean_bulge, mean_bulge_geom


def cluster_bars(drawn_bars, eps=BAR_EPS, min_samples=BAR_MIN_SAMPLES):
    if len(drawn_bars) == 0:
        # print('\tNo drawn bars')
        return None, None, None

    bar_geoms = np.array([
        wc.bar_geom_from_zoo(b['value'][0]['value'][0])
        for b in drawn_bars
    ])
    bar_distances = wc.gen_jaccard_distances(bar_geoms)
    clf_bar = DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed')
    clf_bar.fit(bar_distances)
    bar_labels = clf_bar.labels_
    # if len(np.unique(bar_labels)) > 2:
    #     print('\tMultiple ({}) bar clusters found'.format(len(np.unique(bar_labels))))
    clustered_annotations = np.array(drawn_bars)[bar_labels == 0]
    clustered_bars = [b['value'][0]['value'][0] for b in clustered_annotations]
    if len(clustered_bars) == 0:
        # print('\tNo bar cluster')
        return bar_geoms, None, None
    center_xs = [i['x'] + i['width']/2 for i in clustered_bars]
    center_ys = [i['y'] + i['height']/2 for i in clustered_bars]
    mean_center = (np.mean(center_xs), np.mean(center_ys))

    clustered_intensities = np.fromiter(
        (b['value'][1]['value'] for b in clustered_annotations),
        dtype=float,
    )
    clustered_ns = np.fromiter(
        (b['value'][2]['value'] for b in clustered_annotations),
        dtype=float,
    )
    clustered_cs = np.fromiter(
        (b['value'][3]['value'] for b in clustered_annotations),
        dtype=float,
    )
    mean_bar = {}
    mean_bar['width'] = np.mean([max(i['width'], i['height']) for i in clustered_bars])
    mean_bar['height'] = np.mean([min(i['width'], i['height']) for i in clustered_bars])
    mean_bar['x'] = mean_center[0] - mean_bar['width'] / 2
    mean_bar['y'] = mean_center[1] - mean_bar['height'] / 2
    bar_angles = [
        i['angle'] if i['width'] > i['height'] else (i['angle'] + 90)
        for i in clustered_bars
    ]
    mean_bar['angle'] = avg_angle(bar_angles, factor=2)
    mean_bar['i0'] = clustered_intensities.mean()
    mean_bar['n'] = clustered_ns.mean()
    mean_bar['c'] = clustered_cs.mean()
    mean_bar_geom = wc.bar_geom_from_zoo(mean_bar)
    return bar_geoms, mean_bar, mean_bar_geom


def cluster_components(subject_id):
    # print('Working on subject', subject_id)

    classifications_for_subject = gu.classifications[
        gu.classifications['subject_ids'] == subject_id
    ]
    annotations_for_subject = [
        json.loads(foo) for foo in
        classifications_for_subject['annotations'].values
    ]

    # Exctract annotations
    disks = [a[0] for a in annotations_for_subject if len(a) == 4]
    bulges = [a[1] for a in annotations_for_subject if len(a) == 4]
    bars = [a[2] for a in annotations_for_subject if len(a) == 4]

    drawn_disks = [i for i in disks if len(i['value'][0]['value']) > 0]
    drawn_bulges = [i for i in bulges if len(i['value'][0]['value']) > 0]
    drawn_bars = [i for i in bars if len(i['value'][0]['value']) > 0]

    # print('\tFound {} disks, {} bulges and {} bars'.format(
    #     *map(len, (drawn_disks, drawn_bulges, drawn_bars)))
    # )

    disk_geoms, mean_disk, mean_disk_geom = cluster_disks(drawn_disks)
    bulge_geoms, mean_bulge, mean_bulge_geom = cluster_bulges(drawn_bulges)
    bar_geoms, mean_bar, mean_bar_geom = cluster_bars(drawn_bars)

    # Small checks to test if we have a physical result
    try:
        if mean_bar_geom.area / mean_disk_geom.area > MAX_BAR_FRACTION:
            # print('\tAggregate bar too large relative to disk, ignoring')
            mean_bar = None
        if mean_bar.height / mean_bar.width > MAX_BAR_AXRATIO:
            # print('\tAggregate bar too square, ignoring')
            mean_bar = None
        # if mean_bulge_geom.intersection(mean_bar_geom).area / mean_bulge_geom.union(mean_bar_geom).area > MAX_BULGE_BAR_SIMILARITY:
        #     print('\tBulge and bar very similar for {}'.format(subject_id))
    except AttributeError:
        # one of the components is already None
        pass

    with open('cluster-output-old/{}.json'.format(subject_id), 'w') as f:
        json.dump({'disk': mean_disk, 'bulge': mean_bulge, 'bar': mean_bar}, f)
    return (
        (disk_geoms, mean_disk, mean_disk_geom),
        (bulge_geoms, mean_bulge, mean_bulge_geom),
        (bar_geoms, mean_bar, mean_bar_geom),
    )


def get_log_spirals(subject_id, gal=None, angle=None, pic_array=None, bar_length=0):
    drawn_arms = gu.get_drawn_arms(subject_id, gu.classifications)
    if gal is None or angle is None:
        gal, angle = gu.get_galaxy_and_angle(subject_id)
    if pic_array is None:
        pic_array, deprojected_image = gu.get_image(gal, subject_id, angle)

    path_to_subject = './lib/distances/subject-{}.npy'.format(subject_id)

    distances = gu.get_distances(subject_id)
    if distances is None or distances.shape[0] != len(drawn_arms) or not os.path.exists(path_to_subject):
        # print('\t- Calculating distances')
        distances = metric.calculate_distance_matrix(drawn_arms)
        np.save('./lib/distances/subject-{}.npy'.format(subject_id), distances)

    p = Pipeline(drawn_arms, phi=angle, ba=gal['PETRO_BA90'],
                 image_size=pic_array.shape[0], distances=distances)

    arms = p.get_arms(clean_points=True, bar_length=bar_length)
    # print('Identified {} spiral arms'.format(len(arms)))
    return [arm.reprojected_log_spiral for arm in arms]


def plot_component(pic_array, patches, outfile=None):
    plt.figure(figsize=(8, 8))
    plt.title('Combined galaxy')
    plt.imshow(pic_array, origin='lower', cmap='gray_r')
    ax = plt.gca()
    for p in patches:
        ax.add_patch(p)
    plt.axis('off')
    if outfile is not None:
        plt.savefig(outfile)


if __name__ == "__main__":
    sid_list = sorted(np.loadtxt('lib/subject-id-list.csv', dtype='u8'))
    to_iter = sid_list
    progress_bar = Bar('Calculating models', max=len(to_iter), suffix='%(percent).1f%% - %(eta)ds')
    for subject_id in to_iter:
        gal, angle = gu.get_galaxy_and_angle(subject_id)
        pic_array, deprojected_image = gu.get_image(gal, subject_id, angle)
        pix_size = pic_array.shape[0] / (gal['PETRO_THETA'].iloc[0] * 4)  # pixels per arcsecond

        disk_res, bulge_res, bar_res = cluster_components(subject_id)

        spirals = get_log_spirals(subject_id, gal=gal, angle=angle,
                                  pic_array=pic_array, bar_length=10)
        xtick_labels = np.linspace(-100, 100, 11).astype(int)
        xtick_positions = xtick_labels * pix_size + pic_array.shape[0] / 2
        xtick_mask = (xtick_positions > 0) & (xtick_positions < pic_array.shape[0])

        ytick_labels = np.linspace(-100, 100, 11).astype(int)
        ytick_positions = ytick_labels * pix_size + pic_array.shape[1] / 2
        ytick_mask = (ytick_positions > 0) & (ytick_positions < pic_array.shape[1])

        plt.figure(figsize=(12,9))
        ax0 = plt.subplot2grid((3, 3), (0, 0))
        ax1 = plt.subplot2grid((3, 3), (1, 0))
        ax2 = plt.subplot2grid((3, 3), (2, 0))
        ax3 = plt.subplot2grid((3, 3), (0, 1), colspan=3, rowspan=3)

        ax0.set_title('Drawn disks')
        ax0.imshow(pic_array, origin='lower', cmap='gray_r')
        if disk_res[0] is not None:
            for disk_geom in disk_res[0]:
                ax0.add_patch(
                    PolygonPatch(
                        disk_geom,
                        fc='C0',
                        alpha=0.05,
                        zorder=1,
                    )
                )

        ax1.set_title('Drawn bulges')
        ax1.imshow(pic_array, origin='lower', cmap='gray_r')
        if bulge_res[0] is not None:
            for bulge_geom in bulge_res[0]:
                ax1.add_patch(
                    PolygonPatch(
                        bulge_geom,
                        fc='C1',
                        alpha=0.1,
                        zorder=1,
                    )
                )

        ax2.set_title('Drawn bars')
        ax2.imshow(pic_array, origin='lower', cmap='gray_r')
        if bar_res[0] is not None:
            for bar_geom in bar_res[0]:
                ax2.add_patch(
                    PolygonPatch(
                        bar_geom,
                        fc='C2',
                        alpha=0.1,
                        zorder=1,
                    )
                )

        ax3.set_title('Recovered components')
        ax3.imshow(pic_array, origin='lower', cmap='gray_r')
        if disk_res[2] is not None:
            ax3.add_patch(
                PolygonPatch(
                    disk_res[2],
                    fc='C0',
                    alpha=0.1,
                    zorder=1,
                )
            )
        if bulge_res[2] is not None:
            ax3.add_patch(
                PolygonPatch(
                    bulge_res[2],
                    fc='C1',
                    alpha=0.5,
                    zorder=1,
                )
            )
        if bar_res[2] is not None:
            ax3.add_patch(
                PolygonPatch(
                    bar_res[2],
                    fc='C2',
                    alpha=0.5,
                    zorder=1,
                )
            )
        for i, arm in enumerate(spirals):
            ax3.plot(*arm.T, c='C{}'.format((i+3)%10))

        for i, ax in enumerate((ax0, ax1, ax2, ax3)):
            plt.sca(ax)
            if i > 1:
                plt.xticks(xtick_positions[xtick_mask], xtick_labels[xtick_mask])
                plt.xlabel('Arcseconds from galaxy centre')
            else:
                plt.xticks([])
            plt.yticks(ytick_positions[ytick_mask], ytick_labels[ytick_mask])
            plt.ylabel('Arcseconds from galaxy centre')
        plt.suptitle('Combined galaxy')
        plt.subplots_adjust(bottom=0.1, right=0.8, top=0.9)
        outfile = 'clustered-components-images/{}.png'.format(subject_id)
        plt.savefig(outfile)
        plt.close()
        progress_bar.next()
    progress_bar.finish()
