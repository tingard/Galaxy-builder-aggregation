import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import json
from pprint import pprint
from panoptes_aggregation.extractors.shape_extractor import shape_extractor
from panoptes_aggregation.extractors.utilities import annotation_by_task
from panoptes_aggregation.reducers.shape_reducer_dbscan import shape_reducer_dbscan
from panoptes_aggregation.reducers.shape_reducer_hdbscan import shape_reducer_hdbscan
import lib.galaxy_utilities as gu
import wrangle_classifications as wc

with open('tmp_cls_dump.json') as f:
    classifications = json.load(f)

def get_cls(subject_id):
    classifications_for_subject = [
        c for c in classifications
        if c['links']['subjects'][0] == str(subject_id)
    ]
    # print('Found {} classifications for subject_id {}'.format(
    #     len(classifications_for_subject),
    #     subject_id,
    # ))
    annotations_for_subject = [i['annotations'] for i in classifications_for_subject]
    return annotations_for_subject


def get_disk(subject_id):
    gal, angle = gu.get_galaxy_and_angle(subject_id)

    psf = gu.get_psf(subject_id)
    annotations_for_subject = get_cls(subject_id)
    disks = [a[0] for a in annotations_for_subject if len(a) == 4]
    converted_disks = [wc.convert_shape(d) for d in disks]
    kwargs_extractor = {
        'task': 'disk', 'shape': 'ellipse',
        'details': {'disk_tool0': [None, 'slider_extractor']},
    }
    extracted_disks = [
        shape_extractor(
            annotation_by_task({ 'annotations': [d] }),
            **kwargs_extractor
        )
        for d in converted_disks
    ]
    eps = 50
    for i in range(20):
        kwargs_reducer = {
            'shape': 'ellipse',
            'details': {'disk_tool0': [None, 'slider_reducer']},
            'eps': eps, 'symmetric': True, 'min_samples': 5,
        }
        disk_clustering_result = shape_reducer_dbscan(
            extracted_disks,
            **kwargs_reducer,
        )['frame0']
        if len(np.unique(disk_clustering_result['disk_tool0_cluster_labels'])) > 2: # -1 and 0 wanted
            eps *= 0.95
        else:
            eps *= 1.1
    try:
        label = 0
        disk_kwargs = {
            'xy': (disk_clustering_result['disk_tool0_clusters_x'][label], disk_clustering_result['disk_tool0_clusters_y'][label]),
            'width': disk_clustering_result['disk_tool0_clusters_rx'][label],
            'height': disk_clustering_result['disk_tool0_clusters_ry'][label],
            'angle': -disk_clustering_result['disk_tool0_clusters_angle'][label],
        }
        final_disk = Ellipse(
            **disk_kwargs,
            ec='C{}'.format(label),
            linewidth=3,
            fc='none',
        )
        ba = disk_kwargs['width'] / disk_kwargs['height']
        ba = min(ba, 1/ba)
        return gal['SERSIC_BA'].iloc[0], ba
    except KeyError as e:
        print(e)
        print(disk_clustering_result)

if __name__ == "__main__":
    available_ids = np.loadtxt('lib/subject-id-list.csv')
    out = [('Sersic ID', 'Sersic BA', 'GZB axis ratio')]
    for id in available_ids:
        bas = get_disk(int(id))
        out.append([int(id), *bas])
    with open('disk-axis-ratios.csv','w') as f:
        f.write(','.join('"{}"'.format(s) for s in out[0]))
        f.write('\n')
        f.write('\n'.join(','.join(map(str,i)) for i in out[1:]))
