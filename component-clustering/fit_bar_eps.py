import os
import json
import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.cluster import DBSCAN
import pandas as pd
from astropy.io import fits
import wrangle_classifications as wc
import lib.galaxy_utilities as gu
import lib.python_model_renderer.parse_annotation as pa
import lib.average_shape_helpers as ash
import get_average_shape as gas
import warnings
from astropy.utils.exceptions import AstropyWarning
warnings.simplefilter('ignore', category=AstropyWarning)


def get_pbar(gal):
    n = gal['t03_bar_a06_bar_debiased'] + gal['t03_bar_a07_no_bar_debiased']
    return gal['t03_bar_a06_bar_debiased'] / n


NSA_GZ = fits.open('./lib/NSA_GalaxyZoo.fits')

sid_list = sorted(np.loadtxt('lib/subject-id-list.csv', dtype='u8'))
to_iter = sid_list
geom_dict = {}
gz2_pbar = {}
all_distances = {}
for subject_id in to_iter:
    metadata = gu.meta_map.get(int(subject_id), {})
    gal = NSA_GZ[1].data[
        NSA_GZ[1].data['dr7objid'] == np.int64(metadata['SDSS dr7 id'])
    ]
    gz2_pbar[subject_id] = get_pbar(gal[0]) if len(gal) == 1 else np.nan

    annotations = gu.classifications[
        gu.classifications['subject_ids'] == subject_id
    ]['annotations'].apply(json.loads)
    models = annotations\
        .apply(ash.remove_scaling)\
        .apply(pa.parse_annotation)
    spirals = models.apply(lambda d: d.get('spiral', None))
    geoms = pd.DataFrame(
        models.apply(gas.get_geoms).values.tolist(),
        columns=('disk', 'bulge', 'bar')
    )
    geoms['spirals'] = spirals
    geom_dict[subject_id] = geoms
    all_distances[subject_id] = wc.gen_jaccard_distances(
        geoms['bar'].dropna().values
    )


def func(eps):
    eps = max(1E-4, eps)
    c = 0
    for subject_id in to_iter:
        distances = all_distances[subject_id]
        clf = DBSCAN(eps=eps, min_samples=ash.BAR_MIN_SAMPLES,
                     metric='precomputed')
        clf.fit(distances)
        if gz2_pbar[subject_id] < 0.2 and np.max(clf.labels_) < 0:
            # gz2 says unlikely to have a bar, and we have no bar
            c -= 1
        elif gz2_pbar[subject_id] > 0.5 and np.max(clf.labels_) >= 0:
            # gz2 says very likely to have a bar, and we have a bar
            c -= 1
    return c


res = minimize_scalar(func, bracket=(1E-4, 0.4, 2))

if res['success']:
    print('Optimal bar eps fitted: {}'.format(res['x']))
else:
    raise Exception('Fit did not converge')
