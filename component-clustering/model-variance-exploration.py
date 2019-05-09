import json
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import ShuffleSplit
import lib.galaxy_utilities as gu
import lib.python_model_renderer.parse_annotation as pa
import lib.average_shape_helpers as ash
import get_average_shape as gas
from gzbuilderspirals import get_drawn_arms
from gzbuilderspirals.oo import Pipeline
from progress.bar import Bar
import warnings
from astropy.utils.exceptions import AstropyWarning
warnings.simplefilter('ignore', category=AstropyWarning)


def make_comparison(dr8id, ss_id, val_id):
    comp_file = 'model-variances/{}_components.pickle'.format(dr8id)
    pa_file = 'model-variances/{}_pa.pickle'.format(dr8id)
    if os.path.isfile(comp_file) and os.path.isfile(pa_file):
        return
    all_cls = gu.classifications.query(
        '(subject_ids == {}) or (subject_ids == {})'.format(
            ss_id, val_id
        )
    )
    all_models = all_cls['annotations'].apply(
        json.loads
    ).apply(
        ash.remove_scaling
    ).apply(
        pa.parse_annotation
    )
    all_geoms = pd.DataFrame(
        all_models.apply(gas.get_geoms).values.tolist(),
        columns=('disk', 'bulge', 'bar')
    )

    ss = ShuffleSplit(n_splits=20, test_size=0.5, random_state=0)
    split_models = []
    pas = []

    gal, angle = gu.get_galaxy_and_angle(ss_id)

    bar = Bar(str(dr8id), max=ss.n_splits, suffix='%(percent).1f%% - %(eta)ds')
    for i, (train_index, _) in enumerate(ss.split(all_geoms)):
        models = all_models.iloc[train_index]
        drawn_arms = get_drawn_arms((ss_id, val_id), all_cls.iloc[train_index])
        if len(drawn_arms) > 1:
            p = Pipeline(drawn_arms, phi=angle, ba=gal['PETRO_BA90'],
                         image_size=512, parallel=True)
            pas.append(p.get_pitch_angle(p.get_arms()))
        else:
            pas.append((np.nan, np.nan))
        geoms = all_geoms.iloc[train_index]
        labels = list(map(np.array, gas.cluster_components(geoms)))
        comps = gas.get_aggregate_components(
            geoms, models, labels
        )
        aggregate_disk, aggregate_bulge, aggregate_bar = comps
        split_models.append({
            'disk': aggregate_disk if aggregate_disk else None,
            'bulge': aggregate_bulge if aggregate_bulge else None,
            'bar': aggregate_bar if aggregate_bar else None,
        })
        bar.next()
    bar.finish()
    splits_df = []
    for model in split_models:
        model_comps = {}
        for key in ('disk', 'bulge', 'bar'):
            if model.get(key, None) is None:
                model[key] = {}
            mu = model[key].get('mu', (np.nan, np.nan))
            model_comps['{}-mux'.format(key)] = mu[0]
            model_comps['{}-muy'.format(key)] = mu[1]
            for param in ('roll', 'rEff', 'axRatio', 'i0', 'n', 'c'):
                model_comps['{}-{}'.format(key, param)] = (
                    model[key].get(param, np.nan)
                )
        splits_df.append(model_comps)
    splits_df = pd.DataFrame(splits_df)

    pas = pd.DataFrame(
        pas,
        columns=('pa', 'sigma_pa'),
        index=pd.Series(range(len(pas)), name='split_index')
    )
    splits_df.to_pickle(comp_file)
    pas.to_pickle(pa_file)


if __name__ == '__main__':
    data = np.load('lib/duplicate_galaxies.npy')
    dr8ids, ss_ids, validation_ids = data.T
    for dr8id, ss_id, val_id in data:
        make_comparison(dr8id, ss_id, val_id)
