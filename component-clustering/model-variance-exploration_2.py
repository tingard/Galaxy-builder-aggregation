import numpy as np
import pandas as pd
import lib.galaxy_utilities as gu
import gzbuilderaggregation
from progress.bar import Bar
from multiprocessing import Pool
import argparse
import warnings
from astropy.utils.exceptions import AstropyWarning
warnings.simplefilter('ignore', category=AstropyWarning)

N_SPLITS = 5


def get_pa_from_arms(arms):
    try:
        p = arms[0].get_parent()
        return p.get_pitch_angle(arms)
    except IndexError:
        return (np.nan, np.nan)


def get_splits_df(ss_id, val_id, dr8id):
    gal, angle = gu.get_galaxy_and_angle(ss_id)
    cls_for_gal = gu.classifications.query(
        'subject_ids == {} | subject_ids == {}'.format(ss_id, val_id)
    )
    results = []
    for i in range(N_SPLITS):
        cls_sample = cls_for_gal.sample(30)
        results.append(
            gzbuilderaggregation.make_model(
                cls_sample,
                gal, angle,
            )
        )
    disk_df = pd.DataFrame([
        i[0]['disk'] for i in results if i[0]['disk'] is not None
    ])
    disk_df.columns = 'disk_' + disk_df.columns
    bulge_df = pd.DataFrame([
        i[0]['bulge'] for i in results if i[0]['bulge'] is not None
    ])
    bulge_df.columns = 'bulge_' + bulge_df.columns
    bar_df = pd.DataFrame([
        i[0]['bar'] for i in results if i[0]['bar'] is not None
    ])
    bar_df.columns = 'bar_' + bar_df.columns
    pa_df = pd.DataFrame(
        [get_pa_from_arms(i[-1]) for i in results],
        columns=('pa', 'sigma_pa')
    )

    gal_df = pd.concat((disk_df, bulge_df, bar_df, pa_df), axis=1, sort=False)
    return gal_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            'Perform Shuffle split variance explortation'
            ' on aggregate models'
        )
    )
    parser.add_argument('--nsplits', '-N', metavar='N', default=5,
                        help='Number of splits to use')
    args = parser.parse_args()

    N_SPLITS = int(args.nsplits)

    dr8ids, ss_ids, validation_ids = np.load('lib/duplicate_galaxies.npy').T

    out = []
    to_iter = np.stack((ss_ids, validation_ids, dr8ids), axis=-1)
    bar = Bar('Calculating aggregate models', max=len(dr8ids),
              suffix='%(percent).1f%% - %(eta)ds')
    try:
        for row in to_iter:
            try:
                out.append(get_splits_df(*row))
            except Exception as e:
                print('\n', row[0], e)
            bar.next()
        bar.finish()
    except KeyboardInterrupt:
        pass

    df = pd.concat(out, keys=dr8ids, sort=False)
    df.to_pickle('model-variances.pkl')
