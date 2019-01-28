import numpy as np
import pandas as pd
import json
import lib.galaxy_utilities as gu
from astropy.io import fits


def get_bulge_fraction(gal):
    none = gal['t05_bulge_prominence_a10_no_bulge_debiased'],
    just = gal['t05_bulge_prominence_a11_just_noticeable_debiased'],
    obvious = gal['t05_bulge_prominence_a12_obvious_debiased'],
    dominant = gal['t05_bulge_prominence_a13_dominant_debiased'],
    return none + just < obvious + dominant


def get_bar_fraction(gal):
    n = gal['t03_bar_a06_bar_debiased'] + gal['t03_bar_a07_no_bar_debiased']
    return gal['t03_bar_a06_bar_debiased'] / n


with open('tmp_cls_dump.json') as f:
    classifications = json.load(f)

# open the GZ2 catalogue
NSA_GZ = fits.open('./lib/NSA_GalaxyZoo.fits')

available_sids = pd.read_csv('lib/subject-id-list.csv').values[:, 0]

bulge_fractions = []
bar_fractions = []
bar_lengths = []
for subject_id in available_sids:
    cls_for_s = [c for c in classifications
                 if str(subject_id) in c['links']['subjects']]

    metadata = gu.meta_map.get(int(subject_id), {})
    gal = NSA_GZ[1].data[
        NSA_GZ[1].data['dr7objid'] == np.int64(metadata['SDSS dr7 id'])
    ]
    if len(gal) != 1:
        print('Could not find object for id {}'.format(metadata['SDSS dr7 id']))
        bulge_fractions.append((np.nan, np.nan))
        bar_fractions.append((np.nan, np.nan))
        continue
    # how frequently do people draw bulges?
    has_bulge = []
    for c in cls_for_s:
        try:
            has_bulge.append(len(c['annotations'][1]['value'][0]['value']))
        except IndexError:
            pass

    gz2_bulge = get_bulge_fraction(gal)[0]
    bulge_fractions.append((sum(has_bulge) / len(has_bulge), gz2_bulge))

    # and onto the bars
    has_bar = []
    for c in cls_for_s:
        try:
            has_bar.append(len(c['annotations'][2]['value'][0]['value']))
        except IndexError:
            pass
    gz2_bar = get_bar_fraction(gal)[0]
    bar_fractions.append((sum(has_bar) / len(has_bar), gz2_bar))

    bar_fraction = get_bar_fraction(gal[0])
    with open('cluster-output/{}.json'.format(subject_id)) as f:
        model = json.load(f)
    if model['bar'] is not None:
        bar_length = model['bar']['width']
        bar_lengths.append((subject_id, bar_length))
    else:
        bar_length = np.nan
    bar_lengths.append((subject_id, bar_length))


bulge_df = pd.DataFrame(
    bulge_fractions,
    columns=('GZB fraction', 'GZ2 bulge dominated'), index=available_sids,
).dropna()
bulge_df.to_pickle('bulge_fractions.pkl')


bar_df = pd.DataFrame(
    bar_fractions,
    columns=('GZB fraction', 'GZ2 bar fraction'), index=available_sids
).dropna()
bar_df['Strongly barred'] = bar_df['GZ2 bar fraction'] > 0.5
bar_df['No bar'] = bar_df['GZ2 bar fraction'] < 0.2
bar_df.to_pickle('bar_fractions.pkl')


bar_length_df = pd.DataFrame(
    bar_lengths,
    columns=('subject_id', 'GZB bar length')
).set_index('subject_id')
bar_length_df['GZ2 bar fraction'] = (
    bar_df['GZ2 bar fraction'][bar_length_df.index]
)
bar_length_df['GZB fraction'] = bar_df['GZB fraction'][bar_length_df.index]
bar_length_df.to_pickle('bar_lengths.pkl')
