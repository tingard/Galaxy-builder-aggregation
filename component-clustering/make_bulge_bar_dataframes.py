import numpy as np
import pandas as pd
import json
import lib.galaxy_utilities as gu
from astropy.io import fits
import requests


def get_pnonebulge(gal):
    none = gal['t05_bulge_prominence_a10_no_bulge_debiased']
    # just = gal['t05_bulge_prominence_a11_just_noticeable_debiased'],
    # obvious = gal['t05_bulge_prominence_a12_obvious_debiased'],
    # dominant = gal['t05_bulge_prominence_a13_dominant_debiased'],
    return none


def get_pbulge(gal):
    none = gal['t05_bulge_prominence_a10_no_bulge_debiased']
    just = gal['t05_bulge_prominence_a11_just_noticeable_debiased']
    obvious = gal['t05_bulge_prominence_a12_obvious_debiased']
    dominant = gal['t05_bulge_prominence_a13_dominant_debiased']
    return none + just < obvious + dominant


def get_pbar(gal):
    n = gal['t03_bar_a06_bar_debiased'] + gal['t03_bar_a07_no_bar_debiased']
    return gal['t03_bar_a06_bar_debiased'] / n


skyserver_url = '/'.join((
    'http://skyserver.sdss.org',
    'dr13', 'en', 'tools', 'search', 'x_results.aspx',
))


def get_dr8_id(dr7id):
    payload = {
        'searchtool': 'SQL',
        'TaskName': 'Skyserver.Search.SQL',
        'syntax': 'NoSyntax',
        'cmd': 'SELECT dr8objid FROM PhotoObjDR7 WHERE dr7objid = {}'.format(
            dr7id
        ),
        'format': 'json',
        'TableName': '',
    }
    r = requests.get(
        skyserver_url,
        params=payload,
    )
    res = r.json()
    return res[0]['Rows'][0]['dr8objid'] if len(res[0]['Rows']) > 0 else np.nan


def has_comp(annotation, comp=0):
    try:
        drawn_shapes = annotation[comp]['value'][0]['value']
        return len(drawn_shapes) > 0
    except (IndexError, KeyError):
        return False


# open the GZ2 catalogue
NSA_GZ = fits.open('./lib/NSA_GalaxyZoo.fits')

available_sids = pd.read_csv('lib/subject-id-list.csv').values[:, 0]
bulge_fractions = []
bar_fractions = []
bar_lengths = []
for subject_id in available_sids:
    cls_for_s = gu.classifications.query(
        '(subject_ids == {}) & (workflow_version == 61.107)'.format(
            subject_id
        )
    )
    ann_for_s = cls_for_s['annotations'].apply(json.loads)
    metadata = gu.meta_map.get(int(subject_id), {})
    gal = NSA_GZ[1].data[
        NSA_GZ[1].data['dr7objid'] == np.int64(metadata['SDSS dr7 id'])
    ]
    if len(gal) != 1:
        print('Could not find object for id {}'.format(
            metadata['SDSS dr7 id'])
        )
        bulge_fractions.append((np.nan, np.nan))
        bar_fractions.append((np.nan, np.nan))
    else:
        # how frequently do people draw bulges?
        gzb_bulge = ann_for_s.apply(
            lambda v: has_comp(v, comp=1)
        ).sum() / len(ann_for_s)
        gz2_bulge = get_pbulge(gal[0])
        gz2_no_bulge = get_pnonebulge(gal[0])
        bulge_fractions.append((gzb_bulge, gz2_bulge, gz2_no_bulge))

        gzb_bar = ann_for_s.apply(
            lambda v: has_comp(v, comp=2)
        ).sum() / len(ann_for_s)
        gz2_pbar = get_pbar(gal[0])
        bar_fractions.append((gzb_bar, gz2_pbar))

    with open('cluster-output/{}.json'.format(subject_id)) as f:
        model = json.load(f)
    if model['bar'] is not None:
        bar_length = model['bar']['rEff']
    else:
        bar_length = np.nan
    bar_lengths.append(bar_length)

len(available_sids), len(bulge_fractions), len(bar_fractions), len(bar_lengths)
bulge_df = pd.DataFrame(
    bulge_fractions,
    columns=('GZB fraction', 'GZ2 bulge dominated', 'GZ2 no bulge'),
    index=pd.Series(available_sids, name='subject_id'),
)
bulge_df.to_pickle('bulge_fractions.pkl')


bar_df = pd.DataFrame(
    bar_fractions,
    columns=('GZB fraction', 'GZ2 bar fraction'),
    index=pd.Series(available_sids, name='subject_id'),
)
bar_df['Strongly barred'] = bar_df['GZ2 bar fraction'] > 0.5
bar_df['No bar'] = bar_df['GZ2 bar fraction'] < 0.2
bar_df.to_pickle('bar_fractions.pkl')

bar_lengths
bar_length_df = pd.DataFrame(
    bar_lengths,
    columns=('GZB bar length',),
    index=pd.Series(available_sids, name='subject_id'),
)
bar_length_df.head()
bar_length_df['GZ2 bar fraction'] = (
    bar_df['GZ2 bar fraction']
)
bar_length_df['GZB fraction'] = bar_df['GZB fraction']
bar_length_df.to_pickle('bar_lengths.pkl')
