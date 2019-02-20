import os
import re
import json
import numpy as np
import pandas as pd
import lib.galaxy_utilities as gu
from astropy.io import fits
from gzbuilderspirals import cleaning, pipeline, metric
from gzbuilderspirals.oo import Pipeline


def hart_wavg(gal):
    return (np.hstack((
        gal['t10_arms_winding_a28_tight_debiased'],
        gal['t10_arms_winding_a29_medium_debiased'],
        gal['t10_arms_winding_a30_loose_debiased'],
    )) * (np.arange(3) + 1)).sum()


def hart_mavg(gal):
    return (np.hstack((
        gal['t11_arms_number_a31_1_debiased'],
        gal['t11_arms_number_a32_2_debiased'],
        gal['t11_arms_number_a33_3_debiased'],
        gal['t11_arms_number_a34_4_debiased'],
        gal['t11_arms_number_a36_more_than_4_debiased'],
    )) * (np.arange(5) + 1)).sum()


def get_gal_pa(subject_id):
    drawn_arms = gu.get_drawn_arms(subject_id, gu.classifications)
    gal, angle = gu.get_galaxy_and_angle(subject_id)
    pic_array, deprojected_image = gu.get_image(gal, subject_id, angle)
    distances = gu.get_distances(subject_id)
    if distances is None:
        print('\t- Calculating distances')
        distances = metric.calculate_distance_matrix(drawn_arms)
        np.save('./lib/distances/subject-{}.npy'.format(subject_id), distances)

    p = Pipeline(drawn_arms, phi=angle, ba=gal['PETRO_BA90'], image_size=pic_array.shape[0],
                 distances=distances)
    arms = [p.get_arm(i, clean_points=True) for i in range(max(p.db.labels_) + 1)]

    pa = np.zeros(len(arms))
    sigma_pa = np.zeros(pa.shape)
    length = np.zeros(pa.shape)
    for i, arm in enumerate(arms):
        pa[i] = arm.pa
        length[i] = arm.length
        sigma_pa[i] = arm.sigma_pa
    if len(arms) == 0:
        return np.nan, np.nan, np.stack((pa, sigma_pa, length), axis=1)
    combined_pa = (pa * length).sum() / length.sum()
    combined_sigma_pa = np.sqrt((length**2 * sigma_pa**2).sum()) / length.sum()
    return combined_pa, combined_sigma_pa, np.stack((pa, sigma_pa, length), axis=1)


with open('tmp_cls_dump.json') as f:
    classifications = json.load(f)

# open the GZ2 catalogue
NSA_GZ = fits.open('./lib/NSA_GalaxyZoo.fits')

subject_ids = np.loadtxt('lib/subject-id-list.csv', dtype='u8')

pas = np.zeros((len(subject_ids), 4))
pas2 = []
for i, subject_id in enumerate(subject_ids):
    metadata = gu.meta_map.get(int(subject_id), {})
    gz2_gal = NSA_GZ[1].data[
        NSA_GZ[1].data['dr7objid'] == np.int64(metadata['SDSS dr7 id'])
    ]
    if len(gz2_gal) < 1:
        pa = np.nan
    else:
        wavg = hart_wavg(gz2_gal)
        mavg = hart_mavg(gz2_gal)
        pa = 6.37 * wavg + 1.30 * mavg + 4.34

    gzb_pa, gzb_pa_sigma, details = get_gal_pa(subject_id)
    pas[i] = [subject_id, pa, gzb_pa, gzb_pa_sigma]

    pas2.append((subject_id, *details))
pa_df = pd.DataFrame(
    pas[:, 1:],
    columns=['Hart pitch angle', 'GZB pitch angle', 'GZB sigma'],
    index=pas[:, 0].astype(int)
).dropna()

pa_df.to_pickle('pitch_angle_comparisons.pkl')

arm_pas = [
    i for i in np.load('pitch_angles_list.npy')
    if len(i) > 1
]
arm_pa_id = np.fromiter((a[0] for a in arm_pas), dtype=int)
arm_df = pd.DataFrame([
    (i[0], *j)
    for i in arm_pas
    for j in i[1:]
], columns=('subject_id', 'pa', 'sigma_pa', 'length'))
arm_df['hart_pa'] = pa_df['Hart pitch angle'].loc[
    arm_df['subject_id']
].values

arm_df.to_pickle('arm_pitch_angles.pkl')


sparcfire_path = os.path.abspath('../spiral-aggregation/sparcfire-fits')
available_galaxies = [i for i in os.listdir(sparcfire_path) if '.zip' not in i]
sf = np.zeros((len(available_galaxies), 2), dtype=float)
sids = np.zeros(len(available_galaxies), dtype=int)
for i, g in enumerate(available_galaxies):
    sids[i] = int(re.search('-([0-9]+)_', g).group(1))
    data = pd.read_csv(os.path.join(sparcfire_path, g, 'galaxy.csv'))
    sf_angle = data[' pa_alenWtd_avg_domChiralityOnly'].values[0]
    sf_std = data[' paErr_alenWtd_stdev_domChiralityOnly'].values[0]
    sf[i] = [np.abs(sf_angle), sf_std]

sf_df = pd.DataFrame(sf, index=sids, columns=('pa', 'sigma_pa'))
sf_df.to_pickle('sparcfire_pitch_angles.pkl')
