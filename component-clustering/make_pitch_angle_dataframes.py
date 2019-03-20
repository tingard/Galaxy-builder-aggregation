import os
import re
import numpy as np
import pandas as pd
import lib.galaxy_utilities as gu
from astropy.io import fits
from gzbuilderspirals.oo import Pipeline, Arm
import warnings
from astropy.utils.exceptions import AstropyWarning
warnings.simplefilter('ignore', category=AstropyWarning)


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
    try:
        p = Pipeline.load('lib/pipelines/{}.json'.format(subject_id))
    except FileNotFoundError:
        drawn_arms = gu.get_drawn_arms(subject_id, gu.classifications)
        gal, angle = gu.get_galaxy_and_angle(subject_id)
        pic_array, deprojected_image = gu.get_image(gal, subject_id, angle)
        p = Pipeline(drawn_arms, phi=angle, ba=gal['PETRO_BA90'],
                     image_size=pic_array.shape[0])
    arms = (
        Arm.load(os.path.join('lib/spiral_arms', f))
        for f in os.listdir('lib/spiral_arms')
        if re.match('^{}-[0-9]+.pickle$'.format(subject_id), f)
    )
    arms = [arm for arm in arms if not arm.FLAGGED_AS_BAD]

    pa = np.zeros(len(arms))
    sigma_pa = np.zeros(pa.shape)
    length = np.zeros(pa.shape)
    for i, arm in enumerate(arms):
        pa[i] = arm.pa
        length[i] = arm.length
        sigma_pa[i] = arm.sigma_pa
    if len(arms) == 0:
        return (
            np.nan, np.nan,
            np.stack(
                (np.tile(subject_id, len(pa)), pa, sigma_pa, length),
                axis=1
            )
        )
    combined_pa = (pa * length).sum() / length.sum()
    combined_sigma_pa = np.sqrt((length**2 * sigma_pa**2).sum()) / length.sum()
    return (
        combined_pa, combined_sigma_pa,
        np.stack((np.tile(subject_id, len(pa)), pa, sigma_pa, length), axis=1),
    )


if __name__ == '__main__':
    # open the GZ2 catalogue
    NSA_GZ = fits.open('./lib/NSA_GalaxyZoo.fits')

    subject_ids = np.loadtxt('lib/subject-id-list.csv', dtype='u8')

    pas = []
    arm_pas = []
    for i, subject_id in enumerate(subject_ids):
        if i % 50 == 0:
            print(i)
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
        pas.append([subject_id, pa, gzb_pa, gzb_pa_sigma])
        arm_pas.extend(details.tolist())
    pa_df = pd.DataFrame(
        pas,
        columns=(
            'subject_id', 'Hart pitch angle', 'GZB pitch angle', 'GZB sigma'
        ),
    ).set_index('subject_id')
    pa_df.to_pickle('pitch_angle_comparisons.pkl')
    # pa_df = pd.read_pickle('pitch_angle_comparisons.pkl')
    arm_pa_id = np.fromiter((a[0] for a in arm_pas), dtype=int)
    arm_df = pd.DataFrame(
        arm_pas,
        columns=('subject_id', 'pa', 'sigma_pa', 'length')
    )
    arm_df['hart_pa'] = pa_df['Hart pitch angle'].loc[
        arm_df['subject_id'].values
    ].values

    arm_df.to_pickle('arm_pitch_angles.pkl')

    sparcfire_path = os.path.abspath('../spiral-aggregation/sparcfire-fits')
    available_galaxies = [i for i in os.listdir(sparcfire_path)
                          if '.zip' not in i]
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
