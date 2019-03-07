import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import lib.galaxy_utilities as gu
from gzbuilderspirals import metric
from gzbuilderspirals.oo import Pipeline


def read_file(m_id):
    columns = ('R', )\
        + ('GAS_IC-V', 'GAS_IC-Ve', 'GAS_IC-N')\
        + ('GAS___-V', 'GAS___-Ve', 'GAS___-N')\
        + ('BTH_IC-V', 'BTH_IC-Ve', 'BTH_IC-N')\
        + ('BTH___-V', 'BTH___-Ve', 'BTH___-N')
    df = pd.read_csv(
        os.path.join('rotation_curves', '{}-gasrc.db'.format(m_id)),
        comment='#', sep='\s+', dtype=float, names=columns
    )
    return df


def get_shear(omega, R):
    return -np.gradient(np.log(omega), np.log(R))


def get_predicted_pa(shear):
    return np.arctan(2/5 * np.sqrt(4 - 2*shear) / shear)


def convert_arcsec_to_km(gal):
    H0 = 70.0  # km s^{-1} Mpc^{-1}
    c = 299792.458  # km s^{-1}

    # r = c * z / H0 * tan(theta) in Mpc -> * 3.086E19 to get km
    def arcsec_to_km(theta):
        return c * float(gal['Z']) / H0 * np.tan(theta * np.pi / 648000) * 3.086E19

    return arcsec_to_km


def get_gzb_pa(subject_id, gal, angle):
    gal, angle = gu.get_galaxy_and_angle(subject_id)
    drawn_arms = gu.get_drawn_arms(subject_id, gu.classifications)
    distances = gu.get_distances(subject_id)
    if distances is None:
        distances = metric.calculate_distance_matrix(drawn_arms)
        np.save('lib/distances/subject-{}.npy'.format(subject_id), distances)
    p = Pipeline(drawn_arms, phi=angle, ba=gal['PETRO_BA90'],
                 image_size=512, distances=distances)
    arms = [p.get_arm(i, clean_points=True)
            for i in range(max(p.db.labels_) + 1)]
    pa = np.zeros(len(arms))
    sigma_pa = np.zeros(pa.shape)
    length = np.zeros(pa.shape)
    for i, arm in enumerate(arms):
        pa[i] = arm.pa
        length[i] = arm.length
        sigma_pa[i] = arm.sigma_pa
    if len(arms) == 0:
        combined_pa = np.nan
        combined_sigma_pa = np.nan
    else:
        combined_pa = (pa * length).sum() / length.sum()
        combined_sigma_pa = np.sqrt((length**2 * sigma_pa**2).sum()) / length.sum()
    return combined_pa, combined_sigma_pa


if __name__ == '__main__':
    mangaid = '8247-12703'
    subject_id = 21096811
    gal, angle = gu.get_galaxy_and_angle(subject_id)
    unit_converter = convert_arcsec_to_km(gal)
    df = read_file('8247-12703')
    invalid_mask = df.values == -9999.0
    bad_rows = np.unique(np.where(invalid_mask)[0])
    df.iloc[bad_rows] = np.nan
    df = df.dropna()
    df['R'] = unit_converter(df['R'])
    keys = (
        'GAS_IC-V',
        'GAS___-V',
        'BTH_IC-V',
        'BTH___-V',
    )
    gzb_pa, gzb_sigma_pa = get_gzb_pa(subject_id, gal, angle)
    for key in keys:
        omega = df[key] / (2 * np.pi * df['R'])
        shear = get_shear(omega, df['R'].values)
        sa_pa = np.rad2deg(get_predicted_pa(shear.mean()))
        print('For key: {}'.format(key))
        print('\tSwing amplification PA: {:.4f}°'.format(sa_pa))
        print('\tGZB measured PA: {:.4f} ± {:.4f}°'.format(gzb_pa, gzb_sigma_pa))

    labels = (
        r'$H_\alpha$ velocity, fixed center & inclination',
        r'$H_\alpha$ velocity, varying center & inclination',
        r'$H_\alpha$ and stellar velocity, fixed center & inclination',
        r'$H_\alpha$ and stellar velocity',
    )
    fig, ax = plt.subplots(figsize=(8, 6))
    df.plot('R', ['Gas_IC-V', 'GAS___-V', 'BTH_IC-V', 'BTH___-V'],
            label=labels, ax=ax)
    plt.xlabel('Distance from galaxy centre [km]')
    plt.ylabel(r'Rotational velocity [$\mathrm{km}\mathrm{s}^{-1}$]')
    plt.savefig('{}_rotational-velocity_2.pdf'.format(mangaid))
