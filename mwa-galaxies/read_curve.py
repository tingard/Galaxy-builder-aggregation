import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import lib.galaxy_utilities as gu
from scipy.optimize import least_squares
from gzbuilderspirals import metric
from gzbuilderspirals.oo import Pipeline
import warnings
from astropy.utils.exceptions import AstropyWarning
warnings.simplefilter('ignore', category=AstropyWarning)


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
    p = Pipeline(drawn_arms, phi=angle, ba=gal['PETRO_BA90'],
                 image_size=512, parallel=True)
    arms = p.get_arms()
    combined_pa, combined_sigma_pa = p.get_pitch_angle(arms)
    return combined_pa, combined_sigma_pa


def tanh_model(x, y):
    # x = x / 1E17
    def f(p):
        return p[0]*np.tanh(p[1] * (x - p[2])) - y
    return f


def shear_from_tanh(b, R):
    bR = b * R
    return 1 - (4 * bR * np.exp(2 * bR)) / (np.exp(4 * bR) - 1)


def main(mangaid, subject_id):
    gal, angle = gu.get_galaxy_and_angle(subject_id)
    unit_converter = convert_arcsec_to_km(gal)
    df = read_file('8247-12703')
    invalid_mask = df.values == -9999.0
    mask = np.any(invalid_mask, axis=1)
    df.iloc[mask] = np.nan
    df = df.dropna()
    df['R'] = unit_converter(df['R'])
    keys = (
        'GAS_IC-V',
        'GAS___-V',
        'BTH_IC-V',
        'BTH___-V',
    )
    labels = (
        r'$H_\alpha$ velocity, fixed center & inclination',
        r'$H_\alpha$ velocity, varying center & inclination',
        r'$H_\alpha$ and stellar velocity, fixed center & inclination',
        r'$H_\alpha$ and stellar velocity',
    )

    gal, angle = gu.get_galaxy_and_angle(subject_id)
    drawn_arms = gu.get_drawn_arms(subject_id, gu.classifications)
    arm_pipeline = Pipeline(drawn_arms, phi=angle, ba=gal['PETRO_BA90'],
                            image_size=512, parallel=True)
    arms = arm_pipeline.get_arms()
    gzb_pa, gzb_sigma_pa = arm_pipeline.get_pitch_angle(arms)
    arm_details = [
        {
            'pa': arm.pa, 'sigma_pa': arm.sigma_pa,
            'min_r': unit_converter(
                np.linalg.norm(arm.log_spiral - (256, 256), axis=1).min()
                * float(gal['PETRO_THETA']) * 4 / 512
             ),
             'max_r': unit_converter(
                 np.linalg.norm(arm.log_spiral - (256, 256), axis=1).max()
                 * float(gal['PETRO_THETA']) * 4 / 512
              )
         }
        for arm in arms
    ]
    min_r = min(a['min_r'] for a in arm_details)
    max_r = max(a['max_r'] for a in arm_details)
    fitted = {}
    fig, ax = plt.subplots(figsize=(8, 6))
    sa_pas = []
    for i, (key, label) in enumerate(zip(keys, labels)):
        f = tanh_model(df['R'].values, df[key].values)
        p = least_squares(f, (160, 1E-17, 0), x_scale=(10, 1E-17, 0.1))['x']
        fitted[key] = f(p) + df[key].values
        # Calculate shear from analytic solve of dln(Ω)/dln(R)
        shear = shear_from_tanh(p[1], df['R'].values)
        omega = df[key] / (2 * np.pi * df['R'])
        shear2 = get_shear(omega[:-1], df['R'].values[:-1])

        plt.plot(df['R'], shear, c='C{}'.format(i%10), label=label)
        plt.plot(df['R'][:-1], shear2, '--', c='C{}'.format(i%10))

        sa_pa = np.rad2deg(get_predicted_pa(shear))
        sa_pas.append(sa_pa)
        print('For key: {}'.format(key))
        print('\tRotation-predicted: {:.4f}°'.format(sa_pa[df['R'] > min_r].mean()))
        print('\tGZB measured PA: {:.4f} ± {:.4f}°'.format(gzb_pa, gzb_sigma_pa))

    plt.plot([], [], 'k-', label=r'Analytic differentiation')
    plt.plot([], [], 'k--', label='Numerical differentiation')

    plt.xlabel('Distance from galaxy centre [km]')
    plt.ylabel(r'Shear rate, $\Gamma$')
    plt.legend()
    plt.savefig('{}_shear.pdf'.format(mangaid), bbox_inches='tight')
    plt.close()
    fig, ax = plt.subplots(figsize=(8, 6))
    for sa_pa, label in zip(sa_pas, labels):
        plt.plot(df['R'], sa_pa, label=label)
    for row in arm_details:
        plt.hlines(row['pa'], row['min_r'], row['max_r'])
        plt.fill_between(
            np.linspace(row['min_r'], row['max_r'], 2),
            row['pa'] - row['sigma_pa'],
            row['pa'] + row['sigma_pa'],
            color='k',
            alpha=0.2,
        )
    plt.legend()
    plt.xlabel('Distance from galaxy centre [km]')
    plt.ylabel('Pitch angle [degrees]')
    plt.savefig('{}_pa.pdf'.format(mangaid), bbox_inches='tight')
    fig, ax = plt.subplots(figsize=(8, 6))
    # df.plot('R', keys, label=labels, ax=ax)
    for i, key in enumerate(keys):
        plt.plot(df['R'].values, df[key].values, '--', c='C{}'.format(i%10))
        plt.plot(df['R'].values, fitted[key], c='C{}'.format(i%10))
    for i, label in enumerate(labels):
        plt.plot([], [], c='C{}'.format(i%10), label=label)
    plt.plot([], [], 'k-', label=r'$A\tanh(bR)$ model')
    plt.plot([], [], 'k--', label='Data')
    plt.legend()
    plt.xlabel('Distance from galaxy centre [km]')
    plt.ylabel(r'Rotational velocity [$\mathrm{km}\mathrm{s}^{-1}$]')
    plt.savefig('{}_rotational-velocity_2.pdf'.format(mangaid), bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    mangaid = '8247-12703'
    subject_id = 21096811
    main(mangaid, subject_id)
