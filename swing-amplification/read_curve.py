import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import lib.galaxy_utilities as gu
from scipy.optimize import least_squares
from scipy.optimize import minimize
from scipy.interpolate import interp1d
from gzbuilderspirals import xy_from_r_theta, fit_varying_pa
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


def tanh_model(x, y):
    # x = x / 1E17
    def f(p):
        return p[0]*np.tanh(p[1] * x) - y
    return f


def shear_from_tanh(b, R):
    bR = b * R
    return 1 - (4 * bR * np.exp(2 * bR)) / (np.exp(4 * bR) - 1)


def main(mangaid, subject_id):
    gal, angle = gu.get_galaxy_and_angle(subject_id)
    unit_converter = convert_arcsec_to_km(gal)
    df = read_file(mangaid)
    invalid_mask = df.values == -9999.0
    mask = np.any(invalid_mask, axis=1)
    df.iloc[mask] = np.nan
    df = df.dropna()
    df['R-arcsec'] = df['R']
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
        r'$H_\alpha$ and stellar velocity, varying centre and inclination',
    )
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
    sa_pa_datas = []
    for i, (key, label) in enumerate(zip(keys, labels)):
        f = tanh_model(df['R'].values, df[key].values)
        p = least_squares(
            f,
            (160, 1E-17),
            x_scale=(10, 1E-17)
        )['x']
        fitted[key] = f(p) + df[key].values
        # Calculate shear from analytic solve of dln(Ω)/dln(R)
        shear = shear_from_tanh(p[1], df['R'].values)
        omega = df[key] / (2 * np.pi * df['R'])
        shear_data = get_shear(omega[:-1], df['R'].values[:-1])

        plt.plot(df['R'], shear, c='C{}'.format(i % 10), label=label)
        plt.plot(
            np.stack((df['R'][:-1], df['R'][1:])).mean(axis=0),
            shear_data,
            '--', c='C{}'.format(i % 10)
        )

        sa_pa = np.rad2deg(get_predicted_pa(shear))
        sa_pa_data = np.rad2deg(get_predicted_pa(shear_data))
        sa_pas.append(sa_pa)
        sa_pa_datas.append(sa_pa_data)
        print('For key: {}'.format(key))
        msk = (df['R'] > min_r) & (df['R'] < max_r)
        print('\tRotation-predicted: {:.4f}°'.format(sa_pa[msk].mean()))
        print('\tGZB measured PA: {:.4f} ± {:.4f}°'.format(
            gzb_pa, gzb_sigma_pa
        ))

    plt.plot([], [], 'k-', label=r'Analytic differentiation')
    plt.plot([], [], 'k--', label='Numerical differentiation')

    plt.xlabel('Distance from galaxy centre [km]')
    plt.ylabel(r'Shear rate, $\Gamma$')
    plt.legend()
    plt.savefig('{}_shear.pdf'.format(mangaid), bbox_inches='tight')
    plt.close()

    scale = 4 * float(gal['PETRO_THETA'])
    zoo_coords_r = df['R-arcsec'].values / scale
    np.save('pavr', np.stack((zoo_coords_r, sa_pas[0]), axis=1))

    imshow_kwargs = {
        'cmap': 'gray',
        'origin': 'lower',
        'extent': [-0.5 * scale, 0.5 * scale] * 2,
    }
    pic_array, _ = gu.get_image(gal, subject_id, angle)
    fig, ax = plt.subplots(ncols=1, figsize=(5, 5))
    plt.imshow(pic_array, **imshow_kwargs)
    for i, arm in enumerate(arms):
        varying_arm_t = fit_varying_pa(
            arm, zoo_coords_r, np.stack(sa_pas).mean(axis=0)
        )
        t_predict = np.linspace(varying_arm_t.min(), varying_arm_t.max(), 100)
        f = interp1d(varying_arm_t, zoo_coords_r)
        varying_arm = xy_from_r_theta(f(t_predict), t_predict)

        log_spiral = xy_from_r_theta(*np.flipud(arm.polar_logsp))
        plt.plot(*arm.deprojected_coords.T * scale, '.', markersize=1, alpha=1)
        plt.plot(*log_spiral * scale, c='r', linewidth=3, alpha=0.8)
        plt.plot(*varying_arm * scale, c='g', linewidth=3, alpha=0.8)
    # plots for legend
    plt.plot([], [], c='g', linewidth=3, alpha=0.8,
             label='Swing-amplified spiral')
    plt.plot([], [], c='r', linewidth=3, alpha=0.8,
             label='Logarithmic spiral')
    plt.axis('equal')
    plt.xlabel('Arcseconds from galaxy centre')
    plt.ylabel('Arcseconds from galaxy centre')
    plt.xlim(-25, 25)
    plt.ylim(-25, 25)
    plt.legend()
    plt.savefig('{}_varying-pa.pdf'.format(mangaid), bbox_inches='tight')
    plt.close()
    return

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
    plt.close()

    fig, ax = plt.subplots(figsize=(8, 6))
    # df.plot('R', keys, label=labels, ax=ax)
    for i, key in enumerate(keys):
        plt.fill_between(
            df['R'].values,
            df[key].values - df[key + 'e'].values,
            df[key].values + df[key + 'e'].values,
            color='C{}'.format(i % 10),
            alpha=0.1,
        )
        plt.plot(df['R'].values, df[key].values, '--', c='C{}'.format(i % 10))
        plt.plot(df['R'].values, fitted[key], c='C{}'.format(i % 10))
    for i, label in enumerate(labels):
        plt.plot([], [], c='C{}'.format(i % 10), label=label)
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
