import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
import lib.galaxy_utilities as gu
from gzbuilderspirals import theta_from_pa
import read_curve

keys = (
    'GAS_IC-V', 'GAS___-V',
    'BTH_IC-V', 'BTH___-V',
)
labels = (
    r'$H_\alpha$ velocity, fixed center & inclination',
    r'$H_\alpha$ velocity, varying center & inclination',
    r'$H_\alpha$ and stellar velocity, fixed center & inclination',
    r'$H_\alpha$ and stellar velocity',
)


def main(mangaid, subject_id):
    gal, angle = read_curve.gu.get_galaxy_and_angle(subject_id)
    unit_converter = read_curve.convert_arcsec_to_km(gal)
    df = read_curve.read_file(mangaid)
    invalid_mask = df.values == -9999.0
    mask = np.any(invalid_mask, axis=1)
    df.iloc[mask] = np.nan
    df = df.dropna()
    df['R_arcsec'] = df['R']
    df['R'] = unit_converter(df['R'])

    drawn_arms = gu.get_drawn_arms(subject_id)
    arm_pipeline = read_curve.Pipeline(
        drawn_arms, phi=angle, ba=gal['PETRO_BA90'],
        image_size=512, parallel=True
    )
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

    sa_pas = []
    for i, (key, label) in enumerate(zip(keys, labels)):
        f = read_curve.tanh_model(df['R'].values, df[key].values)
        p = read_curve.least_squares(f, (160, 1E-17, 0), x_scale=(10, 1E-17, 0.1))['x']
        # Calculate shear from analytic solve of dln(Ω)/dln(R)
        shear = read_curve.shear_from_tanh(p[1], df['R'].values)
        sa_pa = np.rad2deg(read_curve.get_predicted_pa(shear))
        sa_pas.append(sa_pa)
        print('For key: {}'.format(key))
        print('\tRotation-predicted: {:.4f}°'.format(sa_pa[df['R'] > min_r].mean()))
        print('\tGZB measured PA: {:.4f} ± {:.4f}°'.format(gzb_pa, gzb_sigma_pa))

    r = df['R_arcsec'] / (float(gal['PETRO_THETA']) * 4)
    th = theta_from_pa(r.values, sa_pas[0])
    plt.plot(th, r)
    plt.show()


if __name__ == '__main__':
    mangaid = '8247-12703'
    subject_id = 21096811

    main(mangaid, subject_id)
