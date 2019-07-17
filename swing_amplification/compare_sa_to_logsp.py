import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import numpy as np
import gzbuilder_analysis.parsing as parsing
import gzbuilder_analysis.spirals as spirals
from scipy.integrate import odeint
from scipy.optimize import minimize
import lib.galaxy_utilities as gu

subject_id = 20902040

galaxy_classifcations = gu.classifications.query(
    'subject_ids == {}'.format(subject_id)
)
drawn_arms = spirals.get_drawn_arms(galaxy_classifcations)

gal, angle = gu.get_galaxy_and_angle(subject_id)
ba = gal['PETRO_BA90']
im = gu.get_image(subject_id)
psf = gu.get_psf(subject_id)
diff_data = gu.get_diff_data(subject_id)
pixel_mask = 1 - np.array(diff_data['mask'])[::-1]
galaxy_data = np.array(diff_data['imageData'])[::-1]
size_diff = diff_data['width'] / diff_data['imageWidth']

# functions for plotting
# tv = lambda v: parsing.transform_val(v, np.array(im).shape[0], gal['PETRO_THETA'])
# ts = lambda v: parsing.transform_shape(v, galaxy_data.shape[0], gal['PETRO_THETA'])
# ts_a = lambda v: parsing.transform_shape(v, galaxy_data.shape[0], gal['PETRO_THETA'])
# imshow_kwargs = dict(cmap='gray', origin='lower', extent=[tv(0), tv(np.array(im).shape[0])]*2)


# Swing amplification model (not using sklearn pipelines)
def _swing_amplification_dydt(r, theta, b):
    R = 2 * b * r
    s = np.sinh(R)
    return (
        2*np.sqrt(2) / 7 * r
        * np.sqrt(1 + R / s) / (1 - R / s)
    )


def fit_swing_amplified_spiral(theta, r):
    def f(p):
        # p = (b, r0)
        y = odeint(_swing_amplification_dydt, p[1], theta, args=(p[0],))[:, 0]
        return np.abs(y - r).sum()

    res = minimize(f, (0.1, 0.1))
    guess_b, guess_r0 = res['x']
    r_guess = odeint(_swing_amplification_dydt, guess_r0, theta,
                     args=(guess_b,))[:, 0]
    guess_sigma = (r - r_guess).std()

    return r_guess, {'b': guess_b, 'r0': guess_r0, 'sigma': guess_sigma}


p = spirals.oo.Pipeline(drawn_arms, phi=angle, ba=ba)
arms = p.get_arms()
for arm in arms:
    t_ = arm.t * arm.chirality
    o = np.argsort(t_)
    t, r = t_[o], arm.R[o]
    r_sa, res = fit_swing_amplified_spiral(t, r)
    logsp_r = arm.logsp_model.predict(arm.t.reshape(-1, 1))
    plt.plot(t, r, '.')
    plt.plot(*(arm.polar_logsp.T * [arm.chirality, 1]).T, label='Log Spiral')
    # plt.plot(t, r_sa, label='Swing amplified spiral')
    print('Logsp score:', mean_squared_error(arm.R, logsp_r))
    print('SwAmp score:', mean_squared_error(t, r_sa))

plt.legend()
