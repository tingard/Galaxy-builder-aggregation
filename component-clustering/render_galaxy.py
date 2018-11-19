import numpy as np
from scipy.signal import convolve2d
from gzbuilderspirals.metric import v_calc_t, v_get_diff
from multiprocessing import Pool
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


# Many of these functions are copied and translated from
# https://github.com/zooniverse/Panoptes-Front-End/blob/master/app/features/modelling/galaxy-builder
# without optimization.

default_disk = {
    'mu': np.zeros(2) + 50,
    'roll': 0,
    'rEff': 100,
    'axRatio': 1,
    'c': 2,
}
default_spiral = {
    'i0': 1, 'spread': 1, 'falloff': 1,
}


# image manipulation
def asinh(px):
    return np.log(px + np.sqrt(1.0 + (px * px)))


def asinh_stretch(px, i=0.6):
    return asinh(px / i) / asinh(i)


# rendering functions
def calc_boxy_ellipse_dist(x, y, mu, roll, rEff, axRatio, c):
    xPrime = x * np.cos(roll) \
        - y * np.sin(roll) + mu[0] \
        - mu[0] * np.cos(roll) + mu[1] * np.sin(roll)
    yPrime = x * np.sin(roll) \
        + y * np.cos(roll) + mu[1] \
        - mu[1] * np.cos(roll) - mu[0] * np.sin(roll)
    # return a scaled version of the radius (multiplier is chosen so svg tool
    # doesn't impact badly on shown model component)
    multiplier = 3.0
    return multiplier * np.sqrt(
        np.power(axRatio / rEff, c) * np.power(np.abs(xPrime - mu[0]), c)
        + np.power(np.abs(yPrime - mu[1]), c) / np.power(rEff, c)
    )


def _b(n):
    # from https://arxiv.org/abs/astro-ph/9911078
    return 2 * n - 1/3 + 4/405/n \
        + 46/25515/n/n + 131/1148175/pow(n, 3) \
        - 2194697/30690717750/pow(n, 4)


def sersic2d(x, y, mu, roll, rEff, axRatio, c, i0, n):
    # https://www.cambridge.org/core/services/aop-cambridge-core/content/view/S132335800000388X
    return 0.5 * i0 * np.exp(
        _b(n) * (1 - np.power(
            calc_boxy_ellipse_dist(x, y, mu, roll, rEff, axRatio, c),
            1.0 / n
        ))
    )


def _sersic_comp(comp, x, y):
    return sersic2d(
        x, y, comp['mu'], comp['roll'], comp['rEff'],
        comp['axRatio'], comp['c'], comp['i0'], comp['n'])


def sersic_comp(comp, image_size=512, oversample_n=1):
    oversample_grid = np.meshgrid(
        np.linspace(0, image_size, image_size*oversample_n),
        np.linspace(0, image_size, image_size*oversample_n)
    )
    return _sersic_comp(
        comp,
        *oversample_grid
    ).reshape(
        image_size,
        oversample_n,
        image_size,
        oversample_n,
    ).mean(3).mean(1)


def length(a):
    return np.sqrt(np.add.reduce(a**2))


def get_distance_from_polyline(a, b):
    m = np.zeros((a.shape[0], b.shape[0] - 1, 3, 2))
    m[:, :, 0, :] = np.transpose(
        np.tile(a, [m.shape[1] + 1, 1, 1]),
        axes=[1, 0, 2]
    )[:, :-1, :]
    m[:, :, 1, :] = np.tile(b, [a.shape[0], 1, 1])[:, :-1, :]
    m[:, :, 2, :] = np.tile(
        np.roll(b, -1, axis=0), [a.shape[0], 1, 1]
    )[:, :-1, :]
    # t[i, j] = ((a[i] - b[j]) . (b[j + 1] - b[j])) / (b[j + 1] - b[j]|**2
    t = v_calc_t(np.array(m))
    return v_get_diff(t, m)


def do_batch_dist(args):
    return np.sqrt(get_distance_from_polyline(args[0], args[1]))


def spiral_arm(arm_points, params=default_spiral, disk=default_disk,
               image_size=512):
    cx, cy = np.meshgrid(np.arange(image_size), np.arange(image_size))
    batches = np.stack((cx, cy), axis=2)

    with Pool() as p:
        distances = np.array(
            p.map(
                do_batch_dist,
                ((batch, arm_points) for batch in batches),
            )
        )

    return params['i0'] * np.exp(-distances**2 * 0.1 / params['spread']) \
        * np.exp(
            -calc_boxy_ellipse_dist(
                cx, cy,
                disk['mu'],
                disk['roll'],
                disk['rEff'],
                disk['axRatio'],
                disk['c']
            ) / params['falloff']
        )


def compare_to_galaxy(arr, psf, galaxy):
    return asinh_stretch(
        galaxy - convolve2d(arr, psf, mode='same', boundary='symm')
    )


def post_process(arr, psf):
    return asinh_stretch(
        convolve2d(arr, psf, mode='same', boundary='symm'),
        0.5
    )


def plot_model(model, psf, galaxy_data):
    image_data = asinh_stretch(galaxy_data)
    difference_data = compare_to_galaxy(model, psf, galaxy_data)
    model_data = post_process(model, psf)
    data_min = np.min(np.stack((image_data, difference_data, model_data)))
    data_max = np.max(np.stack((image_data, difference_data, model_data)))
    most_extreme = max(np.abs(data_min), np.abs(data_max))
    kwargs = {
        'vmin': -most_extreme,
        'vmax': most_extreme,
        'origin': 'lower',
        'cmap': 'RdGy'
    }
    plt.figure(figsize=(17, 4))
    plt.subplot(131, label='galaxy-data')
    plt.imshow(asinh_stretch(galaxy_data), **kwargs)
    plt.colorbar()
    plt.subplot(132, label='model-data')
    plt.imshow(post_process(model, psf), **kwargs)
    plt.colorbar()
    plt.subplot(133, label='difference-data')
    plt.imshow(difference_data, **kwargs)
    plt.colorbar()
    plt.subplots_adjust(0, 0, 1.0, 1.0)
    return difference_data
