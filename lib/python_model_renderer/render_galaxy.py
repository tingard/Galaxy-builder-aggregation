import numpy as np
from copy import deepcopy
from scipy.signal import convolve2d
import matplotlib.pyplot as plt
from shapely.geometry import LineString, MultiPoint
from numba import jit
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

MODEL_COMPARISON_IMSHOW_KWARGS = {
    'vmin': -1,
    'vmax': 1,
    'origin': 'lower',
    'cmap': 'RdGy'
}


# image manipulation
@jit(nopython=True)
def asinh(px):
    return np.log(px + np.sqrt(1.0 + (px * px)))


@jit(nopython=True)
def asinh_stretch(px, i=0.6):
    return asinh(px / i) / asinh(i)


# rendering functions
@jit(nopython=True)
def calc_boxy_ellipse_dist(x, y, mu, roll, rEff, axRatio, c):
    xPrime = x * np.cos(roll) \
        - y * np.sin(roll) + mu[0] \
        - mu[0] * np.cos(roll) + mu[1] * np.sin(roll)
    yPrime = x * np.sin(roll) \
        + y * np.cos(roll) + mu[1] \
        - mu[1] * np.cos(roll) - mu[0] * np.sin(roll)
    # return a scaled version of the radius (multiplier is chosen so svg tool
    # doesn't impact badly on shown model component)
    return 3.0 * np.sqrt(
        np.power(axRatio / rEff, c) * np.power(np.abs(xPrime - mu[0]), c)
        + np.power(np.abs(yPrime - mu[1]), c) / np.power(rEff, c)
    )


@jit(nopython=True)
def _b(n):
    # from https://arxiv.org/abs/astro-ph/9911078
    return 2 * n - 1/3 + 4/405/n \
        + 46/25515/n/n + 131/1148175/n**3 \
        - 2194697/30690717750/n**4


@jit(nopython=True)
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
    if comp is None:
        return np.zeros((image_size, image_size))
    ds = np.linspace(
        0.5/oversample_n - 0.5,
        image_size - 0.5 - 0.5/oversample_n,
        image_size*oversample_n
    )
    cx, cy = np.meshgrid(ds, ds)
    return _sersic_comp(
        comp, cx, cy,
    ).reshape(
        image_size, oversample_n, image_size, oversample_n,
    ).mean(3).mean(1)


def spiral_distance_shapely(points, poly_line, output_shape=(256, 256)):
    m = MultiPoint(points)
    line = LineString(poly_line)
    correct_values = np.fromiter((i.distance(line) for i in m), count=len(m), dtype=float).reshape(output_shape)
    return correct_values


def numpy_squared_distance_to_point(P, poly_line):
    """
    f(t) = (1−t)A + tB − P
    t = [(P - A).(B - A)] / |B - A|^2
    """
    u = P - poly_line[:-1]
    v = poly_line[1:] - poly_line[:-1]
    dot = u[:, 0] * v[:, 0] + u[:, 1] * v[:, 1]
    t = np.clip(dot / (v[:, 0]**2 + v[:, 1]**2), 0, 1)
    sep = (v.T * t).T - u
    return np.min(sep[:, 0]**2 + sep[:, 1]**2)


_npsdtp_vfunc = np.vectorize(
    numpy_squared_distance_to_point,
    signature='(d),(n,d)->()'
)


def spiral_distance_numpy(points, poly_line, output_shape=(256, 256)):
    return np.sqrt(
        _npsdtp_vfunc(
            points, poly_line,
        )
    ).reshape(*output_shape)


@jit(nopython=True, fastmath=True, parallel=True)
def spiral_distance_numba(poly_line, distances=np.zeros((250, 250))):
    output_shape = distances.shape
    for i in range(output_shape[0]):
        for j in range(output_shape[1]):
            best = 1E30
            # for each possible pair of vertices
            for k in range(len(poly_line) - 1):
                ux = j - poly_line[k, 0]
                uy = i - poly_line[k, 1]
                vx = poly_line[k + 1, 0] - poly_line[k, 0]
                vy = poly_line[k + 1, 1] - poly_line[k, 1]
                dot = ux * vx + uy * vy
                t = dot / (vx**2 + vy**2)
                if t < 0.0:
                    t = 0.0
                elif t > 1.0:
                    t = 1.0
                d = (vx*t - ux)**2 + (vy*t - uy)**2
                if d < best:
                    best = d
            distances[i, j] = best
    return np.sqrt(distances)


def spiral_arm(arm_points, params=default_spiral, disk=default_disk,
               image_size=512):
    if disk is None or len(arm_points) < 2:
        return np.zeros((image_size, image_size))

    cx, cy = np.meshgrid(np.arange(image_size), np.arange(image_size))

    disk_arr = _sersic_comp(
        {**disk, 'i0': 1, 'rEff': disk['rEff'] / params['falloff']},
        cx, cy
    )

    arm_distances = spiral_distance_numba(
        arm_points,
        distances=np.zeros(disk_arr.shape),
    )

    return (
        params['i0']
        * np.exp(-arm_distances**2 * 0.1 / max(params['spread'], 1E-10))
        * disk_arr
    )


def calculate_model(parsed_annotation, image_size, oversample_n=5):
    disk_arr = sersic_comp(parsed_annotation['disk'],
                           image_size=image_size,
                           oversample_n=oversample_n)
    bulge_arr = sersic_comp(parsed_annotation['bulge'],
                            image_size=image_size,
                            oversample_n=oversample_n)

    bar_arr = sersic_comp(parsed_annotation['bar'],
                          image_size=image_size,
                          oversample_n=oversample_n)
    spirals_arr = np.add.reduce([
        spiral_arm(
            *s,
            parsed_annotation['disk'],
            image_size=image_size,
        )
        for s in parsed_annotation['spiral']
    ])
    model = disk_arr + bulge_arr + bar_arr + spirals_arr
    return model


# 0.8 multiplier was present in the original rendering code
def compare_to_galaxy(arr, psf, galaxy, pixel_mask=None, stretch=True):
    if pixel_mask is None:
        pixel_mask = np.ones_like(arr)
    D = (
        0.8 * galaxy * pixel_mask
        - convolve2d(arr, psf, mode='same', boundary='symm') * pixel_mask
    )
    return asinh_stretch(D) if stretch else D

def post_process(arr, psf):
    return asinh_stretch(
        convolve2d(arr, psf, mode='same', boundary='symm'),
        0.5
    )


def GZB_score(D):
    N = np.multiply.reduce(D.shape)
    return 100 * np.exp(-300 / N * np.sum(asinh(np.abs(D) / 0.6)**2 / asinh(0.6)))


def plot_model(model, psf, galaxy_data, imshow_kwargs, **kwargs):
    image_data = asinh_stretch(galaxy_data)
    difference_data = compare_to_galaxy(model, psf, galaxy_data)
    model_data = post_process(model, psf)
    imshow_kwargs = {
        **imshow_kwargs,
        'vmin': -1,
        'vmax': 1,
        'origin': 'lower',
        'cmap': 'RdGy',
    }

    fig, (ax0, ax1, ax2) = plt.subplots(
        ncols=3, figsize=(15, 5), sharex=True, sharey=True
    )
    panel0 = ax0.imshow(image_data, **imshow_kwargs)
    ax1.imshow(model_data, **imshow_kwargs)
    ax2.imshow(difference_data, **imshow_kwargs)

    plt.subplots_adjust(right=0.9, wspace=0.1, hspace=0)
    cax = fig.add_axes([0.91, 0.14, 0.02, 0.73])
    plt.colorbar(panel0, cax=cax)

    # add a shared x-axis label and y-axis label
    fig.add_subplot(111, frameon=False)
    # hide tick and tick label of the big axes
    plt.tick_params(labelcolor='none', top=False, bottom=False,
                    left=False, right=False)
    plt.grid(False)
    plt.suptitle(kwargs.get('title', ''))
    plt.xlabel(kwargs.get('xlabel', 'Arcseconds from galaxy centre'))
    plt.ylabel(kwargs.get('ylabel', 'Arcseconds from galaxy centre'))
    return difference_data


def sanitize_model(m):
    return {'spiral': m['spiral'], **{
        k: sanitize_param_dict(v)
        for k, v in m.items()
        if k is not 'spiral'
    }}


def sanitize_param_dict(p):
    if p is None:
        return p
    # rEff > 0
    # 0 < axRatio < 1
    # 0 < roll < np.pi (not 2*pi due to rotational symmetry)
    out = deepcopy(p)
    out['rEff'] = (
        abs(out['rEff'])
        * (abs(p['axRatio']) if abs(p['axRatio']) > 1 else 1)
    )
    out['axRatio'] = min(abs(p['axRatio']), 1 / abs(p['axRatio']))
    out['roll'] = p['roll'] % np.pi
    return out
