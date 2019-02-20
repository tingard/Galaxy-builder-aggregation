import numpy as np
import json
from copy import deepcopy
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
import lib.galaxy_utilities as gu
import lib.python_model_renderer.render_galaxy as rg


subject_id = 20902040

gal, angle = gu.get_galaxy_and_angle(subject_id)
pic_array, deprojected_image = gu.get_image(gal, subject_id, angle)

psf = gu.get_psf(subject_id)
diff_data = gu.get_image_data(subject_id)
galaxy_data = 0.8 * np.array(diff_data['imageData'])[::-1]
image_size = galaxy_data.shape[0]
oversample_n = 3

size_diff = diff_data['width'] / diff_data['imageWidth']

# arcseconds per pixel for zooniverse image
pix_size = pic_array.shape[0] / (gal['PETRO_THETA'].iloc[0] * 4)

# arcseconds per pixel for galaxy data
pix_size2 = galaxy_data.shape[0] / (gal['PETRO_THETA'].iloc[0] * 4)


def transform_coords(c):
    return (c - galaxy_data.shape[0] / 2) / pix_size2


imshow_kwargs = {
    'cmap': 'gray_r', 'origin': 'lower',
    'extent': (
        -pic_array.shape[0]/2 / pix_size,  # left of image, arcsec from center
        pic_array.shape[0]/2 / pix_size,  # right...
        -pic_array.shape[1]/2 / pix_size,  # bottom...
        pic_array.shape[1]/2 / pix_size  # top...
    ),
}


# load in the data
def load_data():
    with open('example-annotation.json', 'r') as f:
        foo = json.load(f)

    annotation = deepcopy(foo)

    for k in ['disk', 'bulge', 'bar']:
        annotation[k]['mu'] = np.array(annotation[k]['mu'])
    for i in range(len(annotation['spiral'])):
        sp = annotation['spiral'][i]
        sp[0] = np.array(sp[0])
        annotation['spiral'][i] = sp
    return annotation


param_list = ('i0', 'rEff', 'n', 'c')
param_bounds = ((0, 0, 0.1, 1), (100, 500, 4, 8))


def _fit_to_leftovers(p, an, rest_of_model, galaxy_data, psf,
                      image_size=image_size, oversample_n=oversample_n):
    new_annotation = {**an, **{param_list[i]: p[i] for i in range(len(p))}}
    new_comp = rg.sersic_comp(new_annotation,
                              image_size=image_size,
                              oversample_n=oversample_n)
    model = rg.convolve2d(rest_of_model + new_comp, psf,
                          mode='same', boundary='symm')
    return (model - galaxy_data).reshape(-1)


def fit_comp(annotation, rest_of_model, galaxy_data, psf,
             image_size=image_size, oversample_n=oversample_n,
             c_to_fit=('i0', 'rEff')):
    p0 = [annotation[i] for i in param_list if i in c_to_fit]
    res = least_squares(
        _fit_to_leftovers,
        p0,
        args=(annotation, rest_of_model, galaxy_data, psf),
        bounds=[i[:len(p0)] for i in param_bounds]
    )
    return res


spiral_params = ('i0', 'spread', 'falloff')
spiral_bounds = ((0, 0, 0.2), (1E4, 1E2, 1E5))


def _fit_spiral_to_leftovers(p, an, disk, rest_of_model, galaxy_data, psf):
    points = an[0]
    new_annotation = {**an[1],
                      **{spiral_params[i]: p[i] for i in range(len(p))}}
    new_comp = rg.spiral_arm(
        points, new_annotation, disk,
        image_size=image_size,
    )
    model = rg.convolve2d(rest_of_model + new_comp, psf, mode='same',
                          boundary='symm')
    return (model - galaxy_data).reshape(-1)


def fit_spirals(spirals, disk, rest_of_model, galaxy_data, psf,
                image_size=image_size, oversample_n=oversample_n):
    # i0, spread, falloff
    res = []
    for arm in spirals:
        p0 = [arm[1][k] for k in spiral_params]
        res.append(
            least_squares(
                _fit_spiral_to_leftovers,
                p0,
                args=(arm, disk, rest_of_model, galaxy_data, psf),
                bounds=spiral_bounds,
            )
        )
    return res


def _fit_all(p, an, disk, rest_of_model, galaxy_data, psf):
    new_annotation = deepcopy(an)
    for k in ('disk', 'bulge', 'bar'):
        new_annotation
    new_annotation = {**an[1],
                      **{spiral_params[i]: p[i] for i in range(len(p))}}
    new_comp = rg.spiral_arm(
        p, new_annotation, disk,
        image_size=image_size,
    )
    model = rg.convolve2d(rest_of_model + new_comp, psf, mode='same',
                          boundary='symm')
    return (model - galaxy_data).reshape(-1)


def fit_everything(annotation, galaxy_data, psf):
    # god help me
    p0 = np.concatenate((
        [annotation['disk'][i] for i in param_list[:2]],
        [annotation['bulge'][i] for i in param_list[:3]],
        [annotation['bar'][i] for i in param_list],
        [
            annotation['spiral'][i][1][k]
            for i in range(len(annotation['spiral']))
            for k in spiral_params
        ],
    ))
    return p0
    pass


if __name__ == '__main__-':
    annotation = load_data()
    # render the model created by the volunteer
    disk = rg.sersic_comp(annotation['disk'],
                          image_size=image_size,
                          oversample_n=oversample_n)
    bulge = rg.sersic_comp(annotation['bulge'],
                           image_size=image_size,
                           oversample_n=oversample_n)
    bar = rg.sersic_comp(annotation['bar'],
                         image_size=image_size,
                         oversample_n=oversample_n)
    spiral_arms = np.add.reduce([
        rg.spiral_arm(*s, annotation['disk'], image_size=image_size)
        for s in annotation['spiral']
    ])
    model = rg.convolve2d(disk + bulge + bar + spiral_arms, psf, mode='same',
                          boundary='symm')

    # let's fine tune the disk parameters
    rest_of_model = rg.convolve2d(bulge + bar + spiral_arms, psf, mode='same',
                                  boundary='symm')
    new_disk_comps = fit_comp(annotation['disk'], galaxy_data - rest_of_model)
    new_disk_annotation = {
        **annotation['disk'], 'i0': new_disk_comps['x'][0], 'rEff': new_disk_comps['x'][1]
    }
    new_disk = rg.sersic_comp(new_disk_annotation,
                              image_size=image_size,
                              oversample_n=oversample_n)
    plt.subplot(121)
    plt.imshow(disk)
    plt.colorbar()
    plt.subplot(122)
    plt.imshow(new_disk)
    plt.colorbar()
    new_model = rg.convolve2d(new_disk + bulge + bar + spiral_arms, psf, mode='same', boundary='symm')
    plt.figure()
    plt.imshow(model - galaxy_data, vmin=-0.2, vmax=0.2, cmap='RdGy'); plt.colorbar()
    plt.figure()
    plt.imshow(new_model - galaxy_data, vmin=-0.2, vmax=0.2, cmap='RdGy'); plt.colorbar()
