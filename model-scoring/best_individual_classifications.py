import numpy as np
from functools import partial
import json
import matplotlib.pyplot as plt
import lib.galaxy_utilities as gu
import lib.python_model_renderer.parse_annotation as pa
import lib.python_model_renderer.render_galaxy as rg
from shapely.geometry import Point, box
from shapely.affinity import rotate as shapely_rotate
from shapely.affinity import scale as shapely_scale
from shapely.affinity import translate as shapely_translate
from descartes import PolygonPatch


def transform_coords(c, galaxy_data=np.zeros((2, 2)), pix_size=1.0):
    return (c - galaxy_data.shape[0] / 2) / pix_size


def transform_patch(p, galaxy_data=np.zeros((2, 2)), pix_size=1.0):
    return shapely_scale(
        shapely_translate(p, xoff=-galaxy_data.shape[0]/2,
                          yoff=-galaxy_data.shape[1]/2),
        xfact=1/pix_size,
        yfact=1/pix_size,
        origin=(0, 0),
    )


def make_transforms(galaxy_data, pix_size):
    return (
        partial(transform_coords, galaxy_data=galaxy_data, pix_size=pix_size),
        partial(transform_patch, galaxy_data=galaxy_data, pix_size=pix_size),
    )


def get_geoms(model_details):
    disk = shapely_rotate(
        shapely_scale(
            Point(*model_details['disk']['mu']).buffer(1.0),
            xfact=model_details['disk']['rEff'],
            yfact=model_details['disk']['rEff'] * model_details['disk']['axRatio']
        ),
        -np.rad2deg(model_details['disk']['roll'])
    ) if model_details['disk'] is not None else None
    bulge = shapely_rotate(
        shapely_scale(
            Point(*model_details['bulge']['mu']).buffer(1.0),
            xfact=model_details['bulge']['rEff'],
            yfact=model_details['bulge']['rEff'] * model_details['bulge']['axRatio']
        ),
        -np.rad2deg(model_details['bulge']['roll'])
    ) if model_details['bulge'] is not None else None
    bar = shapely_rotate(
        box(
            model_details['bar']['mu'][0] - model_details['bar']['rEff'] / model_details['bar']['axRatio'],
            model_details['bar']['mu'][1] - model_details['bar']['rEff'],
            model_details['bar']['mu'][0] + model_details['bar']['rEff'] / model_details['bar']['axRatio'],
            model_details['bar']['mu'][1] + model_details['bar']['rEff'],
        ),
        -np.rad2deg(model_details['bar']['roll'])
    ) if model_details['bar'] is not None else None
    return disk, bulge, bar


def plot_residuals(residuals, galaxy_data, blank_data):
    plt.plot(residuals)
    plt.ylabel('Mean error per pixel')
    plt.xlabel('Annotation Index')
    plt.hlines(
        np.sqrt(np.sum(blank_data**2))/np.multiply.reduce(galaxy_data.shape),
        0, len(residuals)-1,
        'C1', label='Empty model'
    )
    plt.legend()
    plt.yscale('log')


def plot_model(model_data, galaxy_data, psf, model_details, imshow_kwargs,
               transform_coords, transform_patch):
    image_data = rg.asinh_stretch(galaxy_data)
    difference_data = rg.compare_to_galaxy(model_data, psf, galaxy_data)
    model_data = rg.post_process(model_data, psf)

    diff = 0.8 * galaxy_data - rg.convolve2d(model_data, psf, mode='same', boundary='symm')
    score = 100*np.exp(
        -300 / np.multiply.reduce(galaxy_data.shape)
        * rg.asinh_stretch(np.abs(diff))**2
    )
    disk, bulge, bar = get_geoms(model_details)

    imshow_kwargs = {
        **imshow_kwargs,
        'vmin': -1,
        'vmax': 1,
        'origin': 'lower',
        'cmap': 'RdGy',
    }

    fig, ((ax0, ax1), (ax2, ax3)) = plt.subplots(
        ncols=2, nrows=2, figsize=(10, 10), sharex=True, sharey=True
    )
    panel0 = ax0.imshow(image_data, **imshow_kwargs)
    ax1.imshow(image_data, **imshow_kwargs)
    for i, geom in enumerate((disk, bulge, bar)):
        if geom is not None:
            ax1.add_patch(
                PolygonPatch(transform_patch(geom), fc='C{}'.format(i), ec='k',
                             alpha=0.2, zorder=3)
            )
    for arm in model_details['spiral']:
        ax1.plot(*transform_coords(arm[0].T))
    ax2.imshow(model_data, **imshow_kwargs)
    ax3.imshow(difference_data, **imshow_kwargs)

    plt.subplots_adjust(right=0.9, wspace=0.1, hspace=0)
    cax = fig.add_axes([0.91, 0.14, 0.02, 0.73])
    plt.colorbar(panel0, cax=cax)

    # add a shared x-axis label and y-axis label
    fig.add_subplot(111, frameon=False)
    # hide tick and tick label of the big axes
    plt.tick_params(labelcolor='none', top=False, bottom=False,
                    left=False, right=False)
    plt.grid(False)
    plt.xlabel('Arcseconds from galaxy centre')
    plt.ylabel('Arcseconds from galaxy centre')
    plt.suptitle('Model score: {:.2f}'.format(score))


for subject_id in np.loadtxt('lib/subject-id-list.csv', dtype='u8'):
    print('Working on', subject_id)
    annotations = gu.classifications[
        gu.classifications['subject_ids'] == subject_id
    ]['annotations'].apply(json.loads).values.tolist()
    gal, angle = gu.get_galaxy_and_angle(subject_id)
    pic_array, deprojected_image = gu.get_image(gal, subject_id, angle)
    psf = gu.get_psf(subject_id)
    diff_data = gu.get_image_data(subject_id)
    galaxy_data = np.array(diff_data['imageData'])[::-1]
    size_diff = diff_data['width'] / diff_data['imageWidth']
    # arcseconds per pixel for zooniverse image
    pix_size = pic_array.shape[0] / (gal['PETRO_THETA'].iloc[0] * 4)
    # arcseconds per pixel for galaxy data
    pix_size2 = galaxy_data.shape[0] / (gal['PETRO_THETA'].iloc[0] * 4)

    imshow_kwargs = {
        'cmap': 'gray_r', 'origin': 'lower',
        'extent': (
            # left of image in arcseconds from centre
            -pic_array.shape[0]/2 / pix_size,
            pic_array.shape[0]/2 / pix_size,  # right...
            -pic_array.shape[1]/2 / pix_size,  # bottom...
            pic_array.shape[1]/2 / pix_size  # top...
        ),
    }

    tc, tp = make_transforms(galaxy_data, pix_size2)

    residuals = np.zeros(len(annotations))
    model_array = np.zeros(
        (len(annotations), diff_data['width'], diff_data['width'])
    )
    for i, annotation in enumerate(annotations):
        parsed_annotation = pa.parse_annotation(
            annotation, size_diff=size_diff
        )
        model = rg.calculate_model(parsed_annotation, diff_data['width'])
        model_array[i] = model
        difference_data = rg.compare_to_galaxy(model, psf, galaxy_data)
        residuals[i] = np.sqrt(np.sum(difference_data**2)) \
            / np.multiply.reduce(galaxy_data.shape)
    blank_data = rg.compare_to_galaxy(np.zeros_like(galaxy_data), psf, galaxy_data)

    plot_residuals(residuals, galaxy_data, blank_data)
    plt.savefig('model-scores/{}.pdf'.format(subject_id))
    plt.close()

    best_annotation = annotations[np.argmin(residuals)]
    best_annotation_parsed = pa.parse_annotation(best_annotation, size_diff=size_diff)
    model_data = model_array[np.argmin(residuals)]

    plot_model(model_data, galaxy_data, psf, best_annotation_parsed,
               imshow_kwargs, tc, tp)
    plt.savefig('best_residual/{}.pdf'.format(subject_id))
    plt.close()
