import numpy as np
from functools import partial
import json
import matplotlib.pyplot as plt
import lib.galaxy_utilities as gu
import lib.python_model_renderer.parse_annotation as pa
import lib.python_model_renderer.render_galaxy as rg
from sklearn.metrics import mean_squared_error
from shapely.geometry import Point, box
from shapely.affinity import rotate as shapely_rotate
from shapely.affinity import scale as shapely_scale
from shapely.affinity import translate as shapely_translate
from descartes import PolygonPatch
from progress.bar import Bar
import warnings
from astropy.utils.exceptions import AstropyWarning
warnings.simplefilter('ignore', category=AstropyWarning)


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


def __score(D):
    N = np.multiply.reduce(D.shape)
    return 100 * np.exp(-300 / N * np.sum(rg.asinh(np.abs(D) / 0.6)**2 / rg.asinh(0.6)))


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


def plot_model(model_data, galaxy_data, psf, model_details, pixel_mask,
               imshow_kwargs, transform_coords, transform_patch, best_cls):
    image_data = rg.asinh_stretch(galaxy_data)
    difference_data = rg.compare_to_galaxy(model_data, psf, galaxy_data) * pixel_mask
    diff = 0.8 * galaxy_data - rg.convolve2d(model_data, psf, mode='same', boundary='symm')
    diff *= pixel_mask
    score = rg.GZB_score(diff)
    scaled_model_data = rg.post_process(model_data, psf)

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
    ax2.imshow(scaled_model_data, **imshow_kwargs)
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
    plt.suptitle('User: {}. Model score: {:.2f}'.format(
        best_cls['user_name'], score
    ))


def get_best_classification(subject_id, should_plot=False, should_save=False):
    # grab all the required metadata for this galaxy
    psf = gu.get_psf(subject_id)
    diff_data = gu.get_image_data(subject_id)
    pixel_mask = 1 - np.array(diff_data['mask'])[::-1]
    galaxy_data = np.array(diff_data['imageData'])[::-1]
    size_diff = diff_data['width'] / diff_data['imageWidth']

    def _lf(rendered_model, y=galaxy_data):
        Y = rg.convolve2d(rendered_model, psf, mode='same', boundary='symm') * pixel_mask
        return mean_squared_error(Y.flatten(), 0.8 * (y * pixel_mask).flatten())

    classifications = gu.classifications.query(
        'subject_ids == {}'.format(subject_id)
    )
    annotations = classifications['annotations'].apply(json.loads)
    models = annotations.apply(pa.parse_annotation, size_diff=size_diff)
    rendered_models = models.apply(
        rg.calculate_model,
        args=(diff_data['width'],)
    )
    scores = rendered_models.apply(_lf)
    best_index = scores.idxmin()
    best_cls = classifications.loc[best_index]
    best_model = models.loc[best_index]
    best_rendered_model = rendered_models.loc[best_index]

    if should_plot:
        gal, angle = gu.get_galaxy_and_angle(subject_id)
        pic_array, deprojected_image = gu.get_image(gal, subject_id, angle)
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
        plot_model(best_rendered_model, galaxy_data, psf, best_model, pixel_mask,
                   imshow_kwargs, tc, tp, best_cls)
        plt.savefig('best_residual/{}.pdf'.format(subject_id))
        plt.close()
    if should_save:
        with open('best_annotation/{}.json'.format(subject_id), 'w') as f:
            f.write(json.dumps(pa.make_json(best_model)))

    return best_cls


if __name__ == '__main__':
    with open('lib/best-classifications.json') as f:
        d = json.load(f)
    done = d.keys()
    to_iter = np.sort(np.loadtxt('lib/subject-id-list.csv', dtype='u8'))
    bar = Bar('Calculating models', max=len(to_iter), suffix='%(percent).1f%% - %(eta)ds')
    d = {}
    try:
        for subject_id in to_iter:
            # if subject_id in d.keys():
            #     continue
            c = get_best_classification(subject_id, should_plot=True,
                                        should_save=True)
            d[str(subject_id)] = int(c.classification_id)
            bar.next()
    except KeyboardInterrupt:
        with open('lib/best-classifications.json', 'w') as f:
            f.write(json.dumps(d, indent=1))
    bar.finish()
