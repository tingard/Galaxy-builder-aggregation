import string, os, shutil, json
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS
import astropy.units as u
import lib.python_model_renderer.parse_annotation as pa
import lib.python_model_renderer.render_galaxy as rg
import lib.galaxy_utilities as gu
import sdssCutoutGrab as scg
from cutout_utils import save_cutout, source_extract_image


with open('./galfit_template.tpl') as f:
    galfit_formatter = string.Template(f.read())

with open('./sersic_component_template.tpl') as f:
    component_formatter = string.Template(f.read())


def get_original_fits(frame):
    return os.path.abspath(os.path.join(
        'lib/fitsImages/',
        '{run}/{camcol}/frame-{band}-{run:06d}-{camcol}-{field:04d}.fits'.format(band='r', **frame),
    ))


# TODO: as not using montaged image, positions need to change
def create_object(component, original_wcs, cutout_wcs):
    if component is None:
        return ''
    position_sky = original_wcs.all_pix2world(component['mu'][np.newaxis, :], 0)
    position_cutout = cutout_wcs.all_world2pix(position_sky, 0)
    # print(component['mu'], '-> Ra, Dec:', position_sky, '->', position_cutout)
    components_params = {
        'mux': position_cutout[0][0],
        'muy': position_cutout[0][1],
        'magnitude': 20.0, # np.sum(rg.sersic_comp(component, image_size=512, oversample_n=3)),
        'rEff': component['rEff'],
        'n': component['n'],
        'axRatio': component['axRatio'],
        'roll': np.rad2deg(component['roll']),
    }
    fitting_params = {
        'fit_mux': 0,
        'fit_muy': 0,
        'fit_magnitude': 1,
        'fit_rEff': 1,
        'fit_n': 0,
        'fit_axRatio': 0,
        'fit_roll': 0,
    }
    return component_formatter.substitute(**components_params, **fitting_params)


def create_object_list(annotation, gal, original_wcs, cutout_wcs):
    # first the disk
    disk = create_object(annotation['disk'], original_wcs, cutout_wcs)
    bulge = create_object(annotation['bulge'], original_wcs, cutout_wcs)
    bar = create_object(annotation['bar'], original_wcs, cutout_wcs)

    return '\n' + '\n'.join((disk, bulge, bar)) + '\n'


def make_galfit_feedme(annotation, gal, size_diff=1, output_dir='', output_fname='output.fits'):
    output_loc = os.path.join(output_dir, )
    parsed_annotation = pa.parse_annotation(annotation, size_diff)

    coord = np.array((gal['RA'].values[0], gal['DEC'].values[0]))
    cutout_size = 4 * np.tile(gal['PETRO_THETA'].values[0], 2)

    frame = scg.queryFromRaDec(gal['RA'], gal['DEC'])[0]

    # get the original fits file as it contains data for sky and sigma image
    image_loc = get_original_fits(frame)
    image_file = fits.open(image_loc)

    # get the montaged fits as we can use it to recover Ra/Dec for objects
    montage_loc = gu.get_fits_location(gal)
    montage_cutout_loc = os.path.join(output_dir, 'montaged_image_cutout.fits')
    montage_cutout = save_cutout(
        fits.open(montage_loc),
        coord, cutout_size * u.arcsec,
        output_file=montage_cutout_loc,
    )
    montage_cutout_wcs = WCS(montage_cutout_loc)
    # save a cropped version of this fits file
    cutout_loc = os.path.join(output_dir, 'image_cutout.fits')
    cutout = save_cutout(
        image_file,
        coord, cutout_size * u.arcsec,
        output_file=cutout_loc,
    )
    cutout_wcs = WCS(cutout_loc)

    cutout_file = fits.open(cutout_loc)

    # find sources to mask
    objects, segmentation_map = source_extract_image(
        cutout_file[0].data,
        image_file[2].data[0][0]
    )

    segmentation_map[segmentation_map == objects[-1][0] + 1] = 0
    segmentation_map[segmentation_map != 0] = 1

    masked_cutout_loc = os.path.join(output_dir, 'masked_cutout.fits')
    fits.HDUList(
        [fits.PrimaryHDU(cutout_file[0].data * segmentation_map)]
    ).writeto(masked_cutout_loc)

    bad_pixel_loc = os.path.join(output_dir, 'bad_pixel_image.fits')
    fits.HDUList([fits.PrimaryHDU(segmentation_map)]).writeto(bad_pixel_loc)

    # download the at the galaxy center
    psf = scg.getPSF(coord, frame, image_file,
                     fname=os.path.join(output_dir, 'tmp-psf.fits'))
    # save it as a fits image
    psf_loc = os.path.join(output_dir, 'PSF.fits')
    fits.HDUList([fits.PrimaryHDU(psf)]).writeto(psf_loc)

    # get the sigma image for the cutout image
    sigma_loc = os.path.join(output_dir, 'sigma.fits')
    _, sigma = scg.cutFits(fits.open(image_loc), *coord.tolist(), size=cutout_size, sigma=True)
    # save it as a fits image
    fits.HDUList([fits.PrimaryHDU(sigma)]).writeto(sigma_loc)

    params = {
        'input_fits': os.path.abspath(cutout_loc),
        'output_fits': os.path.abspath(os.path.join(output_loc, output_fname)),
        'sigma_image': os.path.abspath(sigma_loc),
        'psf_fits': os.path.abspath(psf_loc),
        'psf_fine_sampling': 1,
        'bad_pixel_mask': os.path.abspath(bad_pixel_loc),
        'param_constraint_file': 'none',
        'region_xmin': 1,
        'region_xmax': cutout_file[0].data.shape[1],  # x is columns
        'region_ymin': 1,
        'region_ymax': cutout_file[0].data.shape[0],  # y is rows
        'convolution_box_width': 100,
        'convolution_box_height': 100,
        'photomag_zero': 26.563,  # r-band photomag zero
        'plate_scale_dy': 0.396,  # arcseconds per pixel
        'plate_scale_dx': 0.396,
        'display_type': None,
        'object_list': create_object_list(parsed_annotation, gal, montage_cutout_wcs, cutout_wcs),
    }
    return galfit_formatter.substitute(params)
    # get fits file, get psf file, get sigma image
    # first, add a disk component


# while zooniverse export is fixed
def get_annotations(subject_id):
    with open('../component-clustering/tmp_cls_dump.json') as f:
        classifications = json.load(f)
    classifications_for_subject = [
        c for c in classifications
        if c['links']['subjects'][0] == str(subject_id)
    ]
    annotations_for_subject = [i['annotations'] for i in classifications_for_subject]
    return annotations_for_subject


def make_galfit_folder(subject_id, overwrite=True, base_path=''):
    loc = os.path.join(base_path, str(subject_id))
    if overwrite and os.path.isdir(loc):
        shutil.rmtree(loc)
    if not os.path.isdir(loc):
        os.mkdir(loc)

    gal, angle = gu.get_galaxy_and_angle(subject_id)
    # pic_array, deprojected_image = gu.get_image(gal, subject_id, angle)

    diff_data = gu.get_image_data(subject_id)
    galaxy_data = np.array(diff_data['imageData'])[::-1]
    size_diff = diff_data['width'] / diff_data['imageWidth']
    annotations = get_annotations(subject_id)
    for i, annotation in enumerate(annotations[:1]):
        res = make_galfit_feedme(
            annotation, gal, size_diff,
            output_dir=loc, output_fname='output-annotation-{}.fits'.format(i)
        )
        with open(os.path.join(loc, 'annotation-{}.feedme'.format(i)), 'w') as f:
            f.write(res)


if __name__ == '__main__':
    subject_id = 20902040
    make_galfit_folder(subject_id, base_path='galfit_files')
