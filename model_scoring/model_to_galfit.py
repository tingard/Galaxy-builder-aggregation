import string
import os
import re
import shutil
import json
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
import lib.python_model_renderer.parse_annotation as pa
import lib.galaxy_utilities as gu
import sdssCutoutGrab as scg
from astropy.coordinates.errors import BoundsError

FIT_NOTHING = {
    'fit_mux': 0,
    'fit_muy': 0,
    'fit_magnitude': 0,
    'fit_rEff': 0,
    'fit_n': 0,
    'fit_axRatio': 0,
    'fit_roll': 0,
    'fit_c': 0,
}

DEFAULT_FITTING_PARAMS = {
    'disk': {
        'fit_magnitude': 1,
        'fit_rEff': 1,
        'fit_axRatio': 1,
        'fit_roll': 1,
    },
    'bulge': {
        'fit_magnitude': 1,
        'fit_rEff': 1,
        'fit_axRatio': 1,
        'fit_roll': 1,
        'fit_n': 1,
    },
    'bar': {
        'fit_magnitude': 1,
        'fit_rEff': 1,
        'fit_axRatio': 1,
        'fit_roll': 1,
        'fit_n': 1,
        'fit_c': 1,
    }
}

with open('./galfit_template.tpl') as f:
    galfit_formatter = string.Template(f.read())

with open('./sersic_component_template.tpl') as f:
    sersic_component_formatter = string.Template(f.read())


def get_original_fits(frame):
    return os.path.abspath(os.path.join(
        'lib/fitsImages/',
        '{run}/{camcol}/frame-{band}-{run:06d}-{camcol}-{field:04d}.fits'.format(band='r', **frame),
    ))


def get_montaged_cutout(data_size, pix_size, gal, fname='downloaded_cutout.fits'):
    fits_cutout_url = (
        'http://legacysurvey.org/viewer/fits-cutout'
        '?ra={RA:.6f}&dec={DEC:.6f}&size={size}&layer=sdss&pixscale={scale}&bands=r'.format(
            size=data_size,
            scale=1/pix_size,
            **gal.iloc[0].to_dict()
        )
    )
    if scg.downloadFile(fits_cutout_url, fname, overwrite=True, decompress=False):
        return fname
    else:
        raise FileNotFoundError('Something went wrong')
        return False


def make_magnitude_from_flux(comp):
    f = comp['i0'] / 2
    return 22.5 - 2.5 * np.log10(f)


def create_object(component, extra_params={}, fitting_params={}):
    if component is None:
        return ''
    component_params = {
        'mux': component['mu'][0],
        'muy': component['mu'][1],
        # 0.8 as I'm a twit and arbitrarily scaled things
        'magnitude': make_magnitude_from_flux(component),
        'rEff': component['rEff'],
        'n': component['n'],
        'axRatio': component['axRatio'],
        'roll': -np.rad2deg(component['roll']) + 90,
        'c': float(component['c']) - 2.0,
        **extra_params,
    }
    fitting_params = {
        **FIT_NOTHING,
        **fitting_params,
    }
    tpl = sersic_component_formatter.substitute(
        **component_params,
        **fitting_params
    )
    if component_params['c'] == 0.0 and fitting_params['fit_c'] == 0:
        # remove the boxyness option as it breaks galfit
        return re.sub(r'^C0\).*?$', '', tpl, flags=re.MULTILINE)
    return tpl


def create_object_list(annotation, gal, wcs,
                       fitting_params=DEFAULT_FITTING_PARAMS):
    # first the disk
    disk = create_object(
        annotation['disk'], fitting_params=fitting_params['disk']
    )
    bulge = create_object(
        annotation['bulge'], fitting_params=fitting_params['bulge']
    )
    bar = create_object(
        annotation['bar'], fitting_params=fitting_params['bar']
    )

    return '\n' + '\n'.join((disk, bulge, bar)) + '\n'


def make_galfit_feedme(subject_id, classification, output_dir='',
                       output_fname='output.fits', overwrite=True):
    sid_loc = os.path.join(output_dir, str(subject_id))
    loc = os.path.join(sid_loc, str(classification['classification_id']))
    if overwrite and os.path.isdir(loc):
        shutil.rmtree(loc)
    if not os.path.isdir(sid_loc):
        os.mkdir(sid_loc)
    if not os.path.isdir(loc):
        os.mkdir(loc)

    # Grab galaxy information
    gal, angle = gu.get_galaxy_and_angle(subject_id)
    coord = np.array((gal['RA'].values[0], gal['DEC'].values[0]))
    diff_data = gu.get_image_data(subject_id)
    size_diff = diff_data['width'] / diff_data['imageWidth']
    bad_pixel_mask = np.array(diff_data['mask']).T[::-1]

    cutout_size = 4 * np.tile(gal['PETRO_THETA'].values[0], 2)
    frame = scg.queryFromRaDec(gal['RA'], gal['DEC'])[0]

    original_image_loc = get_original_fits(frame)
    image_file = fits.open(original_image_loc)
    image_file_loc = os.path.join(loc, 'image_cutout.fits')

    # create the cutout and sigma image
    hdu = image_file[0]
    im_cutout, sigma_cutout = scg.cutFits(image_file, *coord, cutout_size,
                                          sigma=True)
    if im_cutout is False:
        return False
    hdu.data = im_cutout.data
    # Update the FITS header with the cutout WCS
    hdu.header.update(im_cutout.wcs.to_header())
    # Write the cutout to a new FITS file
    hdu.writeto(image_file_loc, overwrite=True)

    image_file = fits.open(image_file_loc)

    if image_file[0].data.shape != bad_pixel_mask.shape:
        print(image_file[0].data.shape, bad_pixel_mask.shape)
        raise BoundsError(
            'Could not make a {0:.2f}" by {0:.2f}"'.format(
                cutout_size[0]
            ) + ' cutout from this fits file'
        )
        return False

    # get galaxy wcs (not needed?)
    image_wcs = WCS(image_file_loc)

    # make fits file for pixel mask
    bad_pixel_loc = os.path.join(loc, 'bad_pixel_image.fits')
    fits.HDUList(
        [fits.PrimaryHDU(bad_pixel_mask)]
    ).writeto(bad_pixel_loc)

    # get sigma image
    sigma_loc = os.path.join(loc, 'sigma.fits')
    fits.HDUList(
        [fits.PrimaryHDU(sigma_cutout.data)]
    ).writeto(sigma_loc, overwrite=True)

    # get PSF
    psf_loc = os.path.join(loc, 'PSF.fits')
    psf = scg.getPSF(coord, frame, image_file, fname=psf_loc)
    fits.HDUList([fits.PrimaryHDU(psf)]).writeto(psf_loc)

    # Parse the volunteer's classification into a model
    model = pa.parse_annotation(json.loads(classification['annotations']),
                                size_diff=size_diff)
    object_list = create_object_list(
        model, gal, image_wcs
    )
    output_loc = os.path.join(loc, 'imgblock.fits')
    feedme = galfit_formatter.substitute({
        'input_fits': os.path.abspath(image_file_loc),
        'output_fits': os.path.abspath(output_loc),
        'sigma_image': os.path.abspath(sigma_loc),
        'psf_fits': os.path.abspath(psf_loc),
        'psf_fine_sampling': 1,
        'bad_pixel_mask': os.path.abspath(bad_pixel_loc),
        'param_constraint_file': 'none',
        'region_xmin': 1,
        'region_xmax': image_file[0].data.shape[1],  # x is columns
        'region_ymin': 1,
        'region_ymax': image_file[0].data.shape[0],  # y is rows
        'convolution_box_width': 100,  # something magical in GALFIT
        'convolution_box_height': 100,  # something magical in GALFIT
        'photomag_zero': 26.563,  # sdss r-band photomag zeropoint
        'plate_scale_dy': 0.396,  # arcseconds per pixel
        'plate_scale_dx': 0.396,
        'display_type': None,
        'object_list': object_list,
    })
    feedme_loc = os.path.join(loc, 'galfit.feedme')
    with open(feedme_loc, 'w') as f:
        f.write(feedme)
    return {
        'base': loc,
        'feedme': feedme_loc,
        'image': image_file_loc,
        'output': output_loc,
        'psf': psf_loc,
        'mask': bad_pixel_loc,
    }
