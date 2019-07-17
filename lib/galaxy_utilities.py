import os
from tempfile import NamedTemporaryFile
import numpy as np
import pandas as pd
import json
import requests
from PIL import Image
from astropy.wcs import WCS
from gzbuilderspirals import get_drawn_arms as __get_drawn_arms
from gzbuilderspirals import deprojecting as dpj
from shapely.geometry import box, Point
from shapely.affinity import rotate as shapely_rotate, scale as shapely_scale

# for when we eval the json
null = None
true = True
false = False


# needed as we want to load files relative to this file's location, not the
# current working directory
def get_path(s):
    return os.path.join(
        os.path.abspath(os.path.dirname(__file__)),
        s
    )


df_nsa = pd.read_pickle(get_path('df_nsa.pkl'))

classifications = pd.read_csv(
    get_path('../classifications/galaxy-builder-classifications_12-6-19.csv')
)

subjects = pd.read_csv(
    get_path('../classifications/galaxy-builder-subjects_12-6-19.csv')
).drop_duplicates(subset='subject_id').set_index('subject_id', drop=False)

try:
    subject_images = pd.read_pickle(get_path('subject_images.pkl'))
except FileNotFoundError:
    subject_images = None

# Some galaxies were montaged when created. Create a list of their coordinates
# for use later
montage_output_path = get_path('montageOutputs')
montages = [f for f in os.listdir(montage_output_path) if not f[0] == '.']
montageCoordinates = np.array([
    [float(j) for j in i.replace('+', ' ').split(' ')]
    if '+' in i
    else [float(j) for j in i.replace('-', ' -').split(' ')]
    for i in [f for f in os.listdir(montage_output_path) if not f[0] == '.']
])

metadata = [eval(i) for i in subjects['metadata'].values]
meta_map = {i: j for i, j in zip(subjects['subject_id'].values, metadata)}


def get_fits_location(gal):
    montagesDistanceMask = np.add.reduce(
        (montageCoordinates - [gal['RA'], gal['DEC']])**2,
        axis=1
    ) < 0.01
    if np.any(montagesDistanceMask):
        # __import__('warnings').warn('Using montaged image')
        montageFolder = montages[
            np.where(montagesDistanceMask)[0][0]
        ]
        fits_name = get_path('{}/{}/{}'.format(
            'montageOutputs',
            montageFolder,
            'mosaic.fits'
        ))
    else:
        fileTemplate = get_path(
            'fitsImages/{0}/{1}/frame-r-{0:06d}-{1}-{2:04d}.fits'
        )
        fits_name = fileTemplate.format(
            int(gal['RUN']),
            int(gal['CAMCOL']),
            int(gal['FIELD'])
        )
    return fits_name


def get_angle(gal, fits_name, image_size=np.array([512, 512])):
    wFits = WCS(fits_name)
    # edit to center on the galaxy
    wFits.wcs.crval = [float(gal['RA']), float(gal['DEC'])]
    wFits.wcs.crpix = image_size

    r = 4 * float(gal['PETRO_THETA']) / 3600
    phi = float(gal['PETRO_PHI90'])

    center_pix, dec_line = np.array(wFits.all_world2pix(
        [gal['RA'], gal['RA']],
        [gal['DEC'], gal['DEC'] + r],
        0
    )).T

    rot = [
        [np.cos(np.deg2rad(phi)), -np.sin(np.deg2rad(phi))],
        [np.sin(np.deg2rad(phi)), np.cos(np.deg2rad(phi))]
    ]
    vec = np.dot(rot, dec_line - center_pix)
    rotation_angle = 90 - np.rad2deg(np.arctan2(vec[1], vec[0])) - 90
    return rotation_angle


def get_galaxy_and_angle(subject_id, imShape=(512, 512)):
    # Grab the metadata of the subject we are working on
    subject = subjects.loc[subject_id]
    # And the NSA data for the galaxy (if it's a galaxy with NSA data,
    # otherwise throw an error)
    try:
        gal = df_nsa.drop_duplicates(
            subset='NSAID'
        ).set_index(
            'NSAID',
            drop=False
        ).loc[
            int(json.loads(subject.metadata).get('NSA id', np.nan))
        ]
    except KeyError:
        gal = {}
        raise KeyError(
            'Metadata does not contain valid NSA id (probably an older galaxy)'
        )

    # Now we need to obtain the galaxy's rotation in Zooniverse image
    # coordinates. This is made trickier by some decisions in the subject
    # creation pipeline.

    # First, use a WCS object to obtain the rotation in pixel coordinates, as
    # would be obtained from `fitsFile[0].data`
    fits_name = get_fits_location(gal)
    angle = get_angle(gal, fits_name, np.array(imShape)) % 180
    return gal, angle


def get_drawn_arms(subject_id, classifications=classifications):
    try:
        qs = ' or '.join('subject_ids == {}'.format(i) for i in subject_id)
    except TypeError:
        qs = 'subject_ids == {}'.format(subject_id)
    return __get_drawn_arms(
        classifications.query(qs)
    )


def get_ds9_region(gal, fits_name):
    s = """# Region file format: DS9 version 4.1
    global color=green dashlist=8 3 width=1 font="helvetica 10 normal roman" \
select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1
    fk5
    ellipse({},{},{}",{}",{})"""
    with open(get_path('regions/{}.reg'.format(id)), 'w') as f:
        f.write(s.format(
            float(gal['RA']),
            float(gal['DEC']),
            float(gal['PETRO_THETA'] * gal['SERSIC_BA']),
            float(gal['PETRO_THETA']),
            float(gal['PETRO_PHI90'])
        ))
    print(
        ' '.join((
            'ds9 {0}',
            '-regions {1}',
            '-pan to {2} {3} wcs degrees',
            '-crop {2} {3} {4} {4} wcs degrees',
            '-asinh -scale mode 99.5'
        )).format(
            fits_name,
            get_path('regions/{}.reg'.format(id)),
            gal['RA'],
            gal['DEC'],
            gal['PETRO_THETA'] * 2 / 3600,
        ),
    )


# def getUrl(id):
#     return eval(
#         subjects[subjects['subject_id'] == id]['locations']
#     )['1']
#
#
# def get_image(gal, id, angle):
#     # We'll now download the Zooniverse image that volunteers actually
#     # classified on
#     if not os.path.isfile(get_path('images/{}.png'.format(id))):
#         url = getUrl(id)
#         imgData = requests.get(url).content
#
#         f = NamedTemporaryFile(
#             suffix='.{}'.format(url.split('.')[-1]),
#             delete=False
#         )
#         f.write(imgData)
#         f.close()
#         pic = Image.open(f.name)
#         os.unlink(f.name)
#         pic.save(get_path('images/{}.png'.format(id)))
#     else:
#         pic = Image.open(get_path('images/{}.png'.format(id)))
#     # Grab the data arrays from the Image objects, and imshow the images (for
#     # debugging purposes)
#     picArray = np.array(pic)
#
#     # Now deproject the image of the galaxy:
#     deprojectedImage = dpj.deproject_array(
#         picArray, angle, gal['PETRO_BA90']
#     )
#     return picArray, deprojectedImage
#
#
# def save_images():
#     df_subjects = subjects.set_index('subject_id')
#     urls = df_subjects.locations.apply(eval).apply(lambda v: dict.get(v, '1', None)).dropna()
#
#     def get_zoo_image(url):
#         if url.split('.')[-1].lower() not in ('png', 'jpeg', 'jpg'):
#             return None
#         imgData = requests.get(url).content
#         f = NamedTemporaryFile(
#             suffix='.{}'.format(url.split('.')[-1]),
#             delete=False
#         )
#         f.write(imgData)
#         f.close()
#         pic = Image.open(f.name)
#         os.unlink(f.name)
#         return pic
#
#     for id, url in urls.items():
#         pic = get_zoo_image(url)
#         if pic is not None:
#             pic.save(get_path('images/{}.png'.format(id)))


def get_image_data(subject_id):
    return get_diff_data(subject_id)


def get_image(subject_id):
    image_path = 'subject_data/{}/image.png'.format(subject_id)
    return Image.open(get_path(image_path))


def get_deprojected_image(subject_id, ba, angle):
    return dpj.deproject_array(
        np.array(get_image(subject_id)),
        angle, ba,
    )


def get_diff_data(subject_id):
    diff_path = 'subject_data/{}/diff.json'.format(subject_id)
    with open(get_path(diff_path)) as f:
        diff = json.load(f)
    return {
        **diff,
        **{k: np.array(diff[k], 'f8') for k in ('psf', 'imageData')},
    }


def get_psf(subject_id):
    model_path = 'subject_data/{}/model.json'.format(subject_id)
    with open(get_path(model_path)) as f:
        model = json.load(f)
    return np.array(model['psf'], 'f8')


def get_distances(subject_id):
    try:
        return np.load(get_path('distances/subject-{}.npy'.format(subject_id)))
    except OSError:
        return None


def bar_geom_from_zoo(a):
    b = box(
        a['x'],
        a['y'],
        a['x'] + a['width'],
        a['y'] + a['height']
    )
    return shapely_rotate(b, a['angle'])


def ellipse_geom_from_zoo(a):
    ellipse = shapely_rotate(
        shapely_scale(
            Point(a['x'], a['y']).buffer(1.0),
            xfact=a['rx'],
            yfact=a['ry']
        ),
        -a['angle']
    )
    return ellipse
