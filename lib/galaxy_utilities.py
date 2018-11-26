# # Spiral extraction methodology
#
# 1. Obtain Sérsic parameters of galaxy from NSA catalog
# 2. Translate the Sérsic $\phi$ into zooniverse image coordinates
#     - Need to re-create the transforms used to generate the image for
#       volunteers (due to poor decision making in the subject creation
#       process)
# 2. Cluster drawn poly-lines
#     1. Deproject drawn arms
#     2. Cluster using DBSCAN and a custom metric
#     3. Use Local Outlier Factor to clean points
#     5. Sort points in cluster
#     6. Fit a smoothing spline to ordered points
# 3. Calculate pitch angles for the resulting spline fits
import os
from tempfile import NamedTemporaryFile
import numpy as np
import pandas as pd
import json
import requests
from PIL import Image
from skimage.transform import rotate, rescale
from gzbuilderspirals import get_drawn_arms
from gzbuilderspirals.galaxySpirals import GalaxySpirals
from gzbuilderspirals import deprojecting as dpj

# needed as we want to load files relative to this file's location, not the
# current working directory
def get_path(s):
    return '{}/{}'.format(
        os.path.abspath(os.path.dirname(__file__)),
        s
    )
print('Loading NSA catalog')
df_nsa = pd.read_pickle(get_path('NSA_filtered.pkl'))

print('Loading Zooniverse classification dump')
classifications = pd.read_csv(
    get_path('../classifications/galaxy-builder-classifications_15-11-18.csv')
)
subjects = pd.read_csv(
    get_path('../classifications/galaxy-builder-subjects_24-7-18.csv')
)
null = None
true = True
false = False

print('Obtaining available frame montages')
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

true = True
false = False
null = None
metadata = [eval(i) for i in subjects['metadata'].values]
meta_map = {i: j for i, j in zip(subjects['subject_id'].values, metadata)}


def get_galaxy_and_angle(id):
    print('Working on galaxy {}'.format(id))
    subjectId = id

    # Grab the metadata of the subject we are working on
    meta = eval(
        subjects[subjects['subject_id'] == subjectId].iloc[0]['metadata']
    )

    # And the NSA data for the galaxy (if it's a galaxy with NSA data,
    # otherwise throw an error)
    try:
        gal = df_nsa[df_nsa['NSAID'] == int(meta['NSA id'])]
    except KeyError:
        gal = {}
        raise KeyError(
            'Metadata does not contain valid NSA id (probably an older galaxy)'
        )

    # Now we need to obtain the galaxy's rotation in Zooniverse image
    # coordinates. This is made trickier by some decision in the subject
    # creation pipeline.

    # First, use a WCS object to obtain the rotation in pixel coordinates, as
    # would be obtained from `fitsFile[0].data`

    montagesDistanceMask = np.add.reduce(
        (montageCoordinates - [gal['RA'].iloc[0], gal['DEC'].iloc[0]])**2,
        axis=1
    ) < 0.01
    usingMontage = np.any(montagesDistanceMask)
    if usingMontage:
        montageFolder = montages[
            np.where(montagesDistanceMask)[0][0]
        ]
        fitsName = get_path('./{}/{}/{}'.format(
            'montageOutputs',
            montageFolder,
            'mosaic.fits'
        ))
        print('\t- USING MONTAGED IMAGE')
    else:
        fileTemplate = 'fitsImages/{0}/{1}/frame-r-{0:06d}-{1}-{2:04d}.fits'
        fitsName = fileTemplate.format(
            int(gal['RUN']),
            int(gal['CAMCOL']),
            int(gal['FIELD'])
        )

    print('\t- Getting galaxy rotation')
    angle = dpj.get_angle(gal, fitsName, np.array([512, 512]))
    return gal, angle


def getUrl(id):
    return eval(
        subjects[subjects['subject_id'] == id]['locations'].iloc[0]
    )['1']


def get_image(gal, id, angle):
    # We'll now download the Zooniverse image that volunteers actually
    # classified on
    url = getUrl(id)
    imgData = requests.get(url).content

    f = NamedTemporaryFile(
        suffix='.{}'.format(url.split('.')[-1]),
        delete=False
    )
    f.write(imgData)
    f.close()
    pic = Image.open(f.name)
    os.unlink(f.name)

    # Grab the data arrays from the Image objects, and imshow the images (for
    # debugging purposes)
    picArray = np.array(pic)

    # Now deproject the image of the galaxy:
    rotatedImage = rotate(picArray, angle)
    stretchedImage = rescale(
        rotatedImage,
        (1/gal['SERSIC_BA'].iloc[0], 1),
        mode='constant',
        multichannel=False
    )
    n = int((stretchedImage.shape[0] - np.array(pic).shape[0]) / 2)
    if n > 0:
        deprojectedImage = stretchedImage[n:-n, :]
    else:
        deprojectedImage = stretchedImage.copy()
    return picArray, deprojectedImage


def get_image_data(subject_id):
    with open('location-map.json') as location_map_file:
        location_map = json.load(location_map_file)
    location = location_map[str(subject_id)]
    subject_set = os.path.expanduser(
        '~/PhD/galaxy-builder/subjectUpload/subject_set_{}'.format(
            location[0]
        )
    )
    difference_fname = os.path.expanduser(
        '{}/difference_subject{}.json'.format(subject_set, location[1])
    )
    with open(difference_fname) as difference_file:
        difference = json.load(difference_file)
    return difference


def get_psf(subject_id):
    with open('location-map.json') as location_map_file:
        location_map = json.load(location_map_file)
    location = location_map[str(subject_id)]
    subject_set = os.path.expanduser(
        '~/PhD/galaxy-builder/subjectUpload/subject_set_{}'.format(
            location[0]
        )
    )
    difference_fname = os.path.expanduser(
        '{}/difference_subject{}.json'.format(subject_set, location[1])
    )
    with open(difference_fname) as difference_file:
        difference = json.load(difference_file)
    return np.array(difference['psf'], 'f8').reshape(11, 11)


def get_galaxy_spirals(gal, angle, id, classifications):
    # Onto the clustering and fitting
    # Extract the drawn arms from classifications for this galaxy
    drawn_arms = get_drawn_arms(id, classifications)
    print('\t Identified {} arms'.format(len(drawn_arms)))
    # We'll make use of the `gzbuilderspirals` class method to cluster arms.
    # First, initialise a `GalaxySpirals` object with the arms and deprojection
    # parameters
    print('\t- Clustering arms')
    galaxy_object = GalaxySpirals(
        drawn_arms,
        ba=gal['SERSIC_BA'].iloc[0],
        phi=-angle
    )
    if os.path.isfile(get_path('distances/subject-{}.npy'.format(id))):
        distances = np.load(get_path('distances/subject-{}.npy'.format(id)))
        if distances.shape[0] != len(drawn_arms):
            distances = galaxy_object.calculate_distances()
            np.save(get_path('distances/subject-{}.npy'.format(id)), distances)
    else:
        distances = galaxy_object.calculate_distances()

    db = galaxy_object.cluster_lines(distances)

    print('\t- Fitting arms and errors')
    # Fit both XY and radial splines to the resulting clusters (described in
    # more detail in the method paper)
    return galaxy_object


def fitThings(galaxy_object):
    galaxy_fit = galaxy_object.fit_arms(spline_degree=3)
    return galaxy_fit
