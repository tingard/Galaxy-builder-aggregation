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
import matplotlib.pyplot as plt
import requests
from PIL import Image
from skimage.transform import rotate, rescale
from gzbuilderspirals import deprojecting as dpj
from gzbuilderspirals import fitting
from gzbuilderspirals import get_drawn_arms
from gzbuilderspirals.galaxySpirals import GalaxySpirals

print('Loading NSA catalog')
df_nsa = pd.read_pickle('NSA_filtered.pkl')

print('Loading Zooniverse classification dump')
classifications = pd.read_csv(
    '../classifications/galaxy-builder-classifications_24-7-18.csv'
)
subjects = pd.read_csv(
    '../classifications/galaxy-builder-subjects_24-7-18.csv'
)
null = None
true = True
false = False

print('Obtaining available frame montages')
# Some galaxies were montaged when created. Create a list of their coordinates
# for use later
montages = [f for f in os.listdir('montageOutputs') if not f[0] == '.']
montageCoordinates = np.array([
    [float(j) for j in i.replace('+', ' ').split(' ')]
    if '+' in i
    else [float(j) for j in i.replace('-', ' -').split(' ')]
    for i in [f for f in os.listdir('montageOutputs') if not f[0] == '.']
])


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
        fitsName = '{}/{}/{}'.format(
            os.path.abspath('montageOutputs'),
            montageFolder,
            'mosaic.fits'
        )
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
    return classifications, picArray, deprojectedImage


def get_galaxy_spirals(gal, angle, id, classifications):
    print('\t- Clustering arms')
    # Onto the clustering and fitting
    # Extract the drawn arms from classifications for this galaxy
    drawnArms = get_drawn_arms(id, classifications)

    # We'll make use of the `gzbuilderspirals` class method to cluster arms.
    # First, initialise a `GalaxySpirals` object with the arms and deprojection
    # parameters
    galaxy_object = GalaxySpirals(
        drawnArms,
        ba=gal['SERSIC_BA'].iloc[0],
        phi=-angle
    )

    # Now calculate a the distance matrix for the drawn arms (this can be slow)
    try:
        distances = np.load('distances/subject-{}.npy'.format(id))
        print('\t- Using saved distances')
    except OSError:
        distances = galaxy_object.calculate_distances()
        np.save('distances/subject-{}.npy'.format(id), distances)

    # Perform the clustering (using the DBSCAN clustering algorithm)
    galaxy_object.cluster_lines(distances)

    print('\t- Fitting arms and errors')
    # Fit both XY and radial splines to the resulting clusters (described in
    # more detail in the method paper)
    return galaxy_object


def fitThings(galaxy_object):
    galaxy_fit = galaxy_object.fit_arms(spline_degree=3)
    return galaxy_fit


def main():
    pass


if __name__ == '__main__':
    main()
