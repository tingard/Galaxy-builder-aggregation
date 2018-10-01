# # Spiral extraction methodology
#
# 1. Obtain Sérsic parameters of galaxy from NSA catalog
# 2. Translate the Sérsic $\phi$ into zooniverse image coordinates
#     - Need to re-create the transforms used to generate the image for volunteers (due to poor decision making in the subject creation process)
# 2. Cluster drawn poly-lines
#     1. Deproject drawn arms
#     2. Cluster using DBSCAN and a custom metric
#     3. Use Local Outlier Factor to clean points
#     5. Sort points in cluster
#     6. Fit a smoothing spline to ordered points
# 3. Calculate pitch angles for the resulting spline fits

print('Doing imports...')
import os
from tempfile import NamedTemporaryFile
import numpy as np
import pandas as pd
import requests
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.patches import Ellipse
from PIL import Image
from astropy.io import fits
from scipy.ndimage.filters import gaussian_filter
from skimage import img_as_float
from skimage.transform import rotate, rescale
from skimage.measure import compare_ssim as ssim
import astropy.units as u
from gzbuilderspirals import deprojecting as dpj
from gzbuilderspirals import getDrawnArms, deprojectArm, rThetaFromXY, xyFromRTheta
from gzbuilderspirals.galaxySpirals import GalaxySpirals
import sdssCutoutGrab as scg
import createSubjectsFunctions as csf

print('Loading NSA catalog and Zooniverse data dump')
# Open the NSA catalog, and the galaxy builder subjects and classifications
nsa = fits.open('../../subjectUpload/nsa_v1_0_1.fits')

nsa_keys = [
    'NSAID', 'ISDSS', 'INED', 'RA', 'DEC', 'Z', 'SERSIC_BA', 'SERSIC_PHI', 'PETRO_THETA',
    'IAUNAME', 'ZDIST', 'RUN', 'CAMCOL', 'FIELD', 'RERUN',
]
nsaRas = nsa[1].data['ra']
nsaDecs = nsa[1].data['dec']

df_nsa = pd.DataFrame(
    {key: nsa[1].data[key].byteswap().newbyteorder() for key in nsa_keys}
)

classifications = pd.read_csv('../classifications/galaxy-builder-classifications_24-7-18.csv')
subjects = pd.read_csv('../classifications/galaxy-builder-subjects_24-7-18.csv')
null = None
true = True
false = False

# Some galaxies were montaged when created. Create a list of their coordinates for use later
montages = [f for f in os.listdir('montageOutputs') if not f[0] == '.']
montageCoordinates = np.array([
    [float(j) for j in i.replace('+', ' ').split(' ')]
    if '+' in i
    else [float(j) for j in i.replace('-', ' -').split(' ')]
    for i in [f for f in os.listdir('montageOutputs') if not f[0] == '.']
])

def main(id):
    print('Working on galaxy {}'.format(id))
    subjectId = id

    # Grab the metadata of the subject we are working on
    meta = eval(subjects[subjects['subject_id'] == subjectId].iloc[0]['metadata'])

    # And the NSA data for the galaxy (if it's a galaxy with NSA data,
    # otherwise throw an error)
    try:
        gal = df_nsa[df_nsa['NSAID'] == int(meta['NSA id'])]
    except KeyError:
        gal = {}
        raise KeyError('Metadata does not contain valid NSA id (probably an older galaxy)')

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
        fitsName = 'fitsImages/{0}/{1}/frame-r-{0:06d}-{1}-{2:04d}.fits'.format(
            int(gal['RUN']),
            int(gal['CAMCOL']),
            int(gal['FIELD'])
        )

    print('\t- Getting galaxy rotation')
    w = dpj.createWCSObject(gal, 512)
    angle = dpj.getAngle(gal, fitsName, np.array([512, 512]))

    # We'll now download the Zooniverse image that volunteers actually classified on
    getUrl = lambda id: eval(subjects[subjects['subject_id'] == id]['locations'].iloc[0])['1']
    url = getUrl(subjectId)
    imgData = requests.get(url).content

    f = NamedTemporaryFile(suffix='.{}'.format(url.split('.')[-1]), delete=False)
    f.write(imgData)
    f.close()
    pic = Image.open(f.name)
    os.unlink(f.name)

    # Grab the data arrays from the Image objects, and imshow the images (for debugging purposes)
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


    print('\t- Custering arms')
    # Onto the clustering and fitting
    # Extract the drawn arms from classifications for this galaxy
    drawnArms = getDrawnArms(subjectId, classifications)

    # We'll make use of the `gzbuilderspirals` class method to cluster arms.
    # First, initialise a `GalaxySpirals` object with the arms and deprojection
    # parameters
    s = GalaxySpirals(drawnArms, ba=gal['SERSIC_BA'].iloc[0], phi=-angle)

    # Now calculate a the distance matrix for the drawn arms (this can be slow)
    try:
        distances = np.load('distances/subject-{}.npy'.format(id))
        print('\t- Using saved distances')
    except OSError:
        distances = s.calculateDistances()
        np.save('distances/subject-{}.npy'.format(id), distances)

    # Perform the clustering (using the DBSCAN clustering algorithm)
    db = s.clusterLines(distances)

    print('\t- Fitting splines')
    # Fit both XY and radial splines to the resulting clusters (described in more detail in the method paper)
    xyFits = s.fitXYSplines()
    result = s.fitRadialSplines()

    # PLOTTING
    # Add a helper function to generate plots of the resulting arms
    def prettyPlot(arm, c, ax=plt.gca(), **kwargs):
        ax.plot(
            *arm.T,
            c='k'.format(i), linewidth=4
        )
        ax.plot(
            *arm.T, linewidth=3, **kwargs
        )
        ax.plot(
            *arm.T,
            c='w', linewidth=2, alpha=0.5
        )
    plt.figure(figsize=(10, 30), dpi=200)
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    ax_annot = plt.subplot2grid((6, 2), (0, 0), colspan=2, rowspan=2)
    ax_cluster = plt.subplot2grid((6, 2), (2, 0))
    ax_sorting = plt.subplot2grid((6, 2), (2, 1))
    ax_isophote = plt.subplot2grid((6, 2), (3, 0))
    ax_deproject = plt.subplot2grid((6, 2), (3, 1))
    ax_final = plt.subplot2grid((6, 2), (4, 0), colspan=2, rowspan=2)

    # panel 1: all drawn arms
    ax_annot.imshow(picArray, cmap='gray', origin='lower')
    for arm in drawnArms:
        ax_annot.plot(*arm.T, '.-', linewidth=1, markersize=2)

    # panel 2: clustered arms
    ax_cluster.imshow(picArray, cmap='gray', origin='lower')
    for i, arm in enumerate(s.arms):
        ax_cluster.plot(
            *arm.pointCloud.T,
            '.', c='k',
            markersize=1,
            alpha=1,
        )
        ax_cluster.plot(
            *arm.pointCloud.T,
            '.', c='C{}'.format(i % 10),
            markersize=1,
            alpha=0.4,
        )
        ax_cluster.plot(
            *arm.cleanedCloud.T,
            '.', c='C{}'.format(i % 10),
            markersize=1,
            alpha=0.8,
        )
    plt.setp(ax_cluster.get_xticklabels(), visible=False)

    # panel 3: sorting lines
    ax_sorting.imshow(picArray, cmap='gray', origin='lower')
    for fit in xyFits:
        ax_sorting.plot(*fit.T)
    plt.setp(ax_sorting.get_xticklabels(), visible=False)
    plt.setp(ax_sorting.get_yticklabels(), visible=False)

    ax_isophote.imshow(picArray, cmap='gray', origin='lower')
    isophote = Ellipse(
        xy=np.array(picArray.shape) / 2,
        width=200 * gal['SERSIC_BA'],
        height=200,
        angle=90 + angle,
        ec='w',
        fc='none'
    )
    ax_isophote.add_artist(isophote)

    ax_deproject.imshow(deprojectedImage, cmap='gray', origin='lower')
    for i, arm in enumerate(result['deprojectedArms']):
        ax_deproject.plot(*arm.pointCloud.T, 'k.', markersize=2, alpha=1)
        ax_deproject.plot(
            *arm.pointCloud.T,
            '.', c='C{}'.format(i), markersize=2, alpha=0.3
        )
        ax_deproject.plot(
            *arm.cleanedCloud.T,
            '.', c='C{}'.format(i), markersize=2, alpha=1
        )
    for i, arm in enumerate(result['deprojectedFit']):
        prettyPlot(
            arm,
            ax=ax_deproject,
            label='Arm {}'.format(i),
            c='C{}'.format(i)
        )
    plt.setp(ax_deproject.get_yticklabels(), visible=False)

    ax_final.imshow(deprojectedImage, cmap='gray', origin='lower')
    for i, arm in enumerate(result['deprojectedArms']):
        p = ax_final.plot(
            *arm.cleanedCloud.T,
            '.',
            c='C{}'.format(i % 10),
            markersize=3,
            label='Cleaned points in arm {}'.format(i)
        )
        c = np.array(to_rgb(p[0].get_color()))*0.7
        p = ax_final.plot(
            *arm.pointCloud[np.logical_not(arm.outlierMask)].T,
            '.',
            c=c,
            markersize=3,
            alpha=1,
            label='Outlier points removed from arm {}'.format(i)
        )

    for i, arm in enumerate(result['radialFit']):
        prettyPlot(
            s.arms[i].deNorm(arm),
            ax=ax_final,
            label='Arm {}'.format(i),
            c='C{}'.format(i)
        )
    ax_final.legend()

    plt.savefig('arm-fits/subject-{}.jpg'.format(id), bbox_inches='tight')
    plt.close()

    # calculate pitch angles
    print('\t- Creating pitch angle plot')
    plt.figure(figsize=(14, 7))
    # first panel is the galaxy, including start and end of arms
    plt.subplot(121)
    plt.imshow(deprojectedImage, cmap='gray', origin='lower')
    for i, arm in enumerate(result['radialFit']):
        prettyPlot(
            s.arms[i].deNorm(arm),
            ax=plt.gca(),
            label='Arm {}'.format(i),
            c='C{}'.format(i)
        )
    plt.plot(
        *[deprojectedImage.shape[0] / 2] * 2,
        'x',
        markersize=10,
        label='center'
    )
    plt.subplot(122)
    getRadius = lambda c: np.sqrt(np.add.reduce(c**2, axis=1))

    for i, arm in enumerate(result['radialFit']):
        if getRadius(arm[0].reshape(1, 2)) > getRadius(arm[-1].reshape(1, 2)):
            arm = arm[::-1]

        pas = np.zeros(arm.shape[0] - 2)
        O = np.array([0, 0])
        for j in range(1, arm.shape[0] - 1):
            pnm1 = arm[j - 1]
            pn = arm[j]
            pnp1 = arm[j + 1]
            unitTangent = (pnp1 - pnm1) / np.linalg.norm(pnp1 - pnm1)

            pas[j - 1] = np.rad2deg(
                np.pi / 2 - np.arccos(
                    np.dot(pn / np.linalg.norm(pn), unitTangent)
                )
            )
        plt.plot(
            getRadius(arm[1: -1]),
            pas,
            '-',
            c='C{}'.format(i % 10),
            label='Arm {}'.format(i)
        )

    plt.xlabel('Distance from center of galaxy (arbitrary units)')
    plt.ylabel('Measured pitch angle (degrees)')
    plt.legend()
    plt.savefig('pitchAngles/subject-{}.jpg'.format(id), bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    with open('subject-id-list.csv', 'r') as f:
        subjectIds = np.array([int(n) for n in f.read().split('\n')])
    np.random.shuffle(subjectIds)
    # we will write the notebooks out to here
    outputFolder = 'output-notebooks'
    if not os.path.exists(outputFolder):
        os.mkdir(outputFolder)

    # iterate over all subject IDs, and run the notebooks!
    for id in subjectIds:
        main(id)
