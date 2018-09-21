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

import os
from tempfile import NamedTemporaryFile
import numpy as np
import pandas as pd
import requests
import matplotlib.pyplot as plt
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

    fitsname = dpj.getFitsName(gal)

    print('\t- Getting galaxy rotation')
    w = dpj.createWCSObject(gal, 512)
    angle = dpj.getAngle(gal, w, np.array([512, 512]))

    # We'll now download the Zooniverse image that volunteers actually classified on

    # In[11]:

    getUrl = lambda id: eval(subjects[subjects['subject_id'] == id]['locations'].iloc[0])['1']
    url = getUrl(subjectId)
    imgData = requests.get(url).content

    f = NamedTemporaryFile(suffix='.{}'.format(url.split('.')[-1]), delete=False)
    f.write(imgData)
    f.close()
    pic = Image.open(f.name)
    os.unlink(f.name)

    # Next, re-create an image (including masking and cutouts) from the FITS file
    fitsImageTmp = NamedTemporaryFile(suffix='.{}'.format(url.split('.')[-1]), delete=False)
    fitsFile = fits.open(fitsname)

    r = float(gal['PETRO_THETA'])/3600
    imageData = scg.cutFits(
        fitsFile,
        float(gal['RA']), float(gal['DEC']),
        size=(4 * r * u.degree, 4 * r * u.degree)
    )

    objects, segmentation_map = csf.sourceExtractImage(
        imageData,
        fitsFile[2].data[0][0]
    )

    # create a true/false masking array
    mask = csf.maskArr(imageData, segmentation_map, objects[-1][0] + 1)

    # create the masked image
    maskedImageData = imageData[:]
    maskedImageData[mask] = 0

    # apply an asinh stretch
    stretchedImageData = csf.stretchArray(maskedImageData[:, ::-1])

    resizeTo = (512, 512)

    im = csf.saveImage(
        stretchedImageData,
        fname=fitsImageTmp.name,
        resize=True,
        size=resizeTo
    )
    fitsImageTmp.close()
    os.unlink(fitsImageTmp.name)

    # Grab the data arrays from the Image objects, and imshow the images (for debugging purposes)
    picArray = np.array(pic)
    imArray = np.array(im)

    # Trial some image transformations to see what needs to be done to get the
    # angle in Zoo coordinates
    picArray = picArray.astype(float)
    t1 = imArray.astype(float)
    t2 = imArray.T.astype(float)
    t3 = imArray[:, ::-1].T.astype(float)
    try:
        ssim_vals = [
            ssim(picArray**2, t**2, data_range=picArray.max()**2 - picArray.min()**2)
            for t in [t1, t2, t3]
        ]
        best = np.argmax(ssim_vals)
    except ValueError:
        best = 2

    # Using this knowledge, transform the angle to (hopefully) work in our
    # Zooniverse image frame
    if best == 1:
        angle = 90 - angle
    if best == 2:
        angle = angle - 90

    # Now deproject the image of the galaxy:
    rotatedImage = rotate(picArray, -angle)
    rotatedImage.shape
    stretchedImage = rescale(rotatedImage, (1, 1/gal['SERSIC_BA'].iloc[0]))

    n = int((stretchedImage.shape[1] - np.array(pic).shape[1]) / 2)

    if n > 0:
        deprojectedImage = stretchedImage[:, n:-n]
    else:
        deprojectedImage = stretchedImage.copy()


    print('\t- Custering arms')
    # Onto the clustering and fitting
    # Extract the drawn arms from classifications for this galaxy
    drawnArms = getDrawnArms(subjectId, classifications)

    # We'll make use of the `gzbuilderspirals` class method to cluster arms.
    # First, initialise a `GalaxySpirals` object with the arms and deprojection
    # parameters
    s = GalaxySpirals(drawnArms, ba=gal['SERSIC_BA'].iloc[0], phi=angle)

    # Now calculate a the distance matrix for the drawn arms (this can be slow)
    distances = s.calculateDistances()

    # Perform the clustering (using the DBSCAN clustering algorithm)
    db = s.clusterLines(distances)

    print('\t- Fitting splines')
    # Fit both XY and radial splines to the resulting clusters (described in more detail in the method paper)
    xyFits = s.fitXYSplines()
    result = s.fitRadialSplines()

    # PLOTTING
    # Add a helper function to generate plots of the resulting arms
    def prettyPlot(arm, ax=plt.gca(), **kwargs):
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

    plt.figure(figsize=(10, 30), dpi=100)
    ax_annot = plt.subplot2grid((6, 2), (0, 0), colspan=2, rowspan=2)
    ax_cluster = plt.subplot2grid((6, 2), (2, 0))
    ax_sorting = plt.subplot2grid((6, 2), (2, 1))
    ax_isophote = plt.subplot2grid((6, 2), (3, 0))
    ax_deproject = plt.subplot2grid((6, 2), (3, 1))
    ax_final = plt.subplot2grid((6, 2), (4, 0), colspan=2, rowspan=2)

    # panel 1: all drawn arms
    for arm in drawnArms:
        ax_annot.plot(*arm.T, '.-', linewidth=0.5, markersize=1, alpha=0.8)
    ax_annot.axis('equal')

    # panel 2: clustered arms
    for arm in s.arms:
        ax_cluster.plot(
            *arm.cleanedCloud.T,
            '.',
            markersize=1, alpha=0.8,
        )
    ax_cluster.axis('equal')

    # panel 3: sorting lines
    for fit in xyFits:
        plt.plot(*fit.T)

    plt.imshow(deprojectedImage, cmap='gray', origin='lower')
    for i, arm in enumerate(result['deprojectedArms']):
        plt.plot(*arm.pointCloud.T, 'k.', markersize=2, alpha=1)
        plt.plot(
            *arm.pointCloud.T,
            '.', c='C{}'.format(i), markersize=2, alpha=0.3
        )
        plt.plot(
            *arm.cleanedCloud.T,
            '.', c='C{}'.format(i), markersize=2, alpha=1
        )

    for i, arm in enumerate(result['radialFit']):
        prettyPlot(s.arms[i].deNorm(arm), label='Arm {}'.format(i), c='C{}'.format(i)),

    plt.axis('off')
    plt.savefig('arm-fits/subject-{}.jpg'.format(subjectId))
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
