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
    fitsName = 'fitsImages/{0}/{1}/frame-r-{0:06d}-{1}-{2:04d}.fits'.format(
        int(gal['RUN']),
        int(gal['CAMCOL']),
        int(gal['FIELD'])
    )

    print('\t- Getting galaxy rotation')
    w = dpj.createWCSObject(gal, 512)
    angle = dpj.getAngle(gal, w, np.array([512, 512]))

    # We'll now download the Zooniverse image that volunteers actually classified on
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
    fitsFile = fits.open(fitsName)

    r = float(gal['PETRO_THETA'])/3600
    imageData = scg.cutFits(
        fitsFile,
        float(gal['RA']), float(gal['DEC']),
        size=(4 * r * u.degree, 4 * r * u.degree)
    )
    try:
        objects, segmentation_map = csf.sourceExtractImage(
            imageData,
            fitsFile[2].data[0][0]
        )
        # create a true/false masking array
        mask = csf.maskArr(imageData, segmentation_map, objects[-1][0] + 1)

        # create the masked image
        maskedImageData = imageData.copy()
        maskedImageData[mask] = 0
    except AttributeError:
        print('Failed for id: ', id)
        print('\tRA:', gal['RA'])
        print('\tDEC:', gal['DEC'])
        with open('failed-rotations.txt', 'a') as f:
            f.write(',{}'.format(id))
        return

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
    t1 = imArray.T.astype(float)
    t2 = imArray[:, ::-1].T.astype(float)

    kwargs = {
        'origin': 'lower',
        'cmap': 'gray',
    }
    plt.figure(figsize=(18, 5))
    plt.subplot(141)
    plt.imshow(picArray, **kwargs)
    plt.title('Image shown to volunteers')
    plt.subplot(142)
    plt.imshow(imArray, **kwargs)
    plt.title('FITS image')
    plt.subplot(143)
    plt.imshow(t1, **kwargs)
    plt.title('FITS image transposed')
    plt.subplot(144)
    plt.imshow(t2, **kwargs)
    plt.title('FITS image transposed and mirrored in y')
    plt.savefig(
        './rotation-transforms/subject-{}png'.format(id),
        bbox_inches='tight'
    )
    plt.close()

if __name__ == '__main__':
    with open('subject-id-list.csv', 'r') as f:
        subjectIds = np.array([int(n) for n in f.read().split('\n')])
    # we will write the notebooks out to here
    outputFolder = 'output-notebooks'
    if not os.path.exists(outputFolder):
        os.mkdir(outputFolder)

    # iterate over all subject IDs, and run the notebooks!
    for id in subjectIds:
        main(id)
