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
from skimage.transform import rotate, rescale
from gzbuilderspirals import deprojecting as dpj
from gzbuilderspirals import getDrawnArms, rThetaFromXY, xyFromRTheta
from gzbuilderspirals.galaxySpirals import GalaxySpirals

print('Loading NSA catalog')
df_nsa = pd.read_pickle('NSA_filtered.pkl')

print('Loading Zooniverse classification dump')
classifications = pd.read_csv('../classifications/galaxy-builder-classifications_24-7-18.csv')
subjects = pd.read_csv('../classifications/galaxy-builder-subjects_24-7-18.csv')
null = None
true = True
false = False

print('Obtaining available frame montages')
# Some galaxies were montaged when created. Create a list of their coordinates for use later
montages = [f for f in os.listdir('montageOutputs') if not f[0] == '.']
montageCoordinates = np.array([
    [float(j) for j in i.replace('+', ' ').split(' ')]
    if '+' in i
    else [float(j) for j in i.replace('-', ' -').split(' ')]
    for i in [f for f in os.listdir('montageOutputs') if not f[0] == '.']
])


def prettyPlot(arm, c, ax=plt.gca(), **kwargs):
    ax.plot(
        *arm.T,
        c=c, linewidth=4
    )
    ax.plot(
        *arm.T, linewidth=3, **kwargs
    )
    ax.plot(
        *arm.T,
        c='w', linewidth=2, alpha=0.5
    )


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


    print('\t- Clustering arms')
    # Onto the clustering and fitting
    # Extract the drawn arms from classifications for this galaxy
    drawnArms = getDrawnArms(subjectId, classifications)

    # We'll make use of the `gzbuilderspirals` class method to cluster arms.
    # First, initialise a `GalaxySpirals` object with the arms and deprojection
    # parameters
    galaxy_object = GalaxySpirals(drawnArms, ba=gal['SERSIC_BA'].iloc[0], phi=-angle)

    # Now calculate a the distance matrix for the drawn arms (this can be slow)
    try:
        distances = np.load('distances/subject-{}.npy'.format(id))
        print('\t- Using saved distances')
    except OSError:
        distances = galaxy_object.calculateDistances()
        np.save('distances/subject-{}.npy'.format(id), distances)

    # Perform the clustering (using the DBSCAN clustering algorithm)
    db = galaxy_object.clusterLines(distances)

    print('\t- Fitting arms and errors')
    # Fit both XY and radial splines to the resulting clusters (described in more detail in the method paper)
    galaxy_fit = galaxy_object.fitArms()
    dpjArms = galaxy_object.deprojectArms()

    splines = [r['xy_fit']['spline'] for r in galaxy_fit]

    # PLOTTING
    # Add a helper function to generate plots of the resulting arms
    plt.figure(figsize=(27, 10), dpi=200)
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    ax_annot = plt.subplot2grid((2, 5), (0, 0), colspan=2, rowspan=2)
    ax_cluster = plt.subplot2grid((2, 5), (0, 2))
    ax_isophote = plt.subplot2grid((2, 5), (1, 2))
    ax_final = plt.subplot2grid((2, 5), (0, 3), colspan=2, rowspan=2)

    # panel 1: all drawn arms
    ax_annot.imshow(picArray, cmap='gray', origin='lower')
    for arm in drawnArms:
        ax_annot.plot(*arm.T, '.-', linewidth=1, markersize=2)

    # panel 2: clustered arms
    ax_cluster.imshow(picArray, cmap='gray', origin='lower')
    for i, arm in enumerate(galaxy_object.arms):
        p = ax_cluster.plot(
            *arm.cleanedCloud.T,
            '.',
            c='C{}'.format(i % 10),
            markersize=2,
            label='Cleaned points in arm {}'.format(i)
        )
        c = np.array(to_rgb(p[0].get_color()))*0.7
        p = ax_cluster.plot(
            *arm.pointCloud[np.logical_not(arm.outlierMask)].T,
            'x',
            c=c,
            markersize=3,
            alpha=1,
            label='Outlier points removed from arm {}'.format(i)
        )

    plt.setp(ax_cluster.get_xticklabels(), visible=False)

    # panel 3: NSA isophote
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

    # panel 3: Final splines
    ax_final.imshow(deprojectedImage, cmap='gray', origin='lower')
    for i, arm in enumerate(dpjArms):
        p = ax_final.plot(
            *arm.cleanedCloud.T,
            '.',
            c='C{}'.format(i % 10),
            markersize=2,
            label='Cleaned points in arm {}'.format(i)
        )
        c = np.array(to_rgb(p[0].get_color()))*0.7
        p = ax_final.plot(
            *arm.pointCloud[np.logical_not(arm.outlierMask)].T,
            'x',
            c=c,
            markersize=4,
            label='Outlier points in arm {}'.format(i)
        )

    for i, arm in enumerate(splines):
        prettyPlot(
            galaxy_object.arms[i].deNorm(arm),
            ax=ax_final,
            label='Arm {}. {} drawn poly-lines'.format(
                i,
                np.where(db.labels_ == i)[0].shape[0]
            ),
            c='C{}'.format(i)
        )
    ax_final.legend()

    plt.savefig('arm-fits/subject-{}.jpg'.format(id), bbox_inches='tight')
    plt.close()
    if len(galaxy_fit) > 0:
        plt.figure(figsize=(16, 8))
        for armN, armFit in enumerate(galaxy_fit):
            # plotting

            # Left panel: plot the spirals and errors over the galaxy image
            plt.subplot(121, label='left_panel')

            # show the galaxy
            plt.imshow(deprojectedImage, cmap='gray_r', origin='lower', extent=[-0.5, 0.5, -0.5, 0.5])

            # plot the log spiral and errors
            p = plt.plot(
                *armFit['xy_fit']['log_spiral'].T,
                color='C{}'.format(armN * 2 + 1),
                linewidth=2, label='Logarithmic Spiral, arm {}'.format(armN)
            )
            c = np.array(to_rgb(p[0].get_color())) * 0.8
            plt.plot(*armFit['xy_fit']['log_spiral_error'][0].T, '--', c=c)
            plt.plot(*armFit['xy_fit']['log_spiral_error'][1].T, '--', c=c)
                     # label='Log Spiral $1\sigma$ error')

            # plot the spline and errors
            p = plt.plot(
                *armFit['xy_fit']['spline'].T,
                linewidth=2,
                color='C{}'.format(armN * 2),
                label='Spline Spiral, arm {}'.format(armN)
            )
            c = np.array(to_rgb(p[0].get_color())) * 0.8
            plt.plot(*armFit['xy_fit']['spline_error'][0].T, '--', c=c)
            plt.plot(*armFit['xy_fit']['spline_error'][1].T, '--', c=c)
            #         label='Spline Spiral $1\sigma$ error')

            plt.xlim(-0.5, 0.5)
            plt.ylim(-0.5, 0.5)
            plt.axis('off')
            plt.legend()
            plt.tight_layout()

            plt.subplot(122, label='right_panel')

            pa_obj = armFit['pitch_angle']
            spline_r = armFit['radial']['spline']['r']
            r_bounds = (np.min(spline_r), np.max(spline_r))

            p = plt.plot(
                spline_r[1:-1], pa_obj['spline'][0],
                color='C{}'.format(armN * 2),
                label='Spline fit pitch angle, arm {}'.format(armN)
            )
            c = np.array(to_rgb(p[0].get_color()))
            plt.fill_between(
                spline_r[1:-1],
                *pa_obj['spline'][1:],
                alpha=0.2, color=c,
            )
            p = plt.hlines(
                pa_obj['log_spiral'][0],
                *r_bounds,
                color=c * 0.8,
                linestyle='--',
                label='Log Spiral pitch angle, arm {}'.format(armN)
            )
            plt.fill_between(
                np.linspace(np.min(spline_r), np.max(spline_r), spline_r.shape[0]),
                pa_obj['log_spiral'][0] - pa_obj['log_spiral'][1],
                pa_obj['log_spiral'][0] + pa_obj['log_spiral'][1],
                alpha=0.1, color='k'
            )

        plt.ylabel('Pitch angle (degrees)')
        plt.xlabel('Distance from center of galaxy (arbitrary units)')
        plt.legend()
        plt.tight_layout()
        plt.savefig('error-plots/subject-{}.jpg'.format(id), bbox_inches='tight')
        plt.close()

if __name__ == '__main__':
    with open('subject-id-list.csv', 'r') as f:
        subjectIds = np.array([int(n) for n in f.read().split('\n')])
    # np.random.shuffle(subjectIds)
    # we will write the notebooks out to here
    outputFolder = 'output-notebooks'
    if not os.path.exists(outputFolder):
        os.mkdir(outputFolder)

    # iterate over all subject IDs, and run the notebooks!
    for id in subjectIds:
        main(id)
