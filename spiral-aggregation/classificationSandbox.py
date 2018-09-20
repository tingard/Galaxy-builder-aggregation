import os
from tempfile import NamedTemporaryFile
import numpy as np
import pandas as pd
import requests
import matplotlib.pyplot as plt
from PIL import Image
from skimage.transform import rotate, rescale
from gzbuilderspirals import deprojecting as dpj
from gzbuilderspirals import getDrawnArms
from gzbuilderspirals.galaxySpirals import GalaxySpirals

"""
Plots to create:
1) All classifications
3) Clustered arms
4) LOF cleaning
5) Deprojected spline fits
6) Pitch angle plots
"""

classifications = pd.read_csv('../classifications/galaxy-builder-classifications_24-7-18.csv')
subjects = pd.read_csv('../classifications/galaxy-builder-subjects_24-7-18.csv')
null = None
true = True
false = False

subjectId = 14813274
gal = pd.DataFrame([{
    'RA': 160.65881, 'DEC': 23.95191, 'Z': 0.043594,
    'RUN': 5137, 'CAMCOL': 6, 'FIELD': 314, 'RERUN': 301,
    'SERSIC_BA': 0.576469, 'SERSIC_PHI': 43.1215,
    'PETRO_THETA': 22.2729,
}])

fitsname = dpj.getFitsName(gal)
print('FITS name:', fitsname)

w = dpj.createWCSObject(gal, 512)
angle = dpj.getAngle(gal, w, np.array([512, 512])) - 90
print('Angle:', angle)

url = 'https://panoptes-uploads.zooniverse.org/production/subject_location/6ee8db8f-2a1e-4d10-bcb9-1e5ec84f8f10.png'
print('URL of galaxy image:')
print(url)
imgData = requests.get(url).content

f = NamedTemporaryFile(suffix='.{}'.format(url.split('.')[-1]), delete=False)
f.write(imgData)
f.close()
pic = Image.open(f.name)
picArray = np.array(pic)
os.unlink(f.name)

rotatedImage = rotate(picArray, -angle)
rotatedImage.shape
stretchedImage = rescale(rotatedImage, (1, 1 / gal['SERSIC_BA'].iloc[0]))

n = int((stretchedImage.shape[1] - np.array(pic).shape[1]) / 2)

if n > 0:
    deprojectedImage = stretchedImage[:, n:-n]
else:
    deprojectedImage = stretchedImage.copy()

drawnArms = getDrawnArms(subjectId, classifications)

# plot all the drawn arms
plt.figure(dpi=200)
for arm in drawnArms:
    plt.plot(*arm.T, '.-', linewidth=0.5, markersize=1, alpha=0.7)
plt.axis('equal')
plt.savefig(
    '../../methodPaper/armClustering/classifications.jpg',
    bbox_inches='tight'
)
plt.close()

print('Found {} arms'.format(len(drawnArms)))

s = GalaxySpirals(drawnArms, ba=gal['SERSIC_BA'].iloc[0], phi=angle)

distances = s.calculateDistances()

db = s.clusterLines(distances)

xyResult = s.fitXYSplines()

result = s.fitRadialSplines(xyResult)

# plot the clustering result
plt.figure(dpi=200)
for l in np.arange(np.max(db.labels_) + 1):
    for arm in drawnArms[db.labels_ == l]:
        plt.plot(
            *arm.T,
            '.-', c='C{}'.format(l % 10),
            linewidth=0.5, markersize=1, alpha=0.8,
        )

plt.axis('equal')
plt.savefig(
    '../../methodPaper/armClustering/groupedArms.jpg',
    bbox_inches='tight',
)
plt.close()


def prettyPlot(arm, **kwargs):
    plt.plot(
        *arm.T,
        c='k'.format(i), linewidth=4
    )
    plt.plot(
        *arm.T, linewidth=3, **kwargs
    )
    plt.plot(
        *arm.T,
        c='w', linewidth=2, alpha=0.5
    )


# plot the resulting deprojected splines
plt.figure(dpi=200)
plt.imshow(deprojectedImage, cmap='gray', origin='lower')
for i, arm in enumerate(result.deprojectedArms):
    plt.plot(*arm.pointCloud.T, 'k.', markersize=2, alpha=1)
    plt.plot(
        *arm.pointCloud.T,
        '.', c='C{}'.format(i % 10), markersize=2, alpha=0.3
    )
    plt.plot(
        *arm.cleanedCloud.T,
        '.', c='C{}'.format(i % 10), markersize=2, alpha=1
    )

for o in result.orderings:
    plt.plot(*o.orderedPoints.T, 'w.', markersize=2, alpha=0.2)

for i, arm in enumerate(result.radialFit):
    prettyPlot(
        s.arms[0].deNorm(arm),
        label='Arm {}'.format(i),
        c='C{}'.format(i % 10)
    )

plt.axis('off')
plt.savefig(
    '../../methodPaper/armClustering/overlaidSplinesDeprojected.jpg',
    bbox_inches='tight'
)
plt.close()


# plot the resulting splines
plt.figure(figsize=(20, 10), dpi=200)
plt.subplot(121)
plt.imshow(pic, cmap='gray', origin='lower')
for i, arm in enumerate(s.arms):
    plt.plot(*arm.pointCloud.T, 'k.', markersize=2, alpha=1)
    plt.plot(
        *arm.pointCloud.T,
        '.', c='C{}'.format(i % 10), markersize=2, alpha=0.3
    )
    plt.plot(
        *arm.cleanedCloud.T,
        '.', c='C{}'.format(i % 10), markersize=2, alpha=1
    )

for o in xyResult.orderings:
    plt.plot(*o.orderedPoints.T, 'w.', markersize=2, alpha=0.2)

for i, ra in enumerate(xyResult.representativeArms):
    plt.plot(*ra.T, c='C{}'.format(i))
t = np.linspace(0, 1, 1000)

for i, arm in enumerate(xyResult.fits):
    prettyPlot(
        s.arms[0].deNorm(arm),
        label='Arm {}'.format(i),
        c='C{}'.format(i % 10)
    )

plt.axis('off')

plt.subplot(122)
plt.imshow(deprojectedImage, cmap='gray', origin='lower')
for i, arm in enumerate(result.deprojectedArms):
    plt.plot(*arm.pointCloud.T, 'k.', markersize=2, alpha=1)
    plt.plot(
        *arm.pointCloud.T,
        '.', c='C{}'.format(i % 10), markersize=2, alpha=0.3
    )
    plt.plot(
        *arm.cleanedCloud.T,
        '.', c='C{}'.format(i % 10), markersize=2, alpha=1
    )

for o in result.orderings:
    plt.plot(*o.orderedPoints.T, 'w.', markersize=2, alpha=0.2)

for i, arm in enumerate(result.radialFit):
    prettyPlot(arm, label='Arm {}'.format(i % 10), c='C{}'.format(i % 10)),

plt.axis('off')
plt.savefig(
    '../../methodPaper/armClustering/overlaidSplines.jpg',
    bbox_inches='tight'
)
plt.close()
