# import sdss_psf
import numpy as np
import sep
# from astropy.wcs import WCS
# from astropy.io import fits
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from PIL import Image
# import os
# import time
from copy import copy
import montage_wrapper as montage
# import sdssCutoutGrab as scg


def stretchArray(arr, a=0.1):
    # shift so lowest value = 0
    shiftedArr = arr - np.amin(arr)
    # normalise from 0 to 1
    normalisedArr = shiftedArr / np.amax(shiftedArr)
    # perform the stretch
    arrOut = np.arcsinh(normalisedArr / a) / np.arcsinh(1 / a)
    return arrOut


def inverseArcsinh(arr, a=0.1):
    return a * np.sinh(arr * np.arcsinh(1 / a))


def makeEllipseMask(e, x, y, threshold=0.01):
    a, b, angl = e['a'], e['b'], e['theta']
    x = x - e['x']
    y = y - e['y']
    return (np.cos(angl)**2 / a**2 + np.sin(angl)**2 / b**2) * x**2 +\
        2 * np.cos(angl) * np.sin(angl) * (1 / a**2 - 1 / b**2) * x * y +\
        (np.sin(angl)**2 / a**2 + np.cos(angl)**2 / b**2) * y**2 < 1 +\
        threshold


def sourceExtractImage(data, bkgArr=None, thresh=0.05, sortType='center'):
    """Extract sources from data array and return enumerated objects sorted
    smallest to largest, and the segmentation map provided by source extractor
    """
    data = data.byteswap().newbyteorder()
    if bkgArr is None:
        bkgArr = np.zeros(data.shape)
    o = sep.extract(data, thresh, segmentation_map=True)
    if sortType == 'size':
        print('Sorting extracted objects by radius from size')
        sizeSortedObjects = sorted(
            enumerate(o[0]), key=lambda src: src[1]['npix']
        )
        return sizeSortedObjects, o[1]
    elif sortType == 'center':
        print('Sorting extracted objects by radius from center')
        centerSortedObjects = sorted(
            enumerate(o[0]),
            key=lambda src: (
                (src[1]['x'] - data.shape[0] / 2)**2 +
                (src[1]['y'] - data.shape[1] / 2)**2
            )
        )[::-1]
        return centerSortedObjects, o[1]


def maskArr(arrIn, segMap, maskID):
    """Return a true/false mask given a segmentation map and segmentation ID
    True signifies the pixel should be masked
    """
    return np.logical_and(segMap != maskID, segMap != 0)


def maskArrWithEllipses(arrIn, objects):
    # plot background-subtracted image
    mask = np.zeros(arrIn.shape, dtype=bool)
    y, x = np.ogrid[0:mask.shape[0], 0:mask.shape[1]]
    for i in range(len(objects)):
        el = copy(objects[i])
        el['a'], el['b'] = 3 * el['a'], 3 * el['b']
        elMask = makeEllipseMask(el, x, y)
        mask[elMask] = True
    return mask


def showObjectsOnArr(arr, objects):
    fix, ax = plt.subplots()
    ax.imshow(
        stretchArray(arr),
        interpolation='nearest',
        cmap='gray',
        origin='lower',
        vmax=0.6
    )
    # plot an ellipse for each object
    for i in range(len(objects) - 1):
        e = Ellipse(xy=(objects[i]['x'], objects[i]['y']),
                    width=6 * objects[i]['a'],
                    height=6 * objects[i]['b'],
                    angle=objects[i]['theta'] * 180. / np.pi)
        e.set_facecolor('none')
        e.set_edgecolor('red')
        ax.add_artist(e)
    eMain = Ellipse(
        xy=(objects[-1]['x'], objects[-1]['y']),
        width=6 * objects[-1]['a'],
        height=6 * objects[-1]['b'],
        angle=objects[-1]['theta'] * 180. / np.pi
    )
    eMain.set_facecolor('none')
    eMain.set_edgecolor('green')
    ax.add_artist(eMain)
    plt.show()


def saveImage(
        arr, fname='testImage.png', resize=False, size=(512, 512),
        preserveAspectRatio=True, resample=Image.LANCZOS):
    # ensure image is normalised to [0, 255]
    print('📷  Saving image to {}'.format(fname))
    arr = (arr.transpose() - np.amin(arr)) / np.amax(arr - np.amin(arr)) * 255
    # cast to uint8 with a weird coordinate swap (idk why)
    im = Image.fromarray(
        np.uint8(np.flipud(np.swapaxes(np.flipud(arr), 0, 1)))
    )
    # want to preserve aspect ratio, so increase the width to provided width
    if preserveAspectRatio:
        correctedSize = (size[0], int(im.size[1] / im.size[0] * size[0]))
    else:
        correctedSize = size[:]
    if resize:
        im = im.resize(correctedSize, resample)
    im.save(fname)
    return im


def montageDir(inputDir, outputDir):
    return montage.wrappers.mosaic(inputDir, outputDir)
