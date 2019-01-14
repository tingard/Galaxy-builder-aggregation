from astropy.io import fits
from astropy.nddata import Cutout2D
from astropy.wcs import WCS
import astropy.units as u
from astropy.coordinates import SkyCoord
import numpy as np
import sep


def save_cutout(fFile, position, size, output_file='example_cutout.fits'):
    position, size = map(np.array, (position, size))
    hdu = fFile[0]
    # Put the cutout image in the FITS HDU
    wcs = WCS(hdu.header)
    try:
        frame = fFile[0].header['SYSTEM'].strip().lower()
    except KeyError:
        frame = 'fk5'
    ra, dec = position * u.degree
    cutout = Cutout2D(
        hdu.data,
        position=SkyCoord(ra=ra, dec=dec, frame=frame),
        size=size * u.arcsec,
        wcs=wcs,
    )
    hdu.data = cutout.data
    # Update the FITS header with the cutout WCS
    hdu.header.update(cutout.wcs.to_header())
    # Write the cutout to a new FITS file
    hdu.writeto(output_file, overwrite=True)
    return hdu.data


def source_extract_image(data, bkgArr=None, thresh=0.05, sortType='center'):
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
