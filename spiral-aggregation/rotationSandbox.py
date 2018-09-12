from astropy.wcs import WCS
import pandas
import requests
from shapely.geometry import LineString
import json
import os
from tempfile import NamedTemporaryFile
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from astropy.io import fits
import pandas as pd
import numpy as np
import spiralExtraction

deNormArm = lambda a: 512*(0.5 + a)
getUrl = lambda id: eval(subjects[subjects['subject_id'] == id]['locations'].iloc[0])['1']


nsa = fits.open('../../subjectUpload/nsa_v1_0_1.fits')
classifications = pandas.read_csv('../classifications/galaxy-builder-classifications_24-7-18.csv')
subjects = pandas.read_csv('../classifications/galaxy-builder-subjects_24-7-18.csv')
null = None
true = True
false = False

most_recent_workflow = classifications.workflow_version == 61.107
classifications[most_recent_workflow].groupby('subject_ids')
classificationsCounts = classifications[most_recent_workflow]['subject_ids'].value_counts()
subjIds = np.array(classificationsCounts[classificationsCounts > 25].index)


def getDrawnArms(id):
    annotationsForSubject = [
        eval(foo) for foo in
        classifications[classifications['subject_ids'] == id]['annotations']
    ]
    try:
        annotationsWithSpiral = [
            c[3]['value'][0]['value']
            for c in annotationsForSubject
            if len(c) > 3 and len(c[3]['value'][0]['value'])
        ]
    except IndexError as e:
        print('{} raised {}'.format(id, e))
        assert False
    spirals = [[a['points'] for a in c] for c in annotationsWithSpiral]
    spiralsWithLengthCut = [
        [[[p['x'], p['y']] for p in a] for a in c]
        for c in spirals if all([len(a) > 5 for a in c])
    ]
    drawnArms = np.array([
        np.array(arm) for classification in spiralsWithLengthCut
        for arm in classification
        if LineString(arm).is_simple
    ])
    return drawnArms


def p(line, *args, ax=plt, **kwargs):
    ax.plot(line[:, 0], line[:, 1], *args, **kwargs)

def plotThing(chosenId):
    drawnArms = getDrawnArms(chosenId)

    functions, labels = spiralExtraction.fit(drawnArms, verbose=True, returnArmLabels=True)

    meta = eval(subjects[subjects['subject_id'] == chosenId].iloc[0]['metadata'])

    nsa_keys = [
        'NSAID', 'ISDSS', 'INED', 'RA', 'DEC', 'Z', 'SERSIC_BA', 'SERSIC_PHI', 'PETRO_THETA',
        'IAUNAME', 'ZDIST', 'RUN', 'CAMCOL', 'FIELD', 'RERUN',
    ]
    nsaRas = nsa[1].data['ra']
    nsaDecs = nsa[1].data['dec']

    df_nsa = pd.DataFrame(
        {key: nsa[1].data[key].byteswap().newbyteorder() for key in nsa_keys}
    )

    gal = df_nsa[df_nsa['NSAID'] == int(meta['NSA id'])]


    # Lookup the source fits file (needed for the rotation matrix)
    fname = 'fitsImages/{0}/{1}/frame-r-{0:06d}-{1}-{2:04d}.fits'.format(
        int(gal['RUN']),
        int(gal['CAMCOL']),
        int(gal['FIELD'])
    )
    print('Source fits file:', fname)

    if not os.path.isfile(fname):
        print('\n\n\tNo file {}, aborting\n\n'.format(fname))
        assert False

    # Load a WCS object from the FITS image
    wFits = WCS(fname)
    print('\n\nWCS from fits image:', wFits)

    # The SDSS pixel scale is 0.396 arc-seconds
    fits_cdelt = 0.396 / 3600

    phi = float(gal['SERSIC_PHI'])

    # cutouts were chosen to be 4x Petrosean radius, and then scaled (including interpolation) to be 512x512 pixels
    scale = 4 * (float(gal['PETRO_THETA']) / 3600) / 512

    # This should be obtained from the image, as some were not square.
    size_pix = np.array([512, 512])

    # Create a new WCS object
    w = WCS(naxis=2)
    w.wcs.crpix = size_pix / 2
    w.wcs.crval = np.array([float(gal['RA']), float(gal['DEC'])])
    w.wcs.ctype = ['RA---TAN', 'DEC--TAN']
    w.wcs.cunit = ['deg', 'deg']

    # Copy the rotation matrix from the source FITS file, adjusting the scaling as needed
    w.wcs.cd = [
        wFits.wcs.cd[0] / fits_cdelt * scale,
        wFits.wcs.cd[1] / fits_cdelt * scale
    ]

    print('\n\nCreated WCS:')
    print(w)

    r = float(gal['PETRO_THETA'])/3600

    cx, cy = 256, 256
    wCx, wCy = float(gal['RA']), float(gal['DEC'])

    # find our line in world coordinates
    x = r * np.sin(np.deg2rad(90 + phi)) + wCx
    y = r * np.cos(np.deg2rad(90 - phi)) + wCy

    ra_line, dec_line = w.wcs_world2pix([wCx, x], [wCy, y], 0)

    axis_vector = np.subtract.reduce(
        np.stack(
            (ra_line, dec_line),
            axis=1
        )
    )

    normal_vector = np.dot([[0, 1], [-1, 0]], axis_vector) * float(gal['SERSIC_BA'])
    origin = np.array([cx, cy])

    angle = 180 * np.arccos(axis_vector[1]/np.linalg.norm(axis_vector))/ np.pi

    print('Angle:', angle)

    url = getUrl(chosenId)
    imgData = requests.get(url).content

    f = NamedTemporaryFile(suffix='.{}'.format(url.split('.')[-1]), delete=False)
    f.write(imgData)
    f.close()
    pic = Image.open(f.name)
    os.unlink(f.name)

    ax = plt.gca()
    plt.imshow(np.array(pic), cmap='gray', origin='lower')

    t = np.linspace(0, 1, 500)
    for i, (Sx, Sy) in enumerate(functions):
        plt.plot(
            deNormArm(Sx(t)),
            deNormArm(Sy(t)),
            linewidth=5,
            label='Arm {}'.format(i)
        )

    e = Ellipse(
        xy=(256, 256),
        width=256 * gal['SERSIC_BA'],
        height=256,
        angle=-angle,
        ec='w',
        fc='none'
    )
    ax.add_artist(e)
    plt.savefig('ellipses/subject-{}.png'.format(chosenId))
    plt.clf()

if __name__ == "__main__":
    import re
    ids = [
        int(re.search('[0-9]+', i).group())
        for i in os.listdir('montages')
        if re.search('[0-9]+', i)
    ]
    print('Available IDs:\n{}'.format('\t'.join(map(str, ids))))

    plt.figure(figsize=(10, 10))
    for chosenId in ids:
        plotThing(chosenId)
