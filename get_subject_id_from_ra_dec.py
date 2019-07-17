import sys
import numpy as np
import warnings
import pandas as pd
import lib.galaxy_utilities as gu
from tqdm import tqdm
from astropy.utils.exceptions import AstropyWarning
from astropy.coordinates import SkyCoord
import astropy.units as u

warnings.simplefilter('ignore', category=AstropyWarning)

# ra, dec = 118.716248079, 45.8225546331


def make_map():
    sid_list = sorted(np.loadtxt('lib/subject-id-list.csv', dtype='u8'))
    to_iter = sid_list[:]
    coords = []
    for subject_id in tqdm(to_iter):
        gal, angle = gu.get_galaxy_and_angle(subject_id)
        coords.append((subject_id, gal['RA'].iloc[0], gal['DEC'].iloc[0]))

    df = pd.DataFrame(coords, columns=('subject_id', 'Ra', 'Dec'))

    df.to_pickle('lib/wcs-coord-map.pkl')
    return df


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('Please provide a coordinate to compare to!')
        print('Calculating data frame')
    try:
        ra = float(sys.argv[1])
        dec = float(sys.argv[2])
        coord = SkyCoord(ra*u.degree, dec*u.degree, frame='fk5')
        df = pd.read_pickle('lib/wcs-coord-map.pkl')
        c = SkyCoord(df['Ra'].values * u.degree, df['Dec'].values * u.degree, frame='fk5')
        sep = c.separation(coord)
        print(df[sep.arcsec < 0.01])
    except Exception as e:
        print(e)
        print('Please provide a valid coordinate to compare to!')
