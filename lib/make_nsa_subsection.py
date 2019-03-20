from astropy.table import Table
import pandas as pd
import os

this_file_location = os.path.dirname(os.path.abspath(__file__))
nsa_catalog = Table.read(
    '/Users/tlingard/PhD/galaxy-builder/subjectUpload/nsa_v1_0_1.fits',
    format='fits'
)

nsa_keys = (
    'NSAID', 'ISDSS', 'INED', 'IAUNAME', # identifiers
    'RA', 'DEC', 'Z', 'ZDIST', # position
    'SERSIC_BA', 'SERSIC_PHI', # sersic photometry
    'PETRO_THETA', # azimuthally averaged petrosean radius
    'PETRO_BA90', 'PETRO_PHI90', # petrosean photometry at 90% light radius
    'PETRO_BA50', 'PETRO_PHI50', # ... at 50% light radius
    'RUN', 'CAMCOL', 'FIELD', 'RERUN',
    'ELPETRO_MASS', 'SERSIC_MASS',
)

pd.DataFrame(
    {k: nsa_catalog[k] for k in nsa_keys}
).to_pickle(
    this_file_location + '/df_nsa.pkl'
)
