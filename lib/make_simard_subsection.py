import os
import sys
import numpy as np
import pandas as pd
from astropy.io import fits
import galaxy_utilities as gu
from tqdm import tqdm

try:
    simard_loc = sys.argv[1]
except IndexError:
    raise IndexError('Please provide the location of the catalogue fits file as an argument')
    sys.exit(0)

if not os.path.isfile(simard_loc) and simard_loc.split('.')[-1] == 'fits':
    print('Invalid input file')
    sys.exit(0)

tqdm.pandas(desc='Calculating fixed n', leave=False)
simard_fits = fits.open(simard_loc)
simard_df = pd.DataFrame(gu.meta_map)\
  .loc['SDSS dr7 id'].dropna()\
  .progress_apply(
      lambda i: simard_fits[1].data[simard_fits[1].data['objID'] == np.int64(i)]
  )\
  .apply(lambda i: i[0] if len(i) > 0 else np.nan).dropna()\
  .apply(lambda i: {k: v for k, v in zip(simard_fits[1].data.dtype.fields.keys(), i)})\
  .apply(pd.Series)
simard_df.to_csv('simard-catalog_fixed-n.csv')


tqdm.pandas(desc='Calculating free n', leave=False)
simard_fits = fits.open(simard_loc)
simard_df = pd.DataFrame(gu.meta_map)\
  .loc['SDSS dr7 id'].dropna()\
  .progress_apply(
      lambda i: simard_fits[2].data[simard_fits[1].data['objID'] == np.int64(i)]
  )\
  .apply(lambda i: i[0] if len(i) > 0 else np.nan).dropna()\
  .apply(lambda i: {k: v for k, v in zip(simard_fits[1].data.dtype.fields.keys(), i)})\
  .apply(pd.Series)
simard_df.to_csv('simard-catalog_free-n.csv')
