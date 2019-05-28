import matplotlib.pyplot as plt
from astropy.io import fits
import numpy as np
import os
loc = 'galfit_files/20902040'
f = fits.open(os.path.join(loc, 'output-annotation-0.fits'))
f[0].data.shape
f2 = fits.open(os.path.join(loc, 'image_cutout.fits'))
f2[0].data.shape
data = [i.data for i in f]
fig, ax = plt.subplots(ncols=3, sharey=True, figsize=(15, 8))

model = data[2]
bad_list = np.loadtxt(os.path.join(loc, 'bad_pixel_table.txt')).astype(int)
print(np.min(bad_list, axis=0), np.max(bad_list, axis=0))
bad_list[0]
data[0][bad_list[:, 0], bad_list[:, 1]] = 0
for i in range(len(ax)):
    data[i][bad_list[:, 0], bad_list[:, 1]] = 0
    im = ax[i].imshow(data[i], cmap='bone')
    plt.colorbar(im, ax=ax[i])
plt.show()
