import os
from PIL import Image
import numpy as np

plot_folder = 'pipeline_plots/'
im_list = [
    'image_deprojection.png',
    'cleaning.png',
    'fits.png',
    'polar_fits.png',
    'fit_comparison.png'
]

for sub_id in os.listdir(plot_folder)[:1]:
    loc = plot_folder + sub_id + '/'
    images = [Image.open(loc + im) for im in im_list]
    out_width = max(i.width for i in images)
    out_height = sum(i.height for i in images)
    out = np.array([])
    montage = Image.new(mode='RGB', size=(out_width, out_height), color='#ffffff')
    cursor = 0
    for image in images:
        montage.paste(
            image,
            box=(int((montage.width - image.width) / 2), cursor)
        )
        cursor += image.height
    montage.save(loc + '/combined.png')
