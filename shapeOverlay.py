import pandas
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Rectangle
import matplotlib.image as mpimg
import matplotlib.transforms as transforms
import numpy as np
from shapely.geometry import LineString
import progressbar as pb


def _update_patch_transfrom_about_center(self):
    x = self.convert_xunits(self._x)
    y = self.convert_yunits(self._y)
    width = self.convert_xunits(self._width)
    height = self.convert_yunits(self._height)
    bbox = transforms.Bbox.from_bounds(x, y, width, height)
    rot_trans = transforms.Affine2D()
    rot_trans.rotate_deg_around(x + 0.5 * width, y + 0.5 * height, self.angle)
    self._rect_transform = transforms.BboxTransformTo(bbox)
    self._rect_transform += rot_trans


Rectangle._update_patch_transform = _update_patch_transfrom_about_center

classifications = pandas.read_csv('galaxy-builder-classifications.csv')
wdx = classifications.workflow_version == 61.107
cdx = 0
null = None
true = True
false = False
alpha = 0.8

widgets = ['Ploting: ', pb.Percentage(), ' ', pb.Bar(marker='0', left='[', right=']'), ' ', pb.ETA()]
pbar = pb.ProgressBar(widgets=widgets, maxval=len(classifications[wdx]['subject_ids'].unique()))
pbar.start()
for subject_id, group in classifications[wdx].groupby('subject_ids'):
    image = mpimg.imread('subjects/{0}.png'.format(subject_id))
    fig, ax = plt.subplots(2, 2, figsize=[15, 15])
    # Disk
    ax[0, 0].imshow(image, cmap='gray')
    next_color = ax[0, 0]._get_lines.get_next_color
    # Bulge
    ax[0, 1].imshow(image, cmap='gray')
    # Bar
    ax[1, 0].imshow(image, cmap='gray')
    # Spiral
    ax[1, 1].imshow(image, cmap='gray')
    for idx, classification in group.iterrows():
        annotations = eval(classification.annotations)
        if len(annotations) == 4:
            disk, bulge, bar, spiral = annotations
            color = next_color()
            if len(disk['value'][0]['value']) > 0:
                # plot disk ellipse
                ellipse_params = disk['value'][0]['value'][0]
                e = Ellipse(
                    [ellipse_params['x'], ellipse_params['y']],
                    2 * ellipse_params['rx'],
                    2 * ellipse_params['ry'],
                    angle=-ellipse_params['angle'],
                    ls='-',
                    fc='none',
                    ec=color,
                    alpha=alpha
                )
                ax[0, 0].add_artist(e)
            if len(bulge['value'][0]['value']) > 0:
                # plot bulge ellipse
                ellipse_params = bulge['value'][0]['value'][0]
                e = Ellipse(
                    [ellipse_params['x'], ellipse_params['y']],
                    2 * ellipse_params['rx'],
                    2 * ellipse_params['ry'],
                    angle=-ellipse_params['angle'],
                    ls='-',
                    fc='none',
                    ec=color,
                    alpha=alpha
                )
                ax[0, 1].add_artist(e)
            if len(bar['value'][0]['value']) > 0:
                # plot bar
                rectangle_params = bar['value'][0]['value'][0]
                r = Rectangle(
                    [rectangle_params['x'], rectangle_params['y']],
                    rectangle_params['width'],
                    rectangle_params['height'],
                    angle=rectangle_params['angle'],
                    ls='-',
                    fc='none',
                    ec=color,
                    alpha=alpha
                )
                ax[1, 0].add_patch(r)
            if len(spiral['value'][0]['value']) > 0:
                for arm in spiral['value'][0]['value']:
                    xy = [[a['x'], a['y']] for a in arm['points']]
                    if (len(xy) > 1) and (LineString(xy).is_simple):
                        x, y = np.array(xy).T
                        ax[1, 1].plot(
                            x,
                            y,
                            ls='-',
                            color=color,
                            alpha=alpha
                        )
        fig.savefig('overlay/{0}.png'.format(subject_id))
        plt.close(fig)
    cdx += 1
    pbar.update(cdx)
pbar.finish()
