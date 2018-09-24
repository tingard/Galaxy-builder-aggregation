import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from PIL import Image

plt.figure(figsize=(10, 30), dpi=100)
plt.subplots_adjust(wspace=0.1, hspace=0.1)
ax_annot = plt.subplot2grid((6, 2), (0, 0), colspan=2, rowspan=2)
ax_cluster = plt.subplot2grid((6, 2), (2, 0))
ax_sorting = plt.subplot2grid((6, 2), (2, 1))
ax_isophote = plt.subplot2grid((6, 2), (3, 0))
ax_deproject = plt.subplot2grid((6, 2), (3, 1))
ax_final = plt.subplot2grid((6, 2), (4, 0), colspan=2, rowspan=2)


for i, ax in enumerate([
    ax_annot, ax_cluster, ax_sorting, ax_isophote, ax_deproject, ax_final
]):
    ax.imshow(
        np.zeros((512, 512))
    )
    if i == 1 or i == 2:
        plt.setp(ax.get_xticklabels(), visible=False)
    if i == 2 or i == 4:
        plt.setp(ax.get_yticklabels(), visible=False)

plt.savefig('spacingPlot.png', bbox_inches='tight')
plt.close()
