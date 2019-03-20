import numpy as np
from progress.bar import Bar
import lib.galaxy_utilities as gu
from gzbuilderspirals.oo import Pipeline
import warnings
from astropy.utils.exceptions import AstropyWarning
warnings.simplefilter('ignore', category=AstropyWarning)


sid_list = sorted(np.loadtxt('lib/subject-id-list.csv', dtype='u8'))
bar = Bar('Calculating models', max=len(sid_list), suffix='%(percent).1f%% - %(eta)ds')

for subject_id in sid_list:
    gal, angle = gu.get_galaxy_and_angle(subject_id)
    drawn_arms = gu.get_drawn_arms(subject_id, gu.classifications)
    p = Pipeline(drawn_arms, phi=angle, ba=gal['PETRO_BA90'],
                 image_size=512)
    p.save('lib/pipelines/{}'.format(subject_id))
    arms = p.get_arms()
    for i, arm in enumerate(arms):
        arm.save('lib/spiral_arms/{}-{}'.format(subject_id, i))
    bar.next()
bar.finish()
