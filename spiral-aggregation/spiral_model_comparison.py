# for all our duplicate galaxies, determine which spiral model best fits them
# and plot it against their mass from the NSA catalogue
import os
import re
import argparse
import numpy as np
from astropy.io import fits
import pandas as pd
import matplotlib.pyplot as plt
import lib.galaxy_utilities as gu
from gzbuilderspirals.oo import Pipeline, Arm
from gzbuilderspirals.fitting import AIC
from progress.bar import Bar
import warnings
from astropy.utils.exceptions import AstropyWarning
warnings.simplefilter('ignore', category=AstropyWarning)
from sklearn.metrics import r2_score

dr8ids, ss_ids, validation_ids = np.load('lib/duplicate_galaxies.npy').T

def make_combined_arms():
    bar = Bar('Obtaining combined spirals',
              max=len(dr8ids), suffix='%(percent).1f%% - %(eta)ds')
    avail = os.listdir('lib/duplicate_spiral_arms')
    for i in range(len(dr8ids)):
        original_id = ss_ids[i]
        validation_id = validation_ids[i]
        gal, angle = gu.get_galaxy_and_angle(original_id)
        mass = float(gal['SERSIC_MASS'])

        original_drawn_arms = gu.get_drawn_arms(original_id, gu.classifications)
        validation_drawn_arms = gu.get_drawn_arms(validation_id, gu.classifications)
        drawn_arms = np.array(
            list(original_drawn_arms) + list(validation_drawn_arms),
        )
        p = Pipeline(drawn_arms, phi=angle, ba=gal['PETRO_BA90'],
                     image_size=512, distances=None, parallel=True)
        arms = p.get_arms()
        for j, arm in enumerate(arms):
            arm.save('lib/duplicate_spiral_arms/{}-{}'.format(dr8ids[i], j))
        bar.next()
    bar.finish()


def make_models():
    bar = Bar('Obtaining model fits',
              max=len(dr8ids), suffix='%(percent).1f%% - %(eta)ds')
    arm_loc = 'lib/duplicate_spiral_arms'
    df = []
    arm_count = []
    for i in range(len(dr8ids)):
        original_id = ss_ids[i]
        validation_id = validation_ids[i]
        gal, angle = gu.get_galaxy_and_angle(original_id)
        mass = float(gal['SERSIC_MASS'])

        arms = [
            Arm.load(os.path.join(arm_loc, f))
            for f in os.listdir(arm_loc)
            if re.match('^{}-[0-9]+.pickle$'.format(dr8ids[i]), f)
        ]
        for i, arm in enumerate(arms):
            if not arm.FLAGGED_AS_BAD:
                arm_count.append(len(arm.arms))
                models, scores = arm.fit_polynomials(n_splits=8, score=r2_score)
                s = {
                    **{k: v.mean() for k,v in scores.items()},
                    **{'{}_std'.format(k): v.std() for k,v in scores.items()}
                }
                s['mass'] = mass
                s['dr8objid'] = str(dr8ids[i])
                df.append(s)
        bar.next()
    bar.finish()
    arm_count = np.array(arm_count)
    print(arm_count.mean(), arm_count.std())
    print(arm_count.min(), arm_count.max())
    df = pd.DataFrame(df)
    df.to_pickle('model-comparison-results.pkl')


def make_arm_plots():
    outfile = 'lib/duplicate_comb_spirals'
    bar = Bar('Plotting arms',
              max=len(dr8ids), suffix='%(percent).1f%% - %(eta)ds')
    arm_loc = 'lib/duplicate_spiral_arms'
    df = []
    for i in range(len(dr8ids)):
        original_id = ss_ids[i]
        gal, angle = gu.get_galaxy_and_angle(original_id)
        pic_array, _ = gu.get_image(gal, original_id, angle)
        arms = [
            Arm.load(os.path.join(arm_loc, f))
            for f in os.listdir(arm_loc)
            if re.match('^{}-[0-9]+.pickle$'.format(dr8ids[i]), f)
        ]
        plt.figure(figsize=(8, 8))
        plt.imshow(pic_array, cmap='gray')
        for i, arm in enumerate(arms):
            # plt.plot(*arm.coords[arm.outlier_mask].T, '.', markersize=2, alpha=0.3)
            # plt.plot(*arm.coords[~arm.outlier_mask].T, 'rx', markersize=1, alpha=0.8)
            plt.plot(*arm.reprojected_log_spiral.T, c=('C2' if not arm.FLAGGED_AS_BAD else 'C1'))
        # plt.plot([pic_array.shape[0] / 2], [pic_array.shape[1] / 2], 'o', markersize=10)

        plt.savefig(os.path.join(outfile, '{}.png'.format(original_id)))
        plt.close()
        bar.next()
    bar.finish()


def make_mass_model_plot():
    df = pd.read_pickle('model-comparison-results.pkl')
    print(df.groupby('dr8objid').mean().mean(axis=0))
    c = ['poly_spiral_1', 'poly_spiral_2', 'poly_spiral_3']#, 'poly_spiral_4']
    plt.hlines(0, df['mass'].min(), df['mass'].max(), 'k', alpha=0.4)
    for k in c:
        plt.plot(
            df['mass'], (df['log_spiral'] - df[k]) / np.abs(df['log_spiral']),
            '.', label=k
        )
    plt.xscale('log')
    plt.legend()
    plt.xlabel(r'Galaxy mass [$M_\odot$]')
    plt.savefig('model-comparison-results.png', bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Calculate duplicate component properties')
    parser.add_argument('--recalculate-arms', '-r', help='Recalculate saved components',
                        action='store_true')
    parser.add_argument('--recalculate-models', '-m', help='Recalculate saved parameters',
                        action='store_true')
    parser.add_argument('--make-arm-plots', '-p', help='Plot combined log spirals',
                        action='store_true')
    args = parser.parse_args()
    if args.recalculate_arms:
        make_combined_arms()
    if args.recalculate_models:
        make_models()
    if args.make_arm_plots:
        make_arm_plots()
    make_mass_model_plot()
