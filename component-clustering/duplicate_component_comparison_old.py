import os
import re
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import lib.galaxy_utilities as gu
import lib.average_shape_helpers as ash
from gzbuilderspirals.oo import Pipeline, Arm
from tqdm import tqdm
import argparse
import warnings
from astropy.utils.exceptions import AstropyWarning
warnings.simplefilter('ignore', category=AstropyWarning)

# Should we recalculate the values (check for a flag in args, or missing files)
parser = argparse.ArgumentParser(
    description='Calculate duplicate component properties'
)
parser.add_argument('-r', '--recalculate',
                    help='Should recalculate saved components',
                    action='store_true')
args = parser.parse_args()
SHOULD_RECALCULATE = (
    args.recalculate
    or not os.path.isfile('duplicate-components.pkl')
    or not os.path.isfile('combined_duplicates_pitch_angle.pkl')
)


def get_model_string(v):
    return '+'.join(k for k in ('disk', 'bulge', 'bar') if v[k] is not None)


dr8ids, ss_ids, validation_ids = np.load('lib/duplicate_galaxies.npy').T

# get the alread-calculated models for individual subjects
if not SHOULD_RECALCULATE:
    df = pd.read_pickle('duplicate-components.pkl')
else:
    df = []
    msg = 'Obtaining original & validation models'
    for i in tqdm(len(dr8ids)):
        res = {
            'original_id': ss_ids[i],
            'validation_id': validation_ids[i],
            'original_model_string': '',
            'validation_model_string': '',
            'original_disk_ba': np.nan,
            'validation_disk_ba': np.nan,
            'original_disk_reff': np.nan,
            'validation_disk_reff': np.nan,
            'original_bulge_ba': np.nan,
            'validation_bulge_ba': np.nan,
            'original_bulge_reff': np.nan,
            'validation_bulge_reff': np.nan,
            'original_bar_ba': np.nan,
            'validation_bar_ba': np.nan,
            'original_bar_reff': np.nan,
            'validation_bar_reff': np.nan,
            'original_pa': np.nan,
            'validation_pa': np.nan,
        }
        gal, angle = gu.get_galaxy_and_angle(ss_ids[i])
        gal_v, angle_v = gu.get_galaxy_and_angle(validation_ids[i])
        original_id = ss_ids[i]
        validation_id = validation_ids[i]
        try:
            with open('cluster-output/{}.json'.format(original_id)) as f:
                original_components = json.load(f)
            with open('cluster-output/{}.json'.format(validation_id)) as f:
                validation_components = json.load(f)
        except OSError:
            continue
        res['original_model_string'] = get_model_string(
            original_components
        )
        res['validation_model_string'] = get_model_string(
            validation_components
        )

        both_have_disks = (original_components['disk'] is not None
                           and validation_components['disk'] is not None)
        both_have_bulges = (original_components['bulge'] is not None
                            and validation_components['bulge'] is not None)
        both_have_bars = (original_components['bar'] is not None
                          and validation_components['bar'] is not None)
        if both_have_disks:
            o_disk = ash.sanitize_param_dict(original_components['disk'])
            v_disk = ash.sanitize_param_dict(validation_components['disk'])
            res['original_disk_ba'] = o_disk['axRatio']
            res['validation_disk_ba'] = v_disk['axRatio']
            res['original_disk_reff'] = o_disk['rEff']
            res['validation_disk_reff'] = v_disk['rEff']
        if both_have_bulges:
            o_bulge = ash.sanitize_param_dict(original_components['bulge'])
            v_bulge = ash.sanitize_param_dict(validation_components['bulge'])
            res['original_bulge_ba'] = o_bulge['axRatio']
            res['validation_bulge_ba'] = v_bulge['axRatio']
            res['original_bulge_reff'] = o_bulge['rEff']
            res['validation_bulge_reff'] = v_bulge['rEff']
        if both_have_bars:
            o_bar = ash.sanitize_param_dict(original_components['bar'])
            v_bar = ash.sanitize_param_dict(validation_components['bar'])
            res['original_bar_ba'] = o_bar['axRatio']
            res['validation_bar_ba'] = v_bar['axRatio']
            res['original_bar_reff'] = o_bar['rEff']
            res['validation_bar_reff'] = v_bar['rEff']

        original_drawn_arms = gu.get_drawn_arms(original_id)
        validation_drawn_arms = gu.get_drawn_arms(validation_id)

        try:
            original_p = Pipeline.load(
                'lib/pipelines/{}.json'.format(original_id)
            )
            original_arms = (
                Arm.load(os.path.join('lib/spiral_arms', i))
                for i in os.listdir('lib/spiral_arms')
                if re.match('^{}-[0-9]+.pickle$'.format(original_id), i)
            )
            original_arms = [arm for arm in original_arms
                             if not arm.FLAGGED_AS_BAD]
        except IOError as e:
            print(e)
            original_p = Pipeline(original_drawn_arms, phi=angle,
                                  ba=gal['PETRO_BA90'], image_size=512,
                                  parallel=True)
            original_arms = original_p.get_arms()

        try:
            validation_p = Pipeline.load(
                'lib/pipelines/{}.json'.format(validation_id)
            )
            validation_arms = (
                Arm.load(os.path.join('lib/spiral_arms', i))
                for i in os.listdir('lib/spiral_arms')
                if re.match('^{}-[0-9]+.pickle$'.format(validation_id), i)
            )
            validation_arms = [arm for arm in validation_arms
                               if not arm.FLAGGED_AS_BAD]
        except IOError as e:
            print(e)
            validation_p = Pipeline(validation_drawn_arms, phi=angle,
                                    ba=gal['PETRO_BA90'], image_size=512,
                                    parallel=True)
            validation_arms = validation_p.get_arms()

        res['original_pa'], res['original_sigma_pa'] = (
            original_p.get_pitch_angle(
                arms=original_arms
            )
        )
        res['validation_pa'], res['validation_sigma_pa'] = (
            validation_p.get_pitch_angle(
                arms=validation_arms
            )
        )
        df.append(res)

    df = pd.DataFrame(df).set_index('original_id')
    df.to_pickle('duplicate-components.pkl')


# Plots for comparison between models
# disk comparison
for comp in ('disk', 'bulge', 'bar'):
    fig, plot_axes = plt.subplots(ncols=2, figsize=(10, 4))
    for p, label, ax in zip(('ba', 'reff'), ('axis ratio', 'size'), plot_axes):
        c_selection = [
            'original_{}_{}'.format(comp, p),
            'validation_{}_{}'.format(comp, p)
        ]
        limits = (
            np.nanmin(df[c_selection].values),
            np.nanmax(df[c_selection].values)
        )

        df.plot(*c_selection, kind='scatter', ax=ax, use_index=False)
        ax.plot(limits, limits, color='k', alpha=0.3)
        ax.set_xlabel('{} {} from sub-sample'.format(comp.capitalize(), label))
        ax.set_ylabel('{} {} from validation subset'.format(
            comp.capitalize(), label
        ))
        ax.set_aspect('equal')
        ax.set_xlim(limits)
        ax.set_ylim(limits)

    plt.savefig('duplicates_plots/{}_comparison.pdf'.format(comp),
                bbox_inches='tight')
    plt.close()

# calculate pitch angles for combined duplicate subjects
if not SHOULD_RECALCULATE:
    pa_df = pd.read_pickle('combined_duplicates_pitch_angle.pkl')
else:
    print('Calculaing pitch angles for combined classifications')
    pa_df = {'pa': [], 'sigma_pa': []}
    msg = 'Obtaining combined spirals'
    for i in tqdm(len(dr8ids)):
        original_id = ss_ids[i]
        validation_id = validation_ids[i]
        gal, angle = gu.get_galaxy_and_angle(original_id)
        gal_v, angle_v = gu.get_galaxy_and_angle(validation_id)
        try:
            arm_locations = [
                os.path.join('lib/duplicate_spiral_arms/', f)
                for f in os.listdir('lib/duplicate_spiral_arms/')
                if re.match(r'^{}-[0-9]+.pickle'.format(dr8ids[i]), f)
            ]
            arms = (Arm.load(f) for f in arm_locations)
            arms = [arm for arm in arms if not arm.FLAGGED_AS_BAD]
            p = arms[0].get_parent() if len(arms) > 0 else None
        except FileNotFoundError:
            original_drawn_arms = gu.get_drawn_arms(original_id)
            validation_drawn_arms = gu.get_drawn_arms(validation_id)
            try:
                if (
                    len(original_drawn_arms) == 0
                    or len(validation_drawn_arms) == 0
                ):
                    drawn_arms = (
                        original_drawn_arms
                        if len(original_drawn_arms) != 0
                        else validation_drawn_arms
                    )
                else:
                    drawn_arms = np.array(
                        list(map(
                            np.array,
                            original_drawn_arms.tolist()
                            + validation_drawn_arms.tolist()
                        ))
                    )
            except Exception as e:
                print(e)
                drawn_arms = np.array([])
            p = Pipeline(drawn_arms, phi=angle, ba=gal['PETRO_BA90'],
                         image_size=512, parallel=True)
            arms = p.get_arms()
        if p is not None:
            pa, sigma_pa = p.get_pitch_angle(arms=arms)
        else:
            pa, sigma_pa = np.nan, np.nan
        pa_df['pa'].append(pa)
        pa_df['sigma_pa'].append(sigma_pa)

    pa_df = pd.DataFrame(
        pa_df,
        index=ss_ids,
    )
    pa_df.to_pickle('combined_duplicates_pitch_angle.pkl')

# spiral arm pitch angle
plt.figure(figsize=(8, 4))
plt.errorbar(
    pa_df['pa'], df['original_pa'] - pa_df['pa'],
    np.sqrt(pa_df['sigma_pa']**2 + df['original_sigma_pa']**2),
    fmt='.',
    label=r'$\Psi_\mathrm{original}$'
)
plt.errorbar(
    pa_df['pa'], df['validation_pa'] - pa_df['pa'],
    np.sqrt(pa_df['sigma_pa']**2 + df['validation_sigma_pa']**2),
    fmt='.',
    label=r'$\Psi_\mathrm{validation}$'
)
plt.xlabel(r'$\Psi_\mathrm{combined}$')
plt.ylabel(r'$\Psi_\mathrm{subset} - \Psi_\mathrm{combined}$')
plt.hlines(0, 0, 70, alpha=0.2)
plt.legend()
plt.savefig('duplicates_plots/pa_comparison.pdf', bbox_inches='tight')


plt.figure(figsize=(5, 5))
plt.errorbar(
    df['original_pa'], df['validation_pa'],
    df['original_sigma_pa'], df['validation_sigma_pa'],
    fmt='.',
)
plt.plot((-1, 45), (-1, 45), color='k', alpha=0.3)
plt.xlabel('Pitch angle from sub-sample')
plt.ylabel('Pitch angle from validation subset')
plt.xlim(-0.1, 45)
plt.ylim(-0.1, 45)
plt.savefig('duplicates_plots/pa_comparison2.pdf', bbox_inches='tight')
