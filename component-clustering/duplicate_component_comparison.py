import os
import sys
import json
import time
import numpy as np
import pandas as pd
from scipy.stats import ttest_ind
import matplotlib.pyplot as plt
import lib.galaxy_utilities as gu
# from make_pitch_angle_dataframes import get_gal_pa
from gzbuilderspirals.oo import Pipeline


def get_model_string(v):
    return '+'.join(k for k in ('disk', 'bulge', 'bar') if v[k] is not None)


def get_ba(comp):
    ratio = comp['rx'] / comp['ry']
    return min(ratio, 1/ratio)


def get_bar_ba(comp):
    ratio = comp['width'] / comp['height']
    return min(ratio, 1/ratio)


def pa_from_arms(arms):
    pa = np.zeros(len(arms))
    sigma_pa = np.zeros(pa.shape)
    length = np.zeros(pa.shape)
    for i, arm in enumerate(arms):
        pa[i] = arm.pa
        length[i] = arm.length
        sigma_pa[i] = arm.sigma_pa
    if len(arms) == 0:
        return np.nan, np.nan
    combined_pa = (pa * length).sum() / length.sum()
    combined_sigma_pa = np.sqrt((length**2 * sigma_pa**2).sum()) / length.sum()
    return combined_pa, combined_sigma_pa


dr8ids, ss_ids, validation_ids = np.load('lib/duplicate_galaxies.npy').T

if True:
    df = pd.read_pickle('duplicate-components.pkl')
else:
    df = []
    old_out = sys.stdout
    with open('/dev/null', 'w') as dev_null:
        sys.stdout = dev_null
        for i in range(len(dr8ids)):
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
            res['original_model_string'] = get_model_string(original_components)
            res['validation_model_string'] = get_model_string(validation_components)

            both_have_disks = (original_components['disk'] is not None
                               and validation_components['disk'] is not None)
            both_have_bulges = (original_components['bulge'] is not None
                                and validation_components['bulge'] is not None)
            both_have_bars = (original_components['bar'] is not None
                              and validation_components['bar'] is not None)
            if both_have_disks:
                o_disk = original_components['disk']
                v_disk = validation_components['disk']
                res['original_disk_ba'] = get_ba(o_disk)
                res['validation_disk_ba'] = get_ba(v_disk)
                res['original_disk_reff'] = max(o_disk['rx'], o_disk['ry'])
                res['validation_disk_reff'] = max(v_disk['rx'], v_disk['ry'])
            if both_have_bulges:
                o_bulge = original_components['bulge']
                v_bulge = validation_components['bulge']
                res['original_bulge_ba'] = get_ba(o_bulge)
                res['validation_bulge_ba'] = get_ba(v_bulge)
                res['original_bulge_reff'] = max(o_bulge['rx'], o_bulge['ry'])
                res['validation_bulge_reff'] = max(v_bulge['rx'], v_bulge['ry'])
            if both_have_bars:
                o_bar = original_components['bar']
                v_bar = validation_components['bar']
                res['original_bar_ba'] = get_bar_ba(o_bar)
                res['validation_bar_ba'] = get_bar_ba(v_bar)
                res['original_bar_reff'] = max(o_bar['width'], o_bar['height'])
                res['validation_bar_reff'] = max(v_bar['width'], v_bar['height'])

            original_drawn_arms = gu.get_drawn_arms(original_id, gu.classifications)
            validation_drawn_arms = gu.get_drawn_arms(validation_id, gu.classifications)

            original_distances = gu.get_distances(original_id)
            validation_distances = gu.get_distances(validation_id)

            original_p = Pipeline(original_drawn_arms, phi=angle, ba=gal['PETRO_BA90'],
                                  image_size=512, distances=original_distances,
                                  parallel=True)
            validation_p = Pipeline(validation_drawn_arms, phi=angle, ba=gal['PETRO_BA90'],
                                    image_size=512, distances=validation_distances,
                                    parallel=True)

            original_arms = [original_p.get_arm(i, clean_points=True)
                             for i in range(max(original_p.db.labels_) + 1)]

            validation_arms = [validation_p.get_arm(i, clean_points=True)
                               for i in range(max(validation_p.db.labels_) + 1)]

            original_pa, original_sigma_pa = pa_from_arms(original_arms)
            validation_pa, validation_sigma_pa = pa_from_arms(validation_arms)
            res['original_pa'], res['original_sigma_pa'] = pa_from_arms(original_arms)
            res['validation_pa'], res['validation_sigma_pa'] = pa_from_arms(
                validation_arms
            )
            df.append(res)

    sys.stdout = old_out
    df = pd.DataFrame(df).set_index('original_id')
    df.to_pickle('duplicate-components.pkl')

_, p_disk_ba_same = ttest_ind(df['original_disk_ba'].values, df['validation_disk_ba'].values)
_, p_disk_reff_same = ttest_ind(df['original_disk_reff'].values, df['validation_disk_reff'].values)
_, p_bulge_ba_same = ttest_ind(df['original_bulge_ba'].values, df['validation_bulge_ba'].values)
_, p_bulge_reff_same = ttest_ind(df['original_bulge_reff'].values, df['validation_bulge_reff'].values)
_, p_bar_ba_same = ttest_ind(df['original_bar_ba'].values, df['validation_bar_ba'].values)
_, p_bar_reff_same = ttest_ind(df['original_bar_reff'].values, df['validation_bar_reff'].values)

if True:
    pa_df = pd.read_pickle('combined_duplicates_pitch_angle.pkl')
else:
    print('Calculaing pitch angles for combined classifications')
    pa_df = {'pa': [], 'sigma_pa': []}
    for i in range(len(dr8ids)):
        print(i)
        original_id = ss_ids[i]
        validation_id = validation_ids[i]
        gal, angle = gu.get_galaxy_and_angle(original_id)
        gal_v, angle_v = gu.get_galaxy_and_angle(validation_id)
        original_drawn_arms = gu.get_drawn_arms(original_id, gu.classifications)
        validation_drawn_arms = gu.get_drawn_arms(validation_id, gu.classifications)
        try:
            if len(original_drawn_arms) == 0 or len(validation_drawn_arms) == 0:
                drawn_arms = original_drawn_arms if len(original_drawn_arms) != 0 else validation_drawn_arms
            else:
                drawn_arms = np.array(
                    list(map(
                        np.array,
                        original_drawn_arms.tolist() + validation_drawn_arms.tolist()
                    ))
                )
        except Exception as e:
            print(original_drawn_arms.shape, validation_drawn_arms.shape)
            print([len(i) for i in original_drawn_arms])
            print(e)
            drawn_arms = np.array([])
        print(drawn_arms.shape)
        p = Pipeline(drawn_arms, phi=angle, ba=gal['PETRO_BA90'],
                     image_size=512, distances=None, parallel=True)
        arms = [p.get_arm(i, clean_points=True)
                for i in range(max(p.db.labels_) + 1)]

        pa, sigma_pa = pa_from_arms(arms)
        pa_df['pa'].append(pa)
        pa_df['sigma_pa'].append(sigma_pa)

    pa_df = pd.DataFrame(
        pa_df,
        index=ss_ids,
    )
    pa_df.to_pickle('combined_duplicates_pitch_angle.pkl')

# disk comparison
fig, (ax_ba, ax_reff) = plt.subplots(ncols=2, figsize=(10, 4))
ax_ba.plot(df['original_disk_ba'], df['validation_disk_ba'], '.')
ax_ba.plot((0.4, 1), (0.4, 1), color='k', alpha=0.3)
ax_ba.set_title(
    r'$p_\mathrm{{samples\ same}} = {:.2f}$'.format(p_disk_ba_same)
)
ax_ba.set_xlabel('Disk axis ratio from sub-sample')
ax_ba.set_ylabel('Disk axis ratio from validation subset')
ax_ba.set_xlim(0.4, 1)
ax_ba.set_ylim(0.4, 1)

ax_reff.plot(df['original_disk_reff'], df['validation_disk_reff'], '.')
ax_reff.plot((50, 240), (50, 240), color='k', alpha=0.3)
ax_reff.set_title(
    r'$p_\mathrm{{samples\ same}} = {:.2f}$'.format(p_disk_reff_same)
)
ax_reff.set_xlabel('Disk size from sub-sample')
ax_reff.set_ylabel('Disk size from validation subset')
ax_reff.set_xlim(50, 240)
ax_reff.set_ylim(50, 240)
plt.savefig('duplicates_plots/disk_comparison.pdf', bbox_inches='tight')

# bulge comparion
fig, (ax_ba, ax_reff) = plt.subplots(ncols=2, figsize=(10, 4))
ax_ba.plot(df['original_bulge_ba'], df['validation_bulge_ba'], '.')
ax_ba.plot((0.4, 1), (0.4, 1), color='k', alpha=0.3)
ax_ba.set_title(
    r'$p_\mathrm{{samples\ same}} = {:.2f}$'.format(p_bulge_ba_same)
)
ax_ba.set_xlabel('Bulge axis ratio from sub-sample')
ax_ba.set_ylabel('Bulge axis ratio from validation subset')
ax_ba.set_xlim(0.4, 1)
ax_ba.set_ylim(0.4, 1)

ax_reff.plot(df['original_bulge_reff'], df['validation_bulge_reff'], '.')
ax_reff.plot((0, 100), (0, 100), color='k', alpha=0.3)
ax_reff.set_title(
    r'$p_\mathrm{{samples\ same}} = {:.2f}$'.format(p_bulge_reff_same)
)
ax_reff.set_xlabel('Bulge size from sub-sample')
ax_reff.set_ylabel('Bulge size from validation subset')
ax_reff.set_xlim(0, 100)
ax_reff.set_ylim(0, 100)
plt.savefig('duplicates_plots/bulge_comparison.pdf', bbox_inches='tight')

# bar comparison
fig, (ax_ba, ax_reff) = plt.subplots(ncols=2, figsize=(10, 4))
ax_ba.plot(df['original_bar_ba'], df['validation_bar_ba'], '.')
ax_ba.plot((0.2, 0.6), (0.2, 0.6), color='k', alpha=0.3)
ax_ba.set_title(
    r'$p_\mathrm{{samples\ same}} = {:.2f}$'.format(p_bar_ba_same)
)
ax_ba.set_xlabel('Bar axis ratio from sub-sample')
ax_ba.set_ylabel('Bar axis ratio from validation subset')
ax_ba.set_xlim(0.2, 0.6)
ax_ba.set_ylim(0.2, 0.6)

ax_reff.plot(df['original_bar_reff'], df['validation_bar_reff'], '.')
ax_reff.plot((0, 130), (0, 130), color='k', alpha=0.3)
ax_reff.set_title(
    r'$p_\mathrm{{samples\ same}} = {:.2f}$'.format(p_bar_reff_same)
)
ax_reff.set_xlabel('Bar size from sub-sample')
ax_reff.set_ylabel('Bar size from validation subset')
ax_reff.set_xlim(0, 120)
ax_reff.set_ylim(0, 120)
plt.savefig('duplicates_plots/bar_comparison.pdf', bbox_inches='tight')

pa_df['id'] = ss_ids
pa_df = pa_df.set_index('id')
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
