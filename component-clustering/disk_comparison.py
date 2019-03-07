import os
import json
import re
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import lib.galaxy_utilities as gu


def compare_all_classifications():
    with open('tmp_cls_dump.json') as f:
        classifications = json.load(f)
    bas = []
    gzb_ax_ratios = {}
    nsa_ax_ratios = []
    for subject_id in np.loadtxt('lib/subject-id-list.csv', dtype='u8'):
        classifications_for_subject = [
            c for c in classifications
            if c['links']['subjects'][0] == str(subject_id)
        ]
        annotations_for_subject = [i['annotations'] for i in classifications_for_subject]
        drawn_disks = [
            a[0]['value'][0]['value'][0]
            for a in annotations_for_subject
            if len(a) == 4 and len(a[0]['value'][0]['value']) > 0
        ]

        # get ax ratio for each disk drawn for this galaxy
        ax_ratios = [min(j, 1/j) for j in (i['rx'] / i['ry'] for i in drawn_disks)]
        ax_ratios = [i for i in ax_ratios if i != 0.5]
        gzb_ax_ratios[subject_id] = ax_ratios

        metadata = gu.meta_map.get(int(subject_id), {})
        NSAID = metadata.get('NSA id', False)
        if NSAID is not False:
            galaxy = gu.df_nsa[gu.df_nsa['NSAID'] == int(NSAID)]
            nsa_ax_ratio = galaxy['PETRO_BA90']

            nsa_ax_ratios.append([subject_id, nsa_ax_ratio])
    nsa_ax_ratios = np.array(nsa_ax_ratios)

    sid_by_ax = nsa_ax_ratios[:, 0][np.argsort(nsa_ax_ratios[:, 1])]

    plt.figure(figsize=(20, 10))
    plt.boxplot([gzb_ax_ratios[i] for i in sid_by_ax])
    x = range(1, len(sid_by_ax) + 1)
    plt.plot(x, np.sort(nsa_ax_ratios[:, 1]), 'x')
    plt.xlabel('Zooniverse Subject ID')
    plt.ylabel('Axis ratio')
    plt.xticks(x, list(map(int, sid_by_ax)), rotation=90)
    plt.savefig('method-paper-plots/GZBvsNSA_ax-ratio_boxplot.pdf', bbox_inches='tight')


def get_gzb_axis_ratios():
    sid_list = sorted(np.loadtxt('lib/subject-id-list.csv', dtype='u8'))
    axr = []
    for sid in sid_list:
        file_loc = os.path.join('cluster-output', '{}.json'.format(sid))
        if os.path.exists(file_loc):
            with open(file_loc) as f:
                components = json.load(f)
        else:
            components = {}
        disk = components.get('disk', {})
        axRatio = disk.get('axRatio', np.nan) if disk is not None else np.nan
        if axRatio > 0.95:
            print(sid);
        axr.append(axRatio)
    return pd.Series(data=axr, index=sid_list, name='GZB disk axis ratio')


def get_nsa_key(key, name='NSA catalog value'):
    sid_list = sorted(np.loadtxt('lib/subject-id-list.csv', dtype='u8'))
    axr = []
    for sid in sid_list:
        metadata = gu.meta_map.get(sid, {})
        NSAID = metadata.get('NSA id', False)
        if NSAID is not False:
            galaxy = gu.df_nsa[gu.df_nsa['NSAID'] == int(NSAID)]
            nsa_val = galaxy[key].values[0]
        else:
            nsa_val = np.nan
        axr.append(nsa_val)
    return pd.Series(data=axr, index=sid_list, name=name)


def compare_clustered_disk():
    keys = ('PETRO_BA90', 'PETRO_BA50', 'SERSIC_BA')
    ax_labels = (
        'Axis ratio from Stokes parameters at 90% light radius',
        'Axis ratio from Stokes parameters at 50% light radius',
        'Axis ratio from 2D Sérsic fit',
    )
    plt.figure()
    gzb_ax_ratios = get_gzb_axis_ratios()
    for key, ax_label in zip(keys, ax_labels):
        print(key)
        nsa_data = get_nsa_key(key, name='NSA {}'.format(key))
        df = pd.concat((nsa_data, gzb_ax_ratios), axis=1).dropna()
        print(df.values.T.shape)
        p_rho, p_p = pearsonr(*df.values.T)
        # print(pd.concat((nsa_data, gzb_ax_ratios), axis=1))
        print('Coefficient: {:.4f}'.format(p_rho))
        print('Probability samples are correlated: {:.8f}'.format(1-p_p))
        plt.scatter(nsa_data, gzb_ax_ratios)
        plt.plot([0, 1], [0, 1], 'k', alpha=0.4)
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.xlabel(ax_label)
        plt.ylabel('Axis ratio from Galaxy builder disk component')
        plt.savefig('method-paper-plots/GZBvsNSA_ax-ratio_{}.pdf'.format(key), bbox_inches='tight')
        plt.clf()


if __name__ == "__main__":
    # compare_all_classifications()
    compare_clustered_disk()
