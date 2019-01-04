import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import json
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
    plt.savefig('GZBvsNSA_ax-ratio_boxplot', bbox_inches='tight')


def compare_clustered_disk():
    data = []
    for componentFile in os.listdir('./cluster-output'):
        if not '.json' in componentFile:
            continue
        metadata = gu.meta_map.get(int(componentFile.split('.json')[0]), {})
        NSAID = metadata.get('NSA id', False)
        if NSAID is not False:
            galaxy = gu.df_nsa[gu.df_nsa['NSAID'] == int(NSAID)]
            nsa_ax_ratio = galaxy['PETRO_BA90']

            with open('./cluster-output/{}'.format(componentFile)) as f:
                components = json.load(f)
            if components.get('disk').get('rx', False):
                gzb_ax_ratio = components['disk']['rx'] / components['disk']['ry']
                gzb_ax_ratio = min(gzb_ax_ratio, 1/gzb_ax_ratio)
                data.append([nsa_ax_ratio, gzb_ax_ratio])
    data = np.array(data)
    print(data.shape)
    plt.scatter(*data.T)
    plt.plot([0, 1], [0, 1], 'k', alpha=0.5)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel('Axis ratio from Stokes parameters at 90% light radius')
    plt.ylabel('Axis ratio from Galaxy builder disk component')
    plt.savefig('GZBvsNSA_ax-ratio_P90', bbox_inches='tight')

if __name__ == "__main__":
    # compare_all_classifications()
    compare_clustered_disk()
