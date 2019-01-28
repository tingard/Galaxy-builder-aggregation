import numpy as np
import matplotlib.pyplot as plt
import os
import json
import lib.galaxy_utilities as gu
from sklearn.metrics import r2_score
from scipy.stats import pearsonr


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
    plt.savefig('method-paper-plots/GZBvsNSA_ax-ratio_boxplot.png', bbox_inches='tight')


def compare_clustered_disk():
    keys = ('PETRO_BA90', 'PETRO_BA50', 'SERSIC_BA')
    ax_labels = (
        'Axis ratio from Stokes parameters at 90% light radius',
        'Axis ratio from Stokes parameters at 50% light radius',
        'Axis ratio from 2D Sérsic fit',
    )
    plt.figure()
    for key, ax_label in zip(keys, ax_labels):
        data = []
        for componentFile in os.listdir('./cluster-output'):
            if not '.json' in componentFile:
                continue
            metadata = gu.meta_map.get(int(componentFile.split('.json')[0]), {})
            NSAID = metadata.get('NSA id', False)
            if NSAID is not False:
                galaxy = gu.df_nsa[gu.df_nsa['NSAID'] == int(NSAID)]
                nsa_ax_ratio = galaxy[key].values

                with open('./cluster-output/{}'.format(componentFile)) as f:
                    components = json.load(f)
                try:
                    if components.get('disk', {}).get('rx', False):
                        gzb_ax_ratio = components['disk']['rx'] / components['disk']['ry']
                        gzb_ax_ratio = min(gzb_ax_ratio, 1/gzb_ax_ratio)
                        data.append([nsa_ax_ratio, gzb_ax_ratio])
                except AttributeError:
                    print(componentFile, 'contained no disk')
        data = np.array(data).astype(float)
        np.save('tst.npy', data)
        p_rho, p_p = pearsonr(data.T[0], data.T[1])
        print('Probability samples are correlated: {:.8f}'.format(1-p_p))
        plt.scatter(*data.T)
        plt.plot([0, 1], [0, 1], 'k', alpha=0.4)
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.xlabel(ax_label)
        plt.ylabel('Axis ratio from Galaxy builder disk component')
        plt.title('Pearson correlation coefficient: {:.4f}'.format(p_rho))
        plt.savefig('method-paper-plots/GZBvsNSA_ax-ratio_{}.pdf'.format(key), bbox_inches='tight')
        plt.savefig('method-paper-plots/GZBvsNSA_ax-ratio_{}.png'.format(key), bbox_inches='tight')
        plt.clf()


if __name__ == "__main__":
    # compare_all_classifications()
    compare_clustered_disk()
