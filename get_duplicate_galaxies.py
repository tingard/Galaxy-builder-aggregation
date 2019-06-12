import os
from getpass import getpass
import pandas as pd
import numpy as np
import lib.galaxy_utilities as gu
from panoptes_client import Panoptes, Project, Subject

def find_duplicates():
    Panoptes.connect(username='tingard', password=getpass())

    gzb_project = Project.find(slug='tingard/galaxy-builder')

    subject_sets = []
    for set in gzb_project.links.subject_sets:
        subject_sets.append(list(set.subjects))

    subjects = [j for i in subject_sets for j in i]

    subject_set_ids = [[np.int64(j.id) for j in i] for i in subject_sets]
    ids = [int(i.id) for i in subjects]
    dr7objids = [np.int64(i.metadata.get('SDSS dr7 id', False)) for i in subjects]

    pairings = sorted(zip(ids, dr7objids), key=lambda i: i[0])
    df = pd.DataFrame(pairings, columns=('subject_id', 'dr7objid'))
    df = df[df['dr7objid'] != 0].groupby('subject_id').max()
    n_sids = len(df)
    n_dr7ids = len(df.groupby('dr7objid'))
    print('{} unique subject ids'.format(n_sids))
    print('{} unique dr7 object ids'.format(n_dr7ids))
    print('{} duplicate galaxies'.format(n_sids - n_dr7ids))


    groups = np.array([np.concatenate(([i[0]], i[1].index.values)) for i in df.groupby('dr7objid') if len(i[1]) > 1])
    # okay, what subject sets are our duplicates?
    s1 = gzb_project.links.subject_sets[
        np.argmax([np.all(np.isin(subject_set_ids[i], groups[:, 1])) for i in range(len(subject_set_ids))])
    ]
    s2 = gzb_project.links.subject_sets[
        np.argmax([np.all(np.isin(subject_set_ids[i], groups[:, 2])) for i in range(len(subject_set_ids))])
    ]
    print(s1, s2)
    return groups

groups = find_duplicates()
np.save('lib/duplicate_galaxies.npy', groups)
np.savetxt('lib/duplicate_galaxies.csv', groups, delimiter=',')
