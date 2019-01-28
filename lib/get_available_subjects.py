import os
import sys
import numpy as np
import pandas as pd

if len(sys.argv) < 3:
    print()
    print('Please specify data export to use, and an output file to write to')
    print()
    sys.exit(0)

if not os.path.isfile(sys.argv[1]):
    print('Could not find specified input file, please try again')

in_file = os.path.abspath(sys.argv[1])
out_file = os.path.abspath(sys.argv[2])

all_classifications = pd.read_csv(in_file)
cls_count_threshold_mask = (
    all_classifications.groupby('subject_ids')['annotations'].count() >= 30
)[all_classifications['subject_ids']].values

right_version_mask = all_classifications['workflow_version'] == 61.107

finished_galaxies = all_classifications[
    cls_count_threshold_mask & right_version_mask
]
subject_ids = finished_galaxies['subject_ids'].unique().astype(int)

print('Identified {} galaxies'.format(len(subject_ids)))

ftype = out_file.split('.')[-1]

if ftype == 'npy':
    np.save(out_file, subject_ids)
elif ftype == 'pkl':
    pd.DataFrame(subject_ids, columns=('subject_id',))\
        .set_index('subject_id')\
        .to_pickle(out_file)
else:
    pd.DataFrame(subject_ids, columns=('subject_id',))\
        .to_csv(out_file, index=False, header=False)
