# we want to create a file "location-map.json" which links a subject set with its original files

import re
import os
import json
import numpy as np
import galaxy_utilities as gu

def grab_data_from_file(fname):
    with open(fname) as meta_file:
        meta = json.load(meta_file)
    ks = ('SDSS dr7 id', 'ra', 'dec', 'NSA id')
    return {k: meta[k] for k in ks}


true = True
false = False
null = None
metadata = gu.subjects['metadata'].apply(json.loads).values
meta_map = {i: j for i, j in zip(gu.subjects['subject_id'].values, metadata)}

subject_ids = np.unique(gu.classifications['subject_ids'].values)

# grab path to all subject json model files
subject_set_path = os.path.expanduser(
    '~/PhD/galaxy-builder/subjectUpload/subject_set_{}'
)
subject_set_difference_files = [
    subject_set_path.format(j) + '/' + i
    for j in range(3)
    for i in os.listdir(subject_set_path.format(j))
    if re.match(r'difference_subject[0-9]+\.json$', i)
]
subject_set_metadata_files = [
    subject_set_path.format(j) + '/' + i
    for j in range(3)
    for i in os.listdir(subject_set_path.format(j))
    if re.match(r'metadata_subject[0-9]+\.json$', i)
]

mwa_set_path = os.path.expanduser(
    '~/PhD/galaxy-builder/subjectUpload/mwa_subject_set/'
)
mwa_set_difference_files = [
    os.path.join(mwa_set_path, i)
    for i in os.listdir(mwa_set_path)
    if re.match(r'metadata_subject[0-9]+\.json$', i)
]
mwa_set_metadata_files = [
    os.path.join(mwa_set_path, i)
    for i in os.listdir(mwa_set_path)
    if re.match(r'metadata_subject[0-9]+\.json$', i)
]

regex = 'subject_set_([0-9]+)/metadata_subject([0-9]+)'
mwa_reges = 'mwa_subject_set/metadata_subject([0-9]+)'
location_keys = [
    re.search(regex, k).groups()
    for k in subject_set_metadata_files
]

subject_set_details = {
    location_keys[i]: grab_data_from_file(k)
    for i, k in enumerate(subject_set_metadata_files)
}
ras = np.array([v.get('ra', None) for v in subject_set_details.values()])
decs = np.array([v.get('dec', None) for v in subject_set_details.values()])
dr7ids = np.array([
    v.get('SDSS dr7 id', None)
    for v in subject_set_details.values()
])

out = {}

for sid, md in meta_map.items():
    try:
        ra = md['ra']
        dec = md['dec']
        dr7objid = md['SDSS dr7 id']
        location = location_keys[
            np.where((ras == ra) & (decs == dec) & (dr7ids == dr7objid))[0][0]
        ]
        out[int(sid)] = location
    except KeyError:
        pass

print(out)
with open('location-map.json', 'w') as f_out:
    json.dump(out, f_out, indent=1)
