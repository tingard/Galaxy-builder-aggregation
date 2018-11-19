import os
import json
import numpy as np
import pandas as pd

with open('subject-id-list.csv', 'r') as f:
    subjectIds = np.array([int(n) for n in f.read().split('\n')])
# np.random.shuffle(subjectIds)
# we will write the notebooks out to here
outputFolder = 'output-notebooks'
if not os.path.exists(outputFolder):
    os.mkdir(outputFolder)

df_nsa = pd.read_pickle('NSA_filtered.pkl')

classifications = pd.read_csv('../classifications/galaxy-builder-classifications_24-7-18.csv')
subjects = pd.read_csv('../classifications/galaxy-builder-subjects_24-7-18.csv')
null = None
true = True
false = False

c = []
for id in subjectIds:
    subjectId = id

    # Grab the metadata of the subject we are working on
    meta = eval(subjects[subjects['subject_id'] == subjectId].iloc[0]['metadata'])

    # And the NSA data for the galaxy (if it's a galaxy with NSA data,
    # otherwise throw an error)
    try:
        gal = df_nsa[df_nsa['NSAID'] == int(meta['NSA id'])]
    except KeyError:
        gal = {}
        raise KeyError('Metadata does not contain valid NSA id (probably an older galaxy)')
    c.append([gal['RA'].iloc[0], gal['DEC'].iloc[0]])

np.savetxt('subject-ra-decs.csv', np.array(c), fmt='%10.5f', delimiter=',')
