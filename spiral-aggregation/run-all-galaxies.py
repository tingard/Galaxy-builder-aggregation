import os
import papermill as pm

# load a list of all the subject IDs we should run on
with open('subject-id-list.csv', 'r') as f:
    subjectIds = list(map(int, f.read().split('\n')))

# we will write the notebooks out to here
outputFolder = 'output-notebooks'
if not os.path.exists(outputFolder):
    os.mkdir(outputFolder)

# iterate over all subject IDs, and run the notebooks!
for id in subjectIds:
    print('Running subject', id)
    print('papermill ./spiral-extraction.ipynb {} -p subjectId {}'.format(
        './{}/subject-{}.ipynb'.format(outputFolder, id),
        id
    ))
    pm.execute_notebook(
        './spiral-extraction.ipynb',
        './{}/subject-{}.ipynb'.format(outputFolder, id),
        parameters=dict(subjectId=id)
    )
