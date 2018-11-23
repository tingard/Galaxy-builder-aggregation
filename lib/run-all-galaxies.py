import os
import sys
import papermill as pm
import argparse


def getPath(s):
    return '{}/{}'.format(
        os.path.abspath(os.path.dirname(__file__)),
        s
    )

parser = argparse.ArgumentParser(
    description='Take a notebook with a subject_id parametrisation ' \
    + 'and run it on all available subject ids'
)
parser.add_argument('input_notebook', type=str,
                    help='The input notebook to run')

parser.add_argument('output_folder', type=str,
                    help='The folder to output all run notebooks to')

parser.add_argument('-o', '--overwrite', action='store_true',
                    help='Overwrite the contents of the folder')

args = parser.parse_args()
if not os.path.isfile(args.input_notebook):
    raise OSError('Invalid input notebook')

if os.path.isdir(args.output_folder) and not args.overwrite:
    raise OSError('Output folder exists, use -o to overwrite')

input_notebook = args.input_notebook
output_folder = args.output_folder

# load a list of all the subject IDs we should run on
with open(getPath('subject-id-list.csv'), 'r') as f:
    subject_ids = list(map(int, f.read().split('\n')))

# we will write the notebooks out to here
if not os.path.exists(output_folder):
    os.mkdir(output_folder)

# iterate over all subject IDs, and run the notebooks!
for id in subject_ids:
    print('Running subject', id)
    # print('papermill ./spiral-extraction.ipynb {} -p subjectId {}'.format(
    #     './{}/subject-{}.ipynb'.format(output_folder, id),
    #     id
    # ))
    pm.execute_notebook(
        input_notebook,
        './{}/subject_id-{}.ipynb'.format(output_folder, id),
        parameters=dict(subject_id=id)
    )
