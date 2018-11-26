"""summary.ipynb"""
import papermill as pm
import os
import matplotlib.pyplot as plt

def get_path(s):
    try:
        return '{}/{}'.format(
            os.path.abspath(os.path.dirname(__file__)),
            s
        )
    except NameError:
        return s

nbs = pm.read_notebooks(get_path('cluster-output'))

notebook_files = [i for i in os.listdir(get_path('cluster-ls output')) if '.ipynb' in i]
for f in notebook_files[:1]:
    nbs.display_output(f, 'clustered_components')
    plt.savefig('clustered-components-images/{}'.format(f.split('.')[0]))
    plt.close()
