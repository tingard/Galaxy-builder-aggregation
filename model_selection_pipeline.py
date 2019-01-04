from importlib import reload
import os
import pandas as pd
import numpy as np
import gzbuilderspirals.pipeline as gzbp
import gzbuilderspirals.metric as gzbm
import gzbuilderspirals as gzb
import lib.galaxy_utilities as gu

gzb = reload(gzb)
gzbp = reload(gzbp)
gzbm = reload(gzbm)

id_list = np.loadtxt('lib/subject-id-list.csv', dtype=int)

model_selection_pipeline = gzbp.model_selection_pipeline
calculate_distance_matrix = gzbm.calculate_distance_matrix
gu = reload(gu)

fit_table = []
for chosenId in id_list:
    gal, angle = gu.get_galaxy_and_angle(chosenId)

    drawn_arms = gu.get_drawn_arms(chosenId, gu.classifications)

    if os.path.exists('./lib/distances/subject-{}.npy'.format(chosenId)):
        distances = np.load('./lib/distances/subject-{}.npy'.format(chosenId))
    else:
        print('\t- Calculating distances')
        distances = calculate_distance_matrix(drawn_arms)
        np.save('./lib/distances/subject-{}.npy'.format(chosenId), distances)

    out = model_selection_pipeline(
        drawn_arms, phi=angle,
        ba=float(gal['SERSIC_BA']),
        distances=distances, verbose=False
    )

    for i, arm in enumerate(out):
        scores = [v.mean() for k, v in arm.items()]
        row = [chosenId, i, list(arm.keys())[np.argmax(scores)], *scores]
        fit_table.append(row)

d = {0: 'subject_id', 1: 'arm_index', 2: 'best_model',
     3: 'log_spiral_mean_score'}
for degree in range(4, len(fit_table[0])):
    d[degree] = 'poly_spiral_{}_mean_score'.format(degree - 3)
print(d)
df = pd.DataFrame(fit_table)
df = df.rename(columns=d)
print(df)
df.to_csv('fit-table.csv')
