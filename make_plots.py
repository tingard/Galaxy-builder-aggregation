import numpy as np
import os
import shutil
import lib.galaxy_utilities as gu
from gzbuilderspirals import plotting, metric
from gzbuilderspirals import cleaning

id_list = np.loadtxt('lib/subject-id-list.csv', dtype=int)

for chosen_id in id_list:
    print('Working on', chosen_id)
    if not os.path.isdir('pipeline_plots2/{}'.format(chosen_id)):
        os.mkdir('pipeline_plots2/{}'.format(chosen_id))
    gal, angle = gu.get_galaxy_and_angle(chosen_id)
    pic_array, deprojected_image = gu.get_image(
        gal, chosen_id, angle
    )
    drawn_arms = gu.get_drawn_arms(chosen_id, gu.classifications)

    if os.path.exists('./lib/distances/subject-{}.npy'.format(chosen_id)):
        distances = np.load('./lib/distances/subject-{}.npy'.format(chosen_id))
    else:
        print('\t- Calculating distances')
        distances = metric.calculate_distance_matrix(drawn_arms)
        np.save('./lib/distances/subject-{}.npy'.format(chosen_id), distances)

    coords, groups_all = cleaning.get_grouped_data(drawn_arms)
    out = plotting.make_pipeline_plots(
        drawn_arms,
        image_arr=pic_array,
        deprojected_array=deprojected_image,
        image_size=pic_array.shape[0],
        phi=angle,
        ba=float(gal['SERSIC_BA']),
        distances=distances,
        # clean_points=True,
        file_loc='pipeline_plots2/{}'.format(chosen_id)
    )

    plotting.combine_plots(
        'pipeline_plots2/{}'.format(chosen_id),
        title='Subject ID: {}'.format(chosen_id)
    )
    try:
        shutil.copy2(
            'pipeline_plots2/{}/combined.png'.format(chosen_id),
            'model-comparison-plots2/{}.png'.format(chosen_id)
        )
    except FileNotFoundError:
        pass
