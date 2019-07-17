import os
import requests
import json
from io import BytesIO
from PIL import Image
import numpy as np
import galaxy_utilities as gu
from multiprocessing import Pool


def download_files(subject_id, v):
    global bar
    diff_loc = v['0']
    image_loc = v['1']
    model_loc = v['2']
    res = [requests.get(i) for i in (diff_loc, model_loc)]
    if all(i.status_code == 200 for i in res):
        try:
            os.mkdir('subject_data/{}'.format(subject_id))
        except FileExistsError:
            pass
        for name, r in zip(('diff.json', 'model.json'), res):
            with open('subject_data/{}/{}'.format(subject_id, name), 'w') as f:
                d = r.json()
                json.dump(d, f)
    image = Image.open(BytesIO(requests.get(image_loc).content))
    image.save('subject_data/{}/image.png'.format(subject_id))


if __name__ == '__main__':
    subject_ids = np.unique(
        gu.classifications.query('workflow_version == 61.107')['subject_ids']
    )

    locations = gu.subjects.set_index(
        'subject_id'
    ).locations.apply(
        json.loads
    )[subject_ids]
    with Pool(4) as p:
        async_res = p.starmap_async(
            download_files,
            locations.iteritems(),
        )
        async_res.wait()
