import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import DBSCAN
from PIL import Image
import spiralExtraction as se
import json

with open('classification-dump.json') as f: classifications = json.load(f)
with open('subject-dump.json') as f: subjects = json.load(f)

np.random.seed(299792458)

# print(json.dumps(list(cls[0]['links']['subjects']), indent=1))
index, foo = next(
    iter(
        filter(
            lambda s: s[1] == '6ee8db8f-2a1e-4d10-bcb9-1e5ec84f8f10',
            (
                (
                    i,
                    list(
                        s['locations'][0].items()
                    )[0][1].split('/')[-1].split('.')[0]
                )
                for i, s in enumerate(subjects)
            )
        )
    )
)
subjectId = subjects[index]['id']
print(subjectId)

cls = list(
    filter(
        lambda c: c['links']['subjects'][0] == subjectId,
        classifications
    )
)

annotations = [c['annotations'] for c in cls]
annotationsWithSpiral = [
    c[3]['value'][0]['value']
    for c in annotations
    if len(c[3]['value'][0]['value'])
]
spirals = [[a['points'] for a in c] for c in annotationsWithSpiral]

spiralsWithLengthCut = [
    [[p['x'], p['y']] for p in a]
    for c in spirals
    for a in c
    if len(a) > 10
]
print([len(a) for a in spiralsWithLengthCut])

plt.figure(figsize=(8, 8))
plt.xticks([])
plt.yticks([])
for index, arm in enumerate(spiralsWithLengthCut):
    plt.plot(
        [i[0] for i in arm], [i[1] for i in arm],
        '.', markersize=2, alpha=0.5
    )
    plt.plot(
        [i[0] for i in arm], [i[1] for i in arm],
        '', linewidth=0.5, alpha=0.5
    )

plt.savefig('../../methodPaper/armClustering/classifications.jpg')
plt.clf()

drawnArms = np.array([
    np.array(arm) for arm in spiralsWithLengthCut
])


imageSize = 512
functions = []
print('Calculating distance matrix between {} arms'.format(drawnArms.shape[0]))
distances = se.calculateDistanceMatrix(drawnArms)

print('Clustering')
db = DBSCAN(
    eps=400,
    min_samples=5,
    metric='precomputed',
    n_jobs=-1,
    algorithm='brute'
).fit(distances)

for label in np.unique(db.labels_):
    if label < 0:
        continue
    flag = True
    for arm in drawnArms[db.labels_ == label]:

        plt.plot(
            arm[:, 0],
            arm[:, 1],
            '.-',
            c='C{}'.format(label),
            markersize=2,
            linewidth=0.5,
            alpha=0.5,
            label='Arms in group {}'.format(label) if flag else None
        )
        flag = False
plt.legend()
plt.savefig('../../methodPaper/armClustering/groupedArms.jpg')
plt.clf()

for label in np.unique(db.labels_):
    if label < 0:
        continue

    pointCloud = np.array([
        point for arm in drawnArms[db.labels_ == label]
        for point in arm
    ])

    clf, mask = se.cleanPoints(pointCloud)
    cleanedCloud = pointCloud[mask]

    if label == 0:
        xx, yy = np.meshgrid(
            np.linspace(
                np.min(pointCloud[:, 0]) - 20,
                np.max(pointCloud[:, 0]) + 20,
                50
            ),
            np.linspace(
                np.min(pointCloud[:, 1]) - 20,
                np.max(pointCloud[:, 1]) + 20,
                50
            )
        )
        Z = clf._decision_function(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        fig, ax = plt.subplots(ncols=1, figsize=(10, 10))
        ax.set_title("Local Outlier Factor (LOF)")
        ct = ax.contourf(xx, yy, Z, cmap=plt.cm.Blues_r)
        ax.plot(
            cleanedCloud[:, 0], cleanedCloud[:, 1],
            'k.', markersize=2, alpha=0.5
        )
        ax.plot(
            pointCloud[np.logical_not(mask), 0],
            pointCloud[np.logical_not(mask), 1],
            'r.', markersize=5, alpha=0.5
        )
        cbar = plt.colorbar(ct)
        plt.savefig('../../methodPaper/armClustering/LOF.jpg')
        plt.clf()

    i = np.argmax([
        np.mean(clf._decision_function(arm))
        for arm in drawnArms[db.labels_ == label]
    ])

    armyArm = drawnArms[db.labels_ == label][i]

    if label == 0:
        fig, ax = plt.subplots(ncols=1, figsize=(12, 10))
        ax.set_title("Local Outlier Factor (LOF)")
        ct = ax.contourf(xx, yy, Z, cmap=plt.cm.Blues_r)
        ax.plot(
            cleanedCloud[:, 0], cleanedCloud[:, 1],
            'k.', markersize=2, alpha=0.5
        )
        ax.plot(
            pointCloud[np.logical_not(mask), 0],
            pointCloud[np.logical_not(mask), 1],
            'r.', markersize=5, alpha=0.5
        )
        cbar = plt.colorbar(ct)
        ax.plot(
            armyArm[:, 0],
            armyArm[:, 1],
            'm',
            label='Most representitive poly-line'
        )
        ax.legend()
        plt.savefig('../../methodPaper/armClustering/armyArm.jpg')
        plt.clf()

    deviationCloud = np.transpose(
        se.getDistAlongPolyline(cleanedCloud, armyArm)
    )

    deviationEnvelope = np.abs(deviationCloud[:, 1]) < 30
    startEndMask = np.logical_and(
        deviationCloud[:, 0] > 0,
        deviationCloud[:, 0] < armyArm.shape[0]
    )

    totalMask = np.logical_and(deviationEnvelope, startEndMask)

    pointOrder = np.argsort(deviationCloud[totalMask, 0])

    Sx, Sy = se.fitSmoothedSpline(
        cleanedCloud[totalMask][pointOrder],
        imageSize=imageSize
    )
    functions.append([Sx, Sy])


pic = Image.open("./images/beta_subject.png")
plt.imshow(np.transpose(np.array(pic)[::-1, ::1]), cmap='gray', origin='lower')

t = np.linspace(0, 1, 5000)

for i, (Sx, Sy) in enumerate(functions):
    for arm in drawnArms[db.labels_ == i]:
        plt.plot(
            arm[:, 0], arm[:, 1],
            '.-', c='C{}'.format(i),
            markersize=1, linewidth=0.4, alpha=0.5
        )
    plt.plot(
        (Sx(t) + 0.5) * 512, (Sy(t) + 0.5) * 512,
        c='C{}'.format(i), linewidth=3
    )
    plt.plot(
        (Sx(t) + 0.5) * 512, (Sy(t) + 0.5) * 512,
        c='w', linewidth=2, alpha=0.5
    )

plt.axis('off')
plt.savefig(
    '../../methodPaper/armClustering/overlaidSplines.jpg',
    bbox_inches='tight'
)
