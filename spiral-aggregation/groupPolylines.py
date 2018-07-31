import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.neighbors import LocalOutlierFactor

np.random.seed(299792458)


# Consider the line extending the segment, parameterized as v + t (w - v).
# We find projection of point p onto the line.
# It falls where t = [(p-v) . (w-v)] / |w-v|^2
# We clamp t from [0,1] to handle points outside the segment vw.

# calculate dot(a) of a(n,2), b(n,2): np.add.reduce(b1 * b2, axis=1)
# calucalte norm(a) of a(n,2), b(n,2): np.add.reduce((a-b)**2, axis=1)
def calcT(a):
    b1 = a[:, 0, :] - a[:, 1, :]
    b2 = a[:, 2, :] - a[:, 1, :]
    dots = np.add.reduce(b1 * b2, axis=1)
    l2 = np.add.reduce((a[:, 1] - a[:, 2])**2, axis=1)
    out = np.clip(dots / l2, 0, 1)
    return out


def getDiff(t, a):
    projection = a[:, 1, :] + np.repeat(
        t.reshape(-1, 1), 2, axis=1
    ) * (a[:, 2, :] - a[:, 1, :])
    out = np.add.reduce(
        (a[:, 0, :] - projection) * (a[:, 0, :] - projection),
        axis=1
    )
    return np.sqrt(np.min(out))


vCalcT = np.vectorize(calcT, signature='(a,b,c)->(a)')
vGetDiff = np.vectorize(getDiff, signature='(a),(a,b,c)->()')


def minimum_distance(a, b):
    m = np.zeros((a.shape[0], b.shape[0] - 1, 3, 2))
    m[:, :, 0, :] = np.transpose(
        np.tile(a, [m.shape[1] + 1, 1, 1]), axes=[1, 0, 2]
    )[:, :-1, :]
    m[:, :, 1, :] = np.tile(b, [a.shape[0], 1, 1])[:, :-1, :]
    m[:, :, 2, :] = np.tile(
        np.roll(b, -1, axis=0), [a.shape[0], 1, 1]
    )[:, :-1, :]
    t = vCalcT(np.array(m))
    return np.sum(vGetDiff(t, m)) / a.shape[0]


def arcDistanceFast(a, b):
    return (
        minimum_distance(a, b) +
        minimum_distance(b, a)
    )


def calculateDistanceMatrix(polyLines):
    distances = np.zeros((len(polyLines), len(polyLines)))
    for i in range(len(polyLines)):
        for j in range(i + 1, len(polyLines)):
            distances[i, j] = arcDistanceFast(polyLines[i], polyLines[j])
    distances += np.transpose(distances)
    return distances


def clusterPolyLines(polyLines):
    print('1 of 2: Calculating distance matrix...')
    distances = calculateDistanceMatrix(polyLines)
    print('2 of 2: Running DBSCAN')
    # initialise fitter and fit!
    db = DBSCAN(
        eps=20, min_samples=3, metric='precomputed',
        n_jobs=-1, algorithm='brute'
    )
    db.fit(distances)

    # Obtain clustering results
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(db.labels_)) - (1 if -1 in db.labels_ else 0)

    print('Estimated number of clusters: %d' % n_clusters_)
    return db


def identifyOutliers(cloud, returnClf=False):
    clf = LocalOutlierFactor(n_neighbors=50)
    y_pred = clf.fit_predict(cloud)
    mask = ((y_pred + 1) / 2).astype(bool)
    cleanedCloud = cloud[mask]
    if returnClf:
        return clf
    return cleanedCloud, mask, clf
