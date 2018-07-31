import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.neighbors import LocalOutlierFactor
from scipy.interpolate import UnivariateSpline


np.random.seed(299792458)


# ------------------------- SECTION: Helper functions -------------------------
def rThetaFromXY(x, y, mux=0, muy=0):
    return (
        np.sqrt((x - mux)**2 + (y - muy)**2),
        np.arctan2((y - muy), (x - mux))
    )


def xyFromRTheta(r, theta, mux=0, muy=0):
    return mux + r * np.cos(theta), muy + r * np.sin(theta)


wrapColor = lambda color, s: '{}{}\033[0m'.format(color, s)
red = lambda s: wrapColor('\033[31m', s)
green = lambda s: wrapColor('\033[32m', s)
yellow = lambda s: wrapColor('\033[33m', s)
blue = lambda s: wrapColor('\033[34m', s)
purple = lambda s: wrapColor('\033[35m', s)


def log(s, flag=True):
    if flag:
        print(s)


# -------------------- SECTION: Polygon distance algorithm --------------------
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
        t.reshape(-1, 1), 2, axis=1) * (a[:, 2, :] - a[:, 1, :])
    out = np.add.reduce(
        (a[:, 0, :] - projection) * (a[:, 0, :] - projection),
        axis=1
    )
    return np.sqrt(np.min(out))


vCalcT = np.vectorize(calcT, signature='(a,b,c)->(a)')
vGetDiff = np.vectorize(getDiff, signature='(a),(a,b,c)->()')


def minimum_distance(a, b):
    # construct our tensor (allowing quick calculation)
    m = np.zeros((a.shape[0], b.shape[0] - 1, 3, 2))
    m[:, :, 0, :] = np.transpose(
        np.tile(a, [m.shape[1] + 1, 1, 1]),
        axes=[1, 0, 2]
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


def calculateDistanceMatrix(cls):
    distances = np.zeros((len(cls), len(cls)))
    for i in range(len(cls)):
        for j in range(i + 1, len(cls)):
            distances[i, j] = arcDistanceFast(cls[i], cls[j])
    distances += np.transpose(distances)
    return distances


# ------------------------- SECTION: DBSCAN clustering ------------------------

def clusterArms(distanceMatrix):
    return DBSCAN(
        eps=20,
        min_samples=3,
        metric='precomputed',
        n_jobs=-1,
        algorithm='brute'
    ).fit(distanceMatrix)


# -------------------------- SECTION: Point cleaning --------------------------
def cleanPoints(pointCloud):
    clf = LocalOutlierFactor(n_neighbors=50)
    y_pred = clf.fit_predict(pointCloud)
    mask = ((y_pred + 1) / 2).astype(bool)
    return clf, mask


# -------------------------- SECTION: Point ordering --------------------------
def findArmyArm(arms, clf):
    i = np.argmax([
        np.sum(clf._decision_function(arm[:10])) / arm.shape[0]
        for arm in arms
    ])
    arm = arms[i]

    t = np.linspace(0, 1, arm.shape[0])

    Sx = UnivariateSpline(t, arm[:, 0], s=512, k=5)
    Sy = UnivariateSpline(t, arm[:, 1], s=512, k=5)

    smoothedArm = np.stack((Sx(t), Sy(t)), axis=1)

    return smoothedArm


def sign(a):
    b1 = a[:, 0, :] - a[:, 1, :]
    b2 = a[:, 2, :] - a[:, 1, :]
    paddedB1 = np.pad(b1, ((0, 0), (0, 1)), 'constant', constant_values=(0,))
    paddedB2 = np.pad(b2, ((0, 0), (0, 1)), 'constant', constant_values=(0,))
    return np.sign(np.cross(paddedB1, paddedB2, axisa=1, axisb=1))[:, 2]


def getDiff2(t, a):
    projection = a[:, 1, :] + np.repeat(
        t.reshape(-1, 1), 2, axis=1) * (a[:, 2, :] - a[:, 1, :])
    out = np.add.reduce(
        (a[:, 0, :] - projection) * (a[:, 0, :] - projection),
        axis=1
    )
    return np.sqrt(out)


vGetDiff2 = np.vectorize(getDiff2, signature='(a),(a,b,c)->(a)')
vSign = np.vectorize(sign, signature='(a,b,c)->(a)')


def getDistAlongPolyline(points, polyLine):
    # construct the tensor
    m = np.zeros((points.shape[0], polyLine.shape[0] - 1, 3, 2))
    m[:, :, 0, :] = np.transpose(
        np.tile(points, [m.shape[1] + 1, 1, 1]), axes=[1, 0, 2]
    )[:, :-1, :]
    m[:, :, 1, :] = np.tile(polyLine, [points.shape[0], 1, 1])[:, :-1, :]
    m[:, :, 2, :] = np.tile(
        np.roll(polyLine, -1, axis=0), [points.shape[0], 1, 1]
    )[:, :-1, :]

    t = vCalcT(np.array(m))
    signs = vSign(np.array(m))
    distances = vGetDiff2(t, m)
    minDistIndex = np.argmin(distances, axis=1)
    asd = np.dstack((np.arange(minDistIndex.shape[0]), minDistIndex))[0]
    return (
        minDistIndex + t[asd[:, 0], asd[:, 1]],
        distances[asd[:, 0], asd[:, 1]] * signs[asd[:, 0], asd[:, 1]]
    )


# ------------------------- SECTION: Final spline fit -------------------------
def fitSmoothedSpline(points, imageSize=512):
    _points = points / imageSize - 0.5
    t = np.linspace(0, 1, points.shape[0])
    Sx = UnivariateSpline(t, _points[:, 0], s=0.25, k=5)
    Sy = UnivariateSpline(t, _points[:, 1], s=0.25, k=5)
    return (Sx, Sy)


# ------------------------ SECTION: Complete Algorithm ------------------------
def fit(drawnArms, imageSize=512, verbose=True):
    log('Calculating distance matrix (this can be slow)', verbose)
    functions = []
    distances = calculateDistanceMatrix(drawnArms)
    log('Clustering arms', verbose)
    db = DBSCAN(
        eps=20,
        min_samples=3,
        metric='precomputed',
        n_jobs=-1,
        algorithm='brute'
    ).fit(distances)

    for label in np.unique(db.labels_):
        if label < 0:
            continue
        log('Working on arm label {}'.format(label), verbose)
        pointCloud = np.array([
            point for arm in drawnArms[db.labels_ == label]
            for point in arm
        ])
        log(
            '\t[1 / 4] Cleaning points ({} total)'.format(pointCloud.shape[0]),
            verbose
        )
        clf, mask = cleanPoints(pointCloud)
        cleanedCloud = pointCloud[mask]
        log('\t[2 / 4] Identifiying most representitive arm', verbose)
        armyArm = findArmyArm(drawnArms[db.labels_ == label], clf)
        log('\t[3 / 4] Sorting points', verbose)
        deviationCloud = np.transpose(
            getDistAlongPolyline(cleanedCloud, armyArm)
        )
        pointOrder = np.argsort(deviationCloud[:, 0])
        log('\t[4 / 4] Fitting Spline', verbose)
        Sx, Sy = fitSmoothedSpline(
            cleanedCloud[pointOrder],
            imageSize=imageSize
        )
        functions.append([Sx, Sy])
    log('done!', verbose)
    return functions
