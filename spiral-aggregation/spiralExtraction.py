import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.neighbors import LocalOutlierFactor
from scipy.interpolate import UnivariateSpline
from shapely.geometry import LineString

np.random.seed(299792458)

null = None
true = True
false = False


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


def getDrawnArms(id, classifications):
    annotationsForSubject = [
        eval(foo) for foo in
        classifications[classifications['subject_ids'] == id]['annotations']
    ]
    try:
        annotationsWithSpiral = [
            c[3]['value'][0]['value']
            for c in annotationsForSubject
            if len(c) > 3 and len(c[3]['value'][0]['value'])
        ]
    except IndexError as e:
        print('{} raised {}'.format(id, e))
        assert False
    spirals = [[a['points'] for a in c] for c in annotationsWithSpiral]
    spiralsWithLengthCut = [
        [[[p['x'], p['y']] for p in a] for a in c]
        for c in spirals if all([len(a) > 5 for a in c])
    ]
    drawnArms = np.array([
        np.array(arm) for classification in spiralsWithLengthCut
        for arm in classification
        if LineString(arm).is_simple
    ])
    return drawnArms


# --------------------------- SECTION: Deprojection ---------------------------
def deprojectArm(phi, ba, arm):
    p = np.deg2rad(phi)
    Xs = (1 / ba) * (arm[:, 0] * np.cos(p) - arm[:, 1] * np.sin(p))
    Ys = 1 * (arm[:, 0] * np.sin(p) + arm[:, 1] * np.cos(p))

    return np.stack((Xs, Ys), axis=1)


# -------------------- SECTION: Polygon distance algorithm --------------------
# function to get distance from a point (y) to a line connected by two vertices
# (p1, p2) from stackoverflow.com/questions/849211/
# Consider the line extending the segment, parameterized as v + t (w - v).
# We find projection of point p onto the line.
# It falls where t = [(p-v) . (w-v)] / |w-v|^2

# calculate dot(a) of a(n,2), b(n,2): np.add.reduce(b1 * b2, axis=1)
# calucalte norm(a) of a(n,2), b(n,2): np.add.reduce((a-b)**2, axis=1)
def calcT(a):
    b1 = a[:, 0, :] - a[:, 1, :]
    b2 = a[:, 2, :] - a[:, 1, :]
    dots = np.add.reduce(b1 * b2, axis=1)
    l2 = np.add.reduce((a[:, 1] - a[:, 2])**2, axis=1)
    return np.clip(dots / l2, 0, 1)


def getDiff(t, a):
    projection = a[:, 1, :] + np.repeat(
        t.reshape(-1, 1), 2, axis=1) * (a[:, 2, :] - a[:, 1, :])
    outsideBounds = np.logical_or(t < 0, t > 1)
    out = np.add.reduce(
        (a[:, 0, :] - projection) * (a[:, 0, :] - projection),
        axis=1
    )
    endPointDistance = np.amin([
        np.add.reduce((a[outsideBounds, 1] - a[outsideBounds, 0])**2, axis=1),
        np.add.reduce((a[outsideBounds, 2] - a[outsideBounds, 0])**2, axis=1)
    ], axis=0)
    # If we have gone beyond endpoints, set distance to be the distance to the
    # end point (rather than to a continuation of the line)
    out[outsideBounds] = endPointDistance
    return np.min(out)
    # Testing to see if penalising long distances more helps improve arm detection
    # return np.sqrt(np.min(out))


vCalcT = np.vectorize(calcT, signature='(a,b,c)->(a)')
vGetDiff = np.vectorize(getDiff, signature='(a),(a,b,c)->()')


def minimum_distance(a, b):
    # construct our tensor (allowing vectorization)
    # m{i, j, k, p}
    # i iterates over each point in a
    # j cycles through each pair of points in b
    # k cycles through (a[i], b[j], b[j+1])
    # p each of which has [x, y]
    m = np.zeros((a.shape[0], b.shape[0] - 1, 3, 2))
    m[:, :, 0, :] = np.transpose(
        np.tile(a, [m.shape[1] + 1, 1, 1]),
        axes=[1, 0, 2]
    )[:, :-1, :]
    m[:, :, 1, :] = np.tile(b, [a.shape[0], 1, 1])[:, :-1, :]
    m[:, :, 2, :] = np.tile(
        np.roll(b, -1, axis=0), [a.shape[0], 1, 1]
    )[:, :-1, :]
    # t[i, j] = ((a[i] - b[j]) . (b[j + 1] - b[j])) / (b[j + 1] - b[j]|**2
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
def findArmyArm(arms, clf, smooth=True):
    i = np.argmax([
        np.sum(clf._decision_function(arm)) / arm.shape[0]
        for arm in arms
    ])
    arm = arms[i]
    if not smooth:
        return arm

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
    s = np.sign(np.cross(paddedB1, paddedB2, axisa=1, axisb=1))[:, 2]
    s[s == 0] = 1.0
    return s


def getDiff2(t, a):
    projection = a[:, 1, :] + np.repeat(
        t.reshape(-1, 1), 2, axis=1) * (a[:, 2, :] - a[:, 1, :])
    outsideBounds = np.logical_or(t < 0, t > 1)
    out = np.add.reduce(
        (a[:, 0, :] - projection) * (a[:, 0, :] - projection),
        axis=1
    )
    endPointDistance = np.amin([
        np.add.reduce((a[outsideBounds, 1] - a[outsideBounds, 0])**2, axis=1),
        np.add.reduce((a[outsideBounds, 2] - a[outsideBounds, 0])**2, axis=1)
    ], axis=0)
    # If we have gone beyond endpoints, set distance to be the distance to the
    # end point (rather than to a continuation of the line)
    out[outsideBounds] = endPointDistance
    return np.sqrt(out)


vGetDiff2 = np.vectorize(getDiff2, signature='(a),(a,b,c)->(a)')
vSign = np.vectorize(sign, signature='(a,b,c)->(a)')


def getDistAlongPolyline(points, polyLine):
    # construct our tensor (allowing vectorization)
    # m{i, j, k, p}
    # i iterates over each point in a
    # j cycles through each pair of points in b
    # k cycles through (a[i], b[j], b[j+1])
    # p represents [x, y]
    m = np.zeros((points.shape[0], polyLine.shape[0] - 1, 3, 2))
    m[:, :, 0, :] = np.transpose(
        np.tile(points, [m.shape[1] + 1, 1, 1]), axes=[1, 0, 2]
    )[:, :-1, :]
    m[:, :, 1, :] = np.tile(polyLine, [points.shape[0], 1, 1])[:, :-1, :]
    m[:, :, 2, :] = np.tile(
        np.roll(polyLine, -1, axis=0), [points.shape[0], 1, 1]
    )[:, :-1, :]

    t = vCalcT(m)
    signs = vSign(m)
    distances = vGetDiff2(t, m)
    minDistIndex = np.argmin(distances, axis=1)
    optimumIndex = np.dstack(
        (np.arange(minDistIndex.shape[0]), minDistIndex)
    )[0]
    return (
        minDistIndex + t[optimumIndex[:, 0], optimumIndex[:, 1]],
        (
            distances[optimumIndex[:, 0], optimumIndex[:, 1]]
            * signs[optimumIndex[:, 0], optimumIndex[:, 1]]
        )
    )


# ------------------------- SECTION: Final spline fit -------------------------
def fitSmoothedSpline(points):
    t = np.linspace(0, 1, points.shape[0])
    Sx = UnivariateSpline(t, points[:, 0], k=5)#, s=0.25)
    Sy = UnivariateSpline(t, points[:, 1], k=5)#, s=0.25)
    return (Sx, Sy)


# ------------------------ SECTION: Complete Algorithm ------------------------
def fit(
    drawnArms, imageSize=512, verbose=True, fullOutput=True, phi=0, ba=1
):
    log('Calculating distance matrix (this can be slow)', verbose)
    functions = []
    clfs = []
    distances = calculateDistanceMatrix(drawnArms)
    log('Clustering arms', verbose)
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
        clfs.append(clf)
        cleanedCloud = pointCloud[mask]
        log('\t[2 / 4] Identifiying most representitive arm', verbose)
        armyArm = findArmyArm(drawnArms[db.labels_ == label], clf)
        log('\t[3 / 4] Sorting points', verbose)
        deviationCloud = np.transpose(
            getDistAlongPolyline(cleanedCloud, armyArm)
        )

        deviationEnvelope = np.abs(deviationCloud[:, 1]) < 30
        startEndMask = np.logical_and(
            deviationCloud[:, 0] > 0,
            deviationCloud[:, 0] < armyArm.shape[0]
        )

        totalMask = np.logical_and(deviationEnvelope, startEndMask)

        pointOrder = np.argsort(deviationCloud[totalMask, 0])

        normalisedPoints = cleanedCloud[totalMask][pointOrder] / imageSize
        normalisedPoints -= 0.5

        log('\t[4 / 4] Fitting Spline', verbose)
        Sx, Sy = fitSmoothedSpline(
            normalisedPoints
        )
        functions.append([Sx, Sy])
    log('done!', verbose)
    if not fullOutput:
        return functions

    returns = {
        'functions': functions,
        'distances': distances,
        'LOF': clfs,
        'labels': db.labels_
    }
    return returns
