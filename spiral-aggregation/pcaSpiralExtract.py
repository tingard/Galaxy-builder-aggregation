import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from scipy.interpolate import UnivariateSpline
# np.random.seed(299792458)


def circularMean(theta):
    return np.angle(np.sum(np.exp(theta * 1j)))


def rThetaFromXY(x, y, mux=0, muy=0):
    return (
        np.sqrt((x - mux)**2 + (y - muy)**2),
        np.arctan2((y - muy), (x - mux))
    )


def xyFromRTheta(r, theta, mux=0, muy=0):
    return mux + r * np.cos(theta), muy + r * np.sin(theta)


def getVectorField(cloud, mask=None, patchSize=20):
    if mask is None:
        mask = np.ones(cloud.shape[0], dtype=bool)
    pca = PCA(n_components=2)
    keyPoints = cloud[mask]
    vectors = np.zeros((keyPoints.shape[0], 2))
    varianceRatios = np.zeros((keyPoints.shape[0], 2))
    for i, point in enumerate(keyPoints):
        radialMask = np.add.reduce((cloud - point)**2, axis=1) < patchSize**2
        pcaResult = pca.fit(cloud[radialMask])
        vectors[i] = pcaResult.components_[0]
        varianceRatios[i] = pcaResult.explained_variance_ratio_
    return np.hstack((keyPoints, vectors)), varianceRatios


def dropStick(startPoint, vectorField,
              stepSize=1, patchRadius=15,
              includeUpstream=True, deltaPos=np.array([1, 0])):
    # vectorField is a 2D array of [x_pos, y_pos, newAx[0], newAx[1]]
    pos = startPoint.copy()
    deltaPos_ = deltaPos.copy()
    path = [pos.copy()]
    while True:
        radialMask = np.add.reduce(
            (vectorField[:, :2] - pos)**2,
            axis=1
        ) < patchRadius**2

        if(np.any(radialMask)):
            vectorDirectionWrtFlow = np.sign(
                np.add.reduce(vectorField[radialMask, 2:] * deltaPos_, axis=1)
            )
            rs, thetas = rThetaFromXY(
                vectorField[radialMask, 2],
                vectorField[radialMask, 3]
            )
            shouldFlipVector = vectorDirectionWrtFlow < 0
            alignedThetas = (thetas + shouldFlipVector * np.pi) % (2 * np.pi)
            theta = circularMean(alignedThetas)
            deltaPos_ = xyFromRTheta(stepSize, theta)
            pos += deltaPos_
            path.append(pos.copy())
        else:
            break
    if includeUpstream:
        secondPath = dropStick(
            startPoint,
            vectorField,
            stepSize=stepSize,
            patchRadius=patchRadius,
            includeUpstream=False,
            deltaPos=deltaPos * -1
        )
        # define "downstream" as always longer than "upstream"
        pathDirection = 1 if len(path) > len(secondPath) else -1
        return np.vstack((secondPath[:0:-1], path))[::pathDirection]
    return np.array(path)


def dropSticks(startPoints, vectorField,
               stepSize=1, patchRadius=15, includeUpstream=True,
               deltaPos=np.array([1, 0])
               ):
    print('Dropping {} sticks'.format(len(startPoints)))
    paths = []
    for startPoint in startPoints:
        paths.append(
            dropStick(
                startPoint,
                vectorField,
                stepSize,
                patchRadius,
                includeUpstream,
                deltaPos
            )
        )
    return paths


def getStartCluster(drawnArms, cloud, figsize=[512, 512], fullOutput=False):
    coordInCloud = lambda coord: np.any(
        np.logical_and(
            cloud[:, 0] == coord[0],
            cloud[:, 1] == coord[1]
        )
    )
    startPoints = np.array([
        [j for j in i if coordInCloud(j)][0]
        for i in drawnArms if coordInCloud(i[0])
    ])
    endPoints = np.array([i[-1] for i in drawnArms if coordInCloud(i[-1])])
    print(startPoints.shape, endPoints.shape)
    try:
        startEndPoints = np.concatenate((startPoints, endPoints))
    except ValueError as e:
        print(e)
        return False
    print(startEndPoints)

    db_startEndPoints = DBSCAN(eps=50, min_samples=2, n_jobs=-1)
    db_startEndPoints.fit(startEndPoints)

    means = np.array([
        (
            np.add.reduce(startEndPoints[db_startEndPoints.labels_ == l])
            / startEndPoints[db_startEndPoints.labels_ == l].shape[0]
        )
        for l in range(np.max(db_startEndPoints.labels_))
    ])
    if len(means) == 0:
        print('ERROR: no start cluster found')
        return np.array([])
    centralGroupLabel = np.argmin(
        np.add.reduce(
            (means - [figsize[0] / 2, figsize[0] / 2])**2,
            axis=1
        )
    )
    if fullOutput:
        return (
            startEndPoints[db_startEndPoints.labels_ == centralGroupLabel],
            db_startEndPoints.labels_ == centralGroupLabel,
            db_startEndPoints.labels_
        )
    return startEndPoints[db_startEndPoints.labels_ == centralGroupLabel]


def findDownstreamCluster(startEndPoints):
    db_downStreamPoints = DBSCAN(eps=15, min_samples=2, n_jobs=-1)
    db_downStreamPoints.fit(startEndPoints)
    clusterSizes = [
        sum(1 for i in db_downStreamPoints.labels_ == clusterLabel if i)
        for clusterLabel in range(max(db_downStreamPoints.labels_ + 1))
    ]
    if len(clusterSizes) == 0:
        print('ERROR: no downstream cluster found')
        return np.zeros(startEndPoints.shape[0], dtype=bool)
    mask = db_downStreamPoints.labels_ == np.argmax(clusterSizes)
    return mask


def calcT(a):
    b1 = a[:, 0, :] - a[:, 1, :]
    b2 = a[:, 2, :] - a[:, 1, :]
    dots = np.add.reduce(b1 * b2, axis=1)
    l2 = np.add.reduce((a[:, 1] - a[:, 2])**2, axis=1)
    out = np.clip(dots / l2, 0, 1)
    return out


def sign(a):
    b1 = a[:, 0, :] - a[:, 1, :]
    b2 = a[:, 2, :] - a[:, 1, :]
    paddedB1 = np.pad(b1, ((0, 0), (0, 1)), 'constant', constant_values=(0,))
    paddedB2 = np.pad(b2, ((0, 0), (0, 1)), 'constant', constant_values=(0,))
    return np.sign(np.cross(paddedB1, paddedB2, axisa=1, axisb=1))[:, 2]


def getDiff(t, a):
    projection = (
        a[:, 1, :]
        + np.repeat(t.reshape(-1, 1), 2, axis=1) * (a[:, 2, :] - a[:, 1, :])
    )
    out = np.add.reduce(
        (a[:, 0, :] - projection) * (a[:, 0, :] - projection),
        axis=1
    )
    return np.sqrt(out)


vCalcT = np.vectorize(calcT, signature='(a,b,c)->(a)')
vGetDiff = np.vectorize(getDiff, signature='(a),(a,b,c)->(a)')
vSign = np.vectorize(sign, signature='(a,b,c)->(a)')


def sortCloudAlongLine(cloud, line):
    m = np.zeros((cloud.shape[0], line.shape[0] - 1, 3, 2))
    m[:, :, 0, :] = np.transpose(
        np.tile(cloud, [m.shape[1] + 1, 1, 1]), axes=[1, 0, 2]
    )[:, :-1, :]
    m[:, :, 1, :] = np.tile(line, [cloud.shape[0], 1, 1])[:, :-1, :]
    m[:, :, 2, :] = np.tile(
        np.roll(line, -1, axis=0), [cloud.shape[0], 1, 1]
    )[:, :-1, :]
    t = vCalcT(np.array(m))
    signs = vSign(np.array(m))
    distances = vGetDiff(t, m)
    minDistIndex = np.argmin(distances, axis=1)
    asd = np.dstack((np.arange(minDistIndex.shape[0]), minDistIndex))[0]
    return (
        minDistIndex + t[asd[:, 0], asd[:, 1]],
        distances[asd[:, 0], asd[:, 1]] * signs[asd[:, 0], asd[:, 1]]
    )


def fitParametrisedSpline(points):
    t = np.linspace(0, 1, points.shape[0])
    Sx = UnivariateSpline(t, points[:, 0], s=0.25, k=5)
    Sy = UnivariateSpline(t, points[:, 1], s=0.25, k=5)
    return Sx, Sy


if __name__ == '__main__':
    print('This script cannot be run')
