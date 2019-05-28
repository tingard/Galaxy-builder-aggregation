import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.linear_model import BayesianRidge
from gzbuilderspirals import r_theta_from_xy, fitting
from gzbuilderspirals.galaxySpirals import GalaxySpirals
import lib.galaxy_utilities as gu


clf_kwargs = {}
clf_kwargs.setdefault('alpha_1', fitting.alpha_1_prior)
clf_kwargs.setdefault('alpha_2', fitting.alpha_2_prior)

clf = BayesianRidge(
    compute_score=True,
    fit_intercept=True,
    copy_X=True,
    normalize=True,
    **clf_kwargs
)


# grab r and theta for each arm in the cluster
def return_groups(self):
    drawn_arms_r_theta = [
        r_theta_from_xy(*self.normalise(a).T)
        for a in self.drawn_arms
    ]
    # grab R array
    R = np.fromiter(
        (j for i in drawn_arms_r_theta for j in np.sort(i[0])),
        dtype=float
    )
    # construct theta array
    t = np.array([])
    groups = np.array([])
    # for each arm cluster...
    for i, (r, theta) in enumerate(drawn_arms_r_theta):
        # unwrap the drawn arm
        r_, t_ = fitting.unwrap_and_sort(r, theta)
        # set the minimum theta of each arm to be in [0, 2pi) (not true in
        # general but should pop out as an artefact of the clustering alg)
        while np.min(t_) < 0:
            t_ += 2*np.pi
        # add this arm to the theta array
        t = np.concatenate((t, t_))
        groups = np.concatenate((groups, np.repeat([i], t_.shape[0])))
    # sort the resulting points by radius
    a = np.argsort(R)
    coords = np.stack((R[a], t[a]))
    return coords, groups


def log_spiral_k_fold_test(R_centered, t_centered, groups, n_splits=5):
    clf = BayesianRidge(
        compute_score=True,
        fit_intercept=True,
        copy_X=True,
        normalize=True,
        **clf_kwargs
    )
    scores = np.zeros(n_splits)
    params = []
    gkf = GroupKFold(n_splits=n_splits)
    for i, (train, test) in enumerate(gkf.split(Rp_centered, t_centered, groups=groups)):
        R_train = R[train]
        t_train = t[train]
        R_test = R[test]
        t_test = t[test]

        X = np.vander(R_train, 2)
        X_test = np.vander(R_test, 2)
        clf.fit(X[:, :-1], t_train, sample_weight=point_weights[train])
        s = clf.score(
            X_test[:, :-1],
            t_test,
            sample_weight=point_weights[test]
        )
        scores[i] = s
    return score, params


def poly_spiral_k_fold_test(R_centered, t_centered, groups, degree=3,
                            n_splits=5):
    clf = BayesianRidge(
        compute_score=True,
        fit_intercept=True,
        copy_X=True,
        normalize=True,
        **clf_kwargs
    )
    from sklearn.pipeline import make_pipeline
    score = 0
    params = []
    gkf = GroupKFold(n_splits=n_splits)
    for train, test in gkf.split(Rp_centered, t_centered, groups=groups):
        R_train = R[train]
        t_train = t[train]
        R_test = R[test]
        t_test = t[test]

        X = np.vander(R_train, degree)
        X_test = np.vander(R_test, degree)
        clf.fit(X[:, :-1], t_train, sample_weight=point_weights[train])
        s = clf.score(
            X_test[:, :-1],
            t_test,
            sample_weight=point_weights[test]
        )
        params.append(clf.coef_)
        score += s / n_splits
    return score, params


if __name__ == '__main__':
    chosenId = 21097008
    # chosenId = 21686558
    gal, angle = gu.get_galaxy_and_angle(chosenId)
    pic_array, deprojected_image = gu.get_image(
        gal, chosenId, angle
    )

    drawn_arms = gu.get_drawn_arms(chosenId, gu.classifications)

    galaxy_object = GalaxySpirals(
        drawn_arms,
        ba=gal['SERSIC_BA'].iloc[0],
        phi=-angle
    )
    try:
        distances
    except NameError:
        distances = galaxy_object.calculate_distances()

    db = galaxy_object.cluster_lines(distances)

    dpj_arms = galaxy_object.deproject_arms()

    coords, groups = return_groups(dpj_arms[0])

    R = coords[0]
    t = coords[1]

    t_centered = t - t.mean()

    R_centered = R - R.mean()
    Rp = np.log(R)
    R p_centered = Rp - Rp.mean()

    point_weights = dpj_arms[0].get_sample_weight(R)
    plt.plot(R_centered, point_weights)
    log_test = log_spiral_k_fold_test(Rp_centered, t_centered, groups))
    poly_test = poly_spiral_k_fold_test(R_centered, t_centered, groups, degree=9))

    plt.figure()
    for coef in
