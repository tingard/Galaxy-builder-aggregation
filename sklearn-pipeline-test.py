import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.compose import TransformedTargetRegressor
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import BayesianRidge
from sklearn.pipeline import make_pipeline
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


def get_data(chosenId):
    global distances
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
    galaxy_object.cluster_lines(distances)
    dpj_arms = galaxy_object.deproject_arms()

    (R, t), groups = return_groups(dpj_arms[0])
    point_weights = dpj_arms[0].get_sample_weight(R)
    return (R, t), groups, point_weights


def weighted_group_cross_val(pipeline, X, y, cv, groups, weights):
    scores = np.zeros(cv.get_n_splits())
    for i, (train, test) in enumerate(cv.split(X, y, groups=groups)):
        X_train, y_train = X[train], y[train]
        X_test, y_test = X[test], y[test]
        group_weights = weights[train] / weights[train].mean()
        pipeline.fit(X_train, y_train,
                     bayesianridge__sample_weight=group_weights)

        scores[i] = pipeline.score(
            X_test,
            y_test,
            sample_weight=weights[test]
        )
    return scores


def log_spiral_pipeline():
    return make_pipeline(
        StandardScaler(),
        PolynomialFeatures(
            degree=1,
            include_bias=False,
        ),
        TransformedTargetRegressor(
            regressor=BayesianRidge(
                compute_score=True,
                fit_intercept=True,
                copy_X=True,
                normalize=True,
                **clf_kwargs
            ),
            func=np.log,
            inverse_func=np.exp
        )
    )


def spiral_fit_pipeline(degree):
    return make_pipeline(
        StandardScaler(),
        PolynomialFeatures(
            degree=degree,
            include_bias=False,
        ),
        BayesianRidge(
            compute_score=True,
            fit_intercept=True,
            copy_X=True,
            normalize=True,
            **clf_kwargs
        )
    )


if __name__ == '__main__':
    chosenId = 21096790
    (R, t), groups, point_weights = get_data(chosenId)
    plt.plot(R, point_weights)
    plt.xlabel('Radius from centre')
    plt.ylabel('Point weight')

    R = R.reshape(-1, 1)
    Rp = np.log(R)

    logsp_pipeline = spiral_fit_pipeline(1)

    gkf = GroupKFold(n_splits=5)

    s = weighted_group_cross_val(
        logsp_pipeline,
        Rp, t,
        cv=gkf,
        groups=groups,
        weights=point_weights
    )
    print('Log spiral gives\n\tMean: {}\n\tSTD: {}'.format(s.mean(), s.std()))

    for degree in range(3, 9):
        poly_model = spiral_fit_pipeline(degree)
        s = weighted_group_cross_val(
            poly_model,
            R, t,
            cv=gkf,
            groups=groups,
            weights=point_weights
        )
        print('Degree {} gives\n\tMean: {}\n\tSTD: {}'.format(
            degree, s.mean(), s.std()))

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='polar')
    ax.set_xlim(min(t), max(t))
    plt.plot(t, R, '.')

    logsp_pipeline.fit(Rp, t)
    T = logsp_pipeline.predict(Rp)
    plt.plot(T, R, label='Log spiral')

    for degree in range(3, 9):
        poly_model = spiral_fit_pipeline(degree)
        poly_model.fit(R, t)
        T = poly_model.predict(R)
        plt.plot(T, R, label='$k={}$'.format(degree))

    plt.legend()
    plt.show()
