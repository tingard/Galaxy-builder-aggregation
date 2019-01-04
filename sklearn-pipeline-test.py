import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.neighbors import LocalOutlierFactor
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


def return_groups(self):
    drawn_arms_r_theta = [
        r_theta_from_xy(*self.normalise(a).T)
        for a in self.drawn_arms
    ]
    R = np.fromiter((j for i in drawn_arms_r_theta for j in np.sort(i[0])),
                    dtype=float)
    t = np.array([])
    groups = np.array([])
    dt = (np.arange(5) - 2) * 2 * np.pi
    theta_mean = 0
    for i, (r, theta) in enumerate(drawn_arms_r_theta):
        # out of all allowed transformations, which puts the mean of the theta
        # values closest to the mean of rest of the points (using the first arm
        # as a template)?
        t_ = np.unwrap(theta)
        if i == 0:
            theta_mean = np.concatenate((t, t_)).mean()
        j = np.argmin(np.abs(t_.mean() + dt - theta_mean))
        t_ += dt[j]
        t = np.concatenate((t, t_))
        groups = np.concatenate((groups, np.repeat([i], t_.shape[0])))
    # sort the resulting points by radius (not really important)
    # a = np.argsort(R)
    coords = np.stack((R, t))
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
    (R, t), groups = return_groups(dpj_arms[1])
    point_weights = dpj_arms[1].get_sample_weight(R)
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


def get_log_spiral_pipeline():
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


def get_polynomial_pipeline(degree):
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
    chosenId = 21096794
    (R, t), groups, point_weights = get_data(chosenId)
    R_normed, t_normed = R/R.std(), t/t.std()
    alg = LocalOutlierFactor(contamination='auto', n_jobs=-1, n_neighbors=40, novelty=True)
    res = np.ones(R.shape).astype(bool)

    def foo(row):
        X = np.stack((R_normed[row].reshape(-1), t_normed[row])).T
        out = np.ones(R.shape[0], dtype=bool)
        out[row] = alg.fit_predict(X) > 0
        return out

    for group in np.unique(groups):
        print('working on group', group)
        npoints = R[groups == group].shape[0]
        testField = groups != group
        X_train = np.stack((R_normed[testField].reshape(-1), t_normed[testField])).T
        X_test = np.stack((R_normed[~testField].reshape(-1), t_normed[~testField])).T

        alg.fit(X_train)
        res[~testField] = alg.predict(X_test) > 0

    plt.plot(R, t, '.')
    plt.plot(R[res], t[res], '.', markersize=4)



    R = R.reshape(-1, 1)
    plt.plot(R, t, '.')
    plt.title(r'Input data $r$, $\theta$')
    plt.xlabel('Radius from center')
    plt.ylabel(r'$\theta$')
    Rp = np.log(R)

    logsp_pipeline = get_polynomial_pipeline(1)

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
        poly_model = get_polynomial_pipeline(degree)
        s = weighted_group_cross_val(
            poly_model,
            R, t,
            cv=gkf,
            groups=groups,
            weights=point_weights
        )
        print('Degree {} gives\n\tMean: {}\n\tSTD: {}'.format(
            degree, s.mean(), s.std()))

    fig = plt.figure(figsize=(15, 8))
    ax = fig.add_subplot(121)
    ax.semilogx(R, t, '.')
    ax_polar = fig.add_subplot(122, projection='polar')
    ax_polar.plot(t, R, '.')

    R_predict = np.linspace(0.01, max(R), 300).reshape(-1, 1)

    logsp_pipeline.fit(Rp, t)
    T = logsp_pipeline.predict(np.log(R_predict))
    ax_polar.plot(T, R_predict, label='Log spiral')
    ax.semilogx(R_predict, T)

    for degree in range(3, 9):
        poly_model = get_polynomial_pipeline(degree)
        poly_model.fit(R, t)
        T = poly_model.predict(R_predict)
        ax.semilogx(R_predict, T)
        ax_polar.plot(T, R_predict, label='$k={}$'.format(degree))

    ax.set_xlabel('Radius from center')
    ax.set_ylabel(r'$\theta$')
    ax_polar.legend()
    plt.show()
