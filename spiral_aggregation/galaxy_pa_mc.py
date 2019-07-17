import os
import shutil
from tqdm import trange
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from gzbuilderspirals.oo import Arm
import theano
import theano.tensor as tt
import pymc3 as pm
import argparse
from sklearn.preprocessing import OrdinalEncoder
import warnings
from astropy.utils.exceptions import AstropyWarning
warnings.simplefilter('ignore', category=AstropyWarning)
warnings.simplefilter('ignore', category=UserWarning)
warnings.simplefilter('ignore', category=FutureWarning)

"""
Example usage:
>>> import galaxy_pa_mc as gpm
>>> gpm.run_simultaneous_hierarchial(True, 30000, 30)
>>> gpm.make_simultaneous_hierarchial_kde_plot()
"""


def get_logsp_trace_from_arms(arms, nsamples=3000):
    X = np.concatenate([
        np.stack(
            ((arm.t * arm.chirality), arm.R, arm.point_weights, np.tile(i, len(arm.R))),
            axis=1
        )
        for i, arm in enumerate(arms)
    ])
    pw_mask = X[:, 2] > 0
    t = X[:, 0][pw_mask]
    R = X[:, 1][pw_mask]
    weights = X[:, 2][pw_mask]

    pa, sigma_pa = arms[0].get_parent().get_pitch_angle(arms)

    with pm.Model() as model:
        # psi has a uniform prior over all reasonable values (-80, 80)
        # not -90 to 90 to avoid infinities / overflows
        psi = pm.Uniform('psi', lower=0, upper=80, testval=pa)
        psi_radians = psi * np.pi / 180

        # model a as a uniform, which we scale later so our varibles have
        # similar magnitude
        a = pm.Uniform('a', lower=0, upper=200, testval=1, shape=len(arms))
        a_arr = tt.concatenate([
            tt.tile(a[i], len(arms[i].t)) for i in range(len(arms))
        ])[pw_mask]

        # define our equation for mu_r
        mu_r = a_arr / 100 * tt.exp(tt.tan(psi_radians) * t)

        # define our expected error on r (different to error on psi)
        base_sigma = pm.HalfCauchy('sigma', beta=5, testval=0.02)
        sigma_y = theano.shared(
            np.asarray(
                np.sqrt(weights),
                dtype=theano.config.floatX
            ),
            name='sigma_y'
        )
        sigmas = base_sigma / sigma_y
        # define our likelihood function
        likelihood = pm.Normal('R', mu=mu_r, sd=sigmas, observed=R)
        # run the sampler

    with model:
        trace = pm.sample(nsamples, tune=1500, cores=2)

    return trace


def get_arms(subject_id, err=True):
    available_arms = os.listdir('lib/spiral_arms')

    arms = [
        Arm.load(os.path.join('lib/spiral_arms', a))
        for a in available_arms
        if str(subject_id) in a
    ]
    if err and len(arms) == 0:
        raise IndexError('No arms found for provided spiral arm')
    return arms


def make_validation_comparisons(outfolder='mc_validation_comparison_plots'):
    dr8ids, ss_ids, validation_ids = np.load('lib/duplicate_galaxies.npy').T
    try:
        os.mkdir(outfolder)
    except FileExistsError:
        pass
    for i in range(len(dr8ids)):
        print('\n\033[1mWorking on {} and {}\033[0m'.format(
            ss_ids[i], validation_ids[i]
        ))
        ss_arms = get_arms(ss_ids[i], err=False)
        val_arms = get_arms(validation_ids[i], err=False)
        if len(ss_arms) == 0 or len(val_arms) == 0:
            print('No arms found')
            continue
        plot_loc = os.path.join(outfolder, str(dr8ids[i]))
        try:
            os.mkdir(plot_loc)
        except FileExistsError:
            pass

        plt.figure(figsize=(8, 8), dpi=200)
        for arm in ss_arms:
            plt.plot(*arm.log_spiral.T, c='C0')
        for arm in val_arms:
            plt.plot(*arm.log_spiral.T, c='C1')
        plt.axis('equal')
        plt.tight_layout()
        plt.savefig(os.path.join(plot_loc, 'arms.png'),
                    bbox_inches='tight')
        plt.close()

        ss_pa = abs(ss_arms[0].get_parent().get_pitch_angle(ss_arms)[0])
        val_pa = abs(val_arms[0].get_parent().get_pitch_angle(val_arms)[0])
        ss_trace = get_logsp_trace_from_arms(ss_arms)

        plt.figure(figsize=(10, 4), dpi=200)
        pm.traceplot(ss_trace, varnames=['psi', 'sigma'],
                     lines={'psi': ss_pa})
        plt.savefig(os.path.join(plot_loc, 'original.png'),
                    bbox_inches='tight')
        plt.clf()

        val_trace = get_logsp_trace_from_arms(val_arms)
        pm.traceplot(val_trace, varnames=['psi', 'sigma'],
                     lines={'psi': val_pa})
        plt.savefig(os.path.join(plot_loc, 'validation.png'),
                    bbox_inches='tight')
        plt.close()


def run_on_all_gals():
    sid_list = pd.read_csv('lib/subject-id-list.csv').values.T[0]
    for subject_id in sid_list:
        traces_dir = os.path.join('uniform-traces', str(subject_id))
        if os.path.isdir(traces_dir):
            continue
        try:
            arms = get_arms(subject_id)
        except IndexError:
            continue
        print('Working on', subject_id)
        trace = get_logsp_trace_from_arms(arms)
        try:
            os.mkdir(traces_dir)
        except FileExistsError:
            pass
        pm.save_trace(trace, directory=traces_dir, overwrite=True)


def make_X_all(sid_list):
    return np.concatenate([
        np.stack(
            (
                (arm.t * arm.chirality), arm.R, arm.point_weights,
                np.tile(gal_no, len(arm.R)),
                np.tile(gal_no * len(sid_list) + arm_no, len(arm.R))
            ),
            axis=1
        )
        for gal_no, subject_id in enumerate(sid_list)
        for arm_no, arm in enumerate(get_arms(subject_id, err=False))
    ])


def hierarchial(subject_id=20902040):
    X_all = np.concatenate([
        np.stack(
            (
                (arm.t * arm.chirality), arm.R, arm.point_weights,
                np.tile(arm_no, len(arm.R))
            ),
            axis=1
        )
        for arm_no, arm in enumerate(get_arms(subject_id, err=False))
    ])
    # remove values with zero weight
    X = X_all[X_all.T[2] > 0]

    t, R, point_weights = X.T[:3]
    # get an arm index
    enc = OrdinalEncoder(dtype=np.int32)
    arm_idx = enc.fit_transform(X[:, [3]]).T[0]

    n_unique_arms = len(np.unique(arm_idx))

    with pm.Model() as hierarchical_model:
        print('Defining model')
        mu_psi = pm.Uniform('mu_psi', lower=0, upper=80, testval=15.0)
        sigma_psi = pm.HalfCauchy('sigma_psi', beta=1)
        psi_offset = pm.Normal('psi_offset', mu=0, sd=1)
        psi = pm.Deterministic('psi', mu_psi + sigma_psi * psi_offset)
        psi_radians = psi * np.pi / 180

        a = pm.Uniform('a', lower=0, upper=200, testval=1,
                       shape=n_unique_arms)

        # define our equation for mu_r
        r_est = (
            a[arm_idx] / 100
            * tt.exp(
                tt.tan(psi_radians)
                * t
            )
        )

        # define our expected error on r here we assume this sigma is the
        # same for all galaxies (not necessarily true)
        base_sigma = pm.HalfCauchy('sigma', beta=5, testval=0.02)
        sigma_y = theano.shared(
            np.asarray(
                np.sqrt(point_weights),
                dtype=theano.config.floatX
            ),
            name='sigma_y'
        )
        sigmas = base_sigma / sigma_y

        # define our likelihood function
        pm.Normal('R_like', mu=r_est, sd=sigmas, observed=R)

    with hierarchical_model:
        trace = pm.sample(2000, tune=1000, cores=2)

    pm.traceplot(trace)
    plt.gcf().set_size_inches(6, 15)
    pm.plots.pairplot(trace, varnames=['mu_psi', 'sigma_psi', 'sigma'])
    plt.gcf().set_size_inches(10, 10)
    pm.plots.plot_posterior(trace, kde_plot=True)
    plt.gcf().set_size_inches(10, 10)
    plt.show()


def run_simultaneous_hierarchial(
    recalculate=False, sample_size=None, max_ngals=None,
    outfolder='hierarchical-model'
):
    enc = OrdinalEncoder(dtype=np.int32)
    sid_list = pd.read_csv('lib/subject-id-list.csv').values.T[0]

    if os.path.isfile('Xall.npy') and not recalculate:
        X_all = np.load('Xall.npy')
    else:
        X_all = make_X_all(sid_list)
        np.save('Xall.npy', X_all)

    # remove all points with weight of zero (or less..?)
    all_gal_idx, all_arm_idx = enc.fit_transform(X_all[:, [3, 4]]).T
    if max_ngals is not None and max_ngals <= all_gal_idx.max():
        gals = np.random.choice(np.arange(all_gal_idx.max() + 1), max_ngals,
                                replace=False)
    else:
        gals = np.arange(len(all_gal_idx.max() + 1))
    X_masked = X_all[
        (X_all.T[2] > 0)
        & np.isin(all_gal_idx, gals)
    ]
    sample = np.random.choice(
        len(X_masked),
        size=sample_size,
        replace=False
    ) if sample_size else np.arange(len(X_masked))
    X = X_masked[sample]

    t, R, point_weights = X.T[:3]
    # encode categorical variables into an index
    enc = OrdinalEncoder(dtype=np.int32)
    gal_idx, arm_idx = enc.fit_transform(X[:, [3, 4]]).T
    n_gals = len(np.unique(gal_idx))
    n_unique_arms = len(np.unique(arm_idx))

    print('{} galaxies, {} spiral arms, {} points'.format(
        n_gals, n_unique_arms, len(X)
    ))

    with pm.Model() as hierarchical_model:
        print('Defining model')
        # Hyperpriors (informative for now)
        mu_psi = pm.Uniform('mu_psi', lower=0, upper=80, testval=15)
        # sigma_psi = pm.Gamma('sigma_psi', alpha=2, beta=10)
        sigma_psi = pm.HalfCauchy('sigma_psi', beta=1)
        psi_offset = pm.Normal('psi_offset', mu=0, sd=1, shape=n_gals)
        psi = pm.Deterministic('psi', mu_psi + sigma_psi * psi_offset)
        psi_radians = psi * np.pi / 180

        a = pm.Uniform('a', lower=0, upper=200, testval=1,
                       shape=n_unique_arms)

        # define our equation for mu_r
        r_est = (
            a[arm_idx] / 100
            * tt.exp(
                tt.tan(psi_radians[gal_idx])
                * t
            )
        )

        # define our expected error on r here we assume this sigma is the
        # same for all galaxies (not necessarily true)
        base_sigma = pm.HalfCauchy('sigma', beta=1, testval=0.02)
        sigma_y = theano.shared(
            np.asarray(
                np.sqrt(point_weights),
                dtype=theano.config.floatX
            ),
            name='sigma_y'
        )
        sigmas = base_sigma / sigma_y

        # define our likelihood function
        likelihood = pm.Normal('R_like', mu=r_est, sd=sigmas, observed=R)

    with hierarchical_model:
        trace = pm.sample(2000, tune=1000, cores=2, target_accept=0.95)
    if outfolder is not None:
        traces_dir = os.path.join('uniform-traces', outfolder)
    try:
        os.mkdir(traces_dir)
    except FileExistsError:
        shutil.rmtree(traces_dir)
    pm.save_trace(trace, directory=traces_dir, overwrite=True)
    pm.traceplot(trace, varnames=['mu_psi', 'sigma_psi', 'sigma'])
    plt.show()


def make_simultaneous_hierarchial_kde_plot(
    n_mean_plots=100, infolder='hierarchical-model'
):
    with pm.Model():
        trace = pm.load_trace(os.path.join('uniform-traces', infolder))
    psi = trace.get_values('psi')
    mu_psi = trace.get_values('mu_psi')
    df = pd.DataFrame(
        trace.get_values('psi'),
        columns=['psi-{}'.format(i) for i in range(psi.shape[1])]
    )
    df['mu_psi'] = mu_psi
    sigma_psi = pd.Series(trace.get_values('sigma_psi'))
    # plotting
    f, (ax0, ax1) = plt.subplots(
        ncols=1, nrows=2, figsize=(10, 6), dpi=200, sharex=True,
        gridspec_kw={'height_ratios': [3, 1]}
    )
    plt.sca(ax0)
    df.drop('mu_psi', axis=1).plot.kde(
        ax=plt.gca(), linestyle='dashed', legend=None
    )
    plt.sca(ax1)
    mean_dist_params = pd.concat((df['mu_psi'], sigma_psi), axis=1)
    for _ in trange(n_mean_plots):
        mu, sigma = mean_dist_params.sample().values.T
        x = np.linspace(0, 50, 500)
        y = np.exp(-(x - mu)**2/(2 * sigma**2)) / np.sqrt(2 * np.pi * sigma**2)
        plt.fill_between(x, np.zeros_like(y), y, alpha=0.008,
                         color='k')
    plt.xlim(0, 50)
    ax1.set_ylim(0, ax1.get_ylim()[1])
    plt.suptitle(r"KDE for individual galaxies' $\psi$ and $\psi_\mu$")
    plt.xlabel('Pitch angle (degrees)')
    plt.savefig('hierarchial_gal_result.png', bbox_inches='tight')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=(
            'Use MCMC to fit log spirals of uniform pitch angle '
            'to multiple arms from a galaxy identified by galaxy builder'
        )
    )
    parser.add_argument('--subject_id', '-s', default=20902040,
                        help='Subject id of galaxy to perform trace on')
    parser.add_argument('--outfolder', '-o', metavar='/path/to/directory',
                        default=False, help='Output directory')
    parser.add_argument(
        '--plot', '-p', action='store_true',
        default='.', help='Should plot trace and show to screen'
    )
    args = parser.parse_args()

    arms = get_arms(args.subject_id)
    pa, sigma_pa = arms[0].get_parent().get_pitch_angle(arms)
    gal_pa_est = pa * arms[0].chirality

    trace = get_logsp_trace_from_arms(arms)

    if args.outfolder:
        traces_dir = os.path.join(str(args.outfolder), str(args.subject_id))
        try:
            os.mkdir(traces_dir)
        except FileExistsError:
            pass
        pm.save_trace(trace, directory=traces_dir, overwrite=True)

    if args.plot:
        pm.traceplot(trace, lines={'psi': gal_pa_est})
        plt.show()
