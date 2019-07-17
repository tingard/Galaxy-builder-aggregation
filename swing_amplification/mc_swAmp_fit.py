import os
import numpy as np
import scipy as sp
from scipy.integrate import odeint
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import pymc3 as pm
import theano.tensor as tt
from scipy import optimize
from IPython.display import display
from numba import jit
from gzbuilder_analysis.spirals import fitting, xy_from_r_theta
from gzbuilder_analysis.spirals.oo import Arm


@jit(nopython=True)
def dydt(r, theta, b):
    R = 2 * b * r
    s = np.sinh(R)
    return (
        2*np.sqrt(2) / 7 * r
        * np.sqrt(1 + R / s) / (1 - R / s)
    )


def theano_dydt(r, theta, b):
    R = 2 * b * r
    s = tt.sinh(R)
    return (2 * np.sqrt(2) / 7) * r * tt.sqrt(1 + R / s) / (1 - R / s)


def theano_rk4(t, t_m1, y, *args):
    dt = t - t_m1
    k1 = dt * theano_dydt(y, t, *args)
    k2 = dt * theano_dydt(y + 0.5 * k1, t, *args)
    k3 = dt * theano_dydt(y + 0.5 * k2, t, *args)
    k4 = dt * theano_dydt(y + k3, t, *args)
    y_np1 = y + (1./6.)*k1 + (1./3.)*k2 + (1./3.)*k3 + (1./6.)*k4
    return y_np1


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


def run_mc():
    # get all arms for a galaxy:
    arms = get_arms(20902040)
    if len(arms == 0):
        return
    X = np.concatenate([
        np.stack(
            (
                (arm.t * arm.chirality), arm.R,
                arm.point_weights, np.tile(i, len(arm.R))
            ),
            axis=1
        )
        for i, arm in enumerate(arms)
    ])
    with pm.Model() as mdl_ode:
        # assume each arm has $b$ drawn from some normal distribution
        # and each has its own r0
        mu_b = pm.Uniform('mu_b', lower=0, upper=1E3)
        sigma_b = pm.HalfCauchy('sigma_b', beta=1)
        b_offset = pm.Normal(mu=0, sigma=1)
        B = pm.Deterministic('B', mu_b + b_offset * sigma_b)

        logB = pm.Normal('logB', mu=0, sd=10, testval=np.log(guess_b))
        B = tt.exp(logB)
        r_0 = pm.Uniform('r_0', lower=0, upper=100, testval=guess_r0)
        sigma = pm.Exponential('sigma', lam=0.05, testval=guess_sigma)

        y_est, updates = theano.scan(
            fn=theano_rk4,
            sequences=[
                {'input': tt.as_tensor(theta), 'taps': [0, -1]}
            ],
            outputs_info=r_0,
            non_sequences=[
                B,
            ]
        )

        y_est = tt.concatenate([[r_0], y_est])
        likelihood = pm.Normal('likelihood', mu=y_est, sd=sigma, observed=y_obs)
