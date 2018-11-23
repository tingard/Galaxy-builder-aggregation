import scipy.stats as st
import scipy.special as sc
import scipy.optimize as so
import numpy as np
import pylab as plt
from astropy.visualization import hist


data = np.load('./sigmas.npy')


fit_gamma = st.gamma.fit(data, floc=0)
print('a: {0}, b: {1}, loc: {2}'.format(fit_gamma[0], 1 / fit_gamma[2], fit_gamma[1]))

fit_gamma_loc = st.gamma.fit(data)
print('a: {0}, b: {1}, loc: {2}'.format(fit_gamma_loc[0], 1 / fit_gamma_loc[2], fit_gamma_loc[1]))


def logpdf(x, a, b, xl):
    return sc.xlogy(a-1.0, x) + sc.xlogy(a, b) - (x * b) - sc.gammaln(a) - np.log(sc.gammaincc(a, xl * b))


def nnlf(theta, x, xl):
    a, b = theta
    return -logpdf(x, a, b, xl).sum(axis=0)


fit_gamma_trunc = so.fmin(nnlf, [1, 1], args=(data, 0.1))
print('a: {0}, b: {1}'.format(*fit_gamma_trunc))

plt.figure(figsize=(10, 8))
hist(data, bins='blocks', density=True, range=(0.1, 5))
x = np.linspace(0.1, 5, 1000)
plt.plot(x, st.gamma.pdf(x, *fit_gamma), label='gamma: floc=0')
plt.plot(x, st.gamma.pdf(x, *fit_gamma_loc), label='gamma: loc free')
plt.plot(x, np.exp(logpdf(x, fit_gamma_trunc[0], fit_gamma_trunc[1], 0.1)), label='truncated gamma')
plt.legend()
plt.savefig('gamma_fitting.png')
plt.show()
