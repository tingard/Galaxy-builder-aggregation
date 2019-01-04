import numpy as np
import matplotlib.pyplot as plt

def lsp(t, a, b):
  return a * np.exp(b * t)

np.random.seed(0)

t = np.linspace(np.pi, 3*np.pi, 300)
t += np.random.randn(t.shape[0]) * 0.01

T = t % (2 * np.pi)

r = lsp(t, 0.2, 0.2) + np.random.randn(T.shape[0]) * 0.1
r = np.abs(r)

a = np.argsort(r)

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='polar')

plt.plot(T, r)
