import numpy as np
from numpy.random import default_rng
import kdvb
from ode import rk4
import sys


dt = 0.01
nstep = 400
nens = 10
nu = 0.07
beta1 = 0.4
beta2 = 0.9
xmin, xmax = -25, 25 
#epbeta1 = 0.05
#epbeta2 = 0.05
#ept = 2.0
epbeta1 = 0.04
epbeta2 = 0.09
ept = 2.0
t = -7

def forecast(rng):
    x = np.load("x.npy")
    n = x.size
    ub = np.zeros([n])
    u = np.zeros([nens, n])
    ub = kdvb.two_solitons(x, t, np.sqrt(0.5 * beta1), np.sqrt(0.5 * beta2))
    ub = kdvb.forecast(ub, nstep, dt, xmax*2, nu=nu, fd=True)

    beta1e = beta1 + epbeta1 * rng.standard_normal(nens)
    beta2e = beta2 + epbeta2 * rng.standard_normal(nens)
    te = t + ept * rng.standard_normal(nens)
    for j in range(nens):
        u[j,] = kdvb.two_solitons(x, te[j], np.sqrt(0.5 * beta1e[j]), np.sqrt(0.5 * beta2e[j]))
        u[j,] = kdvb.forecast(u[j,], nstep, dt, xmax*2, nu=nu, fd=True)
    pf12 = u - ub[None,]
    np.save("pf12.npy", pf12.T)

if __name__     == "__main__":
    seed = 514
    rng = default_rng(seed)
    forecast(rng)
