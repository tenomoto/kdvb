import numpy as np
import kdvb
from ode import rk4
import sys


n = 101
dt = 0.01
nstep = 200
ncycle = 100
nu = 0.07
beta1 = 0.5
beta2 = 1.0
xmin, xmax = -25, 25 

x = np.linspace(xmin, xmax, n)
np.save("x.npy", x)

t = -5
u = np.zeros([ncycle, n])

for i in range(ncycle):
    if i == 0:
        u[0,] = kdvb.two_solitons(x, t, np.sqrt(0.5 * beta1), np.sqrt(0.5 * beta2))
    else:
        u[i,] = kdvb.forecast(u[i-1,], nstep, dt, xmax*2, nu=nu, fd=True)
np.save("utrue.npy", u)
