import numpy as np
from numpy.random import default_rng
#from hop import hop1 as hop
from hop import hop2 as hop
from ode import rk4


x = np.load("x.npy")
u= np.load("utrue.npy")
ncycle, n = u.shape
epobs = 0.05
#obsnetwork = "fixed"
#obsnetwork = "targeted"

def observe(rng, obsnetwork="all"):
    nobs = n if obsnetwork == "all" else 20
    uobs = np.zeros([ncycle, nobs])
    iobs = np.zeros([ncycle, nobs], dtype=int)
    xobs = np.zeros([ncycle, nobs])
    if obsnetwork == "all":
        xobs[0,] = x[:]
        loc = np.arange(n)
        iobs[0,] = loc

    for i in range(ncycle):
        if obsnetwork == "targeted" or (obsnetwork == "fixed" and i == 0):
            ux = np.abs(np.roll(u[i, :], -1) - np.roll(u[i, :], 1))
            loc = np.sort(ux.argsort()[-nobs:])
            iobs[i,] = loc
            xobs[i,] = x[loc]
        else:
            iobs[i,] = iobs[0,]
            xobs[i,] = xobs[0,]
        uobs[i,:] = hop(u[i,], loc) + epobs * rng.standard_normal(nobs)
    #uobs = np.where(uobs < 0.0, 0.0, uobs)
    np.save("uobs.npy", uobs)
    np.save("xobs.npy", xobs)
    np.save("iobs.npy", iobs)

if __name__ == "__main__":
    seed = 514
    rng = default_rng(seed)
    observe(rng)
