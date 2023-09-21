import numpy as np
import kdvb
#from mlef import analysis
from mlef_zeta import analysis
#from hop import hop1 as hop
from hop import hop2 as hop
import sys

uobs = np.load("uobs.npy")
iobs = np.load("iobs.npy")
pf12 = np.load("pf12.npy")
x = np.load("x.npy")

nstep = 200
dt = 0.01
xmin, xmax = -25, 25 
nu = 0.07
epobs = 0.05
n, nens = pf12.shape
ncycle, nobs = uobs.shape
rinv = np.diag(epobs**-2 * np.ones(nobs))
cycle_save = np.arange(0, 10, 3)
jgmax = 4
maxiter = 100
tol = 1.0e-5
scale0 = True
scale = False

#pf12 *= 0.5
#pf12 /= np.sqrt(nens-1)

beta1 = 0.4
beta2 = 0.9
t = -6
ua = np.zeros([ncycle, n])
ub = np.zeros([ncycle, n])
ue = np.zeros([nens, n])
chi2 = np.zeros([ncycle])
innov = np.zeros([ncycle, nobs])
stda = np.zeros([ncycle])

ub[0,] = kdvb.two_solitons(x, t, np.sqrt(0.5 * beta1), np.sqrt(0.5 * beta2))

for i in range(ncycle):
    print(f"cycle {i}/{ncycle}")
    if scale or (i == 0 and scale0):
        pf12 /= np.sqrt(nens-1)
    jfile = f"j{i:03}.txt" if i < jgmax else None
    gfile = f"g{i:03}.txt" if i < jgmax else None
    ua[i,], pa12, chi2[i], innov[i,] = analysis(ub[i,], pf12, uobs[i,], rinv, hop, iobs[i,],
        maxiter=maxiter, tol=tol, jfile=jfile, gfile=gfile, stat=True) 
    if scale:
        pa12 *= np.sqrt(nens-1)
    stda[i] = np.sqrt(np.trace(pa12 @ pa12.T) / nens)
    if i in cycle_save:
        print("saving Pa")
        pa = pa12 @ pa12.T
        np.save(f"pa{i:03}.npy", pa)
        np.save(f"pf12{i:03}.npy", pf12)
        np.save(f"ua{i:03}.npy", ua[i,])
    ue = ua[i,][:,None] + pa12
    if i < ncycle-1:
        ub[i+1,] = kdvb.forecast(ua[i,], nstep, dt, xmax*2, nu=nu, fd=True)
        for j in range(nens):
            ue[:,j] = kdvb.forecast(ue[:,j], nstep, dt, xmax*2, nu=nu, fd=True)
        pf12 = ue - ub[i+1,][:, None]
np.save("ua.npy", ua)
np.save("ub.npy", ub)
np.save("chi2.npy", chi2)
np.save("stda.npy", stda)
np.save("innov.npy", innov.flatten())
