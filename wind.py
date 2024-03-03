import numpy as np
import mlef
import mlef_zeta
import sys

nens = 1000
uobs = 3.0
ustd = 0.3
maxiter = 100
tol = 1.0e-5
scale = False
rinv = 1 / np.full((1, 1), ustd**2 * nens)
hist = False


def genens(n, rng):
    u0, v0 = 2, 4
    us, vs = 2, 2
    u = u0 + us * rng.standard_normal(nens)
    v = v0 + us * rng.standard_normal(nens)
    x = np.stack((u, v))
    np.save("xf.npy", x)
    xf = np.mean(x, axis=1)
    pf12 = x - xf[:,None]
    return xf, pf12

def hop(x, *params):
    return np.sqrt(x[0]**2 + x[1]**2)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage :: python {sys.argv[0]} CG|CGZ|CGJ|EN|EN1|ENJ")
        sys.exit()
    exp = sys.argv[1]
    if exp[:2] == "CG":
        analysis = mlef_zeta.analysis
    else:
        analysis = mlef.analysis
    if exp == "CG" or exp == "CGJ": mlef_zeta.update_zmat = False
    if exp == "EN1": maxiter = 1
    if exp == "ENJ": mlef.jac = "rms"
    if exp == "CGJ": mlef_zeta.jac = "rms"

    seed = 514
    rng = np.random.default_rng(seed)
    xf, pf12 = genens(nens, rng)
    print(f"xf={xf} var={np.sqrt(np.diag(pf12 @ pf12.T))}")

    y = np.full((1), uobs)
    params = []
    if scale:
        pf12 /= np.sqrt(nens - 1)
    print(f"maxiter={maxiter}")
    xa, pa12, niter, success = analysis(xf, pf12, y, rinv, hop, *params,
            maxiter=maxiter, tol=tol,
            jfile=f"obj_{exp}.txt", gfile=f"grad_{exp}.txt")
    np.savetxt(f"xa_{exp}.txt", xa)
    if scale:
        pa12 *= np.sqrt(nens - 1)
    np.save(f"pa12_{exp}.npy", pa12)
    umag = hop(xa, *params) 
    l2 = np.sqrt((umag - uobs)**2)
    print(f"success={success} niter={niter}")
    print(f"xa={xa} |xa|={umag} l2={l2}")
    if hist:
        xa_hist = [xf]
        for maxiter in range(1, niter):
            print(f"maxiter={maxiter}")
            xa, pa12, niter, success = analysis(xf, pf12, y, rinv, hop, *params,
                    maxiter=maxiter, tol=tol)
            xa_hist.append(xa)
        xa = np.loadtxt(f"xa_{exp}.txt")
        xa_hist.append(xa)
        np.savetxt(f"xa_hist_{exp}.txt", np.array(xa_hist))
