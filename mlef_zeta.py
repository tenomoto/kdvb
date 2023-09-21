import numpy as np
from scipy.optimize import minimize
import scipy.linalg as la
from mlef import matsqrt
import sys


update_zmat = True
jac = ""


def calc_zmat(x, pf12, rinv12, hop, *params):
    if jac == "quadratic":
        zmat = rinv12 @ np.diag(2 * xb) @ pf12
    elif jac == "linear":
        zmat = rinv12 @ pf12 # jacobian linear
    elif jac == "rms":
        zmat = rinv12 @ (np.array(x / la.norm(x)) @ pf12).reshape([1, pf12.shape[1]])
    else:
        zmat = rinv12 @ (hop(x[:, None] + pf12, *params) - np.atleast_1d(hop(x, *params))[:, None])
    return zmat


def costfn(zeta, *args):
    x0, y, g12, heinv, heinv12, zmat, rinv, rinv12, hop, hargs = args
    x = x0 + g12 @ zeta
    d = y - hop(x, *hargs)
    cost = 0.5 * (zeta.T @ heinv @ zeta + d.T @ rinv @ d)
    return cost

def gradfn(zeta, *args):
    if update_zmat:
        x0, y, g12, heinv, heinv12, pf12, rinv, rinv12, hop, hargs = args
    else:
        x0, y, g12, heinv, heinv12, zmat, rinv, rinv12, hop, hargs = args
    x = x0 + g12 @ zeta
    if update_zmat:
        zmat = calc_zmat(x, pf12, rinv12, hop, hargs)
    d = y - hop(x, *hargs)
    grad = heinv @ zeta - heinv12 @ zmat.T @ rinv12 @ d
    return grad

zetak = []
def callback(xk):
    global zetak
    zetak.append(xk)

def analysis(xb, pf12, y, rinv, hop, *params, maxiter=100, tol=1.0e-8,
        jfile=None, gfile=None, sfile=None, stat=False, bopt="PR", debug=True):
    global zetak
    rinv12 = np.diag(np.sqrt(rinv.diagonal())) 
    rinv = rinv12 @ rinv12.T
    nens = pf12.shape[1]
    zeta = np.zeros(nens)
    zmat = calc_zmat(xb, pf12, rinv12, hop, *params)
    cmat = zmat.T @ zmat
    neig = min(pf12.shape)
    ic = np.ones(nens)
    w, v = np.linalg.eigh(cmat)
    ic[-neig:] += w[-neig:]
    heinv = v @ np.diag(1.0 / ic) @ v.T
    heinv12 = v @ np.diag(1.0 / np.sqrt(ic)) @ v.T
    g12 = pf12 @ heinv12
    if update_zmat:
        args = xb, y, g12, heinv, heinv12, pf12, rinv, rinv12, hop, params
    else:
        args = xb, y, g12, heinv, heinv12, zmat, rinv, rinv12, hop, params
    zetak.clear()
    res = minimize(costfn, zeta, args=args, method="CG", jac=gradfn,
        callback=callback, options={
            "maxiter":maxiter, "gtol":tol, "disp":True, "return_all":True})
    print(f"success={res.success} niter={res.nit}")
    print(f"message={res.message}")
    objval = np.zeros(len(zetak)+1)
    grad_error = np.zeros(len(zetak)+1)
    zeta = np.zeros(nens)
    objval[0] = costfn(zeta, *args)
    grad_error[0] = np.linalg.norm(gradfn(zeta, *args))
    for i in range(len(zetak)):
        objval[i+1] = costfn(zetak[i], *args)
        grad_error[i+1] = np.linalg.norm(gradfn(zetak[i], *args))
    zeta, niter, success = res.x, res.nit, res.success
    if jfile != None:
        np.savetxt(jfile, objval)
    if gfile != None:
        np.savetxt(gfile, grad_error)
    xopt = xb + g12 @ zeta
# update ensemble
    zmat = rinv12 @ (hop(xopt[:, None] + pf12, *params) - np.atleast_1d(hop(xopt, *params))[:, None])
    cmat = zmat.T @ zmat
# Eigen decompose C=Z^TZ=VWV^T
    w, v = la.eigh(cmat)
    ic = np.ones(nens)
    ic[-neig:] += w[-neig:]
# Invert C: C^-1=V(I + W)^-1V^T
    pa12 = pf12 @ v @ np.diag(1.0 / np.sqrt(ic)) @ v.T
    if stat:
        zv = zmat @ v
        ginv = -zv @ np.diag(1.0 / ic) @ zv.T
        ginv[np.diag_indices_from(ginv)] += 1.0
        d = rinv12 @ (y - hop(xopt, *params))
        chi2 =  d.T @ ginv @ d / y.size
        s = matsqrt(w)
        ginv12 = zv @ np.diag(s) @ zv.T
        ginv12[np.diag_indices_from(ginv12)] += 1.0
        innov = ginv12 @ d# / np.sqrt(y.size)
        return xopt, pa12, chi2, innov
    else:
        return xopt, pa12, niter, success
