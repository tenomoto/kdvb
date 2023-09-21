import numpy as np
import scipy.linalg as la
from newton import optimize
import sys


jac = ""

def calc_ymat(x, pf12, hop, params):
    if jac == "quadratic":
        ymat =  np.diag(2.0 * x) @ pf12 # jacobian
    elif jac == "rms":
        ymat = (np.array(x / la.norm(x)) @ pf12).reshape([1, pf12.shape[1]])
    else:
        ymat = hop(x[:, None] + pf12, *params) - np.atleast_1d(hop(x, *params))[:, None] 
    return ymat

def costfn(wgt, grad, hess, x0, pf12, y, rinv, hop, *params):
    x = x0 + pf12 @ wgt
    hx = hop(x, *params)
    d = y - hx
    cost = 0.5 * (np.dot(wgt, wgt) + d.T @ rinv @ d)
    ymat = calc_ymat(x, pf12, hop, params)
    if type(grad) == np.ndarray:
        grad[:] = wgt - ymat.T @ rinv @ d
    if type(hess) == np.ndarray:
        hess[:, :] = ymat.T @ rinv @ ymat
        hess[np.diag_indices_from(hess)] += 1.0
    return cost

def matsqrt(w, maxiter=10, tol=1.0e-8):
# iteratively calculate square root of a matrix
# of form (I + ZZ^T) given eigenvalues w of Z^TZ
#    print(f"w={w}")
    p0 = -1.0 / (1.0 + w)
    s = np.zeros(w.size)
    ds = np.zeros(w.size)
    g = np.zeros(w.size)
    for k in range(maxiter):
        ds = p0 - g - p0 * w * g
        e = la.norm(s - ds)
        if e < tol:
            return s
        s = 0.5 * (s + ds)
        g = s / (1.0 + w * s)
    print("Exit matsqrt without convergence")
    return s

def analysis(xb, pf12, y, rinv, hop, *params, maxiter=100, tol=1.0e-8,
        jfile=None, gfile=None, sfile=None, stat=False, nmax_j=0):
    nens = pf12.shape[1]
    wgt = np.zeros([nens])
    success, niter, objval, grad_error = optimize(wgt, costfn, xb, pf12, y, rinv,
            hop, *params, maxiter=maxiter, tol=tol)
    if jfile != None:
        np.savetxt(jfile, objval)
    if gfile != None:
        np.savetxt(gfile, grad_error)
    xopt = xb + pf12 @ wgt
# update ensemble
    ymat = hop(xopt[:, None] + pf12, *params) - np.atleast_1d(hop(xopt, *params))[:, None] 
    cmat = ymat.T @ rinv @ ymat
    neig = min(pf12.shape)
# Eigen decompose C=Y^TY=VWV^T
    w, v = la.eigh(cmat)
    ic = np.ones(nens)
    ic[-neig:] += w[-neig:]
# Invert C: C^-1=V(I + W)^-1V^T
    pa12 = pf12 @ v @ np.diag(1.0 / np.sqrt(ic)) @ v.T
    if stat:
# Covariance of innovation y - H(x): (HP_fH^T + R)^-1
# Sherman-Morrison-Woodbury: A^-1-A^-1U(I+V^TA^-1U)^-1V^TA^-1
# A->R, U->Y, V^T->Y^T:      R^-1-R^-1Y(I+Y^TR^-1Y)^-1Y^TR^-1
        covd = rinv - rinv @ ymat @ v @ np.diag(1.0 / ic) @ v.T @ ymat.T @ rinv
        d = y - hop(xopt, *params)
        chi2 =  d.T @ covd @ d / y.size
# Calculate (HP_fH^T + R)^-1/2
        s = matsqrt(w)
# G^-1/2=I+ZV\Sigma_nV^TZ^T=I+R^-1/2V\Sigma_nV^TY^TR^-T/2
# (HP_fH^T + R)^-1/2=G^-1/2R^-1/2=R^-1/2(I+YV\Sigma_nV^TY^TR-1) 
# NB. Only diagonal elements of R^-1
        rinv12 = np.diag(np.sqrt(rinv.diagonal()))
        covd12 = rinv12 @ (np.diag(np.ones(y.size)) + ymat @ v @ np.diag(s) @ v.T @ ymat.T @ rinv)
        innov = covd12 @ d# / np.sqrt(y.size)
        return xopt, pa12, chi2, innov
    else:
        return xopt, pa12, niter, success
