import numpy as np


def optimize(x, objfn, *args, maxiter=100, tol=1.0e-8):
    n = x.size
    grad = np.zeros([n])
    hess = np.zeros([n, n])
    objval = np.zeros([maxiter])
    grad_err = np.zeros([maxiter])
    for i in range(maxiter):
        objval[i] = objfn(x, grad, hess, *args)
        grad_err[i] = np.linalg.norm(grad)
        if grad_err[i] < tol:
            return True, i, objval[:i+1], grad_err[:i+1]
        d = np.linalg.solve(hess, -grad)
        x += d
    print("Exiting without convergence")
    return False, maxiter, objval, grad_err

