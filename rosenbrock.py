import numpy as np
from scipy.optimize import rosen, rosen_der
import scipy.optimize as sco
import matplotlib.pyplot as plt
from matplotlib import ticker
import sys

# Newton
def calc_hess(x, y):
    return np.array([[2 - 400 * y + 1200 * x**2, -400 * x], [-400 * x, 200]])
#    return np.array([[2 - 400 * (y - x**2) + 800 * x, -400 * x], [-400 * x, 200]])

def calc_step(x, y):
    denom = 1 - 200 * (y - x**2)
    step = np.array([1 - x, 2 * x * (1 - x) - (y - x**2) * denom]) / denom
    return step

def calc_dvec(x, y):
    adbc = 1 - 200 * (y - x**2)
    dx = (1 - x) / adbc
    dy = 2 * x * (1 - x) / adbc - (y - x**2)
    return dx, dy

if __name__ == "__main__":
    plt.rcParams["font.size"] = 18

    x0 = np.array([-1, -1])

    # CG
    cg = [x0]
    def callback_cg(xk):
        cg.append(xk)
    result = sco.minimize(rosen, cg[0], method='CG',
                          jac=rosen_der, callback=callback_cg)
    niter_cg = len(cg) - 1
    cg = np.array(cg)
    print("CG")
    print(cg)
    en = [x0]
    tol = 1.0e-5
    maxiter = 10
    i = 0
    while i < maxiter:
        grad = rosen_der(en[i])
        grad_error = np.linalg.norm(grad)
        print(f"grad_error={grad_error}")
        if grad_error < tol:
            break
        hess = calc_hess(*en[i])
        d = np.linalg.solve(hess, -grad)
    #    d = calc_step(*en[i])
        print(f"en({i}) d={d}")
        en.append(en[i] + d)
        i += 1
    niter_en = len(en) - 1
    en = np.array(en)
    print("EN")
    print(en)

    # Preconditioned CG
    hess = calc_hess(*x0)
    w, v = np.linalg.eig(hess)
    hess12 = v @ np.diag(np.sqrt(w)) @ v.T
    hinv12 = v @ np.diag(np.sqrt(1/w)) @ v.T

    def rosen_fun_precond(xin_p, *args):
        xin = hinv12.T @ xin_p
        return rosen(xin)

    def rosen_jac_precond(xin_p, *args):
        xin = hinv12.T @ xin_p
        grad = rosen_der(xin)
        return hinv12.T @ grad

    pcg = [hess12 @ x0]
    def callback_pcg(xk):
        pcg.append(xk)
    result = sco.minimize(rosen_fun_precond, pcg[0], method='CG',
                          jac=rosen_jac_precond, callback=callback_pcg)
    pcg = np.array(pcg) @ hinv12 # (hinv12.T @ pcg.T).T
    niter_pcg = len(pcg) - 1
    print(pcg)

    # Gauss-Newton

    def calc_F(x, y):
        return np.array([1.0 - x, np.sqrt(100.0) * (y - x**2)])

    def calc_J(x, y):
        return np.array([[-1.0, 0.0], [-2 * np.sqrt(100.0) * x, np.sqrt(100.0)]])

    gn = [x0]
    tol = 1.0e-5
    maxiter = 10
    i = 0
    while i < maxiter:
        fun = calc_F(*gn[i])
        jac = calc_J(*gn[i])
        grad = jac.T @ fun
        grad_error = np.linalg.norm(grad)
        print(f"grad_error={grad_error}")
        if grad_error < tol:
            break
        hess = jac.T @ jac
        d = np.linalg.solve(hess, -grad)
        print(f"gn({i}) d={d}")
        gn.append(gn[i] + d)
        i += 1
    niter_gn = len(gn) - 1
    gn = np.array(gn)
    print(gn)

    # Levenberg-Marquart
    lm = [x0]
    tol = 1.0e-5
    maxiter = 200
    gmma_min = 0.7 # gives minimun niter
    i = 0
    while i < maxiter:
    #    fun = calc_F(*lm[i])
    #    jac = calc_J(*lm[i])
    #    grad = jac.T @ fun
        grad = rosen_der(lm[i])
        grad_error = np.linalg.norm(grad)
        if grad_error < tol:
            break
    #    hess = jac.T @ jac
        hess = calc_hess(*lm[i])
        hinv = np.linalg.inv(hess)
        d = -hinv @ grad
        costi = rosen(lm[i])
        costl = costi + grad.T @ d
        costn = rosen(lm[i] + d)
    #    print(f"costi={costi} costn={costn} costl={costl}")
        gmma = np.abs(costn - costl)/ np.abs(costi)
    #    print(f"gmma={gmma}")
        if gmma > gmma_min:
            hinv1 = np.linalg.inv(hess + np.diag(gmma * grad_error * np.ones(grad.size))) 
            d = - hinv1 @ grad
        lm.append(lm[i] + d)
        i += 1
    niter_lm = len(lm) - 1
    lm = np.array(lm)

    fig, ax = plt.subplots(figsize=[8, 8])
    nx, ny = 1001, 1001
    x = np.linspace(-3, 3, nx)
    y = np.linspace(-4, 2.01, ny)
    X, Y = np.meshgrid(x, y)
    Z = rosen([X, Y])
    Z = np.ma.masked_where(Z <=0, Z)
    ax.contour(X, Y, Z, colors="k",
               locator=ticker.LogLocator(subs=(1, 5)))
    ax.plot(en[:, 0], en[:, 1], linewidth=5, label=f"EN {niter_en}")
    ax.plot(pcg[:, 0], pcg[:, 1], linewidth=2, label=f"PCG {niter_pcg}")
    ax.plot(cg[:, 0], cg[:, 1], linewidth=2, label=f"CG {niter_cg}")
    #ax.plot(lm[:, 0], lm[:, 1], linewidth=2, label=f"LM {niter_lm}")
    ax.plot(gn[:, 0], gn[:, 1], linewidth=2, label=f"GN {niter_gn}")
    ax.legend(loc="lower left")
    ax.set_title("Rosenbrock")
    ax.set_aspect("equal")
#    plt.show()
    fig.savefig("rosenbrock.png", bbox_inches="tight", dpi=300)
    fig.savefig("rosenbrock.pdf", bbox_inches="tight")
