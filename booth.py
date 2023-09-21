import numpy as np
import scipy.optimize as sco
import matplotlib.pyplot as plt
from matplotlib import ticker


plt.rcParams["font.size"] = 18

def booth_ab(xin):
    x, y = xin
    a = x + 2*y - 7
    b = 2*x + y - 5
    return a, b

def booth_fun(xin, *args):
    a, b = booth_ab(xin)
    return a**2 + b**2

def booth_jac(xin, *args):
    a, b = booth_ab(xin)
    return np.array([
        2*a + 2*b*2,
        2*a*2 + 2*b
    ])

x0 = np.array([0, 0])

# CG
cg = [x0]
def callback_cg(xk):
    cg.append(xk)
result = sco.minimize(booth_fun, cg[0], method='CG',
                      jac=booth_jac, callback=callback_cg)
niter_cg = len(cg) - 1
cg = np.array(cg)

# Newton
hess = np.array([[10.0, 8.0], [8.0, 10.0]])
hinv = np.array([[5.0/18.0, -2.0/9.0], [-2.0/9.0, 5.0/18.0]])
en = [x0]
en.append(-hinv @ booth_jac(x0))
niter_en = len(en) - 1
en = np.array(en)

# Preconditioned CG
w, v = np.linalg.eig(hess)
hess12 = v @ np.diag(np.sqrt(w)) @ v.T
hinv12 = v @ np.diag(np.sqrt(1/w)) @ v.T

def booth_fun_precond(xin_p, *args):
    xin = hinv12.T @ xin_p
    return booth_fun(xin)

def booth_jac_precond(xin_p, *args):
    xin = hinv12.T @ xin_p
    grad = booth_jac(xin)
    return hinv12.T @ grad

pcg = [hess12 @ x0]
def callback_pcg(xk):
    pcg.append(xk)
result = sco.minimize(booth_fun_precond, pcg[0], method='CG',
                      jac=booth_jac_precond, callback=callback_pcg)
niter_pcg = len(pcg) - 1
pcg = np.array(pcg) @ hinv12 # (hinv12.T @ pcg.T).T


fig, ax = plt.subplots(figsize=[8, 8])
nx, ny = 1001, 1001
x = np.linspace(-1, 4, nx)
y = np.linspace(-1, 4, ny)
X, Y = np.meshgrid(x, y)
Z = booth_fun([X, Y])
Z = np.ma.masked_where(Z <=0, Z)
ax.contour(X, Y, Z, colors="k",
           locator=ticker.LogLocator(subs=(1, 2, 5)))
ax.plot(en[:, 0], en[:, 1], linewidth=5, label=f"EN {niter_en}")
ax.plot(pcg[:, 0], pcg[:, 1], linewidth=2, label=f"PCG {niter_pcg}")
ax.plot(cg[:, 0], cg[:, 1], linewidth=3, label=f"CG {niter_cg}")
ax.legend(loc="lower right")
ax.set_title("Booth")
ax.set_aspect("equal")
#plt.show()
fig.savefig("booth.png", bbox_inches="tight", dpi=300)
fig.savefig("booth.pdf", bbox_inches="tight")
