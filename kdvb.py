from math import tau
import numpy as np
import matplotlib.pyplot as plt
from ode import rk4
import sys

def tendency_fd(p, period=tau, alpha=6.0, beta=1.0):
    f = np.zeros_like(p)
    dx = period / p.size
    f[:] = -0.5 * (alpha * p * (np.roll(p, -1) - np.roll(p, 1))
            + beta * (np.roll(p, -2) - 2 * np.roll(p, -1) 
                + 2 * np.roll(p, 1) - np.roll(p, 2)) / (dx*dx)) / dx
    return f

def diffuse_fd(p, period=tau, nu=0.0):
    f = np.zeros_like(p)
    dx = period / p.size
    f[:] = nu * (np.roll(p, 1) - 2 * p  + np.roll(p, -1)) / (dx * dx)
    return f

def forecast(x0, nstep, dt, period, alpha=6.0, beta=1.0, nu=0.0, fd=False):
    xold = x0.copy()
    x = x0.copy()
    for i in range(nstep):
        x += rk4(tendency_fd, x, dt, period, alpha, beta)
        if nu > 0:
            x += diffuse_fd(xold, period, nu) * dt
        xold[:] = x[:]
    return x

def two_solitons(x, t, k1, k2):
    k1k1 = k1 * k1
    k2k2 = k2 * k2
    t1 = k1 * (x - 4 * k1k1 * t)
    t2 = k2 * (x - 4 * k2k2 * t)
    u = k2k2 - k1k1 + k2k2 * np.cosh(2 * t1) + k1k1 * np.cosh(2 * t2)
    u /= ((k2 - k1) * np.cosh(t1 + t2) + (k2 + k1) * np.cosh(t1 - t2)) ** 2
    u *= 4 * (k2k2 - k1k1)
    return u

if __name__ == "__main__":
    n = 101
    dx = 0.5
    dt = 0.01
    nt = 200
    nu = 0.07
    x = np.linspace(-(n-1)//2*dx, (n-1)//2*dx, n)
    fig, ax = plt.subplots(figsize=[10,5])
    beta1 = 0.5
    beta2 = 1.0
    for t in range(-4, 9, 4): 
        u = two_solitons(x, t, np.sqrt(0.5 * beta1), np.sqrt(0.5 * beta2))
        ax.plot(x, u, label=f"$t={t}$")
    ax.legend()
    fig.savefig("two_solitons.pdf", bbox_inches="tight")
    fig.savefig("two_solitons.png", bbox_inches="tight", dpi=300)
#    plt.show()
