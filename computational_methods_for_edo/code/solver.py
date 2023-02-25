import numpy as np
from numba import njit


@njit(cache=True)
def explicit_euler(x, y, f, n, h):
    for i in range(n):
        yi = y[i, :]
        y[i + 1, :] = yi + h * f(x[i + 1], yi)
    return y


@njit(cache=True)
def vec_norm(v):
    return np.sqrt(np.sum(v ** 2))


@njit(cache=True)
def implicit_euler(x, y, f, n, h):
    for i in range(n):
        yi = y[i, :]
        # Solve the system of equations using fixed-point iteration.
        yn = yi + h * f(x[i + 1], yi)
        yn_ = yi + h * f(x[i + 1], yn)
        for _ in range(10):
            if vec_norm(yn_ - yn) < 1e-6:
                break
            yn = yn_
            yn_ = yi + h * f(x[i + 1], yn)

        y[i + 1, :] = yn_
    return y


@njit(cache=True)
def modified_midpoint(x, y, f, n, h):
    y[1, :] = y[0, :] + h * f(x[0], y[0, :])
    for i in range(1, n):
        yi = y[i, :]
        yp = y[i - 1, :]
        y[i + 1, :] = yp + 2 * h * f(x[i], yi)
    return y


@njit(cache=True)
def runge_kutta_3(x, y, f, n, h):

    for i in range(n):
        yi = y[i, :]
        k1 = f(x[i], yi)
        k2 = f(x[i] + 0.50 * h, yi + 0.50 * k1 * h)
        k3 = f(x[i] + 0.75 * h, yi + 0.75 * k2 * h)
        y[i + 1, :] = yi + h * (2 * k1 + 3 * k2 + 4 * k3) / 9.0
    return y


@njit(cache=True)
def runge_kutta_4(x, y, f, n, h):

    for i in range(n):
        yi = y[i, :]
        k1 = f(x[i], yi)
        k2 = f(x[i] + 0.5 * h, yi + 0.5 * k1 * h)
        k3 = f(x[i] + 0.5 * h, yi + 0.5 * k2 * h)
        k4 = f(x[i] + 1.0 * h, yi + 1.0 * k3 * h)
        y[i + 1, :] = yi + h * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
    return y


@njit(cache=True)
def apply(method, f, a, b, n, y0):
    h = (b - a) / n
    x = np.arange(a, b + h, h)
    y = np.zeros((len(x), len(y0)))
    y[0, :] = y0

    if method == "Euler Explícito":
        return x, explicit_euler(x, y, f, n, h)
    elif method == "Euler Implícito":
        return x, implicit_euler(x, y, f, n, h)
    elif method == "Pto. Médio Mod.":
        return x, modified_midpoint(x, y, f, n, h)
    elif method == "Runge-Kutta 3":
        return x, runge_kutta_3(x, y, f, n, h)
    else:
        return x, runge_kutta_4(x, y, f, n, h)
