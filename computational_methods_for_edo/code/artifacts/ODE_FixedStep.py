from dataclasses import dataclass
import numpy as np
from scipy.optimize import fsolve
from ODE import ODE_Solver
from numba import njit


@dataclass
class FixedStep(ODE_Solver):
    f: callable
    y0: float
    a: float
    b: float
    n: int

    def apply(self, method: str) -> list[np.ndarray, np.ndarray]:
        """
        Returns:
            x (np.ndarray): x values.
            y (np.ndarray): y values.
        """
        n = self.n
        x = np.linspace(self.a, self.b, n)
        h = x[1] - x[0]
        f, y0 = self.f, self.y0

        if method == "Explicit Euler":
            y = explicit_euler(x, f, y0, n, h)
        elif method == "Implicit Euler":
            y = implicit_euler(x, f, y0, n, h)
        elif method == "Central Difference":
            y = central_difference(x, f, y0, n, h)
        elif method == "Improved Euler":
            y = improved_euler(x, f, y0, n, h)
        elif method == "Modified Euler":
            y = modified_euler(x, f, y0, n, h)
        elif method == "Trapezoidal":
            y = trapezoidal(x, f, y0, n, h)
        elif method == "Simpson":
            y = simpson(x, f, y0, n, h)
        elif method == "Runge-Kutta-3":
            y = runge_kutta_3(x, f, y0, n, h)
        elif method == "Runge-Kutta-4":
            y = runge_kutta_4(x, f, y0, n, h)
        elif method == "Runge-Kutta-5":
            y = runge_kutta_5(x, f, y0, n, h)
        else:
            raise ValueError(f"Invalid method: {method}")
        return x, y


@njit
def explicit_euler(x, f, y0, n, h):
    y = np.zeros(n)
    y[0] = y0
    for i in range(n - 1):
        y[i + 1] = y[i] + h * f(x[i], y[i])
    return y


@njit
def implicit_euler(x, f, y0, n, h):
    y = np.zeros(n)
    y[0] = y0
    for i in range(n - 1):
        y[i + 1] = y[i]
        for _ in range(10):
            y[i + 1] = y[i] + h * f(x[i + 1], y[i + 1])
    return y


@njit
def central_difference(x, f, y0, n, h):
    y = np.zeros(n)
    y[0] = y0
    y[1] = y[0] + h * f(x[0], y[0])
    for i in range(1, n - 1):
        y[i + 1] = y[i - 1] + 2 * h * f(x[i], y[i])
    return y


@njit
def improved_euler(x, f, y0, n, h):
    y = np.zeros(n)
    y[0] = y0
    # AKA:Runge-Kutta 2nd order method
    # AKA: Euler modified method or Heun
    for i in range(n - 1):
        k1 = f(x[i], y[i])
        k2 = f(x[i] + h, y[i] + h * k1)
        y[i + 1] = y[i] + h * (k1 + k2) / 2.0
    return y


@njit
def modified_euler(x, f, y0, n, h):
    y = np.zeros(n)
    y[0] = y0
    # AKA. Runge-Kutta 2rd order method.
    for i in range(n - 1):
        k1 = f(x[i], y[i])
        k2 = f(x[i] + h / 2, y[i] + h * k1 / 2.0)
        y[i + 1] = y[i] + h * k2


@njit
def trapezoidal(x, f, y0, n, h):
    y = np.zeros(n)
    y[0] = y0
    # AKA: Crank-Nicolson method
    # FROM: https://people.sc.fsu.edu/~jburkardt/m_src/rkf45/rkf45.html
    for i in range(n - 1):
        aux = f(x[i], y[i])

        for _ in range(10):
            y[i + 1] = y[i] + h * f(x[i + 1], y[i + 1])
            y[i + 1] = y[i] + h * (aux + f(x[i + 1], y[i + 1])) / 2.0
    return y


@njit
def trapezoidal_fixed_point(y_next, f, x_i, y_i, x_next):
    for _ in range(10):
        y_next = y_i + (x_next - x_i) * f(x_next, y_next)
    return y_next


@njit
def simpson(x, f, y0, n, h):
    y = np.zeros(n)
    y[0] = y0
    for i in range(n - 1):
        k1 = f(x[i], y[i])
        k2 = f(x[i] + h / 2.0, y[i] + h * k1 / 2)
        k3 = f(x[i] + h, y[i] + h * k2)
        y[i + 1] = y[i] + h * (k1 + 4 * k2 + k3) / 6
    return y


@njit
def runge_kutta_3(x, f, y0, n, h):
    y = np.zeros(n)
    y[0] = y0
    for i in range(n - 1):
        k1 = f(x[i], y[i])
        k2 = f(x[i] + h / 2, y[i] + h * k1 / 2)
        k3 = f(x[i] + 0.75 * h, y[i] + 0.75 * h * k2)
        y[i + 1] = y[i] + h * (2 * k1 + 3 * k2 + 4 * k3) / 9.0
    return y


@njit
def runge_kutta_4(x, f, y0, n, h):
    y = np.zeros(n)
    y[0] = y0
    for i in range(n - 1):
        k1 = f(x[i], y[i])
        k2 = f(x[i] + h / 2, y[i] + h * k1 / 2)
        k3 = f(x[i] + h / 2, y[i] + h * k2 / 2)
        k4 = f(x[i] + h, y[i] + h * k3)
        y[i + 1] = y[i] + h * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
    return y


@njit
def runge_kutta_5(x, f, y0, n, h):
    y = np.zeros(n)
    y[0] = y0
    # ADAPTED FROM: https://people.sc.fsu.edu/~jburkardt/m_src/rk45/rk45.html
    a = np.array(
        [
            [0.25, 0, 0, 0, 0],
            [3 / 32, 9 / 32, 0, 0, 0],
            [1932 / 2197, -7200 / 2197, 7296 / 2197, 0, 0],
            [439 / 216, -8, 3680 / 513, -845 / 4104, 0],
            [-8 / 27, 2, -3544 / 2565, 1859 / 4104, -11 / 40],
        ]
    )
    b = np.array([16 / 135, 0, 6656 / 12825, 28561 / 56430, -9 / 50, 2 / 55])
    c = np.array([0, 0.25, 3 / 8, 12 / 13, 1, 0.5])
    # fmt: off
    for i in range(n - 1):
        k1 = h * f(x[i] + c[0] * h, y[i])
        k2 = h * f(x[i] + c[1] * h, y[i] + a[0, 0] * k1)
        k3 = h * f(x[i] + c[2] * h, y[i] + a[1, 0] * k1 + a[1, 1] * k2)
        k4 = h * f(x[i] + c[3] * h, y[i] + a[2, 0] * k1 + a[2, 1] * k2 + a[2, 2] * k3)
        k5 = h * f(x[i] + c[4] * h, y[i] + a[3, 0] * k1 + a[3, 1] * k2 + a[3, 2] * k3 + a[3, 3] * k4)
        k6 = h * f(x[i] + c[5] * h, y[i] + a[4, 0] * k1 + a[4, 1] * k2 + a[4, 2] * k3 + a[4, 3] * k4 + a[4, 4] * k5)
        y[i + 1] = y[i] + b[0] * k1 + b[1] * k2 + b[2] * k3 + b[3] * k4 + b[4] * k5 + b[5] * k6
    # fmt: on
    return y
