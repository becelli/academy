import numpy as np
from ODE import ODE_Solver
from dataclasses import dataclass
from numba import jit


@dataclass
class MultipleStep(ODE_Solver):
    f: callable
    a: float
    b: float
    y0: float
    hmax: int = None
    max_it: int = 10000
    tol: float = 1e-3

    def apply(self, method: str) -> list[np.ndarray, np.ndarray]:
        f = self.f
        a, b = self.a, self.b
        y0 = self.y0
        hmax = (self.b - self.a) / 10.0
        hmin = (self.b - self.a) / 1000.0
        max_it = self.max_it
        tol = self.tol

        x, y = None, None
        if method == "Dormand-Prince":
            x, y, flag = Dormand_Prince(f, a, b, y0, hmin, hmax, max_it, tol)
            if flag != "Success":
                x, y = np.zeros(0), np.zeros(0)
                print(flag)
        else:
            raise ValueError("Unknown method: {}".format(method))

        return x, y


@jit
def Dormand_Prince(
    f: callable,
    low: float,
    high: float,
    y0: float,
    hmin: float,
    hmax: float,
    max_it: int,
    tol: float,
):
    """
    https://en.wikipedia.org/wiki/Dormand%E2%80%93Prince_method
    ADAPTED FROM: https://web.archive.org/web/20150907215914/http://adorio-research.org/wordpress/?p=6565
    """

    # fmt: off
    a = np.array(
        [
            [0           , 0            , 0           , 0         , 0            , 0      ],
            [1 / 5       , 0            , 0           , 0         , 0            , 0      ],
            [3 / 40      , 9 / 40       , 0           , 0         , 0            , 0      ],
            [44 / 45     , -56 / 15     , 32 / 9      , 0         , 0            , 0      ],
            [19372 / 6561, -25360 / 2187, 64448 / 6561, -212 / 729, 0            , 0      ],
            [9017 / 3168 , -355 / 33    , 46732 / 5247, 49 / 176  , -5103 / 18656, 0      ],
            [35 / 384    , 0            , 500 / 1113  , 125 / 192 , -2187 / 6784 , 11 / 84],
        ]
    )
    c = np.array([0, 1 / 5, 3 / 10, 4 / 5, 8 / 9, 1, 1])
    b = np.array([35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84, 0])
    d = np.array([5179 / 57600, 0, 7571 / 16695, 393 / 640, -92097 / 339200, 187 / 2100, 1 / 40])
    e = b - d

    # fmt: on
    x = [low]
    y = [y0]
    h = hmax
    flag = "Success"

    for i in range(max_it):
        # Limit the upper bound.
        if x[-1] + h > high:
            h = high - x[-1]

        # fmt: off
        k1 = f(x[-1] + h * c[0], y[-1] + h * (a[0, 0]))
        k2 = f(x[-1] + h * c[1], y[-1] + h * (a[1, 0] * k1))
        k3 = f(x[-1] + h * c[2], y[-1] + h * (a[2, 0] * k1 + a[2, 1] * k2))
        k4 = f(x[-1] + h * c[3], y[-1] + h * (a[3, 0] * k1 + a[3, 1] * k2 + a[3, 2] * k3))
        k5 = f(x[-1] + h * c[4], y[-1] + h * (a[4, 0] * k1 + a[4, 1] * k2 + a[4, 2] * k3 + a[4, 3] * k4))
        k6 = f(x[-1] + h * c[5], y[-1] + h * (a[5, 0] * k1 + a[5, 1] * k2 + a[5, 2] * k3 + a[5, 3] * k4 + a[5, 4] * k5))
        k7 = f(x[-1] + h * c[6], y[-1] + h * (a[6, 0] * k1 + a[6, 1] * k2 + a[6, 2] * k3 + a[6, 3] * k4 + a[6, 4] * k5 + a[6, 5] * k6))
        
        error = abs(e[0] * k1 + e[2] * k3 + e[3] * k4 + e[4] * k5 + e[5] * k6 + e[6] * k7)

        delta = 0.84 * ((tol / (error + tol * 0.1))) ** (1 / 5)
        if error < tol:
            x.append(x[-1] + h)
            y.append(y[-1] + h * (b[0] * k1 + b[2] * k3 + b[3] * k4 + b[4] * k5 + b[5] * k6))

        # fmt: on

        if delta <= 0.1:  # If delta is too large, reduce the step size.
            h = h / 10
        elif delta >= 4.0:  # If delta is too small, increase the step size.
            h = h * 4
        else:  # If delta is acceptable, change the step proprotionally to delta.
            h = delta * h

        # If the step size is too large, reduce it.
        if h > hmax:
            h = hmax
        # If reached the end of the interval, stop.
        if x[-1] == high:
            break
        # If the step size is too small, the method did not converge.
        elif h < hmin:
            flag = "Did not converge. Minimum step size not sufficient."
            break

    if i == max_it:
        flag = "Maximum number of iterations reached."
    elif i == 0:
        flag = "Minimum step size not sufficient."

    return np.array(x), np.array(y), flag
