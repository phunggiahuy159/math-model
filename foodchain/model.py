"""Core model: response functions, the scaled chemostat food-chain ODEs and Jacobian.

Implements the dimensionless system (1.2) of

    El-Sheikh & Mahrouf, "Stability and bifurcation of a simple food chain
    in a chemostat with removal rates", Chaos, Solitons and Fractals 23 (2005)
    1475-1489.

The scaled state is (S, x, y, z):

    S  nutrient concentration
    x  prey   (feeds on the nutrient S)
    y  first  predator (feeds on the prey x)
    z  second predator (feeds on the first predator y)

System (1.2):

    dS/dt = s_in - S - f1(S) * x
    dx/dt = x * (f1(S) - D1) - f2(x) * y
    dy/dt = y * (f2(x) - D2) - f3(y) * z
    dz/dt = z * (f3(y) - D3)

In the paper the scaled input concentration equals 1; we keep it as the
parameter ``s_in`` so it can be used as a bifurcation parameter.

The per-capita response functions are Holling type II / Michaelis-Menten
(Section 4 of the paper):

    f_i(u) = m_i * u / (a_i + u),   i = 1, 2, 3.

The removal rates ``D_i = D + delta_i`` are the sum of the chemostat washout
rate ``D`` and the species death rate ``delta_i`` ("distinct removal rates"),
which is what breaks the usual conservation law of the classical chemostat.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Parameters:
    """Parameters of the scaled food-chain chemostat model (1.2).

    Attributes
    ----------
    m1, m2, m3:
        Maximal per-capita growth rates in the Michaelis-Menten responses.
    a1, a2, a3:
        Half-saturation constants in the Michaelis-Menten responses.
    D1, D2, D3:
        Distinct removal rates (washout + death) of x, y and z.
    s_in:
        Scaled input nutrient concentration (1 in the paper). Exposed so it
        can serve as a bifurcation parameter.
    """

    m1: float = 5.0
    m2: float = 4.0
    m3: float = 3.0
    a1: float = 0.30
    a2: float = 0.50
    a3: float = 0.50
    D1: float = 0.90
    D2: float = 0.80
    D3: float = 0.50
    s_in: float = 1.0

    # ------------------------------------------------------------------
    # Response functions f_i and their derivatives f_i'
    # ------------------------------------------------------------------
    def f1(self, S: float) -> float:
        return self.m1 * S / (self.a1 + S)

    def f2(self, x: float) -> float:
        return self.m2 * x / (self.a2 + x)

    def f3(self, y: float) -> float:
        return self.m3 * y / (self.a3 + y)

    def df1(self, S: float) -> float:
        return self.m1 * self.a1 / (self.a1 + S) ** 2

    def df2(self, x: float) -> float:
        return self.m2 * self.a2 / (self.a2 + x) ** 2

    def df3(self, y: float) -> float:
        return self.m3 * self.a3 / (self.a3 + y) ** 2

    def with_(self, **changes) -> "Parameters":
        """Return a copy of these parameters with some fields replaced."""
        from dataclasses import replace

        return replace(self, **changes)


def rhs(t: float, state, p: Parameters):
    """Right-hand side of system (1.2).

    Signature ``(t, state, p)`` is compatible with ``scipy.integrate.solve_ivp``
    when wrapped, e.g. ``lambda t, y: rhs(t, y, p)``.
    """
    S, x, y, z = state
    f1 = p.f1(S)
    f2 = p.f2(x)
    f3 = p.f3(y)

    dS = p.s_in - S - f1 * x
    dx = x * (f1 - p.D1) - f2 * y
    dy = y * (f2 - p.D2) - f3 * z
    dz = z * (f3 - p.D3)
    return np.array([dS, dx, dy, dz])


def jacobian(state, p: Parameters) -> np.ndarray:
    """Jacobian matrix of system (1.2) evaluated at ``state`` (Section 2).

    J =
      [ -1 - f1'(S) x        -f1(S)                 0              0      ]
      [  f1'(S) x      f1(S) - D1 - f2'(x) y      -f2(x)           0      ]
      [    0                  f2'(x) y       f2(x) - D2 - f3'(y) z -f3(y)  ]
      [    0                    0                  f3'(y) z   f3(y) - D3   ]
    """
    S, x, y, z = state
    f1, f2, f3 = p.f1(S), p.f2(x), p.f3(y)
    df1, df2, df3 = p.df1(S), p.df2(x), p.df3(y)

    return np.array(
        [
            [-1.0 - df1 * x, -f1, 0.0, 0.0],
            [df1 * x, f1 - p.D1 - df2 * y, -f2, 0.0],
            [0.0, df2 * y, f2 - p.D2 - df3 * z, -f3],
            [0.0, 0.0, df3 * z, f3 - p.D3],
        ]
    )
