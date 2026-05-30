"""Computation of the equilibria E0, E1, E2 and E* of system (1.2).

From the equilibrium equations (Section 2 of the paper):

    1 - S - f1(S) x          = 0
    x (f1(S) - D1) - f2(x) y = 0
    y (f2(x) - D2) - f3(y) z = 0
    z (f3(y) - D3)           = 0

there are up to four biologically meaningful equilibria:

    E0 = (s_in, 0,  0,  0)     washout
    E1 = (S1, x1, 0,  0)       prey survives only
    E2 = (S2, x2, y2, 0)       prey + first predator
    E* = (S*, x*, y*, z*)      full coexistence

With Michaelis-Menten responses each successive level fixes one component
through  f_i(.) = D_i, e.g. f3(y*) = D3, f2(x2) = D2, f1(S1) = D1, and the
remaining components follow from the nutrient/biomass balances.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.optimize import brentq, fsolve

from .model import Parameters, rhs


@dataclass
class Equilibrium:
    name: str
    state: np.ndarray
    exists: bool
    note: str = ""

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        s = np.array2string(self.state, precision=4, suppress_small=True)
        flag = "exists" if self.exists else "does NOT exist"
        return f"<{self.name} {flag}: {s}{(' | ' + self.note) if self.note else ''}>"


# ----------------------------------------------------------------------
# Helpers: invert a Michaelis-Menten response, f(u) = m u / (a + u) = D
#   ->  u = a D / (m - D),  defined and positive only when m > D.
# ----------------------------------------------------------------------
def _invert_mm(m: float, a: float, D: float) -> Optional[float]:
    if m <= D:
        return None
    return a * D / (m - D)


def washout(p: Parameters) -> Equilibrium:
    """E0 = (s_in, 0, 0, 0). Always exists."""
    return Equilibrium("E0", np.array([p.s_in, 0.0, 0.0, 0.0]), exists=True)


def prey_only(p: Parameters) -> Equilibrium:
    """E1 = (S1, x1, 0, 0).

    f1(S1) = D1  ->  S1 = a1 D1 / (m1 - D1)
    1 - S1 - f1(S1) x1 = 0,  f1(S1)=D1  ->  x1 = (s_in - S1) / D1
    Exists iff S1 in (0, s_in)  (equivalently f1(s_in) > D1).
    """
    S1 = _invert_mm(p.m1, p.a1, p.D1)
    if S1 is None or S1 >= p.s_in:
        return Equilibrium("E1", np.array([np.nan] * 4), exists=False,
                           note="requires f1(s_in) > D1")
    x1 = (p.s_in - S1) / p.D1
    return Equilibrium("E1", np.array([S1, x1, 0.0, 0.0]), exists=x1 > 0)


def prey_predator(p: Parameters) -> Equilibrium:
    """E2 = (S2, x2, y2, 0).

    f2(x2) = D2  ->  x2 = a2 D2 / (m2 - D2)
    1 - S2 - f1(S2) x2 = 0  ->  solve scalar equation for S2 in (0, s_in)
    y2 = x2 (f1(S2) - D1) / f2(x2) = x2 (f1(S2) - D1) / D2
    Exists iff x2 > 0 and f1(S2) > D1 (so that y2 > 0).
    """
    x2 = _invert_mm(p.m2, p.a2, p.D2)
    if x2 is None:
        return Equilibrium("E2", np.array([np.nan] * 4), exists=False,
                           note="requires m2 > D2")

    # 1 - S - f1(S) x2 = 0 on S in (0, s_in); LHS decreasing, positive at 0.
    g = lambda S: p.s_in - S - p.f1(S) * x2
    if g(0.0) * g(p.s_in) > 0:
        return Equilibrium("E2", np.array([np.nan] * 4), exists=False,
                           note="no nutrient balance root in (0, s_in)")
    S2 = brentq(g, 1e-12, p.s_in)
    y2 = x2 * (p.f1(S2) - p.D1) / p.D2
    return Equilibrium("E2", np.array([S2, x2, y2, 0.0]), exists=y2 > 0,
                       note="" if y2 > 0 else "requires f1(S2) > D1")


def all_coexistence(p: Parameters, n_scan: int = 2000) -> list[np.ndarray]:
    """Return every positive interior equilibrium (S*, x*, y*, z*) of (1.2).

    With Michaelis-Menten responses the interior balance can admit more than
    one root, so we reduce it to a single scalar equation and scan for *all*
    sign changes rather than relying on a single Newton seed.

    f3(y*) = D3  ->  y* = a3 D3 / (m3 - D3)               (needs m3 > D3)
    For S in (0, s_in) the nutrient balance fixes
        x(S) = (s_in - S) / f1(S)
    and substitution into the prey balance gives the scalar equation
        g(S) = (s_in - S) - D1 x(S) - f2(x(S)) y* = 0.
    For each root S* we recover x* = x(S*) and
        z* = y* (f2(x*) - D2) / D3,
    keeping only roots with z* > 0 (equivalently f2(x*) > D2).
    """
    y_star = _invert_mm(p.m3, p.a3, p.D3)
    if y_star is None:
        return []

    def x_of_S(S):
        return (p.s_in - S) / p.f1(S)

    def g(S):
        x = x_of_S(S)
        return (p.s_in - S) - (p.D1 * x + p.f2(x) * y_star)

    eps = 1e-9
    grid = np.linspace(eps, p.s_in - eps, n_scan)
    vals = np.array([g(S) for S in grid])

    roots: list[np.ndarray] = []
    for i in range(len(grid) - 1):
        if vals[i] == 0.0:
            S_star = grid[i]
        elif vals[i] * vals[i + 1] < 0:
            S_star = brentq(g, grid[i], grid[i + 1], xtol=1e-12)
        else:
            continue
        x_star = x_of_S(S_star)
        z_star = y_star * (p.f2(x_star) - p.D2) / p.D3
        state = np.array([S_star, x_star, y_star, z_star])
        if np.all(state > 0) and np.max(np.abs(rhs(0.0, state, p))) < 1e-7:
            roots.append(state)
    return roots


def coexistence(p: Parameters,
                guess: Optional[np.ndarray] = None) -> Equilibrium:
    """The interior coexistence equilibrium E* = (S*, x*, y*, z*).

    See :func:`all_coexistence` for the construction. When several interior
    roots exist this returns the branch that bifurcates from E2 (the one with
    the smallest x*, i.e. closest to x2 where f2(x2) = D2), which is the
    biologically relevant coexistence state continued from the prey-predator
    equilibrium. Pass ``guess`` (an (S, x) or full state) to instead select the
    root nearest a given point -- used for numerical continuation.
    """
    if _invert_mm(p.m3, p.a3, p.D3) is None:
        return Equilibrium("E*", np.array([np.nan] * 4), exists=False,
                           note="requires m3 > D3")

    roots = all_coexistence(p)
    if not roots:
        return Equilibrium("E*", np.array([np.nan] * 4), exists=False,
                           note="no positive interior root (requires f2(x*) > D2)")

    if guess is not None:
        g = np.asarray(guess, dtype=float)
        key = lambda st: np.hypot(st[0] - g[0], st[1] - g[1])
        state = min(roots, key=key)
    else:
        state = min(roots, key=lambda st: st[1])  # smallest x* -> branch from E2
    return Equilibrium("E*", state, exists=True)


def all_equilibria(p: Parameters) -> list[Equilibrium]:
    """Return [E0, E1, E2, E*] for the given parameters."""
    return [washout(p), prey_only(p), prey_predator(p), coexistence(p)]


def residual(eq: Equilibrium, p: Parameters) -> float:
    """Infinity-norm of rhs(eq) -- should be ~0 for a genuine equilibrium."""
    if not np.all(np.isfinite(eq.state)):
        return np.nan
    return float(np.max(np.abs(rhs(0.0, eq.state, p))))
