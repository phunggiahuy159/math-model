"""Local stability analysis: eigenvalues, Routh-Hurwitz, and the analytic
stability conditions of Theorems 2.1-2.4.

For each equilibrium the linearization is governed by the Jacobian of (1.2).
A hyperbolic equilibrium is locally asymptotically stable iff every eigenvalue
has negative real part.

The paper also gives explicit (and biologically transparent) conditions:

    E0 stable  <=>  f1(s_in) < D1
    E1 stable  <=>  f2(x1)   < D2     (and E1 exists, i.e. f1(s_in) > D1)
    E2 stable  <=>  f3(y2)   < D3     (and E2 exists, i.e. f2(x2)   > D2)
    E* stable  <=>  Routh-Hurwitz conditions on the quartic characteristic
                    polynomial hold.

i.e. an upper trophic level invades exactly when its response at the lower
equilibrium exceeds its removal rate.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .equilibria import Equilibrium
from .model import Parameters, jacobian


@dataclass
class StabilityReport:
    name: str
    eigenvalues: np.ndarray
    stable: bool
    max_real_part: float
    has_complex_pair: bool

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        ev = ", ".join(f"{l.real:+.3f}{l.imag:+.3f}j" for l in self.eigenvalues)
        tag = "STABLE" if self.stable else "unstable"
        return f"<{self.name}: {tag} | max Re={self.max_real_part:+.4f} | eig=[{ev}]>"


def classify(eq: Equilibrium, p: Parameters,
             tol: float = 1e-9) -> StabilityReport:
    """Eigenvalue-based stability classification of an equilibrium."""
    if not np.all(np.isfinite(eq.state)):
        return StabilityReport(eq.name, np.array([]), False, np.nan, False)

    J = jacobian(eq.state, p)
    eigs = np.linalg.eigvals(J)
    max_re = float(np.max(eigs.real))
    has_pair = bool(np.any(np.abs(eigs.imag) > tol))
    return StabilityReport(eq.name, eigs, max_re < -tol, max_re, has_pair)


# ----------------------------------------------------------------------
# Routh-Hurwitz for a quartic  lambda^4 + b1 l^3 + b2 l^2 + b3 l + b4
# (the characteristic polynomial of the 4x4 Jacobian at E*).
# All roots have negative real part iff
#   b1>0, b3>0, b4>0, and  b1 b2 b3 > b3^2 + b1^2 b4.
# ----------------------------------------------------------------------
def routh_hurwitz_quartic(coeffs: np.ndarray) -> tuple[bool, dict]:
    """Apply the Routh-Hurwitz criterion to a monic quartic.

    Parameters
    ----------
    coeffs:
        ``[1, b1, b2, b3, b4]`` as returned by ``numpy.poly`` of the Jacobian
        (leading coefficient first).
    """
    c = np.asarray(coeffs, dtype=float)
    c = c / c[0]  # make monic
    _, b1, b2, b3, b4 = c

    cond1 = b1 > 0
    cond3 = b3 > 0
    cond4 = b4 > 0
    determinant = b1 * b2 * b3 - b3 ** 2 - b1 ** 2 * b4  # > 0 required
    cond_det = determinant > 0

    stable = bool(cond1 and cond3 and cond4 and cond_det)
    details = {
        "b1": b1, "b2": b2, "b3": b3, "b4": b4,
        "b1>0": cond1, "b3>0": cond3, "b4>0": cond4,
        "b1 b2 b3 - b3^2 - b1^2 b4": determinant, "RH_det>0": cond_det,
    }
    return stable, details


def routh_hurwitz_at(eq: Equilibrium, p: Parameters) -> tuple[bool, dict]:
    """Routh-Hurwitz stability of the Jacobian at ``eq`` (use for E*)."""
    J = jacobian(eq.state, p)
    coeffs = np.poly(J)  # characteristic polynomial, leading coeff first
    return routh_hurwitz_quartic(coeffs)


# ----------------------------------------------------------------------
# Analytic invasion conditions (Theorems 2.1-2.3).
# ----------------------------------------------------------------------
def analytic_conditions(p: Parameters) -> dict:
    """Closed-form existence/stability conditions for E0, E1, E2."""
    from .equilibria import prey_only, prey_predator

    out = {
        "E0_stable (f1(s_in) < D1)": p.f1(p.s_in) < p.D1,
        "E1_exists (f1(s_in) > D1)": p.f1(p.s_in) > p.D1,
    }

    e1 = prey_only(p)
    if e1.exists:
        x1 = e1.state[1]
        out["E1_stable (f2(x1) < D2)"] = p.f2(x1) < p.D2
        out["E2_exists (f2(x1) > D2)"] = p.f2(x1) > p.D2

    e2 = prey_predator(p)
    if e2.exists:
        y2 = e2.state[2]
        out["E2_stable (f3(y2) < D3)"] = p.f3(y2) < p.D3
        out["E*_exists (f3(y2) > D3)"] = p.f3(y2) > p.D3
    return out
