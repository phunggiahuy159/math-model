"""Food-chain chemostat model with distinct removal rates.

Reference implementation of

    M.M.A. El-Sheikh & S.A.A. Mahrouf,
    "Stability and bifurcation of a simple food chain in a chemostat with
    removal rates", Chaos, Solitons and Fractals 23 (2005) 1475-1489.
"""

from .model import Parameters, rhs, jacobian
from .equilibria import (
    Equilibrium,
    washout,
    prey_only,
    prey_predator,
    coexistence,
    all_coexistence,
    all_equilibria,
    residual,
)
from .stability import (
    StabilityReport,
    classify,
    routh_hurwitz_quartic,
    routh_hurwitz_at,
    analytic_conditions,
)
from .simulate import Trajectory, simulate
from .bifurcation import (
    SweepPoint,
    continue_coexistence,
    find_hopf,
    orbit_envelope,
)

__all__ = [
    "Parameters", "rhs", "jacobian",
    "Equilibrium", "washout", "prey_only", "prey_predator", "coexistence",
    "all_coexistence", "all_equilibria", "residual",
    "StabilityReport", "classify", "routh_hurwitz_quartic", "routh_hurwitz_at",
    "analytic_conditions",
    "Trajectory", "simulate",
    "SweepPoint", "continue_coexistence", "find_hopf", "orbit_envelope",
]
