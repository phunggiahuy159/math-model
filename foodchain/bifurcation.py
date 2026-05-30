"""Hopf bifurcation analysis (Section 4).

The paper takes the input nutrient concentration as the bifurcation parameter
and shows that the coexistence equilibrium E* can lose stability through a Hopf
bifurcation -- a complex-conjugate pair of eigenvalues of the Jacobian crosses
the imaginary axis -- giving rise to sustained oscillations (a limit cycle).

This module provides:

* ``continue_coexistence`` -- track E* and its leading eigenvalue while sweeping
  any single parameter;
* ``find_hopf`` -- locate the parameter value where the dominant complex pair
  crosses zero real part;
* ``orbit_envelope`` -- long-time min/max of each state to expose limit-cycle
  amplitude in a numerical bifurcation diagram.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.optimize import brentq

from .equilibria import coexistence
from .model import Parameters, jacobian
from .simulate import simulate


@dataclass
class SweepPoint:
    value: float
    state: np.ndarray         # E* components (nan if it does not exist)
    exists: bool
    max_real_part: float      # max Re(eigenvalue) of J(E*)
    leading_imag: float       # |Im| of the eigenvalue attaining max Re


def _leading_eigeninfo(state: np.ndarray, p: Parameters):
    eigs = np.linalg.eigvals(jacobian(state, p))
    i = int(np.argmax(eigs.real))
    return float(eigs[i].real), float(abs(eigs[i].imag))


def continue_coexistence(p: Parameters,
                         param: str,
                         values: np.ndarray) -> list[SweepPoint]:
    """Track E* and its leading eigenvalue as ``param`` ranges over ``values``.

    ``param`` is any field of :class:`~foodchain.model.Parameters`, e.g.
    ``"s_in"``, ``"m3"``, ``"D3"``.
    """
    points: list[SweepPoint] = []
    guess = None
    for v in values:
        pv = p.with_(**{param: float(v)})
        eq = coexistence(pv, guess=guess)
        if eq.exists:
            guess = eq.state[:2]  # warm-start next solve with (S*, x*)
            mre, mim = _leading_eigeninfo(eq.state, pv)
            points.append(SweepPoint(float(v), eq.state, True, mre, mim))
        else:
            points.append(SweepPoint(float(v), np.array([np.nan] * 4),
                                     False, np.nan, np.nan))
    return points


def find_hopf(p: Parameters,
              param: str,
              lo: float,
              hi: float,
              n: int = 400) -> Optional[dict]:
    """Locate a Hopf bifurcation of E* in ``param`` on ``[lo, hi]``.

    Scans for a sign change of ``max Re(eigenvalue)`` of J(E*) where the
    crossing eigenvalue is part of a genuine complex pair (nonzero imaginary
    part), then refines the location with a bracketed root find.
    Returns ``None`` if no such crossing is found.
    """
    values = np.linspace(lo, hi, n)
    pts = continue_coexistence(p, param, values)

    def signed_max_re(v: float) -> float:
        eq = coexistence(p.with_(**{param: float(v)}))
        if not eq.exists:
            return np.nan
        mre, _ = _leading_eigeninfo(eq.state, p.with_(**{param: float(v)}))
        return mre

    for a, b in zip(pts[:-1], pts[1:]):
        if not (a.exists and b.exists):
            continue
        if np.isnan(a.max_real_part) or np.isnan(b.max_real_part):
            continue
        if a.max_real_part * b.max_real_part < 0:
            # require the crossing to involve oscillatory (complex) modes
            if max(a.leading_imag, b.leading_imag) < 1e-6:
                continue
            v_star = brentq(signed_max_re, a.value, b.value, xtol=1e-10)
            pv = p.with_(**{param: v_star})
            eq = coexistence(pv)
            eigs = np.linalg.eigvals(jacobian(eq.state, pv))
            i = int(np.argmax(eigs.real))
            return {
                "param": param,
                "value": v_star,
                "equilibrium": eq.state,
                "frequency": float(abs(eigs[i].imag)),  # Hopf angular freq
                "period": float(2 * np.pi / abs(eigs[i].imag))
                if abs(eigs[i].imag) > 0 else np.inf,
                "from_stable_to_unstable": a.max_real_part < 0,
            }
    return None


def orbit_envelope(p: Parameters,
                   param: str,
                   values: np.ndarray,
                   component: int = 2,
                   t_end: float = 1500.0,
                   tail: float = 0.4,
                   y0=(0.5, 0.3, 0.2, 0.1)) -> dict:
    """Numerical bifurcation diagram: long-time min/max of one state component.

    For each parameter value the system is integrated, the transient is
    discarded and the min/max of ``component`` (0=S, 1=x, 2=y, 3=z) on the
    remaining tail is recorded. A collapsing min==max indicates a stable
    equilibrium; a gap indicates a limit cycle.
    """
    mins, maxs = [], []
    for v in values:
        pv = p.with_(**{param: float(v)})
        traj = simulate(pv, y0=y0, t_end=t_end, n_points=int(t_end * 4))
        tail_states = traj.tail(tail).states[component]
        mins.append(float(np.min(tail_states)))
        maxs.append(float(np.max(tail_states)))
    return {"values": np.asarray(values),
            "min": np.asarray(mins),
            "max": np.asarray(maxs),
            "component": component}
