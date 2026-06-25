"""Cached, UI-facing computations built on the ``foodchain`` package.

Everything here is a thin, *deterministic* wrapper around the analysis code so
the Streamlit layer can stay declarative. Functions return plain Python / numpy
data (no Plotly, no Streamlit) which keeps them unit-testable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from foodchain import (
    Parameters,
    all_equilibria,
    analytic_conditions,
    classify,
    coexistence,
    continue_coexistence,
    find_hopf,
    jacobian,
    simulate,
)
from foodchain.equilibria import Equilibrium

# Field order used everywhere in the UI (sliders, tables, CLI overrides).
PARAM_FIELDS = ["m1", "m2", "m3", "a1", "a2", "a3", "D1", "D2", "D3", "s_in"]

STATE_LABELS = ["S (nutrient)", "x (prey)", "y (predator 1)", "z (predator 2)"]
STATE_KEYS = ["S", "x", "y", "z"]


def make_params(values: dict) -> Parameters:
    """Build a :class:`Parameters` from a dict of field -> value."""
    return Parameters(**{k: float(values[k]) for k in PARAM_FIELDS if k in values})


# ----------------------------------------------------------------------
# Equilibria + stability summary
# ----------------------------------------------------------------------
@dataclass
class EqRow:
    name: str
    exists: bool
    state: Optional[np.ndarray]
    stable: Optional[bool]
    max_real_part: Optional[float]
    has_complex_pair: Optional[bool]
    note: str


def equilibria_table(p: Parameters) -> list[EqRow]:
    """Existence + local stability of E0, E1, E2, E* for the current params."""
    rows: list[EqRow] = []
    for eq in all_equilibria(p):
        if eq.exists:
            rep = classify(eq, p)
            rows.append(EqRow(eq.name, True, eq.state, rep.stable,
                              rep.max_real_part, rep.has_complex_pair, eq.note))
        else:
            rows.append(EqRow(eq.name, False, None, None, None, None, eq.note))
    return rows


def coexistence_eigenvalues(p: Parameters) -> Optional[np.ndarray]:
    """Eigenvalues of J(E*) (or None if E* does not exist)."""
    eq = coexistence(p)
    if not eq.exists:
        return None
    return np.linalg.eigvals(jacobian(eq.state, p))


# ----------------------------------------------------------------------
# Trajectory
# ----------------------------------------------------------------------
def run_trajectory(p: Parameters, y0, t_end: float, n_points: int):
    """Integrate the system; returns the ``foodchain.Trajectory``."""
    return simulate(p, y0=tuple(y0), t_end=t_end, n_points=n_points)


def attractor_summary(traj, tail: float = 0.4) -> dict:
    """Classify the long-time behaviour of a trajectory.

    Distinguishes a *decaying* transient (settling to an equilibrium) from a
    *sustained* oscillation (a limit cycle) by comparing the swing in the first
    vs the second half of the tail window: a damped approach to E* keeps
    shrinking, a limit cycle does not. Returns per-component min/max plus the
    verdict and an estimated period.
    """
    states = traj.tail(tail).states  # (4, N)
    n = states.shape[1]
    half = n // 2
    first, second = states[:, :half], states[:, half:]

    span_first = first.max(axis=1) - first.min(axis=1)
    span_second = second.max(axis=1) - second.min(axis=1)
    scale = np.maximum(states.mean(axis=1), 1e-9)
    rel = span_second / scale

    # Sustained iff the late-window swing is non-trivial AND not still shrinking
    # appreciably (a decaying transient has span_second << span_first).
    nontrivial = rel > 0.02
    not_decaying = span_second > 0.5 * span_first
    oscillating = bool(np.any(nontrivial & not_decaying))

    period = _estimate_period(traj) if oscillating else None
    return {
        "min": states.min(axis=1),
        "max": states.max(axis=1),
        "relative_span": rel,
        "oscillating": oscillating,
        "period": period,
    }


def _estimate_period(traj, component: int = 2) -> Optional[float]:
    """Estimate the limit-cycle period from upward mean-crossings of one state."""
    tail = traj.tail(0.5)
    t = tail.t
    s = tail.states[component]
    s = s - s.mean()
    crossings = np.where((s[:-1] < 0) & (s[1:] >= 0))[0]
    if len(crossings) < 2:
        return None
    times = t[crossings]
    return float(np.mean(np.diff(times)))


# ----------------------------------------------------------------------
# Bifurcation continuation in a chosen parameter
# ----------------------------------------------------------------------
def eigenvalue_curve(p: Parameters, param: str, lo: float, hi: float,
                     n: int = 240) -> dict:
    """max Re(lambda) of J(E*) and components of E* as ``param`` varies."""
    values = np.linspace(lo, hi, n)
    pts = continue_coexistence(p, param, values)
    vals = np.array([pt.value for pt in pts])
    exists = np.array([pt.exists for pt in pts])
    max_re = np.array([pt.max_real_part if pt.exists else np.nan for pt in pts])
    leading_imag = np.array([pt.leading_imag if pt.exists else np.nan
                             for pt in pts])
    states = np.array([pt.state if pt.exists else [np.nan] * 4 for pt in pts])
    return {
        "param": param, "values": vals, "exists": exists,
        "max_real_part": max_re, "leading_imag": leading_imag, "states": states,
    }


def hopf_point(p: Parameters, param: str, lo: float, hi: float,
               n: int = 400) -> Optional[dict]:
    """Locate a Hopf bifurcation of E* in ``param`` on ``[lo, hi]``."""
    return find_hopf(p, param, lo, hi, n=n)


def orbit_diagram(p: Parameters, param: str, lo: float, hi: float,
                  n: int = 30, component: int = 2, t_end: float = 800.0,
                  tail: float = 0.4, y0=(0.5, 0.3, 0.2, 0.1)) -> dict:
    """Numerical bifurcation diagram: long-time min/max of a state component.

    Re-implemented here (rather than calling ``foodchain.orbit_envelope``) so
    we can return *both* min/max and a flag of whether each value oscillates,
    which the UI uses to colour stable branches vs limit-cycle envelopes.
    """
    values = np.linspace(lo, hi, n)
    mins, maxs, osc = [], [], []
    for v in values:
        pv = p.with_(**{param: float(v)})
        try:
            traj = simulate(pv, y0=tuple(y0), t_end=t_end,
                            n_points=int(t_end * 4))
            tail_states = traj.tail(tail).states[component]
            mn, mx = float(np.min(tail_states)), float(np.max(tail_states))
            rel = (mx - mn) / max(abs(np.mean(tail_states)), 1e-9)
            mins.append(mn)
            maxs.append(mx)
            osc.append(rel > 0.02)
        except RuntimeError:
            mins.append(np.nan)
            maxs.append(np.nan)
            osc.append(False)
    return {
        "param": param, "values": values, "component": component,
        "min": np.array(mins), "max": np.array(maxs),
        "oscillating": np.array(osc),
    }


def analytic(p: Parameters) -> dict:
    """Closed-form invasion/stability conditions (Theorems 2.1-2.3)."""
    return analytic_conditions(p)
