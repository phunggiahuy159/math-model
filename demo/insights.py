"""Automated "does it match the paper?" checks.

Each :class:`Check` compares a number the demo just computed against a
qualitative claim from El-Sheikh & Mahrouf (2005), so the user can see the
simulation reproducing the theorems rather than taking them on faith.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from foodchain import Parameters, classify, coexistence, jacobian
from foodchain.equilibria import prey_only, prey_predator, washout

from . import compute


@dataclass
class Check:
    title: str
    passed: bool
    detail: str
    reference: str  # which theorem / equation in the paper


def _stable(eq, p) -> Optional[bool]:
    if not eq.exists:
        return None
    return classify(eq, p).stable


def paper_checks(p: Parameters) -> list[Check]:
    """Run every consistency check for the current parameter set."""
    checks: list[Check] = []

    # --- Theorem 2.2: E0 (washout) stable  <=>  f1(s_in) < D1 -------------
    e0 = washout(p)
    cond = p.f1(p.s_in) < p.D1
    obs = _stable(e0, p)
    checks.append(Check(
        "Washout E0 stability",
        passed=(obs == cond),
        detail=(f"f1(s_in)={p.f1(p.s_in):.3f} vs D1={p.D1:.3f}  =>  "
                f"theory predicts E0 {'stable' if cond else 'unstable'}; "
                f"eigenvalues say {'stable' if obs else 'unstable'}."),
        reference="Theorem 2.2",
    ))

    # --- Theorem 2.4 region: E1 invadable by y  <=>  f2(x1) > D2 ----------
    e1 = prey_only(p)
    if e1.exists:
        x1 = e1.state[1]
        cond = p.f2(x1) < p.D2          # E1 stable in y-direction
        obs = _stable(e1, p)
        checks.append(Check(
            "Prey-only E1 stability",
            passed=(obs == cond),
            detail=(f"f2(x1)={p.f2(x1):.3f} vs D2={p.D2:.3f}  =>  "
                    f"theory predicts E1 {'stable' if cond else 'unstable'}; "
                    f"eigenvalues say {'stable' if obs else 'unstable'}."),
            reference="Theorem 2.4 / 2.5",
        ))

    # --- Theorem 2.8: E2 stable  <=>  f3(y2) < D3 ------------------------
    e2 = prey_predator(p)
    if e2.exists:
        y2 = e2.state[2]
        cond = p.f3(y2) < p.D3
        obs = _stable(e2, p)
        checks.append(Check(
            "Prey+predator E2 stability",
            passed=(obs == cond),
            detail=(f"f3(y2)={p.f3(y2):.3f} vs D3={p.D3:.3f}  =>  "
                    f"theory predicts E2 {'stable' if cond else 'unstable'}; "
                    f"eigenvalues say {'stable' if obs else 'unstable'}."),
            reference="Theorem 2.8 / 2.9",
        ))

    # --- E* existence ladder: needs f3(y2) > D3 (predator z can invade) --
    if e2.exists:
        y2 = e2.state[2]
        cond = p.f3(y2) > p.D3
        es = coexistence(p)
        checks.append(Check(
            "Coexistence E* existence",
            passed=(es.exists == cond),
            detail=(f"f3(y2)={p.f3(y2):.3f} vs D3={p.D3:.3f}  =>  z invades "
                    f"when f3(y2) > D3; E* "
                    f"{'exists' if es.exists else 'absent'}."),
            reference="Section 3 (persistence)",
        ))

    # --- Routh-Hurwitz agrees with raw eigenvalues at E* -----------------
    es = coexistence(p)
    if es.exists:
        from foodchain import routh_hurwitz_at
        rh, _ = routh_hurwitz_at(es, p)
        eig = classify(es, p).stable
        checks.append(Check(
            "Routh-Hurwitz vs eigenvalues at E*",
            passed=(rh == eig),
            detail=(f"Routh-Hurwitz says {'stable' if rh else 'unstable'}; "
                    f"direct eigenvalues say {'stable' if eig else 'unstable'} "
                    "(these must agree)."),
            reference="Section 2 (quartic RH criterion)",
        ))

    return checks


def hopf_story(p: Parameters, param: str = "s_in",
               lo: float = 1.0, hi: float = 6.0) -> dict:
    """Narrative + numbers describing the Hopf bifurcation, if one exists.

    Returns a dict the UI renders: whether a Hopf was found, its location,
    frequency/period, and an enrichment-paradox interpretation.
    """
    h = compute.hopf_point(p, param, lo, hi, n=400)
    if h is None:
        return {"found": False,
                "message": (f"No Hopf bifurcation of E* found while sweeping "
                            f"{param} on [{lo}, {hi}].")}
    return {
        "found": True,
        "param": param,
        "value": h["value"],
        "frequency": h["frequency"],
        "period": h["period"],
        "stable_to_unstable": h["from_stable_to_unstable"],
        "equilibrium": h["equilibrium"],
        "message": (
            f"As {param} increases through {h['value']:.3f}, a complex-conjugate "
            f"eigenvalue pair of J(E*) crosses the imaginary axis: E* loses "
            f"stability and a limit cycle is born (angular frequency "
            f"{h['frequency']:.3f}, period {h['period']:.2f}). This is the "
            f"paradox of enrichment -- raising the nutrient supply "
            f"destabilizes the coexistence steady state."),
        "reference": "Section 4 (Hopf-Andronov-Poincare bifurcation)",
    }


def overall_verdict(checks: list[Check]) -> tuple[bool, str]:
    """Aggregate all checks into a single pass/fail headline."""
    if not checks:
        return True, "No applicable checks for this parameter set."
    n_pass = sum(c.passed for c in checks)
    ok = n_pass == len(checks)
    return ok, f"{n_pass}/{len(checks)} consistency checks match the paper."
