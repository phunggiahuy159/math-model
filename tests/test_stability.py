"""Tests for stability classification, Routh-Hurwitz, and Hopf detection."""

import numpy as np

from foodchain import (
    Parameters,
    classify,
    coexistence,
    find_hopf,
    routh_hurwitz_at,
    routh_hurwitz_quartic,
    washout,
)


def test_washout_stability_matches_analytic_condition():
    # Stable washout: weak prey growth f1(s_in) < D1
    p_stable = Parameters(m1=0.5)
    assert classify(washout(p_stable), p_stable).stable
    # Unstable washout: strong prey growth f1(s_in) > D1
    p_unstable = Parameters(m1=5.0)
    assert not classify(washout(p_unstable), p_unstable).stable


def test_routh_hurwitz_agrees_with_eigenvalues_at_Estar():
    for s_in in [1.0, 1.5, 2.0, 3.0, 4.0]:
        p = Parameters(s_in=s_in)
        es = coexistence(p)
        if not es.exists:
            continue
        rh_stable, _ = routh_hurwitz_at(es, p)
        eig_stable = classify(es, p).stable
        assert rh_stable == eig_stable, f"mismatch at s_in={s_in}"


def test_routh_hurwitz_known_stable_polynomial():
    # (l+1)(l+2)(l+3)(l+4) -> all roots negative -> stable
    coeffs = np.poly([-1, -2, -3, -4])
    assert routh_hurwitz_quartic(coeffs)[0]


def test_routh_hurwitz_known_unstable_polynomial():
    # a positive real root -> unstable
    coeffs = np.poly([1.0, -2, -3, -4])
    assert not routh_hurwitz_quartic(coeffs)[0]


def test_hopf_bifurcation_exists_in_s_in():
    p = Parameters()
    hopf = find_hopf(p, "s_in", 1.0, 6.0, n=300)
    assert hopf is not None
    assert 1.5 < hopf["value"] < 4.0
    assert hopf["frequency"] > 0          # genuine oscillatory crossing
    assert hopf["from_stable_to_unstable"]


def test_estar_stable_below_hopf_unstable_above():
    p = Parameters()
    assert classify(coexistence(p.with_(s_in=1.0)), p.with_(s_in=1.0)).stable
    pv = p.with_(s_in=4.0)
    assert not classify(coexistence(pv), pv).stable
