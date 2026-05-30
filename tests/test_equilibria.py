"""Tests that computed equilibria satisfy the equilibrium equations
and the existence conditions of Section 2."""

import numpy as np

from foodchain import (
    Parameters,
    all_equilibria,
    coexistence,
    prey_only,
    prey_predator,
    residual,
    washout,
)


def test_all_equilibria_are_actual_equilibria():
    p = Parameters()
    for eq in all_equilibria(p):
        if eq.exists:
            assert residual(eq, p) < 1e-7, f"{eq.name} not a true equilibrium"


def test_equilibrium_defining_relations():
    p = Parameters()
    e1 = prey_only(p)
    e2 = prey_predator(p)
    es = coexistence(p)

    # f1(S1) = D1
    assert abs(p.f1(e1.state[0]) - p.D1) < 1e-9
    # f2(x2) = D2
    assert abs(p.f2(e2.state[1]) - p.D2) < 1e-9
    # f3(y*) = D3
    assert abs(p.f3(es.state[2]) - p.D3) < 1e-9


def test_washout_always_exists_at_s_in():
    p = Parameters(s_in=2.3)
    e0 = washout(p)
    assert e0.exists
    assert np.allclose(e0.state, [2.3, 0, 0, 0])


def test_existence_conditions_consistent():
    p = Parameters()
    # E1 exists requires f1(s_in) > D1
    assert prey_only(p).exists == (p.f1(p.s_in) > p.D1)


def test_no_coexistence_when_top_predator_cannot_grow():
    # m3 <= D3 makes f3(y) = D3 unsolvable -> no E*
    p = Parameters(m3=0.4, D3=0.5)
    assert not coexistence(p).exists


def test_positive_coexistence_components():
    p = Parameters()
    es = coexistence(p)
    assert es.exists
    assert np.all(es.state > 0)
