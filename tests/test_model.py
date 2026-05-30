"""Tests for the model RHS, Jacobian and basic invariance properties."""

import numpy as np

from foodchain import Parameters, jacobian, rhs, simulate


def test_response_functions_monotone_and_zero_at_origin():
    p = Parameters()
    assert p.f1(0) == 0 and p.f2(0) == 0 and p.f3(0) == 0
    assert p.f1(1.0) > p.f1(0.5) > 0          # increasing
    assert p.df1(1.0) > 0 and p.df2(1.0) > 0 and p.df3(1.0) > 0


def test_jacobian_matches_finite_difference():
    p = Parameters()
    state = np.array([0.6, 0.3, 0.2, 0.4])
    J = jacobian(state, p)

    eps = 1e-6
    J_fd = np.zeros((4, 4))
    for j in range(4):
        e = np.zeros(4)
        e[j] = eps
        J_fd[:, j] = (rhs(0, state + e, p) - rhs(0, state - e, p)) / (2 * eps)
    assert np.allclose(J, J_fd, atol=1e-6)


def test_nonnegativity_is_invariant():
    """Boundary faces are invariant: if a species starts at 0 it stays >= 0."""
    p = Parameters()
    traj = simulate(p, y0=(0.5, 0.3, 0.0, 0.0), t_end=200, n_points=2000)
    assert np.all(traj.x >= -1e-8)
    assert np.all(np.abs(traj.y) < 1e-6)   # y stays at 0
    assert np.all(np.abs(traj.z) < 1e-6)   # z stays at 0


def test_solutions_remain_bounded_and_nonnegative():
    p = Parameters()
    traj = simulate(p, t_end=400, n_points=4000)
    states = traj.states
    assert np.all(states > -1e-6)          # invariance of non-negativity
    assert np.all(states < 1e3)            # dissipativity / boundedness


def test_washout_when_no_growth():
    """If f1(s_in) < D1 no species can grow: everything washes out to E0."""
    p = Parameters(m1=0.5)  # weak prey growth -> f1(1) < D1
    assert p.f1(p.s_in) < p.D1
    traj = simulate(p, t_end=400, n_points=4000)
    end = traj.states[:, -1]
    assert np.allclose(end, [p.s_in, 0, 0, 0], atol=1e-3)
