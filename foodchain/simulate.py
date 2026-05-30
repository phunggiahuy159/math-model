"""Time integration of system (1.2) using ``scipy.integrate.solve_ivp``."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp

from .model import Parameters, rhs


@dataclass
class Trajectory:
    t: np.ndarray
    S: np.ndarray
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray

    @property
    def states(self) -> np.ndarray:
        """(4, N) array stacking S, x, y, z."""
        return np.vstack([self.S, self.x, self.y, self.z])

    def tail(self, fraction: float = 0.5) -> "Trajectory":
        """Return the last ``fraction`` of the trajectory (transient removed)."""
        n = len(self.t)
        k = int(n * (1.0 - fraction))
        return Trajectory(self.t[k:], self.S[k:], self.x[k:], self.y[k:], self.z[k:])


def simulate(p: Parameters,
             y0=(0.5, 0.3, 0.2, 0.1),
             t_end: float = 400.0,
             n_points: int = 4000,
             rtol: float = 1e-8,
             atol: float = 1e-10) -> Trajectory:
    """Integrate (1.2) from initial state ``y0`` over ``[0, t_end]``.

    Uses the stiff-capable ``LSODA`` integrator on a dense output grid of
    ``n_points`` samples. In the post-Hopf regime the limit cycle is a
    relaxation oscillation (the top predator nearly vanishes each period), so
    if ``LSODA`` reports failure we retry with the implicit ``Radau`` method.
    """
    t_eval = np.linspace(0.0, t_end, n_points)

    def _run(method):
        return solve_ivp(
            fun=lambda t, s: rhs(t, s, p),
            t_span=(0.0, t_end),
            y0=np.asarray(y0, dtype=float),
            method=method,
            t_eval=t_eval,
            rtol=rtol,
            atol=atol,
        )

    sol = _run("LSODA")
    if not sol.success:
        sol = _run("Radau")
    if not sol.success:  # pragma: no cover - defensive
        raise RuntimeError(f"integration failed: {sol.message}")

    S, x, y, z = sol.y
    return Trajectory(sol.t, S, x, y, z)
