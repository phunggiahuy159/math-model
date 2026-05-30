"""Hopf bifurcation study in the input concentration s_in (Section 4).

Produces:
  * eigenvalue diagram: max Re(lambda) of J(E*) vs s_in (zero crossing = Hopf)
  * numerical bifurcation diagram: long-time min/max of y vs s_in
    (a min/max gap appearing past the Hopf point reveals the limit cycle)
  * prints the detected Hopf point, frequency and period.

Run:  python -m experiments.run_bifurcation
"""

from __future__ import annotations

import os

import numpy as np

from foodchain import Parameters, continue_coexistence, find_hopf, orbit_envelope
from foodchain.plots import plot_bifurcation_diagram, plot_eigenvalue_sweep

FIGDIR = os.path.join(os.path.dirname(__file__), "..", "figures")


def main() -> None:
    os.makedirs(FIGDIR, exist_ok=True)
    p = Parameters()
    param = "s_in"
    lo, hi = 1.0, 5.0

    # 1) Eigenvalue continuation of E*.
    values = np.linspace(lo, hi, 400)
    sweep = continue_coexistence(p, param, values)
    plot_eigenvalue_sweep(
        sweep, param,
        r"Leading eigenvalue of $E^*$ vs input $s_{in}$",
        os.path.join(FIGDIR, "hopf_eigenvalues.png"),
    )

    # 2) Locate the Hopf point.
    hopf = find_hopf(p, param, lo, hi, n=400)
    if hopf is not None:
        print("Hopf bifurcation detected:")
        print(f"  {param} = {hopf['value']:.5f}")
        print(f"  E*       = {np.round(hopf['equilibrium'], 5)}")
        print(f"  freq     = {hopf['frequency']:.5f} (period {hopf['period']:.3f})")
        print(f"  stable -> unstable as {param} increases: "
              f"{hopf['from_stable_to_unstable']}")
    else:
        print("No Hopf bifurcation found on the scanned range.")

    # 3) Numerical bifurcation diagram (limit-cycle amplitude of y).
    env_values = np.linspace(lo, hi, 60)
    env = orbit_envelope(p, param, env_values, component=2,
                         t_end=1500.0, tail=0.35)
    plot_bifurcation_diagram(
        env, param,
        r"Bifurcation diagram: long-time min/max of $y$ vs $s_{in}$",
        os.path.join(FIGDIR, "bifurcation_diagram_y.png"),
    )
    print(f"\nWrote figures to {os.path.abspath(FIGDIR)}")


if __name__ == "__main__":
    main()
