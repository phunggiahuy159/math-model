"""Integrate the system and save time-series + 3-D phase portraits.

Demonstrates two regimes with the default parameters:
  * s_in = 1.0  -> convergence to a stable coexistence equilibrium E*
  * s_in = 4.0  -> a stable limit cycle born from the Hopf bifurcation

Run:  python -m experiments.run_timeseries
Figures are written to ./figures/.
"""

from __future__ import annotations

import os

from foodchain import Parameters, simulate
from foodchain.plots import plot_phase3d, plot_timeseries

FIGDIR = os.path.join(os.path.dirname(__file__), "..", "figures")


def main() -> None:
    os.makedirs(FIGDIR, exist_ok=True)
    base = Parameters()

    cases = [
        (base.with_(s_in=1.0), "stable_equilibrium", "s_in = 1.0 (stable E*)", 400.0),
        (base.with_(s_in=3.5), "limit_cycle", "s_in = 3.5 (limit cycle after Hopf)", 3000.0),
    ]

    for p, slug, title, t_end in cases:
        traj = simulate(p, t_end=t_end, n_points=int(t_end * 5))
        ts_path = os.path.join(FIGDIR, f"timeseries_{slug}.png")
        ph_path = os.path.join(FIGDIR, f"phase3d_{slug}.png")
        plot_timeseries(traj, f"Time series -- {title}", ts_path)
        # show the attractor on the tail (transient removed) for the phase plot
        plot_phase3d(traj.tail(0.5), f"Phase portrait (x,y,z) -- {title}", ph_path)
        print(f"[{slug}] wrote {ts_path} and {ph_path}")


if __name__ == "__main__":
    main()
