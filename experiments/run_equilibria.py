"""Report all equilibria and their stability for a parameter set.

Run:  python -m experiments.run_equilibria
"""

from __future__ import annotations

import numpy as np

from foodchain import (
    Parameters,
    all_equilibria,
    analytic_conditions,
    classify,
    coexistence,
    residual,
    routh_hurwitz_at,
)


def main(p: Parameters | None = None) -> None:
    p = p or Parameters()
    np.set_printoptions(precision=5, suppress=True)

    print("=" * 70)
    print("Food-chain chemostat -- equilibria and local stability")
    print("=" * 70)
    print(f"Parameters: {p}\n")

    print("Equilibria (state = [S, x, y, z]):")
    for eq in all_equilibria(p):
        if eq.exists:
            rep = classify(eq, p)
            tag = "STABLE  " if rep.stable else "unstable"
            print(f"  {eq.name:3s} {tag}  state={eq.state}  "
                  f"maxRe={rep.max_real_part:+.4f}  res={residual(eq, p):.1e}")
        else:
            print(f"  {eq.name:3s} does not exist  ({eq.note})")

    print("\nRouth-Hurwitz test at E* (coexistence):")
    e_star = coexistence(p)
    if e_star.exists:
        ok, details = routh_hurwitz_at(e_star, p)
        print(f"  E* Routh-Hurwitz stable: {ok}")
        for k, v in details.items():
            print(f"    {k}: {v}")
    else:
        print("  E* does not exist for these parameters.")

    print("\nAnalytic invasion / stability conditions (Theorems 2.1-2.3):")
    for k, v in analytic_conditions(p).items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
