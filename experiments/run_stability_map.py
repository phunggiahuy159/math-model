"""Which equilibrium is the global attractor? -- a successive-invasion map.

Sweeps the top-predator removal rate D3 (and, separately, the input s_in) and
records, for every value, which of E0/E1/E2/E* exists and is locally stable.
This illustrates Theorems 2.1-2.4: each trophic level establishes only when
its response at the equilibrium below exceeds its own removal rate.

Run:  python -m experiments.run_stability_map
"""

from __future__ import annotations

import os

import numpy as np

from foodchain import Parameters, all_equilibria, classify
from foodchain.plots import plt

FIGDIR = os.path.join(os.path.dirname(__file__), "..", "figures")
_NAMES = ["E0", "E1", "E2", "E*"]


def stable_index(p: Parameters) -> int:
    """Index (0..3) of the highest existing & locally stable equilibrium, else -1."""
    best = -1
    for i, eq in enumerate(all_equilibria(p)):
        if eq.exists and classify(eq, p).stable:
            best = i
    return best


def main() -> None:
    os.makedirs(FIGDIR, exist_ok=True)
    base = Parameters()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # (a) sweep D3
    d3 = np.linspace(0.05, 2.5, 300)
    idx = [stable_index(base.with_(D3=float(v))) for v in d3]
    axes[0].step(d3, idx, where="mid", color="#1f77b4")
    axes[0].set_xlabel("D3 (top-predator removal rate)")
    axes[0].set_title("Stable equilibrium vs D3")

    # (b) sweep s_in
    s = np.linspace(0.2, 8.0, 300)
    idx2 = [stable_index(base.with_(s_in=float(v))) for v in s]
    axes[1].step(s, idx2, where="mid", color="#d62728")
    axes[1].set_xlabel("s_in (input concentration)")
    axes[1].set_title("Stable equilibrium vs s_in")

    for ax in axes:
        ax.set_yticks([-1, 0, 1, 2, 3])
        ax.set_yticklabels(["none/limit cycle"] + _NAMES)
        ax.grid(alpha=0.3)

    fig.suptitle("Successive-invasion structure (locally stable equilibrium)")
    fig.tight_layout()
    out = os.path.join(FIGDIR, "stability_map.png")
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
