"""Plotting helpers (matplotlib, non-interactive Agg backend)."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # safe for headless / file output
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .simulate import Trajectory  # noqa: E402

_LABELS = ["S (nutrient)", "x (prey)", "y (predator 1)", "z (predator 2)"]
_COLORS = ["#1f77b4", "#2ca02c", "#ff7f0e", "#d62728"]


def plot_timeseries(traj: Trajectory, title: str, path: str) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    for arr, lab, col in zip(traj.states, _LABELS, _COLORS):
        ax.plot(traj.t, arr, label=lab, color=col, lw=1.6)
    ax.set_xlabel("time t")
    ax.set_ylabel("scaled concentration")
    ax.set_title(title)
    ax.legend(loc="best")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_phase3d(traj: Trajectory, title: str, path: str,
                 components=(1, 2, 3)) -> None:
    """3-D phase portrait of three chosen components (default x, y, z)."""
    states = traj.states
    i, j, k = components
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(states[i], states[j], states[k], lw=0.8, color="#6a3d9a")
    ax.scatter(states[i, -1], states[j, -1], states[k, -1],
               color="red", s=30, label="endpoint")
    ax.set_xlabel(_LABELS[i])
    ax.set_ylabel(_LABELS[j])
    ax.set_zlabel(_LABELS[k])
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_eigenvalue_sweep(sweep_points, param: str, title: str,
                          path: str) -> None:
    """Plot max Re(eigenvalue) of E* vs the swept parameter."""
    vals = [pt.value for pt in sweep_points if pt.exists]
    mre = [pt.max_real_part for pt in sweep_points if pt.exists]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(vals, mre, color="#1f77b4", lw=1.8)
    ax.axhline(0.0, color="k", lw=1.0, ls="--")
    ax.set_xlabel(param)
    ax.set_ylabel(r"max Re($\lambda$) of $J(E^*)$")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_bifurcation_diagram(envelope: dict, param: str, title: str,
                             path: str) -> None:
    labels = ["S", "x", "y", "z"]
    comp = envelope["component"]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(envelope["values"], envelope["max"], ".", ms=3,
            color="#d62728", label="max")
    ax.plot(envelope["values"], envelope["min"], ".", ms=3,
            color="#1f77b4", label="min")
    ax.set_xlabel(param)
    ax.set_ylabel(f"{labels[comp]} (long-time min/max)")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
