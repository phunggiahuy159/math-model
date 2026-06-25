"""Plotly figures and animations for the demo.

All functions take already-computed numpy data (from :mod:`demo.compute`) and
return ``plotly.graph_objects.Figure`` objects. No Streamlit imports here.
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from .compute import STATE_KEYS, STATE_LABELS

# Consistent colour per state variable across every figure.
COLORS = {
    "S": "#1f77b4",   # nutrient
    "x": "#2ca02c",   # prey
    "y": "#ff7f0e",   # predator 1
    "z": "#d62728",   # predator 2
}
_ORDER = ["S", "x", "y", "z"]


def time_series(traj, title: str) -> go.Figure:
    """Static time-series of all four state variables."""
    fig = go.Figure()
    for key, label in zip(STATE_KEYS, STATE_LABELS):
        fig.add_trace(go.Scatter(
            x=traj.t, y=getattr(traj, key), mode="lines",
            name=label, line=dict(color=COLORS[key], width=2),
        ))
    fig.update_layout(
        title=title, xaxis_title="time t",
        yaxis_title="scaled concentration",
        legend=dict(orientation="h", y=1.12), margin=dict(l=10, r=10, t=60, b=10),
        height=420,
    )
    return fig


def time_series_animated(traj, title: str, n_frames: int = 60) -> go.Figure:
    """Animated time-series: a growing trace with a Play button.

    The animation reveals the transient settling onto its attractor; for a
    limit cycle the curve keeps sweeping the same band, for an equilibrium it
    flattens out.
    """
    t = traj.t
    n = len(t)
    idx = np.linspace(2, n, n_frames).astype(int)

    base = []
    for key, label in zip(STATE_KEYS, STATE_LABELS):
        base.append(go.Scatter(
            x=t[: idx[0]], y=getattr(traj, key)[: idx[0]], mode="lines",
            name=label, line=dict(color=COLORS[key], width=2)))
    fig = go.Figure(data=base)

    frames = []
    for k in idx:
        frames.append(go.Frame(data=[
            go.Scatter(x=t[:k], y=getattr(traj, key)[:k])
            for key in STATE_KEYS
        ]))
    fig.frames = frames

    ymax = max(getattr(traj, k).max() for k in STATE_KEYS) * 1.05
    fig.update_layout(
        title=title, xaxis_title="time t", yaxis_title="scaled concentration",
        xaxis=dict(range=[t.min(), t.max()]), yaxis=dict(range=[0, ymax]),
        legend=dict(orientation="h", y=1.12), height=420,
        margin=dict(l=10, r=10, t=60, b=10),
        updatemenus=[dict(
            type="buttons", showactive=False, x=0.0, y=-0.18, xanchor="left",
            buttons=[
                dict(label="Play", method="animate",
                     args=[None, dict(frame=dict(duration=40, redraw=True),
                                      fromcurrent=True,
                                      transition=dict(duration=0))]),
                dict(label="Pause", method="animate",
                     args=[[None], dict(frame=dict(duration=0, redraw=False),
                                        mode="immediate")]),
            ])],
    )
    return fig


def phase_portrait_3d(traj, title: str, equilibrium=None,
                      components=(1, 2, 3)) -> go.Figure:
    """3-D phase portrait of three chosen components (default x, y, z)."""
    states = traj.states
    i, j, k = components
    keys = [_ORDER[c] for c in components]
    fig = go.Figure()
    fig.add_trace(go.Scatter3d(
        x=states[i], y=states[j], z=states[k], mode="lines",
        line=dict(color="#6a3d9a", width=3), name="trajectory"))
    fig.add_trace(go.Scatter3d(
        x=[states[i, -1]], y=[states[j, -1]], z=[states[k, -1]],
        mode="markers", marker=dict(color="red", size=5), name="endpoint"))
    if equilibrium is not None and np.all(np.isfinite(equilibrium)):
        fig.add_trace(go.Scatter3d(
            x=[equilibrium[i]], y=[equilibrium[j]], z=[equilibrium[k]],
            mode="markers",
            marker=dict(color="black", size=6, symbol="diamond"),
            name="E*"))
    fig.update_layout(
        title=title,
        scene=dict(xaxis_title=STATE_LABELS[i], yaxis_title=STATE_LABELS[j],
                   zaxis_title=STATE_LABELS[k]),
        height=480, margin=dict(l=0, r=0, t=50, b=0),
        legend=dict(orientation="h", y=1.05),
    )
    return fig


def eigenvalue_plane(eigs, title: str) -> go.Figure:
    """Complex plane of the eigenvalues of J(E*); RHP = unstable."""
    fig = go.Figure()
    re, im = eigs.real, eigs.imag
    colors = ["#d62728" if r > 1e-9 else "#1f77b4" for r in re]
    fig.add_trace(go.Scatter(
        x=re, y=im, mode="markers",
        marker=dict(size=12, color=colors, line=dict(width=1, color="black")),
        text=[f"{r:+.3f}{i:+.3f}j" for r, i in zip(re, im)],
        hovertemplate="%{text}<extra></extra>", name="eigenvalues"))
    fig.add_vline(x=0, line=dict(color="black", dash="dash"))
    fig.add_hline(y=0, line=dict(color="gray", dash="dot"))
    lim = max(1.0, np.max(np.abs(re)) * 1.3, np.max(np.abs(im)) * 1.3)
    fig.update_layout(
        title=title, xaxis_title="Re(λ)", yaxis_title="Im(λ)",
        xaxis=dict(range=[-lim, lim]), yaxis=dict(range=[-lim, lim]),
        height=360, margin=dict(l=10, r=10, t=50, b=10), showlegend=False)
    fig.add_annotation(x=lim * 0.6, y=lim * 0.9, text="unstable half-plane",
                       showarrow=False, font=dict(color="#d62728", size=11))
    return fig


def eigenvalue_curve(curve: dict, hopf: dict | None, title: str) -> go.Figure:
    """max Re(λ) of E* vs the swept parameter; zero crossing = Hopf."""
    fig = go.Figure()
    vals, mre = curve["values"], curve["max_real_part"]
    fig.add_trace(go.Scatter(
        x=vals, y=mre, mode="lines",
        line=dict(color="#1f77b4", width=2.5),
        name="max Re(λ) of J(E*)"))
    fig.add_hline(y=0, line=dict(color="black", dash="dash"))
    if hopf is not None:
        fig.add_vline(x=hopf["value"], line=dict(color="#d62728", dash="dot"))
        fig.add_annotation(
            x=hopf["value"], y=0, ax=40, ay=-40,
            text=f"Hopf @ {curve['param']}={hopf['value']:.3f}",
            font=dict(color="#d62728"), showarrow=True, arrowcolor="#d62728")
    fig.update_layout(
        title=title, xaxis_title=curve["param"],
        yaxis_title="max Re(λ) of J(E*)",
        height=380, margin=dict(l=10, r=10, t=50, b=10),
        legend=dict(orientation="h", y=1.12))
    return fig


def bifurcation_diagram(diagram: dict, hopf: dict | None,
                        title: str) -> go.Figure:
    """Long-time min/max of a state component vs the swept parameter."""
    label = STATE_LABELS[diagram["component"]]
    vals = diagram["values"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=vals, y=diagram["max"], mode="markers",
        marker=dict(color="#d62728", size=5), name="long-time max"))
    fig.add_trace(go.Scatter(
        x=vals, y=diagram["min"], mode="markers",
        marker=dict(color="#1f77b4", size=5), name="long-time min"))
    if hopf is not None:
        fig.add_vline(x=hopf["value"], line=dict(color="black", dash="dot"))
        fig.add_annotation(x=hopf["value"], y=float(np.nanmax(diagram["max"])),
                           text="Hopf", showarrow=False,
                           font=dict(color="black"))
    fig.update_layout(
        title=title, xaxis_title=diagram["param"],
        yaxis_title=f"{label} (long-time min / max)",
        height=380, margin=dict(l=10, r=10, t=50, b=10),
        legend=dict(orientation="h", y=1.12))
    return fig
