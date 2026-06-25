"""Interactive demo for the food-chain chemostat model.

    streamlit run app.py

A browser UI to explore El-Sheikh & Mahrouf (2005): drag the parameters, watch
the equilibria and their stability update live, animate trajectories, sweep the
input nutrient to trigger a Hopf bifurcation, and see automated checks confirm
the simulation matches the paper's theorems.
"""

from __future__ import annotations

import numpy as np
import streamlit as st

from foodchain import Parameters, coexistence
from demo import compute, insights, viz
from demo.compute import PARAM_FIELDS, STATE_LABELS

st.set_page_config(page_title="Food-chain chemostat demo",
                   layout="wide", initial_sidebar_state="expanded")

# Defaults match foodchain.Parameters() — the verified stable-then-Hopf regime.
_DEFAULTS = Parameters()
_SLIDER_RANGES = {
    "m1": (0.1, 8.0, 0.05), "m2": (0.1, 8.0, 0.05), "m3": (0.1, 8.0, 0.05),
    "a1": (0.05, 8.0, 0.05), "a2": (0.05, 8.0, 0.05), "a3": (0.05, 8.0, 0.05),
    "D1": (0.05, 2.0, 0.01), "D2": (0.05, 2.0, 0.01), "D3": (0.05, 2.0, 0.01),
    "s_in": (0.2, 8.0, 0.05),
}
_HELP = {
    "m1": "max growth rate of prey on nutrient",
    "m2": "max growth rate of predator 1 on prey",
    "m3": "max growth rate of predator 2 on predator 1",
    "a1": "half-saturation, prey response f1",
    "a2": "half-saturation, predator-1 response f2",
    "a3": "half-saturation, predator-2 response f3",
    "D1": "removal rate of prey x (washout + death)",
    "D2": "removal rate of predator 1 y",
    "D3": "removal rate of predator 2 z",
    "s_in": "input nutrient concentration (the bifurcation parameter)",
}


# ----------------------------------------------------------------------
# Cached wrappers (re-run only when inputs change)
# ----------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def _trajectory(values: tuple, y0: tuple, t_end: float, n_points: int):
    p = compute.make_params(dict(zip(PARAM_FIELDS, values)))
    return compute.run_trajectory(p, y0, t_end, n_points)


@st.cache_data(show_spinner=False)
def _eig_curve(values: tuple, param: str, lo: float, hi: float, n: int):
    p = compute.make_params(dict(zip(PARAM_FIELDS, values)))
    return compute.eigenvalue_curve(p, param, lo, hi, n)


@st.cache_data(show_spinner=False)
def _hopf(values: tuple, param: str, lo: float, hi: float):
    p = compute.make_params(dict(zip(PARAM_FIELDS, values)))
    return compute.hopf_point(p, param, lo, hi, n=400)


@st.cache_data(show_spinner=True)
def _orbit(values: tuple, param: str, lo: float, hi: float, n: int,
           component: int, t_end: float):
    p = compute.make_params(dict(zip(PARAM_FIELDS, values)))
    return compute.orbit_diagram(p, param, lo, hi, n=n, component=component,
                                 t_end=t_end)


# ----------------------------------------------------------------------
# Sidebar: parameter controls
# ----------------------------------------------------------------------
st.sidebar.title("Parameters")
st.sidebar.caption("Scaled model (1.2): dS,dx,dy,dz with Michaelis–Menten "
                   "responses fᵢ(u)=mᵢu/(aᵢ+u).")

if st.sidebar.button("Reset to paper defaults"):
    for f in PARAM_FIELDS:
        st.session_state[f] = getattr(_DEFAULTS, f)

st.sidebar.subheader("Growth rates mᵢ")
for f in ["m1", "m2", "m3"]:
    lo, hi, step = _SLIDER_RANGES[f]
    st.sidebar.slider(f, lo, hi, key=f,
                      value=st.session_state.get(f, getattr(_DEFAULTS, f)),
                      step=step, help=_HELP[f])
st.sidebar.subheader("Half-saturations aᵢ")
for f in ["a1", "a2", "a3"]:
    lo, hi, step = _SLIDER_RANGES[f]
    st.sidebar.slider(f, lo, hi, key=f,
                      value=st.session_state.get(f, getattr(_DEFAULTS, f)),
                      step=step, help=_HELP[f])
st.sidebar.subheader("Removal rates Dᵢ")
for f in ["D1", "D2", "D3"]:
    lo, hi, step = _SLIDER_RANGES[f]
    st.sidebar.slider(f, lo, hi, key=f,
                      value=st.session_state.get(f, getattr(_DEFAULTS, f)),
                      step=step, help=_HELP[f])
st.sidebar.subheader("Input nutrient s_in")
lo, hi, step = _SLIDER_RANGES["s_in"]
st.sidebar.slider("s_in", lo, hi, key="s_in",
                  value=st.session_state.get("s_in", _DEFAULTS.s_in),
                  step=step, help=_HELP["s_in"])

values = tuple(float(st.session_state[f]) for f in PARAM_FIELDS)
p = compute.make_params(dict(zip(PARAM_FIELDS, values)))

st.sidebar.divider()
st.sidebar.subheader("Initial condition")
y0 = (
    st.sidebar.number_input("S₀", 0.0, 10.0, 0.5, 0.1),
    st.sidebar.number_input("x₀", 0.0, 10.0, 0.3, 0.1),
    st.sidebar.number_input("y₀", 0.0, 10.0, 0.2, 0.1),
    st.sidebar.number_input("z₀", 0.0, 10.0, 0.1, 0.1),
)
t_end = st.sidebar.slider("integration time t_end", 100.0, 4000.0, 800.0, 100.0)


# ----------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------
st.title("A simple food chain in a chemostat with removal rates")
st.caption("Interactive companion to El-Sheikh & Mahrouf, *Chaos, Solitons & "
           "Fractals* 23 (2005) 1475–1489. Nutrient S → prey x → predator y → "
           "predator z, each with its own removal rate Dᵢ.")

# Quick status banner from the coexistence equilibrium.
eq_star = coexistence(p)
if eq_star.exists:
    rep = compute.classify(eq_star, p)
    if rep.stable:
        st.success(f"**E\\*** coexistence equilibrium exists and is **stable** "
                   f"(max Re λ = {rep.max_real_part:+.4f}). All four species "
                   f"persist at a steady state.")
    else:
        st.warning(f"**E\\*** coexistence equilibrium exists but is "
                   f"**unstable** (max Re λ = {rep.max_real_part:+.4f}). Expect "
                   f"sustained oscillations (a limit cycle).")
else:
    st.info(f"**E\\*** coexistence equilibrium does **not** exist for these "
            f"parameters ({eq_star.note}). A lower trophic level is the "
            f"attractor.")

tab_dyn, tab_eq, tab_bif, tab_paper = st.tabs(
    ["Dynamics", "Equilibria & stability", "Bifurcation",
     "Paper check"])


# ----------------------------------------------------------------------
# Tab 1 — Dynamics
# ----------------------------------------------------------------------
with tab_dyn:
    c1, c2 = st.columns([3, 2])
    animate = c1.toggle("Animate the time series", value=False)
    n_points = int(min(8000, max(2000, t_end * 4)))
    traj = _trajectory(values, tuple(y0), float(t_end), n_points)
    summary = compute.attractor_summary(traj)

    with c1:
        title = "Time series"
        if animate:
            st.plotly_chart(viz.time_series_animated(traj, title),
                            width='stretch')
        else:
            st.plotly_chart(viz.time_series(traj, title),
                            width='stretch')

    with c2:
        st.plotly_chart(
            viz.phase_portrait_3d(
                traj.tail(0.6), "Phase portrait (x, y, z)",
                equilibrium=eq_star.state if eq_star.exists else None),
            width='stretch')

    if summary["oscillating"]:
        per = summary["period"]
        st.info(f"**Long-time behaviour: sustained oscillation** (limit cycle)"
                + (f", period ≈ {per:.2f}." if per else "."))
    else:
        st.info("**Long-time behaviour: settling to a steady state** "
                "(equilibrium).")

    cols = st.columns(4)
    for col, lab, mn, mx in zip(cols, STATE_LABELS, summary["min"],
                                summary["max"]):
        col.metric(lab, f"{mx:.3f}", f"min {mn:.3f}", delta_color="off")


# ----------------------------------------------------------------------
# Tab 2 — Equilibria & stability
# ----------------------------------------------------------------------
with tab_eq:
    st.subheader("Equilibria and local stability")
    rows = compute.equilibria_table(p)
    table = []
    for r in rows:
        if r.exists:
            table.append({
                "equilibrium": r.name,
                "S": f"{r.state[0]:.4f}", "x": f"{r.state[1]:.4f}",
                "y": f"{r.state[2]:.4f}", "z": f"{r.state[3]:.4f}",
                "stable?": "stable" if r.stable else "unstable",
                "max Re(λ)": f"{r.max_real_part:+.4f}",
                "oscillatory modes": "yes" if r.has_complex_pair else "no",
            })
        else:
            table.append({
                "equilibrium": r.name, "S": "—", "x": "—", "y": "—", "z": "—",
                "stable?": "does not exist", "max Re(λ)": "—",
                "oscillatory modes": "—"})
    st.dataframe(table, width='stretch', hide_index=True)
    st.caption("E0 washout · E1 prey only · E2 prey+predator1 · E\\* full "
               "coexistence. Each upper level establishes only when its "
               "response at the level below exceeds its own removal rate.")

    eigs = compute.coexistence_eigenvalues(p)
    if eigs is not None:
        st.plotly_chart(
            viz.eigenvalue_plane(eigs, "Eigenvalues of J(E\\*) in the complex "
                                       "plane"),
            width='stretch')
        st.caption("An eigenvalue crossing into the red right-half plane is "
                   "what destabilizes E\\*. A *complex* pair crossing ⇒ Hopf "
                   "(oscillatory) onset.")
    else:
        st.info("E\\* does not exist for these parameters, so there is no "
                "interior Jacobian to show.")


# ----------------------------------------------------------------------
# Tab 3 — Bifurcation
# ----------------------------------------------------------------------
with tab_bif:
    st.subheader("Bifurcation analysis")
    cc1, cc2, cc3 = st.columns(3)
    bparam = cc1.selectbox("Bifurcation parameter", PARAM_FIELDS,
                           index=PARAM_FIELDS.index("s_in"))
    rlo, rhi, _ = _SLIDER_RANGES[bparam]
    lo_hi = cc2.slider("sweep range", float(rlo), float(rhi),
                       (1.0, min(6.0, float(rhi))), 0.1)
    comp = cc3.selectbox("orbit-diagram variable", STATE_LABELS, index=2)
    comp_idx = STATE_LABELS.index(comp)

    curve = _eig_curve(values, bparam, float(lo_hi[0]), float(lo_hi[1]), 240)
    hopf = _hopf(values, bparam, float(lo_hi[0]), float(lo_hi[1]))

    st.plotly_chart(
        viz.eigenvalue_curve(curve, hopf,
                             f"Leading eigenvalue of E\\* vs {bparam}"),
        width='stretch')
    if hopf is not None:
        st.success(f"**Hopf bifurcation** at {bparam} = {hopf['value']:.4f} "
                   f"(angular frequency {hopf['frequency']:.3f}, period "
                   f"{hopf['period']:.2f}). E\\* changes stability here.")
    else:
        st.info("No Hopf bifurcation (complex pair crossing zero) detected on "
                "this range.")

    st.markdown("##### Numerical bifurcation diagram (limit-cycle amplitude)")
    run_orbit = st.button("Compute orbit diagram (integrates many trajectories)")
    if run_orbit:
        diag = _orbit(values, bparam, float(lo_hi[0]), float(lo_hi[1]), 30,
                      comp_idx, 800.0)
        st.plotly_chart(
            viz.bifurcation_diagram(diag, hopf,
                                    f"Long-time min/max of {comp} vs {bparam}"),
            width='stretch')
        st.caption("Where min and max separate, the steady state has given way "
                   "to a limit cycle — the gap is the oscillation amplitude.")
    else:
        st.caption("Click the button to integrate the system across the sweep "
                   "(slower; a few seconds).")


# ----------------------------------------------------------------------
# Tab 4 — Paper check
# ----------------------------------------------------------------------
with tab_paper:
    st.subheader("Does the simulation match the paper?")
    checks = insights.paper_checks(p)
    ok, headline = insights.overall_verdict(checks)
    (st.success if ok else st.error)(headline)

    for c in checks:
        status = "PASS" if c.passed else "FAIL"
        with st.expander(f"[{status}] {c.title}  ·  ({c.reference})",
                         expanded=not c.passed):
            st.write(c.detail)

    st.divider()
    story = insights.hopf_story(p, "s_in", 1.0, 6.0)
    st.markdown("##### Hopf bifurcation & the paradox of enrichment")
    if story["found"]:
        st.write(story["message"])
        m1, m2, m3 = st.columns(3)
        m1.metric("Hopf at s_in", f"{story['value']:.3f}")
        m2.metric("angular frequency", f"{story['frequency']:.3f}")
        m3.metric("period", f"{story['period']:.2f}")
        st.caption(f"Reference: {story['reference']}")
    else:
        st.write(story["message"])

    st.divider()
    st.markdown(
        "##### What the paper concludes\n"
        "- **Boundedness & dissipativity** (Thm 2.1): all trajectories stay in "
        "a bounded region — visible as every time series settling, never "
        "diverging.\n"
        "- **Successive invasion**: each species establishes only when its "
        "response at the equilibrium below exceeds its removal rate "
        "(f1(s_in)>D1 ⇒ prey; f2(x1)>D2 ⇒ predator 1; f3(y2)>D3 ⇒ predator 2). "
        "Try lowering a growth rate mᵢ or raising a removal rate Dᵢ to collapse "
        "the chain a level at a time.\n"
        "- **Hopf bifurcation** (Sec 4): enriching the chemostat (raising "
        "s_in) destabilizes the coexistence equilibrium into a limit cycle — "
        "the *paradox of enrichment*.")
