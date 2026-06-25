# A simple food chain in a chemostat with removal rates

A Python implementation and numerical study of the model in

> M.M.A. El-Sheikh & S.A.A. Mahrouf, *"Stability and bifurcation of a simple
> food chain in a chemostat with removal rates"*, **Chaos, Solitons and
> Fractals** 23 (2005) 1475-1489.

The code reproduces the paper's qualitative results: existence and local
stability of the four equilibria, the Routh-Hurwitz analysis of coexistence,
and the **Hopf bifurcation** that destabilizes the coexistence equilibrium and
gives rise to a limit cycle as the input nutrient concentration increases.

## The model

A nutrient `S` feeds a prey `x`, which feeds a first predator `y`, which feeds
a second predator `z` -- a 4-level food chain in a chemostat. Because each
species has its own *removal rate* `Di = D + delta_i` (washout + death), the
usual conservation law of the classical chemostat fails. In dimensionless form
(system (1.2) of the paper):

```
dS/dt = s_in - S - f1(S)*x
dx/dt = x*(f1(S) - D1) - f2(x)*y
dy/dt = y*(f2(x) - D2) - f3(y)*z
dz/dt = z*(f3(y) - D3)
```

with Holling type-II / Michaelis-Menten responses (Section 4):

```
fi(u) = mi*u / (ai + u),   i = 1, 2, 3.
```

In the paper the scaled input is `s_in = 1`; here it is kept as a parameter so
it can be used as the **bifurcation parameter**.

### Equilibria (Section 2)

| Name | State | Meaning | Exists when |
|------|-------|---------|-------------|
| `E0` | `(s_in, 0, 0, 0)` | washout | always |
| `E1` | `(S1, x1, 0, 0)` | prey only | `f1(s_in) > D1` |
| `E2` | `(S2, x2, y2, 0)` | prey + predator 1 | `f2(x1) > D2` |
| `E*` | `(S*, x*, y*, z*)` | full coexistence | `f3(y2) > D3` |

Each level invades exactly when its response at the equilibrium below exceeds
its own removal rate. Stability of `E0`, `E1`, `E2` has the closed forms of
Theorems 2.1-2.3; `E*` is analyzed via the **Routh-Hurwitz** criterion on the
quartic characteristic polynomial.

## Project layout

```
foodchain/
  model.py         Parameters, response functions fi, RHS, Jacobian
  equilibria.py    E0, E1, E2, E* with existence conditions
  stability.py     eigenvalue classification, Routh-Hurwitz, analytic conditions
  simulate.py      time integration (scipy solve_ivp / LSODA)
  bifurcation.py   eigenvalue continuation, Hopf detection, orbit envelopes
  plots.py         matplotlib helpers
experiments/
  run_equilibria.py     equilibria + stability table (text)
  run_timeseries.py     time series + 3-D phase portraits
  run_bifurcation.py    Hopf point + bifurcation diagrams
  run_stability_map.py  successive-invasion map
tests/             pytest suite (model, equilibria, stability, Hopf)
main.py            CLI entry point
```

## Setup

```bash
pip install -r requirements.txt
```

## Usage

```bash
# Print the equilibria and their stability for the default parameters
python main.py equilibria

# Override any parameter on the command line
python main.py equilibria --s_in 4 --D3 0.4 --m3 3.5

# Time series + phase portraits (stable E* and the post-Hopf limit cycle)
python main.py timeseries

# Hopf bifurcation analysis in s_in (eigenvalue + bifurcation diagrams)
python main.py bifurcation

# Successive-invasion / stable-equilibrium maps
python main.py stability-map

# Everything
python main.py all
```

## Interactive demo (Streamlit)

An interactive browser app, [`app.py`](app.py), lets you drag the parameters and
watch the model respond live:

```bash
pip install -r requirements.txt   # includes streamlit + plotly
streamlit run app.py
```

It opens at `http://localhost:8501` with four tabs:

- **▶ Dynamics** — time series (with an animation toggle) and a rotatable 3-D
  phase portrait; an automatic verdict of *settling to a steady state* vs
  *sustained oscillation* (with the estimated period).
- **⚖ Equilibria & stability** — a live table of E0/E1/E2/E\*, their existence
  and local stability, plus the eigenvalues of `J(E*)` in the complex plane
  (an eigenvalue entering the red right-half plane is what destabilizes E\*).
- **🔀 Bifurcation** — sweep any parameter: the leading eigenvalue of E\* with
  the Hopf point marked, and an on-demand numerical bifurcation diagram showing
  the limit-cycle amplitude.
- **📄 Paper check** — automated comparisons of the simulation against the
  paper's theorems (E0/E1/E2 stability conditions, E\* existence ladder,
  Routh–Hurwitz vs eigenvalues) with a pass/fail verdict, and a narrative of the
  Hopf bifurcation / paradox of enrichment.

The demo logic lives in [`demo/`](demo/) (`compute.py`, `viz.py`,
`insights.py`) and is covered by `tests/test_demo.py`.

Figures are written to `figures/`.

## What you should see

With the default parameters:

* **`s_in = 1.0`** -> trajectories converge to a **stable coexistence
  equilibrium** `E*`.
* Increasing `s_in` past **~2.27** a complex-conjugate eigenvalue pair of
  `J(E*)` crosses the imaginary axis: a **Hopf bifurcation** (angular frequency
  ~0.67, period ~9.4). `E*` becomes unstable and a **stable limit cycle**
  appears -- an enrichment-driven destabilization, the "paradox of enrichment".
* **`s_in = 4.0`** -> sustained oscillations of all four state variables,
  with amplitude growing as `s_in` increases further.

`python main.py bifurcation` prints the detected Hopf value, angular frequency
and period, and `tests/` asserts that Routh-Hurwitz agrees with the eigenvalue
classification and that the Hopf crossing is genuinely oscillatory.

## Tests

```bash
python -m pytest -q
```

The suite checks: the analytic Jacobian against finite differences,
invariance of non-negativity and boundedness (Theorem 2.1), that every computed
equilibrium satisfies the equilibrium equations, agreement between Routh-Hurwitz
and eigenvalues at `E*`, and the existence of the Hopf bifurcation.

## Notes on parameter choices

The paper states the response functions and the qualitative results but the
numeric parameter set for its figures is not present in machine-readable form in
the source PDF. The defaults in `foodchain/model.py`
(`m1=2`, `m2=3`, `m3=2`, `a1=1`, `a2=4`, `a3=4`, `D1=D2=D3=0.3`) were chosen so
that:

* all four equilibria `E0, E1, E2, E*` exist at `s_in = 1`, with `E*` locally
  asymptotically stable;
* the interior equilibrium `E*` is **unique** over the explored range of `s_in`
  (with Holling type-II responses other parameter sets can produce *several*
  interior equilibria -- `foodchain.all_coexistence` returns every one, and
  `coexistence` returns the branch continued from `E2`);
* a Hopf bifurcation sits at an easily explored `s_in ~ 2.27`.

Change them freely -- the analysis code is fully general.
