# Implementation notes & design choices

A record of how this code maps onto the paper

> M.M.A. El-Sheikh & S.A.A. Mahrouf, *"Stability and bifurcation of a simple
> food chain in a chemostat with removal rates"*, **Chaos, Solitons and
> Fractals** 23 (2005) 1475-1489,

what was checked, and the deliberate deviations.

## 1. Correctness review summary

The implementation was verified against the paper section by section. No bugs
were found.

| Paper | Code | Status |
|-------|------|--------|
| System (1.2) right-hand side | `model.rhs` | matches exactly |
| Jacobian / variational matrix (2.2) | `model.jacobian` | matches; confirmed against finite differences |
| Equilibria E0, E1, E2, E\* (Section 2) | `equilibria.py` | all defining relations correct |
| Thm 2.2: E0 stable ⇔ `f1(1) < D1` | `analytic_conditions` / `classify` | correct (see note 4) |
| Routh–Hurwitz quartic H3/H4 | `routh_hurwitz_quartic` | matches |
| Hopf bifurcation of E\* (Section 4) | `bifurcation.find_hopf` | complex pair crosses the imaginary axis |

Verification performed:

* `pytest` suite: **17/17 pass** (analytic Jacobian vs finite differences,
  invariance of non-negativity and boundedness, equilibrium residuals,
  Routh–Hurwitz ↔ eigenvalue agreement, Hopf existence).
* Equilibria at `s_in = 1`: E0/E1/E2 unstable, **E\* locally asymptotically
  stable** — the expected successive-invasion cascade.
* Hopf detected at **`s_in ≈ 2.265`**, angular frequency `ω ≈ 0.672`,
  period `≈ 9.35`; post-Hopf time series shows a sustained limit cycle.
* A 200-point `s_in` sweep: Routh–Hurwitz agrees with the direct eigenvalue
  classification at **every** point (0 mismatches).

## 2. The model (dimensionless system 1.2)

```
dS/dt = s_in - S - f1(S) x
dx/dt = x (f1(S) - D1) - f2(x) y
dy/dt = y (f2(x) - D2) - f3(y) z
dz/dt = z (f3(y) - D3)
```

with `Di = D + delta_i` the distinct removal rates (washout + death) that break
the classical chemostat conservation law, and Holling type-II responses
`fi(u) = mi u / (ai + u)`.

## 3. Equilibria and their defining relations (Section 2)

Each successive trophic level fixes one component through `f_i(·) = D_i`; the
rest follow from the nutrient/biomass balances:

| Name | State | Construction |
|------|-------|--------------|
| E0 | `(s_in, 0, 0, 0)` | always exists (washout) |
| E1 | `(S1, x1, 0, 0)` | `f1(S1)=D1` → `S1=a1 D1/(m1-D1)`; `x1=(s_in-S1)/D1` |
| E2 | `(S2, x2, y2, 0)` | `f2(x2)=D2`; `S2` from `f1(S)x2 + S = s_in`; `y2=x2(f1(S2)-D1)/D2` |
| E\* | `(S*, x*, y*, z*)` | `f3(y*)=D3`; nutrient balance gives `x*=(s_in-S*)/f1(S*)`; prey balance fixes `S*`; `z*=y*(f2(x*)-D2)/D3` |

Existence is gated by the natural invasion condition at each level (an upper
level establishes exactly when its response at the equilibrium below exceeds its
own removal rate).

## 4. Design choices / deliberate deviations

These are intentional and do **not** affect correctness.

1. **`s_in` is exposed as the bifurcation parameter.**
   The paper fixes the scaled input at `s_in = 1` and, in Section 4, varies an
   abstract parameter `μ` inside `f1(S, μ)` — it never gives `μ` a numeric
   meaning or a numeric parameter set. Here `s_in` is kept as a free parameter
   and used to drive the Hopf bifurcation. This is the classic "paradox of
   enrichment" route and is mathematically equivalent: the Routh–Hurwitz
   analysis of the quartic characteristic polynomial is identical regardless of
   which parameter is continued.

2. **Routh–Hurwitz is applied to `numpy.poly(J)` numerically**, rather than
   transcribing the paper's hand-derived coefficient formulas (`n_i`, `a_i`,
   `d_i`). Those formulas are lengthy and the only machine-readable copy (the
   PDF) is OCR-garbled, so transcribing them would risk introducing errors.
   Computing the characteristic polynomial directly from the Jacobian is more
   robust, and the test suite confirms it agrees with the eigenvalues across a
   parameter sweep.

3. **The interior equilibrium is found by a 1-D scan for all sign changes**
   (`all_coexistence`) instead of a single Newton seed. With Holling type-II
   responses the interior balance can in principle admit several roots;
   `coexistence` then returns the branch continued from E2 (smallest `x*`).

4. **E0 stability follows the theorem, not the paper's typo.** Theorem 2.2 and
   the eigenvalue `λ2 = f1(1) - D1` give: E0 stable ⇔ `f1(1) < D1`. The paper's
   Discussion section states it with the inequality flipped (`> D1`), which is a
   typo; the code uses the correct `f1(s_in) < D1`.

5. **No numeric parameter set is given in the paper**, so the defaults in
   `model.py` (`m1=2, m2=3, m3=2, a1=1, a2=4, a3=4, D1=D2=D3=0.3`) were chosen so
   that all four equilibria exist at `s_in=1` with E\* stable, and a Hopf
   bifurcation sits at an easily explored `s_in ≈ 2.27`.

## 5. Minor cosmetic observation (no action needed)

At the default parameters, E\* at `s_in = 1` is genuinely stable but sits fairly
close to the Hopf boundary (`max Re(λ) ≈ -0.006`). If a more comfortably-stable
baseline is wanted for figures, nudging a parameter (slightly larger `a2`/`a3`,
or smaller `D3`) pushes `s_in = 1` deeper into the stable region. Purely
cosmetic — the qualitative results are unaffected.

## 6. Running the project

The development machine has no system `numpy`/`scipy`/`matplotlib`. Use a
`uv`-managed virtual environment:

```bash
cd math-model
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python numpy scipy matplotlib pytest
.venv/bin/python -m pytest -q
.venv/bin/python main.py all       # equilibria | bifurcation | timeseries | stability-map
```
