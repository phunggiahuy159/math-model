"""Tests for the interactive-demo helpers (compute + insights).

These cover the demo-specific logic layered on top of ``foodchain``: attractor
classification, the bifurcation continuation wrappers, and the paper-matching
checks. The Streamlit/Plotly layers are smoke-tested separately via
``streamlit.testing`` and are not imported here.
"""

import numpy as np

from foodchain import Parameters
from demo import compute, insights


def test_attractor_settles_below_hopf():
    p = Parameters(s_in=1.0)
    traj = compute.run_trajectory(p, (0.5, 0.3, 0.2, 0.1), 800, 4000)
    summary = compute.attractor_summary(traj)
    assert summary["oscillating"] is False


def test_attractor_oscillates_above_hopf():
    p = Parameters(s_in=5.0)
    traj = compute.run_trajectory(p, (0.5, 0.3, 0.2, 0.1), 1500, 6000)
    summary = compute.attractor_summary(traj)
    assert summary["oscillating"] is True
    assert summary["period"] is not None and summary["period"] > 0


def test_equilibria_table_has_four_rows():
    rows = compute.equilibria_table(Parameters())
    assert [r.name for r in rows] == ["E0", "E1", "E2", "E*"]
    estar = rows[-1]
    assert estar.exists and estar.stable  # default regime: stable coexistence


def test_eigenvalue_curve_crosses_zero_at_hopf():
    p = Parameters()
    curve = compute.eigenvalue_curve(p, "s_in", 1.0, 6.0, n=120)
    mre = curve["max_real_part"]
    finite = mre[np.isfinite(mre)]
    assert finite.min() < 0 < finite.max()   # E* changes stability on the range


def test_hopf_point_detected():
    h = compute.hopf_point(Parameters(), "s_in", 1.0, 6.0, n=300)
    assert h is not None
    assert 2.0 < h["value"] < 3.0
    assert h["frequency"] > 0


def test_paper_checks_all_pass_for_defaults():
    checks = insights.paper_checks(Parameters())
    assert checks  # non-empty
    assert all(c.passed for c in checks)
    ok, _ = insights.overall_verdict(checks)
    assert ok


def test_hopf_story_reports_enrichment_paradox():
    story = insights.hopf_story(Parameters(), "s_in", 1.0, 6.0)
    assert story["found"]
    assert 2.0 < story["value"] < 3.0
    assert "enrichment" in story["message"].lower()
