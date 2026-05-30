"""Command-line entry point for the food-chain chemostat project.

Examples
--------
    python main.py equilibria          # print equilibria + stability table
    python main.py timeseries          # time series + phase portraits
    python main.py bifurcation         # Hopf analysis + bifurcation diagrams
    python main.py stability-map       # successive-invasion map
    python main.py all                 # run everything

Override parameters from the CLI, e.g.:
    python main.py equilibria --s_in 4 --D3 0.4 --m3 3.5

Parameter overrides apply to the `equilibria` report; the figure-producing
commands use the default parameter set defined in foodchain/model.py.
"""

from __future__ import annotations

import argparse

from foodchain import Parameters

_PARAM_FIELDS = ["m1", "m2", "m3", "a1", "a2", "a3", "D1", "D2", "D3", "s_in"]


def build_parameters(args) -> Parameters:
    overrides = {f: getattr(args, f) for f in _PARAM_FIELDS
                 if getattr(args, f) is not None}
    return Parameters(**overrides)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command",
                        choices=["equilibria", "timeseries", "bifurcation",
                                 "stability-map", "all"],
                        help="which analysis to run")
    for f in _PARAM_FIELDS:
        parser.add_argument(f"--{f}", type=float, default=None,
                            help=f"override parameter {f}")
    args = parser.parse_args()
    p = build_parameters(args)

    if args.command in ("equilibria", "all"):
        from experiments.run_equilibria import main as run
        run(p)
    if args.command in ("timeseries", "all"):
        from experiments.run_timeseries import main as run
        run()
    if args.command in ("bifurcation", "all"):
        from experiments.run_bifurcation import main as run
        run()
    if args.command in ("stability-map", "all"):
        from experiments.run_stability_map import main as run
        run()


if __name__ == "__main__":
    main()
