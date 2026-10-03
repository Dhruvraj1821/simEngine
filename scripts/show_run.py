r"""Print a prototype run as a table plus one sparkline per species.

Usage (PowerShell, venv active):
    python scripts\show_run.py --years 100 --every-years 10 --width 60
"""

from __future__ import annotations

import argparse
import sys

from simengine.engine.prototype import (
    SPECIES_NAMES,
    TICK_YEARS,
    default_world,
    format_table,
    simulate,
)
from simengine.textviz import sparkline


def main() -> None:
    parser = argparse.ArgumentParser(description="Text viewer for the prototype run")
    parser.add_argument("--years", type=float, default=100.0)
    parser.add_argument("--every-years", type=float, default=10.0)
    parser.add_argument("--width", type=int, default=60)
    args = parser.parse_args()
    if args.years <= 0 or args.every_years <= 0 or args.width < 1:
        parser.error("years, every-years and width must be positive")

    # Block characters need UTF-8 even when output is redirected on Windows.
    sys.stdout.reconfigure(encoding="utf-8")

    ticks_total = round(args.years / TICK_YEARS)
    every = max(round(args.every_years / TICK_YEARS), 1)

    state, params = default_world()
    ticks, history = simulate(state, params, ticks_total, every=1)

    chosen = [
        i for i, t in enumerate(ticks) if t % every == 0 or i == len(ticks) - 1
    ]
    print(format_table([ticks[i] for i in chosen], history[chosen]))
    print()
    print(f"biomass over {args.years:g} years, one tick per season (min / max)")
    for s, name in enumerate(SPECIES_NAMES):
        series = history[:, s]
        line = sparkline(series, args.width)
        print(f"{name:<6} {line:<{args.width}}  {series.min():9.2f} / {series.max():9.2f}")


if __name__ == "__main__":
    main()