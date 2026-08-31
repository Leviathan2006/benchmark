"""Enumerate the experiment grid as a flat job list.

This module only *enumerates* jobs -- it does not run anything. ``--dry-run``
prints the grid size and a sample of jobs; there is deliberately no execution
path wired here yet. Actually running a job is :func:`rollout_error.runner.run_cell`.

Grid axes
---------
scenario     x  architecture  x  training mode  x  weight mode  x  gamma  x  seed

* ``scenario``   -- one YAML from ``configs/scenarios/`` (``diff.yaml`` is the
  null control).
* ``architecture`` -- one YAML from ``configs/architectures/``.
* ``training mode`` -- one YAML from ``configs/training/``; ``one`` / ``sup`` /
  ``div`` are APEBench-native, ``wsup`` is our horizon-weighted unrolled variant.
* ``weight mode`` / ``gamma`` -- only vary for ``wsup``. For ``one`` / ``sup`` /
  ``div`` the weight mode is fixed to ``uniform`` and gamma to ``1.0`` so every
  row has the same schema.
* ``seed`` -- ``0 .. num_seeds - 1`` (default ``num_seeds`` from ``base.yaml``).
"""

from __future__ import annotations

import argparse
import dataclasses
import itertools
import json
from pathlib import Path
from typing import Iterable

import yaml

from .losses import DISCOUNTED_GAMMA_GRID, WeightMode

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_ROOT = REPO_ROOT / "configs"

# Weight modes / gammas explored for the wsup training mode.
WSUP_WEIGHT_MODES = (WeightMode.DISCOUNTED.value, WeightMode.NORMALIZED.value)
WSUP_GAMMA_GRID = DISCOUNTED_GAMMA_GRID


@dataclasses.dataclass(frozen=True)
class Job:
    """One cell of the grid = one call to :func:`rollout_error.runner.run_cell`."""

    scenario: str
    architecture: str
    train_mode: str
    weight_mode: str
    gamma: float
    seed: int

    def as_dict(self) -> dict:
        return dataclasses.asdict(self)


def _load_yaml(path: Path) -> dict:
    with path.open() as fh:
        return yaml.safe_load(fh) or {}


def load_base_config() -> dict:
    return _load_yaml(CONFIG_ROOT / "base.yaml")


def _stems(subdir: str) -> list[str]:
    return sorted(p.stem for p in (CONFIG_ROOT / subdir).glob("*.yaml"))


def _weight_axis(train_mode: str) -> Iterable[tuple[str, float]]:
    """(weight_mode, gamma) pairs for a given training mode.

    Non-weighted modes get exactly one ``(uniform, 1.0)`` pair so the results
    schema is uniform across the whole grid.
    """
    if train_mode != "wsup":
        yield (WeightMode.UNIFORM.value, 1.0)
        return
    for wm in WSUP_WEIGHT_MODES:
        if wm == WeightMode.DISCOUNTED.value:
            for g in WSUP_GAMMA_GRID:
                yield (wm, float(g))
        else:
            # normalized mode has no gamma; pin to 1.0 for schema stability
            yield (wm, 1.0)


def enumerate_jobs(
    *,
    scenarios: list[str] | None = None,
    architectures: list[str] | None = None,
    train_modes: list[str] | None = None,
    num_seeds: int | None = None,
) -> list[Job]:
    """Cartesian product of the grid axes, flattened to a list of :class:`Job`."""
    base = load_base_config()
    scenarios = scenarios or _stems("scenarios")
    architectures = architectures or _stems("architectures")
    train_modes = train_modes or _stems("training")
    num_seeds = num_seeds if num_seeds is not None else int(base.get("num_seeds", 5))

    jobs: list[Job] = []
    for scenario, arch, train_mode in itertools.product(scenarios, architectures, train_modes):
        for weight_mode, gamma in _weight_axis(train_mode):
            for seed in range(num_seeds):
                jobs.append(
                    Job(
                        scenario=scenario,
                        architecture=arch,
                        train_mode=train_mode,
                        weight_mode=weight_mode,
                        gamma=gamma,
                        seed=seed,
                    )
                )
    return jobs


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scenarios", nargs="*", default=None, help="scenario config stems (default: all)")
    p.add_argument("--architectures", nargs="*", default=None, help="architecture config stems (default: all)")
    p.add_argument("--train-modes", nargs="*", default=None, help="training config stems (default: all)")
    p.add_argument("--num-seeds", type=int, default=None, help="override num_seeds from base.yaml")
    p.add_argument("--dry-run", action="store_true", help="print grid size and a sample, do not emit the full list")
    p.add_argument("--json", action="store_true", help="print the full job list as JSON lines")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    jobs = enumerate_jobs(
        scenarios=args.scenarios,
        architectures=args.architectures,
        train_modes=args.train_modes,
        num_seeds=args.num_seeds,
    )

    if args.dry_run:
        print(f"grid size: {len(jobs)} jobs")
        by_mode: dict[str, int] = {}
        for j in jobs:
            by_mode[j.train_mode] = by_mode.get(j.train_mode, 0) + 1
        for mode, n in sorted(by_mode.items()):
            print(f"  {mode:>6}: {n}")
        print("sample:")
        for j in jobs[:5]:
            print(f"  {j.as_dict()}")
        return 0

    if args.json:
        for j in jobs:
            print(json.dumps(j.as_dict()))
        return 0

    print(f"{len(jobs)} jobs enumerated. Use --dry-run for a summary or --json for the list.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
