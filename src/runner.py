"""Run one grid cell against APEBench and append the results rows.

STATUS: stub. The APEBench call is not wired yet -- :func:`run_cell` raises
``NotImplementedError`` after documenting exactly what it will do. Everything
around it (config resolution, results schema, row writing) is real so the
integration is a fill-in-the-middle job.

Results schema
--------------
One row per (rollout step) per cell. Columns:

    scenario       str    scenario config stem (e.g. "adv")
    arch           str    architecture config stem (e.g. "fno")
    train_mode     str    "one" | "sup" | "div" | "wsup"
    weight_mode    str    "uniform" | "discounted" | "normalized"
    gamma          float  discount base (1.0 when not applicable)
    seed           int
    step           int    rollout step, 1-based
    nrmse          float  per-step nRMSE at this step
    pearson_corr   float  per-step Pearson correlation at this step

Aggregations (geometric/arithmetic mean nRMSE, correlation times) are recomputed
from these rows at analysis time -- see ``notebooks/02_figures.ipynb`` -- so the
stored table stays the minimal per-step record.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from .sweep import CONFIG_ROOT, Job, load_base_config

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

RESULTS_COLUMNS = [
    "scenario",
    "arch",
    "train_mode",
    "weight_mode",
    "gamma",
    "seed",
    "step",
    "nrmse",
    "pearson_corr",
]


@dataclasses.dataclass(frozen=True)
class CellConfig:
    """Fully resolved config for one cell: base + scenario + arch + training."""

    job: Job
    base: dict
    scenario: dict
    architecture: dict
    training: dict

    @property
    def network_config(self) -> str:
        """APEBench ``network_config`` string, e.g. ``"FNO;12;30;4;gelu"``."""
        return self.architecture["network_config"]

    @property
    def train_config(self) -> str:
        """APEBench ``train_config`` string, e.g. ``"one"`` / ``"sup-5"`` / ``"div-5"``.

        For ``wsup`` this is the horizon-weighted unrolled config; the concrete
        string form depends on how the trainax extension registers itself.
        """
        return self.training["train_config"]


def _load_yaml(path: Path) -> dict:
    with path.open() as fh:
        return yaml.safe_load(fh) or {}


def resolve_config(job: Job) -> CellConfig:
    """Merge the four YAML layers for ``job`` into a :class:`CellConfig`."""
    return CellConfig(
        job=job,
        base=load_base_config(),
        scenario=_load_yaml(CONFIG_ROOT / "scenarios" / f"{job.scenario}.yaml"),
        architecture=_load_yaml(CONFIG_ROOT / "architectures" / f"{job.architecture}.yaml"),
        training=_load_yaml(CONFIG_ROOT / "training" / f"{job.train_mode}.yaml"),
    )


def run_cell(job: Job, *, results_dir: Path | None = None, overwrite: bool = False) -> Path:
    """Train one cell on APEBench and write its per-step results rows.

    Planned implementation
    ----------------------
    1. ``cfg = resolve_config(job)``.
    2. Instantiate the scenario:
       ``scenario_cls = apebench.scenarios.difficulty.scenario_dict[cfg.scenario["name"]]``
       then ``scenario = scenario_cls(num_spatial_dims=cfg.base["num_spatial_dims"],
       **cfg.scenario["difficulty_params"])`` with any ``base`` overrides
       (num_points, num_train_samples, train_temporal_horizon, ...).
    3. Call the scenario:
       ``data, trained_nets = scenario(task_config=cfg.base["task_config"],
       network_config=cfg.network_config, train_config=cfg.train_config,
       num_seeds=1)`` -- seeded from ``job.seed`` (APEBench seeds are
       ``range(num_seeds)``; run one seed per cell so failures are isolated).
       For ``train_mode == "wsup"`` the trainax loss aggregation is swapped for
       :func:`rollout_error.losses.rollout_weights` with
       ``(job.weight_mode, job.gamma)``.
    4. Roll the trained net out over ``test_temporal_horizon`` and build
       ``pred_traj`` / ``target_traj`` from ``data``.
    5. ``report = rollout_error.metrics.rollout_report(pred_traj, target_traj)``.
    6. Expand ``report["nrmse_per_step"]`` / ``report["pearson_per_step"]`` into
       one row per step via :func:`rows_from_report` and append with
       :func:`append_rows`.

    Returns the parquet path the rows were written to.
    """
    raise NotImplementedError(
        "APEBench integration not wired yet -- see the docstring for the planned steps. "
        "Scaffolding only; no training is run in this repo."
    )


def rows_from_report(job: Job, report: dict[str, Any]) -> pd.DataFrame:
    """Expand a :func:`rollout_error.metrics.rollout_report` into per-step rows."""
    nrmse = list(report["nrmse_per_step"])
    pearson = list(report["pearson_per_step"])
    if len(nrmse) != len(pearson):
        raise ValueError("nrmse / pearson per-step length mismatch")

    records = [
        {
            "scenario": job.scenario,
            "arch": job.architecture,
            "train_mode": job.train_mode,
            "weight_mode": job.weight_mode,
            "gamma": job.gamma,
            "seed": job.seed,
            "step": step,
            "nrmse": float(nrmse[step - 1]),
            "pearson_corr": float(pearson[step - 1]),
        }
        for step in range(1, len(nrmse) + 1)
    ]
    return pd.DataFrame.from_records(records, columns=RESULTS_COLUMNS)


def append_rows(df: pd.DataFrame, results_dir: Path | None = None, name: str = "phase1.parquet") -> Path:
    """Append ``df`` to ``results_dir/name`` (parquet), creating it if absent."""
    results_dir = results_dir or RESULTS_DIR
    results_dir.mkdir(parents=True, exist_ok=True)
    path = results_dir / name

    if path.exists():
        existing = pd.read_parquet(path)
        df = pd.concat([existing, df], ignore_index=True)
    df.to_parquet(path, index=False)
    return path


def load_results(results_dir: Path | None = None, name: str = "phase1.parquet") -> pd.DataFrame:
    """Read the results table, or an empty frame with the right columns."""
    results_dir = results_dir or RESULTS_DIR
    path = results_dir / name
    if not path.exists():
        return pd.DataFrame(columns=RESULTS_COLUMNS)
    return pd.read_parquet(path)
