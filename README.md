# rollout-error

Does reweighting the unrolled-training loss by horizon reduce rollout error in
autoregressive neural PDE emulators, beyond what standard unrolled training
already gives?

An autoregressive emulator `f_theta` approximates one solver step. Applied to its
own output `K` times it drifts from the reference trajectory. Unrolled training
("sup") already backpropagates through the `K`-step composition with a uniform
per-step loss. This repo tests a single change on top of that: a per-horizon
weight `w_k`.

    L_w(theta) = sum_{k=1}^{K} w_k * || u_hat_{t+k} - u_{t+k} ||

Three weightings (`src/losses.py`):

| mode         | `w_k`                          | swept over |
|--------------|--------------------------------|------------|
| `uniform`    | `1`                            | -- (must reproduce `sup`) |
| `discounted` | `gamma ** k`                   | gamma in {0.8, 0.9, 1.0, 1.1, 1.25} |
| `normalized` | `1 / EMA(error at horizon k)`  | EMA decay |

`gamma` is swept on **both** sides of 1.0. We are not assuming that
down-weighting far horizons is the useful direction.

## Benchmark

[APEBench](https://github.com/tum-pbs/apebench). We use the `diff_*` (difficulty)
scenario interface, which parameterizes hardness by

    gamma_s = alpha_s * N^s * 2^(s-1) * D,   alpha_s = a_s * dt / L^s

so a scenario is equally hard across spatial dimension and resolution. The
unrolled forward pass, seed handling, and diverted-chain bookkeeping come from
APEBench's `trainax` dependency. Our weighting is a custom aggregation supplied
to `trainax`'s rollout-loss reduction, **not** a hand-rolled rollout loop.

## Layout

```
configs/
  base.yaml                     APEBench base-scenario defaults (mirrors the library)
  scenarios/
    adv.yaml                    diff_adv   -- linear advection
    diff.yaml                   diff_diff  -- NULL CONTROL (dissipative; effect should vanish)
    burgers_sc.yaml             diff_burgers_sc
    kdv.yaml                    diff_kdv
    ks.yaml                     diff_ks    -- primary scenario (spatiotemporal chaos)
  architectures/{conv,unet,fno}.yaml   APEBench network_config strings
  training/
    one.yaml                    train_config "one"    -- one-step baseline
    sup.yaml                    train_config "sup-5"  -- unrolled, uniform (direct baseline)
    div.yaml                    train_config "div-5"  -- diverted-chain (second baseline)
    wsup.yaml                   train_config "wsup-5" -- horizon-weighted (the method)
src/
  losses.py                     WeightMode enum, rollout_weights(), weighted_rollout_loss()
  metrics.py                    per-step nRMSE (geometric + arithmetic), correlation time @0.8/@0.9
  sweep.py                      config product -> flat job list; --dry-run prints grid size
  runner.py                     run one cell, write per-step results rows (STUB: APEBench call not wired)
results/                        parquet output (gitignored)
notebooks/
  01_sanity_check.ipynb         what to verify before the sweep
  02_figures.ipynb              figures + summary table
scripts/run_phase1.sh           sweep driver (not run during scaffolding)
tests/test_losses.py            weighting-strategy tests (+ xfail stubs for the trainax integration)
```

## Metrics

`report_metrics="mean_nRMSE"` in APEBench is a **geometric** mean over rollout
steps. That is an evaluation summary; the loss weights `w_k` are a training
objective. They are not the same thing and `src/metrics.py` keeps them apart:
every aggregation returns the geometric and the arithmetic mean, always, so the
APEBench-comparable number sits next to the arithmetic one in every table.

For chaotic scenarios (`diff_kdv`, `diff_ks`) nRMSE saturates and the informative
metric is **correlation time** -- the first rollout step where the
prediction/target Pearson correlation drops below 0.8 (and 0.9).

## Status

Scaffold only. No training has been run. `runner.run_cell` raises
`NotImplementedError` with the planned APEBench call sequence in its docstring;
the trainax weighting hook is the fill-in-the-middle work.

## Running (once implemented)

```bash
pip install -e .

# grid size, no execution
python -m rollout_error.sweep --dry-run

# one cell
python - <<'PY'
from rollout_error.sweep import Job
from rollout_error.runner import run_cell
run_cell(Job(scenario="ks", architecture="fno", train_mode="wsup",
             weight_mode="discounted", gamma=0.9, seed=0))
PY

# full phase-1 sweep
scripts/run_phase1.sh
```

## Sanity signal

The null control `diff_diff` should show **no** separation between `sup` and any
`wsup` configuration: pure diffusion damps high-frequency error on its own, so
horizon weighting has nothing to fix. If `wsup` beats `sup` there, the gains on
the other scenarios are an artefact, not a real mitigation -- treat a clean null
control as a precondition for reporting anything else.
