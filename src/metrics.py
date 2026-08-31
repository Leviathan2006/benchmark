"""Rollout evaluation metrics.

All metrics operate on an already-computed rollout: ``pred_traj`` and
``target_traj`` shaped ``(K, ...)`` or ``(B, K, ...)`` where ``K`` is the number
of autoregressive steps.

Aggregation note
----------------
APEBench's base scenario reports ``report_metrics="mean_nRMSE"``, and that
default is a **geometric** mean over the rollout steps. That is a different
reduction from the horizon *weighting* studied in :mod:`rollout_error.losses`:
the geometric mean is an evaluation summary, the weights ``w_k`` are a training
objective. They must never be silently conflated. To keep both visible, every
aggregation helper here returns **both** the geometric and the arithmetic mean,
always computed, so a downstream table can show the APEBench-comparable number
next to the arithmetic one.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

_EPS = 1e-8


def _split_horizon(traj: jax.Array) -> tuple[jax.Array, int]:
    """Return ``(traj_with_K_first, K)``.

    Accepts ``(K, ...)`` or ``(B, K, ...)``; a batched input is left as-is and
    the horizon axis reported as 1.
    """
    if traj.ndim >= 3:
        return traj, traj.shape[1]
    return traj, traj.shape[0]


def nrmse_per_step(pred_traj: jax.Array, target_traj: jax.Array) -> jax.Array:
    """Normalised RMSE at each rollout step.

        nRMSE_k = || pred_k - target_k ||_2 / (|| target_k ||_2 + eps)

    Norms are over all non-horizon axes (mean over batch when present).
    Returns shape ``(K,)``.
    """
    if pred_traj.shape != target_traj.shape:
        raise ValueError(f"shape mismatch: {pred_traj.shape} vs {target_traj.shape}")

    horizon_axis = 1 if pred_traj.ndim >= 3 else 0
    reduce_axes = tuple(i for i in range(pred_traj.ndim) if i != horizon_axis)

    num = jnp.sqrt(jnp.mean((pred_traj - target_traj) ** 2, axis=reduce_axes))
    den = jnp.sqrt(jnp.mean(target_traj ** 2, axis=reduce_axes)) + _EPS
    return num / den


def aggregate_nrmse(per_step: jax.Array) -> dict:
    """Summarise a per-step nRMSE curve.

    Returns
    -------
    dict with keys:
        geometric_mean : APEBench-comparable ``mean_nRMSE`` reduction
                         (exp of the mean log).
        arithmetic_mean : plain mean over steps.
        final : nRMSE at the last rollout step.
        num_steps : K.
    """
    per_step = jnp.asarray(per_step)
    geo = jnp.exp(jnp.mean(jnp.log(per_step + _EPS)))
    arith = jnp.mean(per_step)
    return {
        "geometric_mean": geo,
        "arithmetic_mean": arith,
        "final": per_step[-1],
        "num_steps": int(per_step.shape[0]),
    }


def pearson_correlation_per_step(pred_traj: jax.Array, target_traj: jax.Array) -> jax.Array:
    """Pearson correlation between prediction and target at each rollout step.

    Correlation is computed over the flattened non-horizon axes. Returns shape
    ``(K,)`` with values in ``[-1, 1]``.
    """
    if pred_traj.shape != target_traj.shape:
        raise ValueError(f"shape mismatch: {pred_traj.shape} vs {target_traj.shape}")

    horizon_axis = 1 if pred_traj.ndim >= 3 else 0
    k = pred_traj.shape[horizon_axis]

    p = jnp.moveaxis(pred_traj, horizon_axis, 0).reshape(k, -1)
    t = jnp.moveaxis(target_traj, horizon_axis, 0).reshape(k, -1)

    p = p - jnp.mean(p, axis=1, keepdims=True)
    t = t - jnp.mean(t, axis=1, keepdims=True)

    num = jnp.sum(p * t, axis=1)
    den = jnp.sqrt(jnp.sum(p ** 2, axis=1) * jnp.sum(t ** 2, axis=1)) + _EPS
    return num / den


# Correlation-time thresholds. "Correlation time" is the first rollout step at
# which the prediction/target correlation drops below the threshold -- a standard
# way to report how far a chaotic-system emulator stays trustworthy.
CORRELATION_THRESHOLDS = (0.8, 0.9)


def correlation_time(corr_per_step: jax.Array, threshold: float) -> int:
    """First step index (1-based) at which correlation falls below ``threshold``.

    Returns ``K`` (the horizon length) if the correlation never drops below the
    threshold within the available rollout, so the value is always well defined
    and monotone in threshold strictness.
    """
    corr = jnp.asarray(corr_per_step)
    k = int(corr.shape[0])
    below = corr < threshold
    if not bool(jnp.any(below)):
        return k
    return int(jnp.argmax(below)) + 1


def correlation_times(corr_per_step: jax.Array) -> dict:
    """Correlation time at every threshold in :data:`CORRELATION_THRESHOLDS`.

    Keys are formatted as ``"corr_time@0.8"`` etc.
    """
    return {
        f"corr_time@{thr}": correlation_time(corr_per_step, thr)
        for thr in CORRELATION_THRESHOLDS
    }


def rollout_report(pred_traj: jax.Array, target_traj: jax.Array) -> dict:
    """One-call bundle: per-step nRMSE + both aggregations + correlation times.

    Intended to produce the numbers that :func:`rollout_error.runner.run_cell`
    writes as a results row.
    """
    nrmse = nrmse_per_step(pred_traj, target_traj)
    corr = pearson_correlation_per_step(pred_traj, target_traj)
    return {
        "nrmse_per_step": nrmse,
        "pearson_per_step": corr,
        **{f"nrmse_{k}": v for k, v in aggregate_nrmse(nrmse).items()},
        **correlation_times(corr),
    }
