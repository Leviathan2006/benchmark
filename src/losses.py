"""Horizon-weighted rollout losses for autoregressive PDE emulator training.

The training objective is

    L_w(theta) = sum_{k=1}^{K} w_k * || u_hat_{t+k} - u_{t+k} ||

where ``u_hat_{t+k}`` is the k-fold self-composition of the emulator ``f_theta``
starting from a shared initial window and ``u_{t+k}`` is the reference solver
trajectory. The three weighting modes below only decide ``w_k``; they are
otherwise agnostic to the network, the scenario, and the norm used per step.

Where this plugs into APEBench
------------------------------
The unrolled forward pass (self-composition of ``f_theta``, gradient
accumulation across the chain, diverted-chain bookkeeping, seed handling) lives
in APEBench's ``trainax`` dependency. ``trainax`` already exposes uniform-weight
unrolled training as the ``sup-XX`` train-config and diverted-chain training as
``div-XX``. The intended extension point for a new weighting scheme is the
per-step loss aggregation inside ``trainax`` -- i.e. supply ``w_k`` to the
reduction that ``trainax`` performs over the rollout chain -- *not* a
from-scratch rollout loop in this repo.

This module therefore provides two things:

* ``rollout_weights`` -- pure function producing the weight vector ``w`` (and the
  threaded EMA state) for a given mode. This is what a ``trainax`` aggregation
  hook would call.
* ``weighted_rollout_loss`` -- a reference reduction over already-computed
  prediction / target trajectories, used by tests and by offline analysis. It is
  deliberately not wired to any live training loop yet.

Nothing here starts training.
"""

from __future__ import annotations

import enum
from typing import Optional

import jax
import jax.numpy as jnp

# Sweep grid for the discounted mode. Intentionally spans both sides of 1.0:
# gamma < 1 down-weights far-horizon terms, gamma > 1 up-weights them. We do not
# assume which direction helps.
DISCOUNTED_GAMMA_GRID = (0.8, 0.9, 1.0, 1.1, 1.25)

_EPS = 1e-8


class WeightMode(str, enum.Enum):
    """Strategy selector for the per-horizon weights ``w_k``."""

    #: w_k = 1 for all k. Should reproduce trainax ``sup-XX`` exactly.
    UNIFORM = "uniform"

    #: w_k = gamma ** k, with gamma sweepable on both sides of 1.0.
    DISCOUNTED = "discounted"

    #: w_k = 1 / running_estimate(error_at_horizon_k), EMA-based. The running
    #: estimate is carried in ``ema_state`` and updated on every call.
    NORMALIZED = "normalized"

    @classmethod
    def coerce(cls, mode: "str | WeightMode") -> "WeightMode":
        if isinstance(mode, cls):
            return mode
        try:
            return cls(str(mode).lower())
        except ValueError as exc:  # pragma: no cover - trivial
            valid = ", ".join(m.value for m in cls)
            raise ValueError(f"unknown weight mode {mode!r}; expected one of {valid}") from exc


def new_ema_state(num_steps: int, decay: float = 0.99) -> dict:
    """Fresh EMA state for :data:`WeightMode.NORMALIZED`.

    The state is a plain dict so it can be threaded through a training loop and
    checkpointed alongside optimizer state without any custom pytree
    registration.

    Keys
    ----
    error_ema : jax.Array, shape (num_steps,)
        Running estimate of the per-horizon residual norm. Initialised to ones so
        the first step behaves like uniform weighting.
    decay : float
        EMA decay; ``error_ema <- decay * error_ema + (1 - decay) * observed``.
    count : int
        Number of updates applied (useful for bias correction / debugging).
    """
    return {
        "error_ema": jnp.ones((num_steps,), dtype=jnp.float32),
        "decay": float(decay),
        "count": 0,
    }


def _per_step_residual_norm(pred_traj: jax.Array, target_traj: jax.Array) -> jax.Array:
    """Reduce trajectories to one scalar residual per rollout step.

    ``pred_traj`` / ``target_traj`` are expected as ``(K, ...)`` or
    ``(B, K, ...)``; every axis except the horizon axis is reduced with an
    L2 norm (mean over batch). Returns shape ``(K,)``.
    """
    if pred_traj.shape != target_traj.shape:
        raise ValueError(
            f"pred/target shape mismatch: {pred_traj.shape} vs {target_traj.shape}"
        )

    resid = pred_traj - target_traj

    # Convention for the reference reduction: >=3d is (B, K, spatial...), so the
    # horizon axis is 1; otherwise (K, spatial...) with horizon axis 0. The
    # trainax integration passes an explicit horizon axis and does not rely on
    # this heuristic.
    horizon_axis = 1 if resid.ndim >= 3 else 0
    reduce_axes = tuple(i for i in range(resid.ndim) if i != horizon_axis)
    if not reduce_axes:
        return jnp.abs(resid)
    return jnp.sqrt(jnp.mean(resid**2, axis=reduce_axes) + _EPS)


def rollout_weights(
    mode: "str | WeightMode",
    num_steps: int,
    *,
    gamma: float = 1.0,
    ema_state: Optional[dict] = None,
    observed_error: Optional[jax.Array] = None,
) -> tuple[jax.Array, Optional[dict]]:
    """Return ``(weights, new_ema_state)`` for the requested weighting mode.

    Parameters
    ----------
    mode
        One of :class:`WeightMode`.
    num_steps
        Rollout horizon ``K``.
    gamma
        Discount base for :data:`WeightMode.DISCOUNTED`. Ignored otherwise.
        Values on both sides of 1.0 are meaningful.
    ema_state
        Required for :data:`WeightMode.NORMALIZED`; see :func:`new_ema_state`.
        Returned updated. ``None`` for the other modes.
    observed_error
        Shape ``(num_steps,)``. The per-horizon residual norm observed on the
        current batch, used to update the EMA for the normalized mode. When
        omitted, the normalized mode falls back to the current EMA estimate
        without updating it (useful for eval).

    Notes
    -----
    Weights are returned unnormalised. Whether the downstream reduction wants
    ``sum_k w_k = 1`` or ``w_1 = 1`` is the caller's choice; see
    :func:`weighted_rollout_loss`, which normalises to a mean.
    """
    mode = WeightMode.coerce(mode)
    k = jnp.arange(1, num_steps + 1, dtype=jnp.float32)

    if mode is WeightMode.UNIFORM:
        return jnp.ones((num_steps,), dtype=jnp.float32), None

    if mode is WeightMode.DISCOUNTED:
        return jnp.asarray(gamma, dtype=jnp.float32) ** k, None

    if mode is WeightMode.NORMALIZED:
        if ema_state is None:
            raise ValueError("normalized mode requires an ema_state (see new_ema_state)")
        error_ema = jnp.asarray(ema_state["error_ema"], dtype=jnp.float32)
        if error_ema.shape != (num_steps,):
            raise ValueError(
                f"ema_state['error_ema'] has shape {error_ema.shape}, expected {(num_steps,)}"
            )
        new_state = dict(ema_state)
        if observed_error is not None:
            observed_error = jax.lax.stop_gradient(jnp.asarray(observed_error, dtype=jnp.float32))
            decay = jnp.asarray(ema_state["decay"], dtype=jnp.float32)
            error_ema = decay * error_ema + (1.0 - decay) * observed_error
            new_state["error_ema"] = error_ema
            new_state["count"] = int(ema_state.get("count", 0)) + 1
        weights = 1.0 / (error_ema + _EPS)
        return weights, new_state

    raise AssertionError(f"unhandled mode {mode!r}")  # pragma: no cover


def weighted_rollout_loss(
    pred_traj: jax.Array,
    target_traj: jax.Array,
    mode: str,
    gamma: float = 1.0,
    ema_state: dict | None = None,
) -> tuple[jax.Array, dict]:
    """Reference reduction of a horizon-weighted rollout loss.

        L_w = (1 / sum_k w_k) * sum_{k=1}^{K} w_k * || u_hat_{t+k} - u_{t+k} ||

    This is the offline / test-time counterpart of the ``trainax`` aggregation
    hook. It does *not* perform any rollout: ``pred_traj`` and ``target_traj``
    are assumed to already hold the K-step self-composed prediction and the
    reference trajectory respectively.

    Parameters
    ----------
    pred_traj, target_traj
        Matching arrays shaped ``(K, ...)`` or ``(B, K, ...)``.
    mode
        See :class:`WeightMode`.
    gamma
        Discount base for the discounted mode.
    ema_state
        EMA state for the normalized mode; updated in place-by-return. Pass the
        dict from :func:`new_ema_state`.

    Returns
    -------
    loss : jax.Array
        Scalar weighted loss.
    aux : dict
        ``{"per_step": (K,), "weights": (K,), "ema_state": dict | None,
        "mode": str, "gamma": float}`` for logging.
    """
    per_step = _per_step_residual_norm(pred_traj, target_traj)
    num_steps = int(per_step.shape[0])

    weights, new_ema = rollout_weights(
        mode,
        num_steps,
        gamma=gamma,
        ema_state=ema_state,
        observed_error=per_step if WeightMode.coerce(mode) is WeightMode.NORMALIZED else None,
    )

    weight_sum = jnp.sum(weights) + _EPS
    loss = jnp.sum(weights * per_step) / weight_sum

    aux = {
        "per_step": per_step,
        "weights": weights,
        "ema_state": new_ema if new_ema is not None else ema_state,
        "mode": WeightMode.coerce(mode).value,
        "gamma": float(gamma),
    }
    return loss, aux
