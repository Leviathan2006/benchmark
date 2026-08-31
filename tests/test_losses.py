"""Tests for the horizon-weighting strategies.

Structure over green: tests that depend on the (not-yet-wired) trainax
integration are marked ``xfail`` so they run, fail cleanly, and document the
contract they will eventually enforce. The pure-weighting tests should pass now.

No training is run here.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from rollout_error.losses import (
    DISCOUNTED_GAMMA_GRID,
    WeightMode,
    new_ema_state,
    rollout_weights,
    weighted_rollout_loss,
)

K = 5


# --------------------------------------------------------------------------- #
# uniform mode
# --------------------------------------------------------------------------- #
def test_uniform_weights_are_ones():
    w, state = rollout_weights(WeightMode.UNIFORM, K)
    assert state is None
    np.testing.assert_allclose(np.asarray(w), np.ones(K))


@pytest.mark.xfail(
    reason="requires the trainax rollout-loss integration: needs to compare a "
    "wsup(uniform) training step against a sup-XX training step on identical "
    "batches/seeds and assert equal gradients. Not wired yet.",
    strict=False,
)
def test_uniform_mode_matches_trainax_sup():
    # PLANNED:
    #   from rollout_error.runner import resolve_config, run_cell
    #   grads_sup  = _one_train_step(train_mode="sup",  weight_mode="uniform")
    #   grads_wsup = _one_train_step(train_mode="wsup", weight_mode="uniform")
    #   assert tree_allclose(grads_sup, grads_wsup)
    raise AssertionError("trainax integration not implemented")


# --------------------------------------------------------------------------- #
# discounted mode
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("gamma", DISCOUNTED_GAMMA_GRID)
def test_discounted_weights_are_gamma_pow_k(gamma):
    w, state = rollout_weights(WeightMode.DISCOUNTED, K, gamma=gamma)
    assert state is None
    expected = np.array([gamma ** k for k in range(1, K + 1)])
    np.testing.assert_allclose(np.asarray(w), expected, rtol=1e-6)


def test_discounted_gamma_one_equals_uniform():
    w_disc, _ = rollout_weights(WeightMode.DISCOUNTED, K, gamma=1.0)
    w_unif, _ = rollout_weights(WeightMode.UNIFORM, K)
    np.testing.assert_allclose(np.asarray(w_disc), np.asarray(w_unif))


def test_discounted_direction_not_assumed():
    """gamma > 1 must up-weight far horizons, gamma < 1 must down-weight them."""
    w_lt, _ = rollout_weights(WeightMode.DISCOUNTED, K, gamma=0.8)
    w_gt, _ = rollout_weights(WeightMode.DISCOUNTED, K, gamma=1.25)
    w_lt = np.asarray(w_lt)
    w_gt = np.asarray(w_gt)
    assert w_lt[-1] < w_lt[0]
    assert w_gt[-1] > w_gt[0]


# --------------------------------------------------------------------------- #
# normalized mode
# --------------------------------------------------------------------------- #
def test_normalized_requires_ema_state():
    with pytest.raises(ValueError):
        rollout_weights(WeightMode.NORMALIZED, K)


def test_normalized_weights_are_inverse_error_estimate():
    state = new_ema_state(K, decay=0.5)
    observed = jnp.array([0.1, 0.2, 0.4, 0.8, 1.6])
    w, new_state = rollout_weights(
        WeightMode.NORMALIZED, K, ema_state=state, observed_error=observed
    )
    # after one update with decay 0.5: ema = 0.5*1 + 0.5*observed
    expected_ema = 0.5 * np.ones(K) + 0.5 * np.asarray(observed)
    np.testing.assert_allclose(np.asarray(new_state["error_ema"]), expected_ema, rtol=1e-6)
    np.testing.assert_allclose(np.asarray(w), 1.0 / (expected_ema + 1e-8), rtol=1e-5)
    assert new_state["count"] == 1


def test_normalized_state_threaded_not_mutated():
    state = new_ema_state(K)
    original = np.asarray(state["error_ema"]).copy()
    _, new_state = rollout_weights(
        WeightMode.NORMALIZED, K, ema_state=state,
        observed_error=jnp.ones(K) * 2.0,
    )
    # input dict's array is untouched; a new dict is returned
    np.testing.assert_allclose(np.asarray(state["error_ema"]), original)
    assert new_state is not state


def test_normalized_eval_mode_does_not_update():
    state = new_ema_state(K)
    _, new_state = rollout_weights(WeightMode.NORMALIZED, K, ema_state=state)
    np.testing.assert_allclose(
        np.asarray(new_state["error_ema"]), np.asarray(state["error_ema"])
    )
    assert new_state["count"] == 0


# --------------------------------------------------------------------------- #
# weighted_rollout_loss reduction
# --------------------------------------------------------------------------- #
def _traj(scale_per_step):
    target = jnp.zeros((len(scale_per_step), 16))
    pred = jnp.stack([jnp.full((16,), s) for s in scale_per_step])
    return pred, target


def test_weighted_rollout_loss_returns_array_and_aux_dict():
    pred, target = _traj([0.1, 0.2, 0.3, 0.4, 0.5])
    loss, aux = weighted_rollout_loss(pred, target, mode="uniform")
    assert loss.ndim == 0
    assert set(aux) == {"per_step", "weights", "ema_state", "mode", "gamma"}
    assert aux["mode"] == "uniform"
    assert aux["per_step"].shape == (K,)


def test_weighted_rollout_loss_uniform_is_mean_of_per_step():
    pred, target = _traj([0.1, 0.2, 0.3, 0.4, 0.5])
    loss, aux = weighted_rollout_loss(pred, target, mode="uniform")
    np.testing.assert_allclose(np.asarray(loss), np.mean(np.asarray(aux["per_step"])), rtol=1e-6)


def test_weighted_rollout_loss_discounted_downweights_tail():
    pred, target = _traj([1.0, 1.0, 1.0, 1.0, 10.0])
    loss_small_gamma, _ = weighted_rollout_loss(pred, target, mode="discounted", gamma=0.5)
    loss_uniform, _ = weighted_rollout_loss(pred, target, mode="uniform")
    # heavy tail error contributes less when gamma < 1
    assert float(loss_small_gamma) < float(loss_uniform)


def test_weighted_rollout_loss_threads_ema_state():
    pred, target = _traj([0.1, 0.2, 0.3, 0.4, 0.5])
    state = new_ema_state(K, decay=0.9)
    loss, aux = weighted_rollout_loss(pred, target, mode="normalized", ema_state=state)
    assert aux["ema_state"]["count"] == 1
    assert aux["ema_state"] is not state


def test_shape_mismatch_raises():
    pred = jnp.zeros((K, 16))
    target = jnp.zeros((K, 8))
    with pytest.raises(ValueError):
        weighted_rollout_loss(pred, target, mode="uniform")


def test_unknown_mode_raises():
    pred, target = _traj([0.1] * K)
    with pytest.raises(ValueError):
        weighted_rollout_loss(pred, target, mode="exponential-decay-typo")
