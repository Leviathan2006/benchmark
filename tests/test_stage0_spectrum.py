"""Tests for the Stage 0 error-spectrum estimator.

Two of these are load-bearing for the whole experiment:

* ``Parseval`` -- if the binned power does not reconstruct the real-space power,
  every PSD, flatness and beta downstream is wrong.
* ``white noise`` -- the estimator applied to white Gaussian noise must return
  flatness ~= 1 and beta ~= 0. This validates the estimator end to end against
  the one case where the answer is known a priori, and it is the same null the
  gate decision is read against.

The rest pin the conventions (band centres, sign of beta, bootstrap behaviour)
that those two depend on.

No apebench, no jax, no training: the estimator is pure numpy by design.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rollout_error.stage0_error_spectrum import (
    banded_power,
    build_band_grid,
    decide_verdict,
    fit_powerlaw,
    is_saturated,
    make_synthetic_system,
    measure_error_spectrum,
    spectral_flatness,
    summarize_spectra,
    to_density,
)


# --------------------------------------------------------------------------- #
# Parseval -- the correctness gate for everything else
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("shape", [(64,), (65,), (128,), (16, 16), (8, 12), (9, 7)])
@pytest.mark.parametrize("spacing", ["log", "linear"])
def test_parseval_banded_power_matches_real_space(shape, spacing):
    """dc_power + sum(bands) == mean(field**2), exactly.

    Covers 1D and 2D, even and odd lengths (odd has no Nyquist mode, so the
    one-sided weighting differs).
    """
    rng = np.random.default_rng(0)
    field = rng.standard_normal(shape)

    grid = build_band_grid(shape, num_bands=6, spacing=spacing)
    psd_sum, dc = banded_power(field, grid)

    total = float(dc + psd_sum.sum())
    expected = float(np.mean(field**2))
    np.testing.assert_allclose(total, expected, rtol=1e-10, atol=1e-12)


def test_parseval_holds_with_nonzero_mean():
    """A field with a large DC offset: the offset must land in dc_power only."""
    rng = np.random.default_rng(1)
    field = rng.standard_normal((64,)) + 7.0

    grid = build_band_grid((64,), num_bands=6)
    psd_sum, dc = banded_power(field, grid)

    np.testing.assert_allclose(
        float(dc + psd_sum.sum()), float(np.mean(field**2)), rtol=1e-10
    )
    assert dc > 40.0, "DC offset (7**2 = 49) should dominate the k=0 mode"


def test_parseval_batched():
    """Batch axes are independent; Parseval holds per sample."""
    rng = np.random.default_rng(2)
    fields = rng.standard_normal((5, 3, 32))

    grid = build_band_grid((32,), num_bands=5)
    psd_sum, dc = banded_power(fields, grid)

    assert psd_sum.shape == (5, 3, grid.num_bands)
    np.testing.assert_allclose(
        dc + psd_sum.sum(-1), np.mean(fields**2, axis=-1), rtol=1e-10
    )


def test_every_nondc_mode_lands_in_exactly_one_band():
    grid = build_band_grid((128,), num_bands=16)
    assert np.all(grid.band_index[grid.k_flat > 0] >= 0)
    assert np.all(grid.band_index[grid.k_flat == 0] == -1)
    assert set(np.unique(grid.band_index[grid.k_flat > 0])) == set(range(grid.num_bands))


# --------------------------------------------------------------------------- #
# White noise -- validates the estimator against a known answer
# --------------------------------------------------------------------------- #
def test_white_noise_is_flat_and_beta_zero():
    """The primary null. flatness ~= 1, beta ~= 0."""
    rng = np.random.default_rng(3)
    fields = rng.standard_normal((256, 128))

    grid = build_band_grid((128,), num_bands=8)
    psd_sum, _ = banded_power(fields, grid)
    density = to_density(psd_sum, grid).mean(axis=0)

    flat = spectral_flatness(density)
    beta, se = fit_powerlaw(grid.centers, density)

    assert flat > 0.95, f"white noise should be flat, got flatness={flat:.4f}"
    assert abs(beta) < 0.15, f"white noise should give beta~=0, got {beta:.4f}"
    assert abs(beta) < 3 * se + 0.1


def test_white_noise_2d_is_flat():
    """The 2D radial binning must not manufacture a slope from annulus area.

    An annulus at radius k contains ~k modes, so binning by SUM would produce a
    spurious positive slope. Dividing by the mode count is what makes this pass.
    """
    rng = np.random.default_rng(4)
    fields = rng.standard_normal((64, 32, 32))

    grid = build_band_grid((32, 32), num_bands=6)
    psd_sum, _ = banded_power(fields, grid)
    density = to_density(psd_sum, grid).mean(axis=0)

    beta, _ = fit_powerlaw(grid.centers, density)
    assert spectral_flatness(density) > 0.9
    assert abs(beta) < 0.2, f"2D white noise gave beta={beta:.4f}; check mode counts"


def test_white_error_through_the_full_driver_is_flat_at_every_depth():
    """An emulator whose only error is white must measure as white at every depth.

    This asserts the ESTIMATOR's output (beta ~= 0, flatness ~= 1), not the
    verdict string. The verdict's ``|beta| < 2*SE`` clause is a ~95%-probability
    event per depth by construction, so pinning the exact verdict on sampled data
    would be a coin-flip test across several depths. The decision rule itself is
    tested separately on constructed inputs, where every quantity is exact.
    """
    rng = np.random.default_rng(5)

    def ref(u):
        return u.copy()

    def emu(u):
        return u + 1e-3 * rng.standard_normal(u.shape)

    df = measure_error_spectrum(
        emulator=emu,
        ref_stepper=ref,
        initial_states=rng.standard_normal((32, 128)),
        depths=(1, 2, 5),
        num_origins=2,
        num_bands=8,
        num_bootstrap=100,
        seed=0,
        metadata={"scenario": "white", "arch": "-", "train_config": "-", "seed": 0},
    )

    per_k = df.drop_duplicates(subset=["k"])
    assert (per_k["flatness"] > 0.90).all(), per_k[["k", "flatness"]]
    assert (per_k["beta"].abs() < 0.25).all(), per_k[["k", "beta"]]
    # Whatever else it concludes, white error must never be read as drifting.
    assert decide_verdict(df)[0] != "NONZERO_DRIFTING"


# --------------------------------------------------------------------------- #
# Power-law recovery -- the estimator must read back a spectrum we planted
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("beta_true", [0.0, 1.0, 2.0])
def test_recovers_planted_power_law(beta_true):
    """Synthesise a field with PSD ~ k**(-beta_true) and read beta back."""
    rng = np.random.default_rng(6)
    n, num_samples = 256, 400
    kfreq = np.fft.rfftfreq(n, d=1.0 / n)

    shaping = np.zeros_like(kfreq)
    shaping[1:] = kfreq[1:] ** (-beta_true / 2.0)  # amplitude ~ k^(-beta/2)

    a = rng.standard_normal((num_samples, kfreq.size))
    b = rng.standard_normal((num_samples, kfreq.size))
    fields = np.fft.irfft((a + 1j * b) * shaping, n=n, axis=-1)

    grid = build_band_grid((n,), num_bands=12, spacing="log")
    psd_sum, _ = banded_power(fields, grid)
    density = to_density(psd_sum, grid).mean(axis=0)

    beta, _ = fit_powerlaw(grid.centers, density)
    assert abs(beta - beta_true) < 0.15, f"planted {beta_true}, recovered {beta:.3f}"


def test_log_spacing_less_biased_than_linear_for_steep_spectra():
    """Why ``spacing='log'`` is the default.

    Linear bands put k=1..w in one band and plot it at the band centre, which
    biases a steep spectrum upward in exactly one place and tilts the slope. Log
    bands turn that into a near-constant factor that hits only the intercept.
    """
    rng = np.random.default_rng(7)
    n, num_samples, beta_true = 256, 400, 2.5
    kfreq = np.fft.rfftfreq(n, d=1.0 / n)
    shaping = np.zeros_like(kfreq)
    shaping[1:] = kfreq[1:] ** (-beta_true / 2.0)

    a = rng.standard_normal((num_samples, kfreq.size))
    b = rng.standard_normal((num_samples, kfreq.size))
    fields = np.fft.irfft((a + 1j * b) * shaping, n=n, axis=-1)

    errs = {}
    for spacing in ("log", "linear"):
        grid = build_band_grid((n,), num_bands=12, spacing=spacing)
        psd_sum, _ = banded_power(fields, grid)
        density = to_density(psd_sum, grid).mean(axis=0)
        errs[spacing] = abs(fit_powerlaw(grid.centers, density)[0] - beta_true)

    assert errs["log"] <= errs["linear"] + 1e-9
    assert errs["log"] < 0.2


# --------------------------------------------------------------------------- #
# Conventions the above depend on
# --------------------------------------------------------------------------- #
def test_beta_sign_convention_decreasing_psd_is_positive_beta():
    """PSD ~ k**(-beta): power falling with k must give beta > 0."""
    centers = np.array([1.0, 2.0, 4.0, 8.0, 16.0])
    beta, _ = fit_powerlaw(centers, centers**-2.0)
    np.testing.assert_allclose(beta, 2.0, rtol=1e-8)

    beta_up, _ = fit_powerlaw(centers, centers**1.5)
    np.testing.assert_allclose(beta_up, -1.5, rtol=1e-8)


def test_flatness_bounds():
    flat = np.ones(10)
    assert spectral_flatness(flat) == pytest.approx(1.0)

    peaked = np.array([1e6, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    assert spectral_flatness(peaked) < 0.3

    assert 0.0 <= spectral_flatness(np.array([1.0, 2.0, 3.0, 4.0])) <= 1.0


def test_band_centers_are_positive_and_increasing():
    """log(k_center) must be well defined -- DC is excluded for this reason."""
    grid = build_band_grid((160,), num_bands=16)
    assert np.all(grid.centers > 0)
    assert np.all(np.diff(grid.centers) > 0)
    assert grid.counts.sum() == pytest.approx(160.0 - 1.0)  # all modes but DC


def test_grid_rejects_too_few_bands():
    with pytest.raises(ValueError):
        build_band_grid((64,), num_bands=1)
    with pytest.raises(ValueError):
        build_band_grid((64,), num_bands=8, spacing="quadratic")


def test_banded_power_rejects_shape_mismatch():
    grid = build_band_grid((32,), num_bands=5)
    with pytest.raises(ValueError):
        banded_power(np.zeros((16,)), grid)


# --------------------------------------------------------------------------- #
# Bootstrap
# --------------------------------------------------------------------------- #
def test_bootstrap_ci_brackets_the_mean_and_is_deterministic():
    rng = np.random.default_rng(8)
    grid = build_band_grid((64,), num_bands=6)
    per_ic = np.abs(rng.standard_normal((40, grid.num_bands))) + 1.0

    a = summarize_spectra(per_ic, grid, num_bootstrap=200, seed=0)
    b = summarize_spectra(per_ic, grid, num_bootstrap=200, seed=0)

    np.testing.assert_allclose(a.density_lo, b.density_lo)  # seeded => reproducible
    assert np.all(a.density_lo <= a.density_mean + 1e-12)
    assert np.all(a.density_hi >= a.density_mean - 1e-12)
    assert np.isfinite(a.beta_se_boot)


def test_single_ic_reports_nan_ci_rather_than_a_fake_one():
    grid = build_band_grid((64,), num_bands=6)
    est = summarize_spectra(np.ones((1, grid.num_bands)), grid, num_bootstrap=50)
    assert np.all(np.isnan(est.density_lo))
    assert np.isnan(est.beta_se_boot)


# --------------------------------------------------------------------------- #
# End-to-end through the driver, with a known answer
# --------------------------------------------------------------------------- #
def test_synthetic_system_recovers_constant_beta_across_depth():
    """The synthetic reference stepper is unitary, so accumulated error keeps its
    spectral shape: beta must come back ~= beta_true at EVERY depth, and the
    verdict must be NONZERO_CONSTANT (not drifting)."""
    beta_true = 1.5
    ref, emu, ics = make_synthetic_system(
        num_points=128, beta_true=beta_true, seed=11
    )

    df = measure_error_spectrum(
        emulator=emu,
        ref_stepper=ref,
        initial_states=ics(24),
        depths=(1, 2, 5, 10),
        num_origins=2,
        num_bands=10,
        num_bootstrap=200,
        seed=0,
        metadata={"scenario": "synthetic", "arch": "-", "train_config": "-", "seed": 0},
    )

    per_k = df.drop_duplicates(subset=["k"]).set_index("k")
    for k, row in per_k.iterrows():
        assert abs(row["beta"] - beta_true) < 0.3, (
            f"depth {k}: recovered beta={row['beta']:.3f}, planted {beta_true}"
        )

    # The planted spectrum is coloured, so the gate must not read it as white.
    assert decide_verdict(df)[0] != "WHITE_STOP"


def test_measure_output_schema_is_tidy():
    ref, emu, ics = make_synthetic_system(num_points=64, seed=12)
    df = measure_error_spectrum(
        emulator=emu, ref_stepper=ref, initial_states=ics(6),
        depths=(1, 2), num_origins=1, num_bands=6, num_bootstrap=50,
        metadata={"scenario": "s", "arch": "a", "train_config": "one", "seed": 0},
    )

    required = {
        "scenario", "arch", "train_config", "seed", "k", "band", "k_center",
        "psd_sum", "psd_density", "psd_density_lo", "psd_density_hi",
        "psd_true_density", "psd_ratio", "flatness", "beta", "beta_se_ols",
        "beta_se_boot", "n_modes",
    }
    assert required.issubset(df.columns)
    assert len(df) == len(df.drop_duplicates(subset=["k", "band"]))
    assert set(df["k"]) == {1, 2}


def test_nondeterministic_reference_stepper_is_rejected():
    """If S is stochastic then e_k is not the emulator's error and the whole
    measurement is meaningless -- fail loudly rather than produce numbers."""
    rng = np.random.default_rng(13)

    def bad_ref(u):
        return u + rng.standard_normal(u.shape)

    with pytest.raises(RuntimeError, match="deterministic"):
        measure_error_spectrum(
            emulator=lambda u: u, ref_stepper=bad_ref,
            initial_states=rng.standard_normal((4, 32)),
            depths=(1,), num_origins=1, num_bands=5, num_bootstrap=10,
        )


# --------------------------------------------------------------------------- #
# Verdict logic
# --------------------------------------------------------------------------- #
def _verdict_frame(betas, ses, flatness=0.5):
    rows = []
    for k, (b, s) in zip([1, 2, 5, 10], zip(betas, ses)):
        rows.append({
            "k": k, "band": 0, "beta": b, "beta_se_boot": s,
            "beta_se_ols": s, "flatness": flatness,
        })
    return pd.DataFrame(rows)


def test_verdict_flags_drift_loudly():
    df = _verdict_frame([0.5, 1.2, 2.4, 3.6], [0.05] * 4)
    code, msg = decide_verdict(df)
    assert code == "NONZERO_DRIFTING"
    assert "asymptotic" in msg.lower() or "large-k" in msg.lower()


def test_verdict_constant_when_beta_stable():
    df = _verdict_frame([1.50, 1.52, 1.49, 1.51], [0.05] * 4)
    assert decide_verdict(df)[0] == "NONZERO_CONSTANT"


def test_verdict_white_requires_both_flatness_and_zero_beta():
    assert decide_verdict(_verdict_frame([0.01] * 4, [0.2] * 4, 0.99))[0] == "WHITE_STOP"
    # Flat-ish beta but a peaked spectrum is NOT white.
    assert decide_verdict(_verdict_frame([0.01] * 4, [0.2] * 4, 0.30))[0] != "WHITE_STOP"


def test_verdict_drift_needs_to_exceed_noise():
    """Big error bars => the same beta spread is no longer 'clearly' drifting."""
    assert decide_verdict(_verdict_frame([0.5, 1.2, 2.4, 3.6], [0.05] * 4))[0] == (
        "NONZERO_DRIFTING"
    )
    assert decide_verdict(_verdict_frame([0.5, 1.2, 2.4, 3.6], [3.0] * 4))[0] != (
        "NONZERO_DRIFTING"
    )


# --------------------------------------------------------------------------- #
# Saturation diagnostic + conservative SE selection
# --------------------------------------------------------------------------- #
def test_white_noise_is_never_saturated_and_both_ses_agree():
    """Regression test for the two Stage 0 fixes, on a case where nothing is
    ambiguous.

    White error against a stationary reference is the one setting where we know
    the right answer for every quantity at once:

      * it is nowhere near the predictability horizon (the reference does not
        decorrelate at all), so SATURATED must be False at every depth;
      * beta is genuinely ~0, so the bootstrap SE and the OLS SE must reach the
        SAME conclusion about drift. If they disagree HERE, the SE machinery is
        broken rather than merely reflecting a hard dataset.

    Fix 1 (max(boot, ols)) is only defensible if the two agree when they should;
    this pins that. The cases where they legitimately disagree are covered by
    ``test_verdict_uses_the_conservative_se``.
    """
    rng = np.random.default_rng(21)

    def ref(u):
        return u.copy()

    def emu(u):
        return u + 1e-3 * rng.standard_normal(u.shape)

    df = measure_error_spectrum(
        emulator=emu,
        ref_stepper=ref,
        initial_states=rng.standard_normal((32, 128)),
        depths=(1, 2, 5),
        num_origins=2,
        num_bands=8,
        num_bootstrap=200,
        seed=0,
        metadata={"scenario": "white", "arch": "-", "train_config": "-", "seed": 0},
    )

    per_k = df.drop_duplicates(subset=["k"]).set_index("k")

    # No depth may be flagged saturated: the reference is the identity, so
    # u_true never decorrelates from u_pred.
    assert not per_k["saturated"].any(), per_k[
        ["saturated", "rel_error_power", "pearson_corr"]
    ]
    # Error is tiny relative to signal -- nowhere near the 2.0 independent-draw
    # value -- and correlation is essentially 1.
    assert (per_k["rel_error_power"] < 0.5).all(), per_k["rel_error_power"]
    assert (per_k["pearson_corr"] > 0.9).all(), per_k["pearson_corr"]

    # Both SE estimates must agree that there is no significant drift here.
    lo, hi = per_k.index.min(), per_k.index.max()
    drift = abs(per_k.loc[hi, "beta"] - per_k.loc[lo, "beta"])
    for col in ("beta_se_boot", "beta_se_ols"):
        se = float(np.sqrt(per_k.loc[lo, col] ** 2 + per_k.loc[hi, col] ** 2))
        assert drift <= 2.0 * se, (
            f"{col}: drift {drift:.4f} > 2*SE {2 * se:.4f} on white noise"
        )
    assert decide_verdict(df)[0] != "NONZERO_DRIFTING"


def test_verdict_uses_the_conservative_se_when_the_two_disagree():
    """Fix 1: the SMALLER SE must not be the one that decides the gate.

    Constructed so the bootstrap SE alone calls it drifting and the OLS SE alone
    does not -- the exact ambiguity seen on the real smoke run. The verdict must
    follow the larger (OLS) SE and report NOT drifting.
    """
    rows = []
    for k, beta in zip([1, 5], [1.00, 1.60]):
        rows.append({
            "k": k, "band": 0, "beta": beta, "flatness": 0.5,
            "beta_se_boot": 0.12,   # 2*sqrt(2)*0.12 = 0.339 < 0.60 -> drifting
            "beta_se_ols": 0.35,    # 2*sqrt(2)*0.35 = 0.990 > 0.60 -> not
        })
    df = pd.DataFrame(rows)

    code, msg = decide_verdict(df)
    assert code == "NONZERO_CONSTANT", "conservative SE should suppress the drift claim"
    # Both comparisons must be visible in the output, plus the disagreement note.
    assert "2*SE(bootstrap)" in msg and "2*SE(ols)" in msg and "2*SE(max, USED)" in msg
    assert "DISAGREE" in msg


def test_saturated_drift_is_flagged_not_reported_clean():
    """Fix 2: a drift whose endpoint is past the predictability horizon must be
    called out as possibly chaotic decorrelation, not reported as clean."""
    rows = []
    for k, beta, rel, corr in [
        (1, 0.50, 0.02, 0.99),
        (50, 2.50, 1.98, 0.05),   # saturated: rel ~ 2.0 AND corr ~ 0
    ]:
        rows.append({
            "k": k, "band": 0, "beta": beta, "flatness": 0.4,
            "beta_se_boot": 0.05, "beta_se_ols": 0.05,
            "rel_error_power": rel, "pearson_corr": corr,
            "saturated": is_saturated(rel, corr),
        })
    df = pd.DataFrame(rows)

    code, msg = decide_verdict(df)
    assert code == "NONZERO_DRIFTING"
    assert "SATURATION WARNING" in msg
    assert "decorrelation" in msg.lower()
    assert "k=50 is SATURATED" in msg


def test_unsaturated_drift_keeps_the_clean_message():
    """The same drift with healthy diagnostics must NOT carry the warning."""
    rows = []
    for k, beta, rel, corr in [(1, 0.50, 0.02, 0.99), (50, 2.50, 0.30, 0.85)]:
        rows.append({
            "k": k, "band": 0, "beta": beta, "flatness": 0.4,
            "beta_se_boot": 0.05, "beta_se_ols": 0.05,
            "rel_error_power": rel, "pearson_corr": corr,
            "saturated": is_saturated(rel, corr),
        })
    code, msg = decide_verdict(pd.DataFrame(rows))
    assert code == "NONZERO_DRIFTING"
    assert "SATURATION WARNING" not in msg
    assert "asymptotic" in msg.lower()


@pytest.mark.parametrize(
    "rel, corr, expected",
    [
        (0.02, 0.99, False),   # healthy
        (2.00, 0.00, True),    # textbook independent draws
        (1.75, 0.90, True),    # within 15% of 2.0 -> saturated on power alone
        (2.25, 0.90, True),    # within 15% on the high side
        (1.60, 0.90, False),   # outside the 15% band
        (0.30, 0.15, True),    # correlation collapsed -> saturated on corr alone
        (np.nan, np.nan, False),  # not measured is not saturated
    ],
)
def test_is_saturated_thresholds(rel, corr, expected):
    assert is_saturated(rel, corr) is expected


def test_measure_reports_saturation_columns():
    ref, emu, ics = make_synthetic_system(num_points=64, seed=22)
    df = measure_error_spectrum(
        emulator=emu, ref_stepper=ref, initial_states=ics(6),
        depths=(1, 2), num_origins=1, num_bands=6, num_bootstrap=50,
        metadata={"scenario": "s", "arch": "a", "train_config": "one", "seed": 0},
    )
    for col in ("rel_error_power", "pearson_corr", "saturated"):
        assert col in df.columns
    assert df["pearson_corr"].between(-1.0, 1.0).all()
    assert (df["rel_error_power"] >= 0).all()


def test_verdict_tolerates_frames_without_the_diagnostics():
    """Older parquet files (and hand-built test frames) have no saturation
    columns. Missing must read as 'not measured', never as 'saturated'."""
    df = _verdict_frame([0.5, 1.2, 2.4, 3.6], [0.05] * 4)
    assert "saturated" not in df.columns
    code, msg = decide_verdict(df)
    assert code == "NONZERO_DRIFTING"
    assert "SATURATION WARNING" not in msg
