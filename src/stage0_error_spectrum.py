"""Stage 0 (GATE): radially-binned power spectrum of an emulator's own error,
resolved by rollout depth k.

The question this answers
------------------------
The "error-matched perturbation" method injects training noise whose covariance
matches the model's own error, rather than isotropic white noise. That is only
worth building if:

  (a) the error spectrum is NOT flat -- otherwise matching it reduces exactly to
      existing white-noise injection and the method is a no-op. PRIMARY GATE.
  (b) the spectrum measured at k=1 is representative of the error the model
      actually accumulates -- accumulated error is repeatedly pushed through the
      emulator Jacobian, which amplifies some modes preferentially, so the
      spectrum at k=20 need not resemble k=1.

Hence everything here is resolved by rollout depth, never computed only at k=1.

Estimator conventions (these matter; see tests)
-----------------------------------------------
* ``rfftn`` over the spatial axes, with one-sided weights: every mode gets
  weight 2 except last-axis index 0 and (for even length) the Nyquist index,
  which get weight 1. This accounts for the conjugate modes rfftn does not
  store.
* Normalisation ``1 / M**2`` with ``M = prod(spatial_shape)``, chosen so that

      dc_power + sum_over_bands(psd_sum) == mean(field**2)

  exactly. That identity is the Parseval unit test; if it fails every number
  downstream is wrong.
* The DC mode (k=0) is held OUT of the bands and reported separately. It carries
  no shape information and would make ``log(k)`` undefined.
* Two per-band quantities are stored, and they are not interchangeable:
    - ``psd_sum``     -- summed power in the band. Parseval is defined on this.
    - ``psd_density`` -- ``psd_sum / (modes represented in band)``. This is the
      actual spectral density and is what beta and flatness are computed from.
      In 2D an annulus contains ~k modes, so using the sum would manufacture a
      spurious slope; the density removes it. In 1D the two differ by a constant
      and beta is unaffected.
* ``beta`` is defined by ``PSD(k) ~ k**(-beta)``, so ``beta = -slope`` of an OLS
  fit of ``log(psd_density)`` on ``log(k_center)``. beta ~= 0 means white.

Statistics
----------
Spectra are averaged over held-out initial conditions AND over multiple time
origins t0, then a bootstrap over INITIAL CONDITIONS (the independent unit --
time origins within a trajectory are correlated and are not resampled) gives a
95% CI per band and a standard error on beta.

Two standard errors on beta are computed and they routinely disagree by 2-3x:

  ``beta_se_boot``  resampling initial conditions
  ``beta_se_ols``   residual scatter of the log-log fit (treats bands as
                    independent, which they are not)

Neither dominates the other, and whichever is smaller makes every "clearly
nonzero" / "clearly drifting" claim easier to make -- so choosing one for
convenience silently decides borderline gates, and the same data can support
opposite conclusions. The verdict therefore uses ``max`` of the two per depth
(the conservative choice) and PRINTS BOTH comparisons every run, so the
disagreement is visible rather than something to reverse-engineer.

Saturation
----------
For a chaotic scenario, once k passes the predictability horizon u_pred_k and
u_true_k are independent draws from the same attractor: relative error power
-> 2.0 and correlation -> 0. Beyond that point e_k's spectrum is the
ATTRACTOR's spectrum, not the model's structured error, and in a beta-vs-k
table that is indistinguishable from genuine drift. Both diagnostics are
measured on the same held-out batch as the spectrum, reported per depth, and a
drift verdict whose endpoints are saturated is explicitly flagged as ambiguous
rather than reported as a clean result.

Dependencies: numpy + pandas (+ matplotlib for the figure). apebench/jax are
imported lazily and only by the adapter in Part B, so the estimator and its
tests run with neither installed.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import pandas as pd

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

DEFAULT_DEPTHS = (1, 2, 5, 10, 20, 50)
DEFAULT_NUM_BANDS = 16
_EPS = 1e-30

# Saturation thresholds. Past the predictability horizon of a chaotic system,
# u_pred and u_true are independent draws from the same attractor: the expected
# relative error power is then 2.0 and the correlation is 0. See is_saturated().
SATURATION_REL_POWER = 2.0
SATURATION_REL_TOL = 0.15   # "within 15% of 2.0"
SATURATION_CORR = 0.2


# =========================================================================== #
# Part A -- pure spectral estimator (no apebench, no jax)
# =========================================================================== #


@dataclasses.dataclass(frozen=True)
class BandGrid:
    """Precomputed radial binning for a fixed spatial shape.

    Built once and reused for every field, so the wavenumber grid and band
    assignment cannot drift between the error spectrum and the signal spectrum.
    """

    spatial_shape: tuple[int, ...]
    num_bands: int               # ACTUAL number of non-empty bands
    requested_bands: int
    spacing: str
    domain_length: float
    weights: np.ndarray          # (*rfftn spatial shape) one-sided weights
    k_flat: np.ndarray           # (n_modes,) |wavevector| per stored mode
    band_index: np.ndarray       # (n_modes,) band id, -1 for DC
    centers: np.ndarray          # (num_bands,) band centre wavenumber
    counts: np.ndarray           # (num_bands,) modes REPRESENTED (sum of weights)

    @property
    def ndim(self) -> int:
        return len(self.spatial_shape)

    @property
    def spatial_axes(self) -> tuple[int, ...]:
        return tuple(range(-self.ndim, 0))

    @property
    def num_points(self) -> int:
        return int(np.prod(self.spatial_shape))


def build_band_grid(
    spatial_shape: Sequence[int],
    num_bands: int = DEFAULT_NUM_BANDS,
    domain_length: float = 1.0,
    spacing: str = "log",
) -> BandGrid:
    """Wavenumber grid + radial bands for ``spatial_shape``.

    Works for any number of spatial dims; 1D and 2D are the intended cases.

    ``domain_length`` sets the physical wavenumber units (k = m / L). It shifts
    the intercept of the power-law fit by a constant and leaves ``beta``
    unchanged, so the default of 1.0 (k = integer mode index) is safe if the
    scenario's domain extent is unknown.

    ``spacing`` defaults to ``"log"`` deliberately. With LINEAR bands the lowest
    band spans k = 1..w, and for a steep spectrum its mode-averaged power is
    dominated by k=1 while being plotted at the band centre -- a large upward
    bias in exactly one band, which tilts the fitted slope. With LOG bands every
    band spans the same RATIO of wavenumbers, so the within-band averaging bias
    is very nearly a constant multiplicative factor: it moves the intercept and
    leaves ``beta`` alone. ``"linear"`` is kept for comparison.

    Band centres are the mode-count-weighted GEOMETRIC mean of |k| in the band,
    not the midpoint of the edges, for the same reason.

    Empty bands (common at low k with log spacing, where few modes exist) are
    dropped, so ``num_bands`` on the returned grid may be smaller than requested.
    Every non-DC mode still lands in exactly one kept band, so Parseval holds.
    """
    spatial_shape = tuple(int(n) for n in spatial_shape)
    ndim = len(spatial_shape)
    if ndim == 0:
        raise ValueError("spatial_shape must have at least one axis")
    if num_bands < 2:
        raise ValueError("num_bands must be >= 2 to fit a slope")
    if spacing not in ("log", "linear"):
        raise ValueError(f"spacing must be 'log' or 'linear', got {spacing!r}")

    # Wavenumber per axis. Last axis is half-spectrum (rfft), the rest full.
    axis_freqs = []
    for i, n in enumerate(spatial_shape):
        d = domain_length / n
        f = np.fft.rfftfreq(n, d=d) if i == ndim - 1 else np.fft.fftfreq(n, d=d)
        axis_freqs.append(f)

    mesh = np.meshgrid(*axis_freqs, indexing="ij")
    k_mag = np.sqrt(sum(m**2 for m in mesh))

    # One-sided weights: modes whose conjugate partner is not stored count twice.
    weights = np.full(k_mag.shape, 2.0)
    weights[..., 0] = 1.0
    if spatial_shape[-1] % 2 == 0:
        weights[..., -1] = 1.0  # Nyquist is self-conjugate

    k_flat = k_mag.reshape(-1)
    w_flat = weights.reshape(-1)

    # Bands over the non-DC modes only.
    positive = k_flat > 0
    k_pos = k_flat[positive]
    k_min, k_max = float(k_pos.min()), float(k_pos.max())

    pad = 1e-9
    if spacing == "log":
        edges = np.geomspace(k_min * (1 - pad), k_max * (1 + pad), num_bands + 1)
    else:
        edges = np.linspace(0.0, k_max * (1 + pad), num_bands + 1)

    raw = np.digitize(k_flat, edges, right=False) - 1
    raw = np.clip(raw, 0, num_bands - 1)
    raw = np.where(positive, raw, -1)

    # Drop empty bands and renumber the survivors contiguously.
    kept = [b for b in range(num_bands) if np.any(raw == b)]
    if len(kept) < 3:
        raise ValueError(
            f"only {len(kept)} non-empty bands for shape {spatial_shape} with "
            f"num_bands={num_bands}, spacing={spacing!r}; need >= 3 to fit a slope"
        )
    remap = np.full(num_bands, -1, dtype=int)
    for new, old in enumerate(kept):
        remap[old] = new
    band_index = np.where(raw >= 0, remap[np.where(raw >= 0, raw, 0)], -1)

    counts = np.array(
        [w_flat[band_index == b].sum() for b in range(len(kept))], dtype=float
    )
    # Mode-count-weighted geometric mean of |k| within each band.
    centers = np.array(
        [
            float(
                np.exp(
                    np.sum(w_flat[band_index == b] * np.log(k_flat[band_index == b]))
                    / w_flat[band_index == b].sum()
                )
            )
            for b in range(len(kept))
        ]
    )

    return BandGrid(
        spatial_shape=spatial_shape,
        num_bands=len(kept),
        requested_bands=num_bands,
        spacing=spacing,
        domain_length=domain_length,
        weights=weights,
        k_flat=k_flat,
        band_index=band_index,
        centers=centers,
        counts=counts,
    )


def banded_power(fields: np.ndarray, grid: BandGrid) -> tuple[np.ndarray, np.ndarray]:
    """Radially-binned power of ``fields``.

    Parameters
    ----------
    fields
        Real array shaped ``(..., *grid.spatial_shape)``; leading axes are batch.

    Returns
    -------
    psd_sum : ``(..., num_bands)`` summed power per band.
    dc_power : ``(...)`` power in the k=0 mode, excluded from the bands.

    Satisfies ``dc_power + psd_sum.sum(-1) == mean(fields**2)`` over the spatial
    axes, up to floating point. This is the Parseval identity the tests pin.
    """
    fields = np.asarray(fields)
    if fields.shape[-grid.ndim :] != grid.spatial_shape:
        raise ValueError(
            f"trailing axes {fields.shape[-grid.ndim:]} do not match grid "
            f"spatial_shape {grid.spatial_shape}"
        )

    spec = np.fft.rfftn(fields, axes=grid.spatial_axes)
    power = grid.weights * (np.abs(spec) ** 2) / (grid.num_points**2)

    batch_shape = power.shape[: power.ndim - grid.ndim]
    flat = power.reshape(*batch_shape, -1)

    dc_power = flat[..., grid.band_index == -1].sum(axis=-1)
    psd_sum = np.stack(
        [flat[..., grid.band_index == b].sum(axis=-1) for b in range(grid.num_bands)],
        axis=-1,
    )
    return psd_sum, dc_power


def to_density(psd_sum: np.ndarray, grid: BandGrid) -> np.ndarray:
    """Convert summed band power to power per represented mode."""
    return psd_sum / grid.counts


def spectral_flatness(density: np.ndarray) -> float:
    """Wiener entropy: geometric mean / arithmetic mean over bands.

    1.0 = perfectly flat (white). -> 0 = strongly peaked.
    """
    d = np.asarray(density, dtype=float)
    d = np.clip(d, _EPS, None)
    geo = float(np.exp(np.mean(np.log(d))))
    arith = float(np.mean(d))
    return geo / arith if arith > 0 else 0.0


def fit_powerlaw(centers: np.ndarray, density: np.ndarray) -> tuple[float, float]:
    """OLS fit of ``log(density)`` on ``log(centers)``.

    Returns ``(beta, se_beta)`` for the convention ``PSD(k) ~ k**(-beta)``, i.e.
    ``beta = -slope``. ``se_beta`` is the OLS residual standard error; it assumes
    independent bands and therefore understates the true uncertainty. Compare it
    against the bootstrap standard error, which does not make that assumption.
    """
    x = np.log(np.asarray(centers, dtype=float))
    y = np.log(np.clip(np.asarray(density, dtype=float), _EPS, None))
    n = x.size
    if n < 3:
        return float("nan"), float("nan")

    x_bar = x.mean()
    sxx = float(np.sum((x - x_bar) ** 2))
    if sxx <= 0:
        return float("nan"), float("nan")

    slope = float(np.sum((x - x_bar) * (y - y.mean())) / sxx)
    intercept = float(y.mean() - slope * x_bar)
    resid = y - (slope * x + intercept)
    sse = float(np.sum(resid**2))
    se_slope = float(np.sqrt((sse / (n - 2)) / sxx))
    return -slope, se_slope


@dataclasses.dataclass
class SpectrumEstimate:
    """Bootstrap summary of a set of per-IC spectra."""

    centers: np.ndarray
    density_mean: np.ndarray
    density_lo: np.ndarray
    density_hi: np.ndarray
    sum_mean: np.ndarray
    flatness: float
    beta: float
    beta_se_ols: float
    beta_se_boot: float
    num_ics: int


def summarize_spectra(
    per_ic_sum: np.ndarray,
    grid: BandGrid,
    num_bootstrap: int = 1000,
    seed: int = 0,
    ci: float = 95.0,
) -> SpectrumEstimate:
    """Bootstrap over initial conditions.

    ``per_ic_sum`` is ``(num_ics, num_bands)`` -- already averaged over time
    origins, because origins within one trajectory are correlated and must not
    be treated as independent draws.
    """
    per_ic_sum = np.asarray(per_ic_sum, dtype=float)
    if per_ic_sum.ndim != 2:
        raise ValueError(f"expected (num_ics, num_bands), got {per_ic_sum.shape}")
    num_ics = per_ic_sum.shape[0]

    per_ic_density = to_density(per_ic_sum, grid)
    density_mean = per_ic_density.mean(axis=0)
    beta, beta_se_ols = fit_powerlaw(grid.centers, density_mean)

    rng = np.random.default_rng(seed)
    lo_q, hi_q = (100.0 - ci) / 2.0, 100.0 - (100.0 - ci) / 2.0

    if num_ics < 2:
        # Cannot bootstrap a single sample; report the point estimate and say so.
        return SpectrumEstimate(
            centers=grid.centers,
            density_mean=density_mean,
            density_lo=np.full_like(density_mean, np.nan),
            density_hi=np.full_like(density_mean, np.nan),
            sum_mean=per_ic_sum.mean(axis=0),
            flatness=spectral_flatness(density_mean),
            beta=beta,
            beta_se_ols=beta_se_ols,
            beta_se_boot=float("nan"),
            num_ics=num_ics,
        )

    idx = rng.integers(0, num_ics, size=(num_bootstrap, num_ics))
    boot_density = per_ic_density[idx].mean(axis=1)  # (num_bootstrap, num_bands)
    boot_beta = np.array([fit_powerlaw(grid.centers, d)[0] for d in boot_density])

    return SpectrumEstimate(
        centers=grid.centers,
        density_mean=density_mean,
        density_lo=np.percentile(boot_density, lo_q, axis=0),
        density_hi=np.percentile(boot_density, hi_q, axis=0),
        sum_mean=per_ic_sum.mean(axis=0),
        flatness=spectral_flatness(density_mean),
        beta=beta,
        beta_se_ols=beta_se_ols,
        beta_se_boot=float(np.nanstd(boot_beta, ddof=1)),
        num_ics=num_ics,
    )


# =========================================================================== #
# Part B -- APEBench adapter  *** THE ONLY UNVERIFIED SURFACE ***
# =========================================================================== #
#
# I could not read the installed apebench/exponax to confirm these signatures,
# so instead of guessing silently, each function below introspects what it
# actually receives and raises with a full structural dump when the shape is not
# what it expects. A wrong guess here fails loudly on your first run rather than
# producing a plausible-looking but meaningless spectrum.
#
# Everything above this line is independent of apebench and is covered by tests.


def _describe(obj, depth: int = 0, max_depth: int = 2) -> str:
    """Best-effort structural description, for adapter error messages."""
    pad = "  " * depth
    t = type(obj).__name__
    if depth >= max_depth:
        return f"{pad}{t}"
    if isinstance(obj, (list, tuple)):
        head = f"{pad}{t}[{len(obj)}]"
        kids = [_describe(o, depth + 1, max_depth) for o in list(obj)[:3]]
        return "\n".join([head, *kids])
    if isinstance(obj, dict):
        head = f"{pad}{t}{{{list(obj)[:6]}}}"
        kids = [_describe(v, depth + 1, max_depth) for v in list(obj.values())[:3]]
        return "\n".join([head, *kids])
    shape = getattr(obj, "shape", None)
    if shape is not None:
        return f"{pad}{t} shape={tuple(shape)}"
    attrs = [a for a in dir(obj) if not a.startswith("_")][:12]
    return f"{pad}{t} callable={callable(obj)} attrs={attrs}"


def load_scenario(name: str, **overrides):
    """Instantiate an apebench ``diff_*`` scenario by short name."""
    import apebench  # noqa: PLC0415  (lazy: keeps the estimator apebench-free)

    table = {
        "diff_ks": "KuramotoSivashinsky",
        "diff_burgers_sc": "BurgersSingleChannel",
        "diff_burgers": "Burgers",
        "diff_adv": "Advection",
        "diff_diff": "Diffusion",
    }
    if name not in table:
        raise ValueError(f"unknown scenario {name!r}; expected one of {sorted(table)}")

    cls = getattr(apebench.scenarios.difficulty, table[name])
    overrides.setdefault("num_spatial_dims", 1)
    return cls(**overrides)


def extract_emulator(trained_nets, seed_index: int = 0) -> Callable:
    """Pull one callable ``f_theta`` out of whatever ``scenario(...)`` returned.

    VERIFY ON FIRST RUN. apebench returns the trained networks as an
    equinox-module pytree whose exact nesting I could not confirm offline. The
    accepted shapes below are the plausible ones; anything else raises with a
    dump of what was actually received so you can add the right case in one line.
    """
    candidate = trained_nets

    # Common case: indexed by seed.
    if isinstance(candidate, (list, tuple)):
        if len(candidate) == 0:
            raise ValueError("trained_nets is empty")
        if seed_index >= len(candidate):
            raise IndexError(
                f"seed_index={seed_index} out of range for {len(candidate)} nets"
            )
        candidate = candidate[seed_index]

    # Sometimes a further single-element nesting (e.g. one entry per train_config).
    while isinstance(candidate, (list, tuple)) and len(candidate) == 1:
        candidate = candidate[0]

    if callable(candidate):
        return candidate

    raise TypeError(
        "Could not extract a callable emulator from trained_nets.\n"
        "Structure received:\n"
        f"{_describe(trained_nets, max_depth=3)}\n"
        "Add the correct unwrapping to extract_emulator() -- do not work around "
        "this by guessing, the whole measurement depends on f_theta being the "
        "trained one-step map."
    )


def get_test_trajectories(scenario, num_ics: int, seed: int = 0) -> np.ndarray:
    """Held-out reference trajectories, shaped ``(num_ics, T+1, *spatial)``.

    Uses the scenario's own test split -- never training trajectories. The
    channel axis (apebench states are ``(channels, *spatial)``) is squeezed only
    when it is singleton; multi-channel scenarios raise rather than silently
    folding channels into the spectrum.
    """
    import jax  # noqa: PLC0415

    ic_set = None
    for meth in ("get_test_ic_set", "get_test_ic", "get_ic_set"):
        if hasattr(scenario, meth):
            fn = getattr(scenario, meth)
            try:
                ic_set = fn(jax.random.PRNGKey(seed))
            except TypeError:
                ic_set = fn()
            break
    if ic_set is None:
        raise AttributeError(
            "Could not find a test-IC accessor on the scenario.\n"
            f"{_describe(scenario, max_depth=1)}\n"
            "Set the correct method name in get_test_trajectories()."
        )

    ic_set = np.asarray(ic_set)[:num_ics]

    ref_stepper = scenario.get_ref_stepper()
    horizon = int(getattr(scenario, "test_temporal_horizon", 200))

    traj = _rollout_numpy(ref_stepper, ic_set, horizon)
    return _squeeze_channel(traj)


def _squeeze_channel(traj: np.ndarray) -> np.ndarray:
    """Drop a singleton channel axis; refuse to collapse a real one."""
    # (num_ics, T+1, C, *spatial)
    if traj.ndim >= 4 and traj.shape[2] == 1:
        return traj[:, :, 0]
    if traj.ndim >= 4 and traj.shape[2] > 1:
        raise NotImplementedError(
            f"multi-channel state (C={traj.shape[2]}); decide explicitly whether "
            "to spectrum each channel separately before proceeding"
        )
    return traj


def _rollout_numpy(stepper: Callable, state, num_steps: int) -> np.ndarray:
    """Apply ``stepper`` ``num_steps`` times, returning the stack INCLUDING init.

    Deliberately hand-written rather than using ``exponax.rollout``: I could not
    verify that helper's ``include_init`` semantics offline, and an off-by-one in
    the time index would silently misalign u_pred_k with u_true_k. Swap it in
    once you have confirmed the convention.
    """
    import jax  # noqa: PLC0415

    batched = jax.vmap(stepper)
    out = [np.asarray(state)]
    cur = state
    for _ in range(num_steps):
        cur = batched(cur)
        out.append(np.asarray(cur))
    return np.stack(out, axis=1)


def as_batched_numpy_step(fn: Callable, add_channel: bool = True) -> Callable:
    """Adapt an apebench/exponax stepper to the driver's calling convention.

    apebench states are jax arrays shaped ``(C, *spatial)`` and its steppers act
    on ONE state. The driver works with batched numpy ``(num_ics, *spatial)``
    (channel already squeezed). This wrapper re-inserts the channel axis, vmaps
    over the batch, and converts back -- so the shape contract is stated in one
    place instead of being implicitly assumed at three call sites.
    """
    import jax  # noqa: PLC0415
    import jax.numpy as jnp  # noqa: PLC0415

    vfn = jax.vmap(fn)

    def step(batch: np.ndarray) -> np.ndarray:
        arr = jnp.asarray(batch)
        if add_channel:
            arr = arr[:, None, ...]
        out = np.asarray(vfn(arr))
        if add_channel:
            if out.shape[1] != 1:
                raise NotImplementedError(
                    f"stepper returned {out.shape[1]} channels; decide explicitly "
                    "how to spectrum a multi-channel state"
                )
            out = out[:, 0, ...]
        return out

    return step


def train_emulator(
    scenario,
    network_config: str = "Conv;26;10;relu",
    train_config: str = "one",
    seed: int = 0,
):
    """Train and return ``(f_theta, raw_return)``."""
    data, trained_nets = scenario(
        task_config="predict",
        network_config=network_config,
        train_config=train_config,
        num_seeds=1,
    )
    return extract_emulator(trained_nets, seed_index=0), (data, trained_nets)


# =========================================================================== #
# Part C -- synthetic system (pipeline validation without apebench)
# =========================================================================== #


def make_synthetic_system(
    num_points: int = 128,
    beta_true: float = 1.5,
    noise_scale: float = 1e-2,
    advection_speed: float = 0.15,
    seed: int = 0,
):
    """A system whose error spectrum has a KNOWN, depth-independent exponent.

    The reference stepper is a pure translation (unitary in Fourier: it changes
    phases, never magnitudes). The emulator is that stepper plus an injected
    perturbation with spectrum ``k**(-beta_true)``. Accumulated error is then a
    sum of independently drawn, rigidly translated perturbations, whose powers
    add without changing spectral SHAPE -- so the recovered ``beta`` must be
    ``~= beta_true`` at every depth k.

    That makes this an end-to-end check of the estimator AND of the
    drift-detection logic (it must report CONSTANT, not drifting). It is a
    pipeline test only and says nothing about any real PDE emulator.
    """
    rng = np.random.default_rng(seed)
    kfreq = np.fft.rfftfreq(num_points, d=1.0 / num_points)
    shaping = np.zeros_like(kfreq)
    shaping[1:] = kfreq[1:] ** (-beta_true / 2.0)  # amplitude ~ k^(-beta/2)
    phase_shift = np.exp(-2j * np.pi * kfreq * advection_speed)

    def ref_stepper(u: np.ndarray) -> np.ndarray:
        return np.fft.irfft(np.fft.rfft(u, axis=-1) * phase_shift, n=num_points, axis=-1)

    def emulator(u: np.ndarray) -> np.ndarray:
        base = ref_stepper(u)
        amp = rng.standard_normal((*u.shape[:-1], kfreq.size))
        pha = rng.standard_normal((*u.shape[:-1], kfreq.size))
        noise_spec = (amp + 1j * pha) * shaping
        noise = np.fft.irfft(noise_spec, n=num_points, axis=-1)
        return base + noise_scale * noise

    def initial_conditions(n: int) -> np.ndarray:
        a = rng.standard_normal((n, kfreq.size))
        b = rng.standard_normal((n, kfreq.size))
        env = np.zeros_like(kfreq)
        env[1:6] = 1.0  # fourier;5 -- low-mode initial data, as apebench uses
        return np.fft.irfft((a + 1j * b) * env, n=num_points, axis=-1)

    return ref_stepper, emulator, initial_conditions


# =========================================================================== #
# Part D -- the measurement
# =========================================================================== #


def measure_error_spectrum(
    emulator: Callable,
    ref_stepper: Callable,
    initial_states: np.ndarray,
    depths: Sequence[int] = DEFAULT_DEPTHS,
    num_origins: int = 4,
    origin_stride: int = 1,
    num_bands: int = DEFAULT_NUM_BANDS,
    domain_length: float = 1.0,
    spacing: str = "log",
    num_bootstrap: int = 1000,
    seed: int = 0,
    metadata: dict | None = None,
    verify_stepper: bool = True,
) -> pd.DataFrame:
    """Error PSD vs rollout depth. Returns tidy rows, one per (k, band).

    ``initial_states`` is ``(num_ics, *spatial)`` drawn from the HELD-OUT split.

    For each depth k and each time origin t0, the reference is advanced to t0
    and both maps are then rolled k steps from that same state, so ``e_k`` is
    exactly the emulator's own divergence from ground truth and nothing else.
    """
    initial_states = np.asarray(initial_states, dtype=float)
    num_ics = initial_states.shape[0]
    spatial_shape = initial_states.shape[1:]
    depths = sorted(int(k) for k in depths)
    max_depth = max(depths)

    grid = build_band_grid(
        spatial_shape,
        num_bands=num_bands,
        domain_length=domain_length,
        spacing=spacing,
    )
    metadata = dict(metadata or {})

    if verify_stepper:
        _assert_deterministic(ref_stepper, initial_states[:1])

    # per_ic accumulators: depth -> (num_ics, num_bands), averaged over origins
    err_acc = {k: np.zeros((num_ics, grid.num_bands)) for k in depths}
    true_acc = {k: np.zeros((num_ics, grid.num_bands)) for k in depths}

    # Saturation diagnostics, averaged over origins.
    #
    # For a chaotic scenario, once k passes the predictability horizon u_pred_k
    # and u_true_k are independent draws from the same attractor. Then
    # E||e_k||^2 = E||u_pred||^2 + E||u_true||^2 = 2 E||u_true||^2 and their
    # correlation goes to zero. At that point e_k's spectrum IS the attractor's
    # spectrum, not the model's structured error -- and in the beta-vs-k table
    # that is indistinguishable from genuine drift. These two numbers are what
    # tell the two apart, so they are measured on the same held-out batch as the
    # spectrum rather than reconstructed later.
    err_power_acc = {k: 0.0 for k in depths}
    true_power_acc = {k: 0.0 for k in depths}
    corr_acc = {k: 0.0 for k in depths}

    origin_state = initial_states.copy()
    for origin in range(num_origins):
        true_traj = _iterate(ref_stepper, origin_state, max_depth)
        pred_traj = _iterate(emulator, origin_state, max_depth)

        for k in depths:
            error = pred_traj[k] - true_traj[k]
            err_sum, _ = banded_power(error, grid)
            true_sum, _ = banded_power(true_traj[k], grid)
            err_acc[k] += err_sum / num_origins
            true_acc[k] += true_sum / num_origins

            err_power_acc[k] += float(np.mean(error**2)) / num_origins
            true_power_acc[k] += float(np.mean(true_traj[k] ** 2)) / num_origins
            corr_acc[k] += (
                float(np.mean(_pearson_batch(pred_traj[k], true_traj[k])))
                / num_origins
            )

        # Advance t0 along the TRUE trajectory. Origins are averaged over, never
        # bootstrapped -- consecutive origins are strongly correlated, so treating
        # them as independent draws would understate the CI. Only the ICs are
        # resampled. ``origin_stride`` just buys more decorrelation per origin.
        if origin < num_origins - 1:
            origin_state = _iterate(ref_stepper, origin_state, origin_stride)[
                origin_stride
            ]

    rows: list[dict] = []
    for k in depths:
        est = summarize_spectra(
            err_acc[k], grid, num_bootstrap=num_bootstrap, seed=seed + k
        )
        true_density = to_density(true_acc[k].mean(axis=0), grid)

        rel_power = float(err_power_acc[k] / (true_power_acc[k] + _EPS))
        corr = float(corr_acc[k])
        saturated = is_saturated(rel_power, corr)

        for b in range(grid.num_bands):
            rows.append(
                {
                    **metadata,
                    "k": k,
                    "band": b,
                    "k_center": float(grid.centers[b]),
                    "n_modes": float(grid.counts[b]),
                    "psd_sum": float(est.sum_mean[b]),
                    "psd_density": float(est.density_mean[b]),
                    "psd_density_lo": float(est.density_lo[b]),
                    "psd_density_hi": float(est.density_hi[b]),
                    "psd_true_density": float(true_density[b]),
                    "psd_ratio": float(
                        est.density_mean[b] / (true_density[b] + _EPS)
                    ),
                    "flatness": est.flatness,
                    "beta": est.beta,
                    "beta_se_ols": est.beta_se_ols,
                    "beta_se_boot": est.beta_se_boot,
                    "rel_error_power": rel_power,
                    "pearson_corr": corr,
                    "saturated": saturated,
                    "num_ics": est.num_ics,
                    "num_origins": num_origins,
                    "spacing": grid.spacing,
                    "num_bands": grid.num_bands,
                }
            )

    return pd.DataFrame(rows)


def _iterate(step: Callable, state: np.ndarray, num_steps: int) -> dict[int, np.ndarray]:
    """``{0: state, 1: step(state), ...}`` up to ``num_steps``."""
    out = {0: np.asarray(state, dtype=float)}
    cur = state
    for i in range(1, num_steps + 1):
        cur = np.asarray(step(cur), dtype=float)
        out[i] = cur
    return out


def _assert_deterministic(stepper: Callable, probe: np.ndarray) -> None:
    """The reference stepper must be deterministic, or u_true_k is not ground truth."""
    a = np.asarray(stepper(probe), dtype=float)
    b = np.asarray(stepper(probe), dtype=float)
    if not np.allclose(a, b, rtol=1e-10, atol=1e-12):
        raise RuntimeError(
            "reference stepper is not deterministic; e_k would not be the "
            "emulator's error. Check that no RNG is threaded into it."
        )


def _pearson_batch(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    """Pearson r per sample over the flattened spatial axes. Returns ``(num_ics,)``."""
    a = np.asarray(pred, dtype=float).reshape(np.shape(pred)[0], -1)
    b = np.asarray(true, dtype=float).reshape(np.shape(true)[0], -1)
    a = a - a.mean(axis=1, keepdims=True)
    b = b - b.mean(axis=1, keepdims=True)
    num = np.sum(a * b, axis=1)
    den = np.sqrt(np.sum(a**2, axis=1) * np.sum(b**2, axis=1)) + _EPS
    return num / den


def is_saturated(rel_error_power: float, pearson_corr: float) -> bool:
    """Has the rollout passed the predictability horizon at this depth?

    Two independent draws from the same distribution give
    ``E||e||^2 = 2 * E||u_true||^2`` and zero correlation. Either signal on its
    own is enough to distrust the measured spectrum as a statement about *model*
    error rather than about the attractor.
    """
    rel = float(rel_error_power)
    corr = float(pearson_corr)
    near_two = bool(
        np.isfinite(rel)
        and abs(rel - SATURATION_REL_POWER)
        <= SATURATION_REL_TOL * SATURATION_REL_POWER
    )
    decorrelated = bool(np.isfinite(corr) and corr < SATURATION_CORR)
    return near_two or decorrelated


# =========================================================================== #
# Part E -- verdict, figure, CLI
# =========================================================================== #

FLATNESS_WHITE = 0.90     # above this, the spectrum is effectively flat
SIGMA = 2.0               # how many stderrs count as "clearly"


def _beta_table(df: pd.DataFrame) -> pd.DataFrame:
    cols = ["k", "flatness", "beta", "beta_se_ols", "beta_se_boot"]
    optional = ["rel_error_power", "pearson_corr", "saturated"]
    cols = cols + [c for c in optional if c in df.columns]
    tab = df[cols].drop_duplicates(subset=["k"]).sort_values("k").reset_index(drop=True)
    # Frames built by hand for unit tests may omit the diagnostics. Absent means
    # "not measured", which must never be read as "saturated".
    for c in optional:
        if c not in tab.columns:
            tab[c] = False if c == "saturated" else np.nan
    return tab


def _se(row) -> float:
    """The SE the verdict actually uses: the CONSERVATIVE max of the two.

    The bootstrap SE (resampling initial conditions) and the OLS SE (residual
    scatter of the log-log fit) measure different things and routinely disagree
    by 2-3x on real data. Whichever is SMALLER makes every "clearly nonzero" and
    "clearly drifting" claim easier to make -- so picking one for convenience
    silently decides borderline gates, and the same data can yield opposite
    conclusions. Taking the max means a drift claim has to survive BOTH notions
    of uncertainty. Both are always printed so the disagreement is visible
    instead of something to reverse-engineer from the numbers.
    """
    boot = float(row["beta_se_boot"])
    ols = float(row["beta_se_ols"])
    vals = [v for v in (boot, ols) if np.isfinite(v) and v > 0]
    return float(max(vals)) if vals else float("nan")


def _combined(a: float, b: float) -> float:
    """Two SEs added in quadrature: the SE of their difference."""
    return float(np.sqrt(float(a) ** 2 + float(b) ** 2))


def _saturation_note(tab: pd.DataFrame, lo: int, hi: int) -> str:
    """Describe saturated endpoints of the drift comparison; '' if both clean."""
    lines = []
    for idx, name in ((lo, "low "), (hi, "high")):
        if bool(tab["saturated"].iloc[idx]):
            lines.append(
                f"  {name}-k endpoint k={int(tab['k'].iloc[idx])} is SATURATED: "
                f"rel_error_power={float(tab['rel_error_power'].iloc[idx]):.3f} "
                f"(2.0 == independent draws), "
                f"corr={float(tab['pearson_corr'].iloc[idx]):.3f}\n"
            )
    return "".join(lines)


def decide_verdict(df: pd.DataFrame) -> tuple[str, str]:
    """Apply the pre-registered decision rule. Returns ``(code, message)``."""
    tab = _beta_table(df)
    if tab.empty:
        return "UNKNOWN", "no rows"

    betas = tab["beta"].to_numpy(dtype=float)
    flat = tab["flatness"].to_numpy(dtype=float)
    ses = tab.apply(_se, axis=1).to_numpy(dtype=float)
    ses_boot = tab["beta_se_boot"].to_numpy(dtype=float)
    ses_ols = tab["beta_se_ols"].to_numpy(dtype=float)

    all_white = bool(
        np.all(np.abs(betas) < SIGMA * ses) and np.all(flat > FLATNESS_WHITE)
    )
    if all_white:
        return (
            "WHITE_STOP",
            "Error spectrum is flat at every depth: |beta| < 2*SE (conservative "
            f"SE = max(boot, ols))\nand flatness > {FLATNESS_WHITE} throughout.\n"
            "  => Matching it reduces to existing white-noise injection.\n"
            "  => The method is a no-op. STOP; do not build Rung 1/2.\n",
        )

    lo, hi = 0, len(tab) - 1
    k_lo, k_hi = int(tab["k"].iloc[lo]), int(tab["k"].iloc[hi])
    drift = float(abs(betas[hi] - betas[lo]))

    se_used = _combined(ses[lo], ses[hi])
    se_boot = _combined(ses_boot[lo], ses_boot[hi])
    se_ols = _combined(ses_ols[lo], ses_ols[hi])

    drifting = drift > SIGMA * se_used
    by_boot = drift > SIGMA * se_boot
    by_ols = drift > SIGMA * se_ols

    def _mark(flag: bool) -> str:
        return "DRIFTING" if flag else "not significant"

    comparison = (
        f"  drift |beta(k={k_hi}) - beta(k={k_lo})| = {drift:.3f}\n"
        f"    vs 2*SE(bootstrap) = {SIGMA * se_boot:.3f}  -> {_mark(by_boot)}\n"
        f"    vs 2*SE(ols)       = {SIGMA * se_ols:.3f}  -> {_mark(by_ols)}\n"
        f"    vs 2*SE(max, USED) = {SIGMA * se_used:.3f}  -> {_mark(drifting)}\n"
    )
    if by_boot != by_ols:
        comparison += (
            "    NOTE: the two SE estimates DISAGREE on this comparison. The\n"
            "          conservative max is what the verdict uses.\n"
        )

    sat = _saturation_note(tab, lo, hi)

    if drifting:
        msg = (
            "Error spectrum is coloured AND beta DRIFTS with depth: "
            f"beta(k={k_lo})={betas[lo]:.3f} -> beta(k={k_hi})={betas[hi]:.3f}\n"
            + comparison
        )
        if sat:
            msg += (
                "\n*** SATURATION WARNING -- THIS DRIFT IS NOT UNAMBIGUOUS ***\n"
                + sat
                + "  => Past the predictability horizon, u_pred and u_true are\n"
                "     independent draws from the same attractor, so e_k's spectrum IS\n"
                "     the ATTRACTOR's spectrum, not the model's structured error.\n"
                "  => This drift may be chaotic decorrelation rather than a\n"
                "     correctable error structure the method could target.\n"
                "  => Re-measure using depths BELOW the saturation onset before\n"
                "     treating this as evidence for the method.\n"
            )
        else:
            msg += (
                "  => The method IS justified, but the cheap one-step measurement is\n"
                "     NOT a valid proxy: target the LARGE-k (asymptotic) spectrum.\n"
                "  => This is the most interesting outcome and changes the design of\n"
                "     Rung 1/2. Do not centre a grid search on the k=1 exponent.\n"
            )
        return "NONZERO_DRIFTING", msg

    msg = (
        f"Error spectrum is coloured (beta ~= {float(np.mean(betas)):.3f}) and "
        "roughly CONSTANT across depth.\n"
        + comparison
        + "  => The method is justified AND the cheap one-step measurement is a\n"
        "     valid proxy for the accumulated error.\n"
        "  => Proceed to Rung 1/2; centre any beta grid search near "
        f"{float(np.mean(betas)):.2f}.\n"
    )
    if sat:
        msg += (
            "\n*** SATURATION WARNING ***\n"
            + sat
            + "  => beta looks stable, but at least one endpoint is past the\n"
            "     predictability horizon, where the spectrum reflects the attractor\n"
            "     rather than model error. Confirm on unsaturated depths.\n"
        )
    return "NONZERO_CONSTANT", msg


def print_summary(df: pd.DataFrame, label: str = "") -> str:
    tab = _beta_table(df)
    title = f"Stage 0 summary{f'  [{label}]' if label else ''}"
    print("\n" + "=" * 92)
    print(title)
    print("=" * 92)
    print(
        f"{'k':>4}  {'flatness':>8}  {'beta':>7}  {'SE(boot)':>8}  {'SE(ols)':>8}  "
        f"{'SE(used)':>8}  {'relErrPow':>9}  {'corr':>6}  {'SAT':>4}"
    )
    print("-" * 92)
    for _, r in tab.iterrows():
        rel = float(r["rel_error_power"])
        corr = float(r["pearson_corr"])
        rel_s = f"{rel:9.3f}" if np.isfinite(rel) else f"{'--':>9}"
        corr_s = f"{corr:6.3f}" if np.isfinite(corr) else f"{'--':>6}"
        sat_s = "YES" if bool(r["saturated"]) else "no"
        print(
            f"{int(r['k']):>4}  {r['flatness']:>8.4f}  {r['beta']:>7.3f}  "
            f"{r['beta_se_boot']:>8.3f}  {r['beta_se_ols']:>8.3f}  "
            f"{_se(r):>8.3f}  {rel_s}  {corr_s}  {sat_s:>4}"
        )
    print(
        "\n  SE(used) = max(boot, ols), the conservative choice; both shown so a\n"
        "  disagreement between them is visible rather than hidden.\n"
        f"  SAT = rel_error_power within {SATURATION_REL_TOL:.0%} of "
        f"{SATURATION_REL_POWER} (independent draws) or corr < {SATURATION_CORR}."
    )

    code, msg = decide_verdict(df)
    print("\n" + "=" * 92)
    print(f"VERDICT: {code}")
    print("=" * 92)
    print(msg)
    print("=" * 92 + "\n")
    return code


def plot_spectra(df: pd.DataFrame, out_path: Path, title: str = "") -> Path:
    """log-log error PSD vs wavenumber, one line per depth, with a flat reference."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    depths = sorted(df["k"].unique())
    cmap = plt.get_cmap("viridis")

    for i, k in enumerate(depths):
        sub = df[df["k"] == k].sort_values("k_center")
        color = cmap(i / max(len(depths) - 1, 1))
        ax.plot(
            sub["k_center"], sub["psd_density"],
            marker="o", ms=3.5, lw=1.8, color=color,
            label=f"k = {int(k)}  (beta = {sub['beta'].iloc[0]:.2f})",
        )
        if np.isfinite(sub["psd_density_lo"]).all():
            ax.fill_between(
                sub["k_center"], sub["psd_density_lo"], sub["psd_density_hi"],
                color=color, alpha=0.16, lw=0,
            )

    # Flat reference: the null hypothesis this experiment exists to reject.
    ref = df[df["k"] == depths[-1]].sort_values("k_center")
    ax.plot(
        ref["k_center"],
        np.full(len(ref), ref["psd_density"].mean()),
        ls="--", lw=1.6, color="crimson", label="flat reference (beta = 0)",
    )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("wavenumber  k")
    ax.set_ylabel("error PSD (power per mode)")
    ax.set_title(title or "Stage 0: error spectrum vs rollout depth")
    ax.grid(True, which="both", alpha=0.25, lw=0.5)
    ax.legend(fontsize=8, framealpha=0.9)
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def run_synthetic(args) -> pd.DataFrame:
    ref, emu, ics = make_synthetic_system(
        num_points=args.num_points, beta_true=args.beta_true, seed=args.seed
    )
    states = ics(args.num_ics)
    return measure_error_spectrum(
        emulator=emu,
        ref_stepper=ref,
        initial_states=states,
        depths=args.depths,
        num_origins=args.num_origins,
        origin_stride=args.origin_stride,
        num_bands=args.num_bands,
        spacing=args.spacing,
        num_bootstrap=args.num_bootstrap,
        seed=args.seed,
        metadata={
            "scenario": f"synthetic(beta_true={args.beta_true})",
            "arch": "none",
            "train_config": "none",
            "seed": args.seed,
        },
    )


def run_apebench(args) -> pd.DataFrame:
    frames = []
    for scenario_name in args.scenarios:
        scenario = load_scenario(
            scenario_name,
            num_spatial_dims=1,
            **({"num_points": args.num_points} if args.num_points else {}),
        )
        f_theta, _ = train_emulator(
            scenario,
            network_config=args.arch,
            train_config=args.train_config,
            seed=args.seed,
        )
        traj = get_test_trajectories(scenario, num_ics=args.num_ics, seed=args.seed)
        states = traj[:, 0]  # t0 = 0; measure_error_spectrum walks origins forward

        frames.append(
            measure_error_spectrum(
                emulator=as_batched_numpy_step(f_theta),
                ref_stepper=as_batched_numpy_step(scenario.get_ref_stepper()),
                initial_states=states,
                depths=args.depths,
                num_origins=args.num_origins,
                origin_stride=args.origin_stride,
                num_bands=args.num_bands,
                spacing=args.spacing,
                num_bootstrap=args.num_bootstrap,
                seed=args.seed,
                metadata={
                    "scenario": scenario_name,
                    "arch": args.arch,
                    "train_config": args.train_config,
                    "seed": args.seed,
                },
            )
        )
    return pd.concat(frames, ignore_index=True)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--synthetic", action="store_true",
                   help="run the known-answer synthetic system (no apebench needed)")
    p.add_argument("--smoke", action="store_true",
                   help="tiny config: few points/ICs, depths up to 5")
    p.add_argument("--scenarios", nargs="*", default=["diff_ks", "diff_burgers_sc"])
    p.add_argument("--arch", default="Conv;26;10;relu")
    p.add_argument("--train-config", default="one")
    p.add_argument("--depths", nargs="*", type=int, default=list(DEFAULT_DEPTHS))
    p.add_argument("--num-bands", type=int, default=DEFAULT_NUM_BANDS)
    p.add_argument("--spacing", choices=["log", "linear"], default="log",
                   help="log bands keep the binning bias out of the fitted slope")
    p.add_argument("--num-ics", type=int, default=30)
    p.add_argument("--num-origins", type=int, default=4)
    p.add_argument("--origin-stride", type=int, default=1,
                   help="reference steps between successive time origins t0")
    p.add_argument("--num-points", type=int, default=None)
    p.add_argument("--num-bootstrap", type=int, default=1000)
    p.add_argument("--beta-true", type=float, default=1.5,
                   help="synthetic mode only: the exponent the estimator must recover")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=str(RESULTS_DIR))
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.smoke:
        args.depths = [d for d in (1, 2, 5) if d <= max(args.depths)] or [1, 2, 5]
        args.num_ics = min(args.num_ics, 8)
        args.num_origins = min(args.num_origins, 2)
        args.num_points = args.num_points or 64
        args.num_bands = min(args.num_bands, 8)
        args.num_bootstrap = min(args.num_bootstrap, 200)
        args.scenarios = args.scenarios[:1]

    print(json.dumps({
        "mode": "synthetic" if args.synthetic else "apebench",
        "smoke": args.smoke, "depths": args.depths, "num_ics": args.num_ics,
        "num_origins": args.num_origins, "num_bands": args.num_bands,
        "num_points": args.num_points, "seed": args.seed,
        "scenarios": None if args.synthetic else args.scenarios,
    }, indent=2))

    df = run_synthetic(args) if args.synthetic else run_apebench(args)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = "smoke" if args.smoke else "full"
    kind = "synthetic" if args.synthetic else "apebench"
    parquet = out_dir / f"stage0_error_spectrum_{kind}_{tag}.parquet"
    df.to_parquet(parquet, index=False)
    print(f"\nwrote {parquet}  ({len(df)} rows)")

    for scenario_name, sub in df.groupby("scenario", sort=False):
        code = print_summary(sub, label=str(scenario_name))
        fig = out_dir / f"stage0_spectrum_{kind}_{tag}_{scenario_name}.png".replace(
            "/", "_"
        ).replace(" ", "")
        plot_spectra(sub, fig, title=f"Stage 0 error spectrum -- {scenario_name}")
        print(f"figure: {fig}   verdict: {code}")

    if args.synthetic:
        print(
            "\nNOTE: synthetic mode validates the ESTIMATOR AND PIPELINE only.\n"
            f"      The error was constructed with beta_true={args.beta_true}, so a\n"
            "      correct estimator must recover that at every k and report\n"
            "      NONZERO_CONSTANT. It is not evidence about any real emulator.\n"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
