"""rollout_error: horizon-weighted rollout-loss objectives for autoregressive
neural PDE emulators, benchmarked on APEBench.

Submodules
----------
losses   : weighting strategies for the unrolled training loss
metrics  : per-step nRMSE (geometric + arithmetic) and correlation-time
sweep    : config product -> job list enumeration
runner   : run one (scenario, arch, train_mode, weight_mode, gamma, seed) cell
"""

__version__ = "0.0.1"
