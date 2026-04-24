"""
Channel Model Module

Defines channel configurations and per-shot noise application for
experiments beyond the baseline uniform erasure channel.

Currently supports:
  - Pure erasure (backward-compatible default, depolar_rate=0)
  - Mixed erasure + depolarizing (Phase 1 journal extension)

Designed so the BP-OSD decoder requires zero changes — all channel
variation enters via the existing update_channel_probs() per-qubit
probability interface.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class ChannelConfig:
    """
    Configuration for the noise channel applied during simulation.

    Parameters
    ----------
    depolar_rate : float, default=0.0
        Per-qubit depolarizing (Pauli) error probability δ applied to
        *non-erased* qubits each shot (unheralded).
        Setting depolar_rate=0 recovers the pure erasure channel exactly.

    Notes
    -----
    Mixed channel model (per qubit, per shot):
      - With probability p_erasure : qubit is erased (heralded).
        The decoder is told p=0.5 for this qubit (maximum uncertainty).
      - With probability depolar_rate : qubit suffers a random Pauli error
        (X, Y, or Z each with probability δ/3), unheralded.
        For the independent X/Z decoder pair:
          P(X component error) = P(X) + P(Y) = 2δ/3
          P(Z component error) = P(Z) + P(Y) = 2δ/3
      These two events are independent.

    Physical motivation: erasure_rate models heralded qubit loss (atom
    drop, photon loss) while depolar_rate models residual gate infidelity
    that is NOT heralded.  The ratio δ/ε ≈ 0.3–0.5 for current
    neutral-atom hardware (Evered et al., Nature 2023).
    """

    depolar_rate: float = 0.0

    @property
    def is_pure_erasure(self) -> bool:
        """True when depolar_rate == 0 (matches original codebase behaviour)."""
        return self.depolar_rate == 0.0

    def to_dict(self) -> dict:
        """Serialise for CSV/JSON metadata columns."""
        return {
            "channel_model": "mixed" if not self.is_pure_erasure else "erasure_uniform",
            "depolar_rate": self.depolar_rate,
        }


# ---------------------------------------------------------------------------
# Per-shot channel functions
# These are called inside the hot Monte Carlo loop — keep them allocation-
# light. Pre-allocate scratch arrays outside the loop where possible.
# ---------------------------------------------------------------------------

def apply_channel(
    x_error: np.ndarray,
    z_error: np.ndarray,
    erasure_mask: np.ndarray,
    channel_config: ChannelConfig,
    rng: np.random.Generator,
) -> None:
    """
    Apply erasure and (optionally) depolarizing errors in-place for one shot.

    Parameters
    ----------
    x_error, z_error : ndarray of uint8, shape (N,)
        Error arrays to be filled in-place (assumed zero on entry).
    erasure_mask : bool ndarray, shape (N,)
        True where qubit was erased this shot.
    channel_config : ChannelConfig
    rng : numpy Generator
    """
    N = len(x_error)

    # --- Erasure errors: random Pauli on erased qubits ---
    # Maximally mixed state: X and Z components independent Bernoulli(0.5)
    if erasure_mask.any():
        rand_x = rng.integers(0, 2, size=N, dtype=np.uint8)
        rand_z = rng.integers(0, 2, size=N, dtype=np.uint8)
        x_error[erasure_mask] = rand_x[erasure_mask]
        z_error[erasure_mask] = rand_z[erasure_mask]

    # --- Depolarizing errors on non-erased qubits (unheralded) ---
    if not channel_config.is_pure_erasure:
        delta = channel_config.depolar_rate
        # P(X component | non-erased) = P(X) + P(Y) = 2*delta/3
        p_xz = 2.0 * delta / 3.0
        non_erased = ~erasure_mask
        x_pauli = (rng.random(N) < p_xz).astype(np.uint8)
        z_pauli = (rng.random(N) < p_xz).astype(np.uint8)
        x_error[non_erased] ^= x_pauli[non_erased]
        z_error[non_erased] ^= z_pauli[non_erased]


def build_channel_probs(
    N: int,
    erasure_mask: np.ndarray,
    channel_config: ChannelConfig,
) -> np.ndarray:
    """
    Build the per-qubit error-probability vector passed to BP-OSD each shot.

    Erased qubits     → p = 0.5  (LLR = 0, maximum uncertainty)
    Non-erased qubits → p = max(2δ/3, 1e-10)
      Pure erasure: 1e-10  (effectively zero, same as original code)
      Mixed channel: 2δ/3  (the marginal X (or Z) component error rate)

    Parameters
    ----------
    N : int
        Code length (number of physical qubits).
    erasure_mask : bool ndarray, shape (N,)
    channel_config : ChannelConfig

    Returns
    -------
    probs : float64 ndarray, shape (N,)
    """
    if channel_config.is_pure_erasure:
        background = 1e-10
    else:
        background = max(2.0 * channel_config.depolar_rate / 3.0, 1e-10)

    probs = np.full(N, background, dtype=np.float64)
    probs[erasure_mask] = 0.5
    return probs
