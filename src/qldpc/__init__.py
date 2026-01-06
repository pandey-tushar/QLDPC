"""
QLDPC: Quantum Low-Density Parity-Check Code Simulation Framework

A high-performance simulation framework for quantum error correction codes,
specifically optimized for Bivariate Bicycle Codes and erasure channel analysis.
"""

__version__ = "0.1.0"

from .code import BivariateBicycleCode
from .simulator import QuantumSimulator
from .decoder import DecoderConfig
from .toric_code import ToricCode

__all__ = ["BivariateBicycleCode", "QuantumSimulator", "DecoderConfig", "ToricCode"]

