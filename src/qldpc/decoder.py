"""
Decoder Configuration Module

Provides configuration and wrapper for BP-OSD decoders.
"""

from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np
import warnings
from scipy.sparse import issparse
from bposd import bposd_decoder
from bposd.css import css_code


@dataclass
class DecoderConfig:
    """
    Configuration for BP-OSD decoder.
    
    Parameters
    ----------
    bp_method : str, default="min_sum"
        Belief propagation method ("min_sum" or "sum_product")
    osd_method : str, default="osd_cs"
        Ordered Statistics Decoding method
    osd_order : int, default=10
        OSD order (higher = more accurate but slower)
    max_iter : int, default=50
        Maximum number of BP iterations
    """
    bp_method: str = "min_sum"
    osd_method: str = "osd_cs"
    osd_order: int = 10
    max_iter: int = 50


def create_decoder(hx, hz, config: DecoderConfig, initial_error_rate: float = 0.1):
    """
    Creates and initializes a BP-OSD decoder for the given code.
    
    Parameters
    ----------
    hx : sparse matrix
        X-type parity check matrix
    hz : sparse matrix
        Z-type parity check matrix
    config : DecoderConfig
        Decoder configuration parameters
    initial_error_rate : float, default=0.1
        Initial error rate estimate (will be updated per shot)
        
    Returns
    -------
    tuple
        (css_code object, bposd_decoder object)
    """
    qcode, bpd_x, _ = create_decoders(hx, hz, config, initial_error_rate)
    return qcode, bpd_x


def create_decoders(
    hx,
    hz,
    config: DecoderConfig,
    initial_error_rate: float = 0.1,
) -> Tuple[css_code, bposd_decoder, bposd_decoder]:
    """
    Create decoders for both X and Z error components.

    Returns
    -------
    tuple
        (css_code object, bpd_x, bpd_z) where:
        - bpd_x decodes X errors using Hz checks
        - bpd_z decodes Z errors using Hx checks
    """
    # Convert sparse matrices to dense arrays if needed
    # bposd.css_code requires dense arrays for logical operator computation
    if issparse(hx):
        hx = hx.toarray().astype(int)
    if issparse(hz):
        hz = hz.toarray().astype(int)

    # Create CSS code object (computes logical operators automatically)
    qcode = css_code(hx=hx, hz=hz)

    # Create BP-OSD decoders
    # Suppress deprecation warning for bposd_decoder (legacy API still works)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            category=UserWarning,
            message=".*old syntax for the `bposd_decoder`.*",
        )
        bpd_x = bposd_decoder(
            qcode.hz,
            error_rate=initial_error_rate,
            bp_method=config.bp_method,
            osd_method=config.osd_method,
            osd_order=config.osd_order,
            max_iter=config.max_iter,
        )
        bpd_z = bposd_decoder(
            qcode.hx,
            error_rate=initial_error_rate,
            bp_method=config.bp_method,
            osd_method=config.osd_method,
            osd_order=config.osd_order,
            max_iter=config.max_iter,
        )

    return qcode, bpd_x, bpd_z

