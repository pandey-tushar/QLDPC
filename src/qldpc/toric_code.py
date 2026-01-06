"""
Toric surface code (code-capacity) construction.

We use the standard toric code on an LxL periodic lattice with:
  - N = 2*L*L data qubits (edges: horizontal + vertical)
  - Z checks on plaquettes (detect X errors)
  - X checks on vertices  (detect Z errors)

This module provides (Hx, Hz, lz) for use in baseline decoders.
"""

from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix


def _h_idx(L: int, x: int, y: int) -> int:
    """Horizontal edge index at (x,y)."""
    return (y % L) * L + (x % L)


def _v_idx(L: int, x: int, y: int) -> int:
    """Vertical edge index at (x,y)."""
    return L * L + (y % L) * L + (x % L)


def toric_code_matrices(L: int):
    """
    Return (Hx, Hz, lz) for a toric code of linear size L.

    Hx: vertex checks (L^2 x 2L^2)
    Hz: plaquette checks (L^2 x 2L^2)
    lz: 2 logical Z operators as a (2 x 2L^2) uint8 array
    """
    if L < 2:
        raise ValueError("L must be >= 2")

    n = 2 * L * L
    m = L * L

    # Vertex checks (X-type): each vertex touches 4 incident edges
    rows, cols, data = [], [], []
    for y in range(L):
        for x in range(L):
            r = y * L + x
            incident = [
                _h_idx(L, x, y),
                _h_idx(L, x - 1, y),
                _v_idx(L, x, y),
                _v_idx(L, x, y - 1),
            ]
            for c in incident:
                rows.append(r)
                cols.append(c)
                data.append(1)
    Hx = csr_matrix((data, (rows, cols)), shape=(m, n), dtype=np.uint8)

    # Plaquette checks (Z-type): each plaquette has 4 boundary edges
    rows, cols, data = [], [], []
    for y in range(L):
        for x in range(L):
            r = y * L + x
            boundary = [
                _h_idx(L, x, y),
                _v_idx(L, x + 1, y),
                _h_idx(L, x, y + 1),
                _v_idx(L, x, y),
            ]
            for c in boundary:
                rows.append(r)
                cols.append(c)
                data.append(1)
    Hz = csr_matrix((data, (rows, cols)), shape=(m, n), dtype=np.uint8)

    # Logical Z operators: one wraps x-direction (use vertical edges at x=0),
    # another wraps y-direction (use horizontal edges at y=0).
    lz = np.zeros((2, n), dtype=np.uint8)
    for y in range(L):
        lz[0, _v_idx(L, 0, y)] = 1
    for x in range(L):
        lz[1, _h_idx(L, x, 0)] = 1

    return Hx, Hz, lz


class ToricCode:
    """
    Toric surface code on an LxL periodic lattice.
    
    The toric code is a well-known quantum error correction code with:
    - N = 2*L*L physical qubits (edges on a torus)
    - K = 2 logical qubits
    - Distance d = L
    
    Parameters
    ----------
    L : int
        Linear size of the toric lattice (L >= 2)
        
    Attributes
    ----------
    L : int
        Linear size parameter
    N : int
        Total number of physical qubits (2 * L * L)
    K : int
        Number of logical qubits (always 2 for toric code)
    d : int
        Code distance (equal to L)
    
    Examples
    --------
    >>> code = ToricCode(L=12)
    >>> Hx, Hz = code.get_matrices()
    >>> print(f"Code: [[{code.N}, {code.K}, {code.d}]]")
    Code: [[288, 2, 12]]
    """
    
    def __init__(self, L: int):
        if L < 2:
            raise ValueError("L must be >= 2 for toric code")
        self.L = L
        self.N = 2 * L * L
        self.K = 2  # Toric code always encodes 2 logical qubits
        self.d = L  # Distance equals linear size
        
    def get_matrices(self):
        """
        Construct the parity check matrices Hx and Hz for the toric code.
        
        Returns
        -------
        Hx : csr_matrix
            X-type parity check matrix (vertex checks) of shape (L*L, N)
        Hz : csr_matrix
            Z-type parity check matrix (plaquette checks) of shape (L*L, N)
        """
        Hx, Hz, _ = toric_code_matrices(self.L)
        return Hx, Hz
    
    def get_logical_z(self):
        """
        Get the logical Z operators for the toric code.
        
        Returns
        -------
        Lz : ndarray
            Logical Z operators as a (2, N) uint8 array
            - Lz[0]: wraps in x-direction (vertical edges at x=0)
            - Lz[1]: wraps in y-direction (horizontal edges at y=0)
        """
        _, _, lz = toric_code_matrices(self.L)
        return lz


