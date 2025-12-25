"""
Code Construction Module

Implements the Bivariate Bicycle Code construction for quantum error correction.
Target: [[144, 12, 12]] 'Gross Code'
"""

import numpy as np
from scipy.sparse import csr_matrix, hstack


class BivariateBicycleCode:
    """
    Generates the parity check matrices for Bivariate Bicycle Codes.
    
    The Bivariate Bicycle Code is a quantum LDPC code constructed using
    the lifted product of cyclic matrices. This implementation targets
    the [[144, 12, 12]] Gross Code.
    
    Parameters
    ----------
    L : int, default=12
        First dimension parameter for the torus geometry
    M : int, default=6
        Second dimension parameter for the torus geometry
        
    Attributes
    ----------
    L : int
        First dimension parameter
    M : int
        Second dimension parameter
    N : int
        Total number of physical qubits (2 * L * M)
    """
    
    def __init__(self, L=12, M=6):
        self.L = L
        self.M = M
        self.N = 2 * L * M
        
    def cyclic_matrix(self, shifts):
        """
        Generates a cyclic permutation matrix block.
        
        Parameters
        ----------
        shifts : list of tuples
            List of (du, dv) shift pairs where:
            - du: shift in the L dimension
            - dv: shift in the M dimension
            
        Returns
        -------
        csr_matrix
            Sparse cyclic permutation matrix of shape (L*M, L*M)
        """
        size = self.L * self.M
        # Map (u, v) -> index
        # Basis: u * M + v
        rows, cols, data = [], [], []
        
        for u in range(self.L):
            for v in range(self.M):
                r_idx = u * self.M + v
                for (du, dv) in shifts:
                    c_u = (u + du) % self.L
                    c_v = (v + dv) % self.M
                    c_idx = c_u * self.M + c_v
                    
                    rows.append(r_idx)
                    cols.append(c_idx)
                    data.append(1)
        
        return csr_matrix((data, (rows, cols)), shape=(size, size), dtype=int)

    def get_matrices(self):
        """
        Constructs the parity check matrices Hx and Hz for the Bivariate Bicycle Code.
        
        The code is defined by polynomials:
        - A = x^3 + y + y^2
        - B = y^3 + x + x^2
        
        Where powers of x correspond to shifts in L dimension,
        and powers of y correspond to shifts in M dimension.
        
        Returns
        -------
        Hx : csr_matrix
            X-type parity check matrix of shape (L*M, N)
        Hz : csr_matrix
            Z-type parity check matrix of shape (L*M, N)
        """
        # Polynomials for the Gross Code [[144, 12, 12]]
        # A = x^3 + y + y^2
        # B = y^3 + x + x^2
        
        # Powers of x correspond to shift in L (first dim)
        # Powers of y correspond to shift in M (second dim)
        # Format: (shift_L, shift_M)
        poly_A = [(3, 0), (0, 1), (0, 2)]
        poly_B = [(0, 3), (1, 0), (2, 0)]
        
        A = self.cyclic_matrix(poly_A)
        B = self.cyclic_matrix(poly_B)
        
        # Lifted Product Construction
        # Hx = [A | B]
        # Hz = [B.T | A.T]
        Hx = hstack([A, B])
        Hz = hstack([B.transpose(), A.transpose()])
        
        return Hx, Hz

