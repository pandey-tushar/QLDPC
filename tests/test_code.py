"""
Unit tests for code construction module
"""

import unittest
import numpy as np
from scipy.sparse import csr_matrix
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from qldpc.code import BivariateBicycleCode


class TestBivariateBicycleCode(unittest.TestCase):
    """Test cases for BivariateBicycleCode"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.code = BivariateBicycleCode(L=12, M=6)
    
    def test_initialization(self):
        """Test code initialization"""
        self.assertEqual(self.code.L, 12)
        self.assertEqual(self.code.M, 6)
        self.assertEqual(self.code.N, 144)
    
    def test_matrix_shapes(self):
        """Test that matrices have correct shapes"""
        Hx, Hz = self.code.get_matrices()
        self.assertEqual(Hx.shape, (72, 144))
        self.assertEqual(Hz.shape, (72, 144))
    
    def test_matrix_sparsity(self):
        """Test that matrices are sparse"""
        Hx, Hz = self.code.get_matrices()
        self.assertIsInstance(Hx, csr_matrix)
        self.assertIsInstance(Hz, csr_matrix)
    
    def test_cyclic_matrix(self):
        """Test cyclic matrix generation"""
        shifts = [(1, 0), (0, 1)]
        matrix = self.code.cyclic_matrix(shifts)
        self.assertEqual(matrix.shape, (72, 72))
        self.assertIsInstance(matrix, csr_matrix)


if __name__ == "__main__":
    unittest.main()

