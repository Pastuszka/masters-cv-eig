import unittest
import cveig
import numpy as np
from scipy.sparse import csr_array
from numpy.testing import assert_almost_equal


class TestTestStat(unittest.TestCase):

    def test_test_stat(self):
        A = csr_array(np.array([[1, 2], [2, 3]]))
        A_test = csr_array(np.array([[0, 1], [1, 2]]))
        x = np.array([1, 2])
        eps = 0.1
        z = 4.216370
        output = cveig.test_stat(A, A_test, x, eps)
        self.assertIsNone(assert_almost_equal(output, z, 6))


class TestRegMatrix(unittest.TestCase):

    def test_norm_reg_matrix(self):
        A = csr_array(np.array([[1, 2], [2, 3]]))
        correct = np.array([[0.1428571, 0.2519763], [0.2519763, 0.3333333]])
        output = cveig.norm_reg_matrix(A).todense()
        self.assertIsNone(assert_almost_equal(output, correct))


class TestSpectral(unittest.TestCase):

    def test_gspectral(self):
        A = csr_array(np.array([[1.0, 2, 3, 4],
                                [2, 2, 1, 3],
                                [3, 1, 3, 4],
                                [4, 3, 4, 4]]))

        kmax = 2
        w = np.array([11.551895, 1.407232])
        v = np.array([[0.4544204, -0.02074883],
                      [0.3507702,  0.78901746],
                      [0.5027302, -0.61092529],
                      [0.6463163,  0.06157284]])
        w_out, v_out = cveig.gspectral(A, kmax)
        self.assertIsNone(assert_almost_equal(w_out, w, 6))
        self.assertIsNone(assert_almost_equal(v_out, v, 6))
