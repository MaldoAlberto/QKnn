'''
Usage:
    python test_functions.py
'''
import unittest
import sys
import logging
import qiskit.circuit
from qiskit.circuit.library import CRYGate, CRZGate
from qknn.functions import index_positions, diffuser, qram

LOGGER = logging.getLogger(__name__)


class TestFunctions(unittest.TestCase): # pylint: disable=too-many-instance-attributes
    """Tests for the Q-kNN `functions` module."""

    def test_index_position(self):
        """Test `index_position()`"""
        LOGGER.info(sys._getframe().f_code.co_name) # pylint: disable=protected-access
        self.assertTrue([2], index_positions(4))
        self.assertTrue([1, 4], index_positions(18))
        self.assertTrue([1, 2, 4], index_positions(22))

    def test_diffuser(self):
        """Test `diffuser()`"""
        nqubits = 2
        gate = diffuser(nqubits=nqubits)
        self.assertEqual(nqubits, gate.num_qubits)
        self.assertEqual("$U_s$", gate.name)
        # evidently `to_matrix` is designed to just raise exceptions, which is too bad.
        with self.assertRaises(qiskit.circuit.exceptions.CircuitError):
            gate.to_matrix()

    def test_qram(self):
        """Test `qram()`"""
        # TODO: test CRY, CRZ, what the circuit does...
        empty_qram = qram(size_QRAM=1, features=0)
        self.assertEqual(type(empty_qram), qiskit.circuit.quantumcircuit.QuantumCircuit)
        empty_crz_qram = qram(size_QRAM=1, features=0, train_set=[], controlled_rotation=CRZGate)
        self.assertEqual(type(empty_crz_qram), qiskit.circuit.quantumcircuit.QuantumCircuit)

    def test_oracle_st(self):
        """Test `oracle_st()`"""
        pass


if __name__ == '__main__':
    unittest.main()
