'''
Usage:
    python test_functions.py
'''
import unittest
import sys
import logging
import qiskit
from qknn.functions import index_positions,diffuser,oracle_st,qknn

LOGGER = logging.getLogger(__name__)


class TestFunctions(unittest.TestCase): # pylint: disable=too-many-instance-attributes
    """Tests for the Q-kNN `functions` module."""

    def test_index_position(self):
        """Test `index_position()`"""
        LOGGER.info(sys._getframe().f_code.co_name) # pylint: disable=protected-access
        self.assertTrue([2], index_positions(4))
        self.assertTrue([1, 4], index_positions(18))
        self.assertTrue([1, 2, 4], index_positions(22))
        self.asserIs(list, index_positions(1))
        
        
    def test_diffuser(self):
        """Test `diffuser()`"""
        LOGGER.info(sys._getframe().f_code.co_name) # pylint: disable=protected-access
        self.assertIs(qiskit.circuit.gate.Gate, diffuser(2))
        
    def test_qram(self):
        """Test `qram()`"""
        LOGGER.info(sys._getframe().f_code.co_name) # pylint: disable=protected-access
        self.assertIs(qiskit.circuit.quantumcircuit.QuantumCircuit, qram())
        
    def test_oracle_st(self):
        """Test `oracle_st()`"""
        LOGGER.info(sys._getframe().f_code.co_name) # pylint: disable=protected-access
        self.assertIs(qiskit.circuit.quantumcircuit.QuantumCircuit, oracle_st())
        
        
    def test_qknn(self):
        """Test `qknn()`"""
        LOGGER.info(sys._getframe().f_code.co_name) # pylint: disable=protected-access
        self.assertIs(qiskit.circuit.quantumcircuit.QuantumCircuit, qknn())
        

if __name__ == '__main__':
    unittest.main()
