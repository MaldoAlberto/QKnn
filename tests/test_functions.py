'''
Usage:
    python test_functions.py
'''
import unittest
import sys
import logging
import qiskit
from qknn.functions import index_positions, diffuser, qram, oracle_st, qknn,qknn_experiments

LOGGER = logging.getLogger(__name__)


class TestFunctions(unittest.TestCase): # pylint: disable=too-many-instance-attributes
    """Tests for the Q-kNN `functions` module."""

    def test_index_position(self):
        """Test `index_position()`"""
        LOGGER.info(sys._getframe().f_code.co_name) # pylint: disable=protected-access
        self.assertTrue([2], index_positions(4))
        self.assertTrue([1, 4], index_positions(18))
        self.assertTrue([1, 2, 4], index_positions(22))
        self.assertIs(list, type(index_positions(1)))


    def test_diffuser(self):
        """Test `diffuser()`"""
        LOGGER.info(sys._getframe().f_code.co_name) # pylint: disable=protected-access
        self.assertIs(qiskit.circuit.gate.Gate, type(diffuser(2)))

    def test_qram(self):
        """Test `qram()`"""
        LOGGER.info(sys._getframe().f_code.co_name) # pylint: disable=protected-access
        self.assertIs(qiskit.circuit.quantumcircuit.QuantumCircuit, type(qram()))

    def test_oracle_st(self):
        """Test `oracle_st()`"""
        LOGGER.info(sys._getframe().f_code.co_name) # pylint: disable=protected-access
        self.assertIs(qiskit.circuit.quantumcircuit.QuantumCircuit, type(oracle_st()))

    def test_qknn(self):
        """Test `qknn()`"""
        LOGGER.info(sys._getframe().f_code.co_name) # pylint: disable=protected-access
        self.assertIs(qiskit.circuit.quantumcircuit.QuantumCircuit, type(qknn()))



    def test_mae(self):
        """Test `mae()`"""
        LOGGER.info(sys._getframe().f_code.co_name) # pylint: disable=protected-access
        self.assertIs(tuple, type(qknn_experiments().mae_acc()))
        
        
    def test_draw_qknn(self):
        """Test `draw_qknn()`"""
        LOGGER.info(sys._getframe().f_code.co_name) # pylint: disable=protected-access
        self.assertIs(qiskit.circuit.quantumcircuit.QuantumCircuit, type(qknn_experiments().draw_qknn()))
        
        
    def test_print_results(self):
        """Test `print_results()`"""
        LOGGER.info(sys._getframe().f_code.co_name) # pylint: disable=protected-access
        self.assertIs(None, qknn_experiments().print_results())
        

if __name__ == '__main__':
    unittest.main()
