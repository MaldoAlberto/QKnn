'''
Usage:
    python test_functions.py
'''
import unittest
import sys
import logging
from qknn.functions import index_positions

LOGGER = logging.getLogger(__name__)


class TestFunctions(unittest.TestCase): # pylint: disable=too-many-instance-attributes
    """Tests for the Q-kNN `functions` module."""

    def test_index_position(self):
        """Test `index_position()`"""
        LOGGER.info(sys._getframe().f_code.co_name) # pylint: disable=protected-access
        self.assertTrue([2], index_positions(4))
        self.assertTrue([1, 4], index_positions(18))
        self.assertTrue([1, 2, 4], index_positions(22))


if __name__ == '__main__':
    unittest.main()
