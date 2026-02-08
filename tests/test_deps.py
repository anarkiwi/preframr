import unittest
import torchtune


class TestRegDatasetLoader(unittest.TestCase):
    def test_import(self):
        _logger = torchtune.utils.get_logger()
