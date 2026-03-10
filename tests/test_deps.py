import unittest
import pyarrow

pyarrow.PyExtensionType = pyarrow.ExtensionType
import torchtune


class TestRegDatasetLoader(unittest.TestCase):
    def test_import(self):
        from torchtune.generation import generate
        from torchtune.models.gemma._component_builders import gemma
        from torchtune.models.llama2._component_builders import llama2
        from torchtune.models.llama3_2._component_builders import llama3_2
        from torchtune.models.mistral._component_builders import mistral
        from torchtune.models.phi3._component_builders import phi3
        from torchtune.models.qwen2._component_builders import qwen2
