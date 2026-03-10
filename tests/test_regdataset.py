import os
import random
import tempfile
import unittest
import numpy as np
import pandas as pd

from preframr.regdataset import RegDataset
from preframr.stfconstants import UNICODE_BASE


class FakeArgs:
    def __init__(self, seq_len=128, tkvocab=0, diffq=64, tkmodel=None):
        self.reglog = None
        self.reglogs = ""
        self.seq_len = seq_len
        self.tkvocab = tkvocab
        self.tkmodel = tkmodel
        self.max_files = 1
        self.diffq = diffq
        self.token_csv = None


class TestRegDatasetLoader(unittest.TestCase):
    pass
