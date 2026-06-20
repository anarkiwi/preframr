"""Pin the load-bearing property that pytorch-lightning is imported only by `preframr.train.model.lightning` and `preframr.train.model.factory`. The other submodules (bodies, losses) MUST stay PL-free so a future predict-image variant can include them without dragging the full train wheel set. Re-importing either module with `pytorch_lightning` cleared from `sys.modules` must succeed without re-adding it."""

import importlib
import sys
import unittest


class TestPLIsolation(unittest.TestCase):
    def _import_clean(self, modname):
        for k in list(sys.modules):
            if k == modname or k.startswith(modname + "."):
                del sys.modules[k]
        sys.modules.pop("pytorch_lightning", None)
        return importlib.import_module(modname)

    def test_bodies_no_pl(self):
        self._import_clean("preframr.train.model.bodies")
        self.assertNotIn("pytorch_lightning", sys.modules)

    def test_losses_no_pl(self):
        self._import_clean("preframr.train.model.losses")
        self.assertNotIn("pytorch_lightning", sys.modules)


if __name__ == "__main__":
    unittest.main()
