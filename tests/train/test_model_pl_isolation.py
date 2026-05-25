"""Pin the load-bearing property that pytorch-lightning is imported only by `preframr.train.model.lightning` and `preframr.train.model.factory`. The other submodules (bodies, heads, losses, tier_map) MUST stay PL-free so a future predict-image variant can include them without dragging the full train wheel set. Re-importing any of those four modules with `torch.modules` cleared of pytorch_lightning must succeed without re-adding it."""

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

    def test_heads_no_pl(self):
        self._import_clean("preframr.train.model.heads")
        self.assertNotIn("pytorch_lightning", sys.modules)

    def test_losses_no_pl(self):
        self._import_clean("preframr.train.model.losses")
        self.assertNotIn("pytorch_lightning", sys.modules)

    def test_tier_map_no_pl(self):
        self._import_clean("preframr.train.model.tier_map")
        self.assertNotIn("pytorch_lightning", sys.modules)


if __name__ == "__main__":
    unittest.main()
