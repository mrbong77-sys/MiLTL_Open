"""Consolidated results sheet — AUROC/confusion metrics + DeLong significance. DGX-free."""
import unittest

import numpy as np

from scripts.consolidate_results import _auroc, _metrics, delong_p


class TestConsolidate(unittest.TestCase):
    def test_auroc_perfect_and_random(self):
        y = np.array([0, 0, 1, 1])
        self.assertAlmostEqual(_auroc([0.1, 0.2, 0.8, 0.9], y), 1.0)
        self.assertAlmostEqual(_auroc([0.9, 0.8, 0.2, 0.1], y), 0.0)

    def test_metrics_keys_and_ranges(self):
        rng = np.random.default_rng(1)
        y = (rng.random(100) < 0.5).astype(int)
        s = y * 0.5 + rng.random(100) * 0.5
        m = _metrics(s, y)
        for k in ("AUROC", "ACC", "SEN", "SPE", "PPV", "NPV"):
            self.assertIn(k, m)
            self.assertGreaterEqual(m[k], 0.0); self.assertLessEqual(m[k], 1.0)

    def test_delong_significant_when_far_apart(self):
        rng = np.random.default_rng(2)
        y = (rng.random(300) < 0.5).astype(int)
        strong = y * 0.7 + rng.random(300) * 0.3
        weak = y * 0.1 + rng.random(300) * 0.9
        p = delong_p(y, strong, weak)
        self.assertLess(p, 0.05)                         # significant difference

    def test_delong_not_significant_when_near_identical(self):
        rng = np.random.default_rng(3)
        y = (rng.random(300) < 0.5).astype(int)
        s1 = y * 0.5 + rng.random(300) * 0.5
        s2 = s1 + rng.normal(0, 1e-3, 300)               # nearly identical
        p = delong_p(y, s1, s2)
        self.assertGreater(p, 0.05)                      # not significant (no false positive)


if __name__ == "__main__":
    unittest.main()
