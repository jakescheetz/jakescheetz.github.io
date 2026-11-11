"""train_series.py (§9 H10): binning by end step, per-bin fields, flag prevalence. No ROM needed."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import train_series as S  # noqa: E402


class Bins(unittest.TestCase):
    def test_bins_use_nominal_edges_and_end_steps(self):
        eps = [{"global_step": g, "P": float(g // 131072), "T": 1 if g > 500_000 else 0,
                "hacks_active": ["O"] if g == 917504 else [], "stuck_active": ["K"]}
               for g in range(131072, 1_200_000, 131072)]
        ms = {m: m + 448 for m in S.NOMINAL}
        b = S.bins_of(eps, ms)
        self.assertEqual([e["global_step"] for e in b[1_000_448]], [786432, 917504])   # (750k, 1M]
        self.assertEqual([e["global_step"] for e in b[250_448]], [131072])             # (0, 250k]
        pb = S.per_bin(b[1_000_448])
        self.assertEqual((pb["n"], pb["P"], pb["T"]), (2, 6.5, 1.0))
        self.assertEqual(pb["flag_prevalence"], {"K": 1.0, "O": 0.5})
        self.assertIsNone(pb["P_terms"])                      # batch 1 rows carry no P_terms


if __name__ == "__main__":
    unittest.main()
