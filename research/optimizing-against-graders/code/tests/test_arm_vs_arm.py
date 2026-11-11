"""r0_compare.py arm-vs-arm mode and proxy attribution (experiment-e1.md §9 "A2″ review"). No ROM needed."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import r0_compare as RC  # noqa: E402


def write_eval(root, run, fire, ckpt="eval-1000.pt", states=("D1",), n_eps=8, n_win=10):
    """An eval dir whose windows fire `fire(state, seed, w)` -> list of detector letters."""
    d = Path(root) / run / "eval"
    d.mkdir(parents=True)
    with open(d / "windows.jsonl", "w") as f:
        for st in states:
            for seed in range(1000, 1000 + n_eps):
                for w in range(n_win):
                    f.write(json.dumps({"ckpt": ckpt, "state": st, "seed": seed, "fired": fire(st, seed, w)}) + "\n")
    return d


class TestArmVsArm(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        t = self.tmp.name
        # A0: D in every window, M in every window. Oracle arm: M in every window, no D. R0: nothing.
        self.a0 = write_eval(t, "A0-D1-s0", lambda st, s, w: ["D", "M"])
        self.ref = write_eval(t, "A2pp-D1-s0", lambda st, s, w: ["M"])
        self.r0 = write_eval(t, "R0", lambda st, s, w: [], ckpt="random-0.pt", n_eps=24)

    def tearDown(self):
        self.tmp.cleanup()

    def test_vs_arm_matches_cells_and_tests_both_ways(self):
        rows = RC.compare([self.a0], vs_dirs=[self.ref])
        by = {r["detector"]: r for r in rows}
        self.assertTrue(by["D"]["counts_after_holm"])          # A0 > A2″ on D
        self.assertFalse(by["M"]["counts_after_holm"])         # tie on M
        self.assertEqual(by["D"]["r0_episodes"], 8)            # reference = the other arm's 8 episodes

    def test_attribution(self):
        vs_r0 = RC.compare([self.a0], self.r0)
        vs_ref = RC.compare([self.a0], vs_dirs=[self.ref])
        att = RC.attribute(vs_r0, vs_ref, RC.compare([self.ref], self.r0))
        self.assertEqual(att["proxy_attributed"], ["D"])
        self.assertEqual(att["learner_artifact"], ["M"])      # beats R0, not A2″, and A2″ has it above R0
        self.assertEqual(att["undetermined"], [])
        # without the reference-vs-R0 evidence, M can't be called an artifact
        att2 = RC.attribute(vs_r0, vs_ref)
        self.assertEqual((att2["learner_artifact"], att2["undetermined"]), ([], ["M"]))

    def test_undetermined_when_reference_underpowered(self):
        # A0: M in 3/8 episodes (beats R0's 0/24: p = C(8,3)/C(32,3) ≈ 0.011). Reference: M in 2/8 (vs R0:
        # p ≈ 0.056, not present), and A0 doesn't beat it. So M is undetermined, not an artifact.
        t = tempfile.mkdtemp(dir=self.tmp.name)
        a0 = write_eval(t, "A0-D1-s1", lambda st, s, w: ["M"] if s - 1000 < 3 else [])
        ref = write_eval(t, "A2pp-D1-s1", lambda st, s, w: ["M"] if s - 1000 < 2 else [])
        vs_r0, vs_ref, ref_r0 = RC.compare([a0], self.r0), RC.compare([a0], vs_dirs=[ref]), RC.compare([ref], self.r0)
        self.assertTrue(any(r["counts_after_holm"] for r in vs_r0 if r["detector"] == "M"))
        self.assertFalse(any(r["counts_after_holm"] for r in ref_r0 if r["detector"] == "M"))
        att = RC.attribute(vs_r0, vs_ref, ref_r0)
        self.assertEqual(att["undetermined"], ["M"])
        self.assertEqual(att["learner_artifact"], [])
        self.assertEqual(att["cells"], {"eval-1000.pt|D1|M": "undetermined"})

    def test_reference_pools_seeds(self):
        ref2 = write_eval(self.tmp.name, "A2pp-D1-s1", lambda st, s, w: [])
        rows = RC.compare([self.a0], vs_dirs=[self.ref, ref2])
        m = next(r for r in rows if r["detector"] == "M")
        self.assertEqual(m["r0_episodes"], 16)
        self.assertAlmostEqual(m["r0_share"], 0.5)


if __name__ == "__main__":
    unittest.main()
