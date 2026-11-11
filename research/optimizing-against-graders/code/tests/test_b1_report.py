"""b1_report.py: collapse rule, share of P from traces, seed bootstrap, H0 tests on synthetic rows, SVG output.
No ROM needed.  Run:  .venv/bin/python -m unittest discover -s tests"""
import gzip
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import b1_report as B  # noqa: E402

MARKS = [250880, 500736, 1000448, 2000896, 3000320, 4000768, 5000192]


def ep(step, state, seed, T, P, lp=None, events=()):
    return {"ckpt": f"eval-{step}.pt", "step": step, "state": state, "seed": seed, "T": T, "P": P,
            "leg_progress": T if lp is None else lp, "events": list(events)}


def world(a0_T, a2_T, a0_P, seeds=(1, 2, 3)):
    """info + rows for D1-trained A0 / A2pp runs; *_T(seed, mark_index) and a0_P(mark_index) give the values."""
    info, rows = {}, {}
    for arm, tf in (("A0", a0_T), ("A2pp", a2_T)):
        for s in seeds:
            run = f"{arm}-D1-s{s}"
            info[run] = {"arm": arm, "state": "D1", "seed": s}
            rows[run] = [ep(m, "D1", 1000 + k, tf(s, i), a0_P(i) if arm == "A0" else 1.0)
                         for i, m in enumerate(MARKS) for k in range(8)]
    return info, rows


class Collapse(unittest.TestCase):
    def test_one_below_best_earlier(self):
        info, rows = world(lambda s, i: 0, lambda s, i: [0, 1, 2, 3, 1.5, 2, 3][i] if s == 2 else min(i, 3),
                           lambda i: 1.0)
        self.assertEqual(B.collapsed(info, rows), {("A2pp-D1-s2", MARKS[4]), ("A2pp-D1-s2", MARKS[5])})


class SeedBoot(unittest.TestCase):
    def test_ci_contains_mean_and_is_zero_width_for_constants(self):
        lo, hi = B.seed_boot([[1.0] * 8, [1.0] * 8])
        self.assertEqual((lo, hi), (1.0, 1.0))
        lo, hi = B.seed_boot([[0, 1] * 4, [1, 1] * 4, [0, 0] * 4])
        self.assertLessEqual(lo, 0.5)
        self.assertGreaterEqual(hi, 0.5)


class ShareOfP(unittest.TestCase):
    def test_inside_merged_ranges(self):
        with tempfile.TemporaryDirectory() as d:
            tf = Path(d) / "t.jsonl.gz"
            with gzip.open(tf, "wt") as f:
                f.write(json.dumps({"header": 1}) + "\n")
                for s in range(1, 11):
                    f.write(json.dumps({"ep_step": s, "p": 1.0 if s <= 4 else 0.5}) + "\n")
            tot, ins = B.p_inside(tf, [{"type": "D", "step": 1, "end": 3}, {"type": "D", "step": 3, "end": 5},
                                       {"type": "M", "step": 9, "end": 10}])
            self.assertEqual(tot, 7.0)
            self.assertEqual(ins, {"D": 4.5, "M": 1.0})       # steps 1–5 once each (overlap not double counted)


class H0(unittest.TestCase):
    def test_supported_when_P_rises_and_T_lags_oracle(self):
        # A0: P rises, T stays 0; A2″ T climbs to 3: (i) and (ii) hold, (iii) not greater holds (T flat)
        info, rows = world(lambda s, i: 0, lambda s, i: min(i, 3), lambda i: 10.0 * (i + 1))
        res = B.h0_tests(info, rows, B.collapsed(info, rows), states=("D1",))
        st = res["states"]["D1"]
        self.assertTrue(st["i"]["holds"])
        self.assertTrue(all(r["holds"] for r in st["T"]["ii"]))
        self.assertTrue(st["T"]["iii"]["holds"])
        self.assertEqual(st["T"]["verdict"], "H0 supported")

    def test_falsifier_when_A0_tracks_oracle(self):
        info, rows = world(lambda s, i: min(i, 3), lambda s, i: min(i, 3), lambda i: 10.0 * (i + 1))
        st = B.h0_tests(info, rows, set(), states=("D1",))["states"]["D1"]
        self.assertFalse(any(r["holds"] for r in st["T"]["ii"]))
        self.assertEqual(st["T"]["verdict"], "falsifier met")

    def test_iii_degenerate_when_P50_only_at_5M(self):
        info, rows = world(lambda s, i: 0, lambda s, i: 3, lambda i: 100.0 if i == 6 else 1.0)
        st = B.h0_tests(info, rows, set(), states=("D1",))["states"]["D1"]
        self.assertTrue(st["T"]["iii"]["degenerate"])
        self.assertFalse(st["T"]["iii"]["holds"])

    def test_collapsed_reference_dropped(self):
        info, rows = world(lambda s, i: 0, lambda s, i: (0 if s == 1 and i >= 4 else min(i, 3)), lambda i: 1.0 + i)
        coll = B.collapsed(info, rows)
        st = B.h0_tests(info, rows, coll, states=("D1",))["states"]["D1"]
        self.assertEqual([r["A2pp_seeds_kept"] for r in st["T"]["ii"]], [3, 2, 2, 2])


class H0Undefined(unittest.TestCase):
    def test_iii_undefined_when_P5M_not_above_start(self):
        info, rows = world(lambda s, i: 0, lambda s, i: 3, lambda i: 0.0)
        st = B.h0_tests(info, rows, set(), states=("D1",))["states"]["D1"]
        self.assertTrue(st["T"]["iii"]["undefined"])
        self.assertIsNone(st["T"]["iii"]["mark_P50"])
        self.assertFalse(st["T"]["iii"]["holds"])
        self.assertNotEqual(st["T"]["verdict"], "H0 supported")
        info, rows = world(lambda s, i: 0, lambda s, i: 3, lambda i: [5, 9, 9, 9, 9, 9, 4][i])   # P(5M) <= P(0.25M)
        self.assertTrue(B.h0_tests(info, rows, set(), states=("D1",))["states"]["D1"]["T"]["iii"]["undefined"])


class CollapsedReference(unittest.TestCase):
    """r0_compare --drop: collapsed A2″ seed-ckpts leave the reference; no seed left -> 'reference collapsed'."""

    def make(self, root, run, arm, fire_d, ckpts=("eval-100.pt", "eval-200.pt")):
        d = Path(root) / run
        (d / "eval").mkdir(parents=True)
        (d / "config.json").write_text(json.dumps({"arm": arm, "seed": int(run[-1]) if run[-1].isdigit() else 0}))
        with open(d / "eval" / "windows.jsonl", "w") as f:
            for ck in ckpts:
                for seed in range(8):
                    for w in range(4):
                        f.write(json.dumps({"ckpt": ck, "state": "D1", "seed": seed,
                                            "fired": ["D"] if fire_d(ck, seed, w) else []}) + "\n")
        return d / "eval"

    def test_drop_and_label(self):
        import r0_compare as R
        with tempfile.TemporaryDirectory() as root:
            a0 = [self.make(root, f"A0-D1-s{i}", "A0", lambda ck, s, w: w < 3) for i in (1, 2)]
            a2 = [self.make(root, f"A2pp-D1-s{i}", "A2pp", lambda ck, s, w: False) for i in (1, 2)]
            r0 = self.make(root, "R0", "random", lambda ck, s, w: False, ckpts=("eval-0.pt",))
            R.DROP.clear()
            R.DROP.update({("A2pp-D1-s1", "eval-200.pt"), ("A2pp-D1-s2", "eval-200.pt"), ("A2pp-D1-s1", "eval-100.pt")})
            try:
                rows = R.compare(a0, r0, vs_dirs=a2)
                self.assertEqual({r["ckpt"]: r["ref_seeds"] for r in rows if r["detector"] == "D"}, {"eval-100.pt": 1})
                attr = R.attribute(R.compare(a0, r0), rows, R.compare(a2, r0), R.ref_cells(a2))
                self.assertEqual(attr["cells"]["eval-200.pt|D1|D"], "undetermined_reference_collapsed")
                self.assertEqual(attr["cells"]["eval-100.pt|D1|D"], "proxy_attributed")
                self.assertEqual(R.ref_cells(a2), {("eval-100.pt", "D1")})
            finally:
                R.DROP.clear()


class Fig(unittest.TestCase):
    def test_svg_writes_hollow_and_ticks(self):
        info, rows = world(lambda s, i: 0, lambda s, i: [0, 1, 2, 3, 1.5, 2, 3][i], lambda i: 1.0 + i)
        for r in rows.values():
            for e in r:
                e["kl_init"] = 0.1 * MARKS.index(e["step"])
        with tempfile.TemporaryDirectory() as d:
            B.fig1(info, rows, Path(d), "f", [("D1", "D1")], [], B.collapsed(info, rows),
                   {"D1→D1": [(500736, "D", 1), (1000448, "D·s2", 2)]}, True)
            svg = (Path(d) / "f.svg").read_text()
            self.assertIn("(collapsed)", svg)
            self.assertIn(">D·s2<", svg)
            self.assertIn("prefers-color-scheme: dark", svg)
            self.assertTrue((Path(d) / "f.md").exists() and (Path(d) / "f.json").exists())


if __name__ == "__main__":
    unittest.main()
