import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import seal_b2 as S  # noqa: E402


def jl(path, rs):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rs))


class SealB2(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        h = Path(self.tmp.name)
        self.saved = {k: getattr(S, k) for k in ("HOME", "RUNS", "SEALED_RAW", "SEALED_OUT", "MANIFEST")}
        S.HOME, S.RUNS, S.SEALED_RAW = h, h / "roms/runs", h / "roms/sealed-b2"
        S.SEALED_OUT, S.MANIFEST = h / "out/sealed-b2", h / "out/sealed-b2.sha256"
        run = S.RUNS / "b2-A0-D1-s1"
        jl(run / "config.json", [])
        (run / "config.json").write_text(json.dumps({"state": "D1"}))
        marks = ["ckpt-0.pt"] + [f"eval-{i}.pt" for i in range(1, 8)]
        dev = [{"ckpt": c, "state": st, "seed": 1000 + i, "P": 1.0} for c in marks for st in ("D1", "D2") for i in range(8)]
        jl(run / "eval/episodes.jsonl", dev)
        jl(run / "eval/windows.jsonl", [{"ckpt": c, "state": st} for c in marks for st in ("D1", "D2")])
        (run / "eval/summary.json").write_text(json.dumps({"summary": {f"{c}|{st}": {} for c in marks for st in ("D1", "D2")}}))
        held = [{**r, "state": {"D1": "HA", "D2": "HB"}[r["state"]]} for r in dev]
        jl(S.SEALED_RAW / "b2-A0-D1-s1/eval-heldout/episodes.jsonl", held)
        jl(S.SEALED_RAW / "b2-A0-D1-s1/eval-heldout/windows.jsonl", [{"ckpt": c, "state": "HA"} for c in marks])
        (S.SEALED_RAW / "b2-A0-D1-s1/eval-heldout/summary.json").write_text(
            json.dumps({"summary": {f"{c}|{st}": {} for c in marks for st in ("HA", "HB")}}))
        self.run = run

    def tearDown(self):
        for k, v in self.saved.items():
            setattr(S, k, v)
        self.tmp.cleanup()

    def test_split_views_counts_and_manifest(self):
        report, n_dev, n_held = S.split_run("b2-A0-D1-s1", "e1-a")
        self.assertEqual((len(n_dev), set(n_dev.values()), len(n_held), set(n_held.values())), (16, {8}, 16, {8}))
        prim = [json.loads(l) for l in open(self.run / "eval-dev-primary/episodes.jsonl")]
        cross = [json.loads(l) for l in open(self.run / "eval-dev-cross/episodes.jsonl")]
        self.assertEqual(({r["state"] for r in prim}, {r["state"] for r in cross}), ({"D1"}, {"D2"}))
        self.assertEqual((len(prim), len(cross)), (56, 56), "tested views: 7 marks x 8, no ckpt-0")
        m0 = [json.loads(l) for l in open(self.run / "eval-dev-mark0/episodes.jsonl")]
        self.assertEqual({r["ckpt"] for r in m0}, {"ckpt-0.pt"})
        self.assertTrue(all(r["u_pool"] == ["D1", "D2"] and r["vm"] == "e1-a" for r in prim))
        self.assertTrue(all("P" not in s for s in report), "the report must print counts only")
        files = S.sealed_files()
        names = [S.rel(p).split("b2-A0-D1-s1/")[1] for p in files]
        self.assertEqual(names, [f"eval-heldout{v}/{f}" for v in ("-test", "")
                                 for f in ("episodes.jsonl", "summary.json", "windows.jsonl")])
        ht = [json.loads(l) for l in open(S.SEALED_RAW / "b2-A0-D1-s1/eval-heldout-test/episodes.jsonl")]
        self.assertEqual((len(ht), any(r["ckpt"] == "ckpt-0.pt" for r in ht)), (112, False))

    def test_manifest_note_lines_and_verify(self):
        saved = S.CODE, S.NOTES
        try:
            S.CODE = S.HOME / "out"
            S.NOTES = {"sealed-b2/dry/": "dry-run output"}
            (S.SEALED_OUT / "dry").mkdir(parents=True)
            (S.SEALED_OUT / "dry/x.json").write_text("{}")
            sys.argv = ["seal_b2.py", "split"]
            S.main()
            lines = S.MANIFEST.read_text().splitlines()
            self.assertEqual(lines[0], "# ~/out/sealed-b2/dry: dry-run output")
            self.assertTrue(any(l.endswith("sealed-b2/dry/x.json") for l in lines))
            sys.argv = ["seal_b2.py", "verify"]
            with self.assertRaises(SystemExit) as e:
                S.main()
            self.assertEqual(e.exception.code, 0)
        finally:
            S.CODE, S.NOTES = saved

    def test_heldout_row_in_runs_tree_is_a_leak(self):
        jl(self.run / "eval/episodes.jsonl", [{"ckpt": "ckpt-0.pt", "state": "HA", "seed": 1000}])
        with self.assertRaises(SystemExit):
            S.split_run("b2-A0-D1-s1", "e1-a")

    def test_heldout_trace_in_runs_tree_is_a_leak(self):
        t = self.run / "eval/traces/eval-1-HB-1000.jsonl.gz"
        t.parent.mkdir(parents=True)
        t.write_bytes(b"")
        self.assertEqual(S.leak_check("b2-A0-D1-s1"), [str(t)])


if __name__ == "__main__":
    unittest.main()
