"""b2_reads.py: H4′ tripwires and per-run summaries on synthetic run dirs. No ROM needed."""
import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def row(step, seed, P, T=0, lp=0.0, L=False, obs=2, sha="ab =ckpt-0", state="D1"):
    return {"ckpt": f"eval-{step}.pt" if step else "ckpt-0.pt", "step": step, "state": state, "seed": seed,
            "P": P, "T": T, "leg_progress": lp, "obs_version": obs, "init_sha256": sha,
            "P_terms": {"explore": P - 1, "level": 1.0, "heal": 0.0, "seen": 0.0},
            "events": [{"type": "L"}] if L else []}


class Reads(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="b2reads"))
        os.environ["POKEMON_RUNS_DIR"] = str(self.tmp)
        import b2_reads
        self.B = b2_reads

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def run_dir(self, run, arm, state, reads, train=(), vm=None, obs=2, rh=None):
        d = self.tmp / run
        d.mkdir()
        (d / "config.json").write_text(json.dumps({
            "run_id": run, "arm": arm, "state": state, "total_steps": 2_500_000, "obs_version": obs,
            "reward_hash": rh or self.B.REWARD_HASH[arm], **({"vm": vm} if vm else {})}))
        for step, rows in reads.items():
            o = d / "eval-interim" / (f"eval-{step}" if step else "ckpt-0")
            o.mkdir(parents=True)
            (o / "episodes.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        (d / "episodes.jsonl").write_text("".join(json.dumps(t) + "\n" for t in train))
        (d / "metrics.jsonl").write_text("".join(json.dumps({"sps_update": 400.0}) + "\n" for _ in range(30)))
        return d

    def test_A0_pass_and_summary(self):
        reads = {0: [row(0, s, 30 + s % 2) for s in range(8)],
                 1_000_448: [row(1_000_448, s, 70 + s) for s in range(8)],
                 2_500_096: [row(2_500_096, s, 95 + s) for s in range(8)]}
        train = [{"global_step": 500_000, "P": 40.0, "T": 0, "P_terms": {"explore": 39, "level": 1, "heal": 0}}]
        self.run_dir("A0-D1-s0", "A0", "D1", reads, train, vm="e1-a")
        t = self.B.tripwire("A0-D1-s0")
        self.assertEqual({r["id"]: r["status"] for r in t["tripwires"]}, {"A-1": "pass", "A-2": "pass", "C": "pass"})
        s = json.loads((self.tmp / "A0-D1-s0" / "summary-marks.json").read_text())
        m1 = next(m for m in s["marks"] if m["step"] == 1_000_448)
        self.assertAlmostEqual(m1["eval"]["x_R0"], round(73.5 / 29.7, 2))
        self.assertAlmostEqual(m1["eval"]["g"], round((73.5 - 30.5) / (417 - 30.5), 4))
        self.assertEqual(m1["train"]["episodes"], 1)                 # training episodes since the last mark
        self.assertAlmostEqual(s["sps_ratio"], round(400 / 411, 3))
        self.assertIn("interim, validity only", self.B.brief("A0-D1-s0"))

    def test_A0_trips_below_2xR0_and_on_a_fall(self):
        reads = {0: [row(0, s, 30, state="D2") for s in range(8)],
                 1_000_448: [row(1_000_448, s, 50 + s, state="D2") for s in range(8)]}
        self.run_dir("A0-D2-s0", "A0", "D2", reads)
        self.assertIn("A-1", self.B.tripwire("A0-D2-s0")["tripped"])          # 53.5 < 62.3
        reads = {0: [row(0, s, 30) for s in range(8)], 1_000_448: [row(1_000_448, s, 120 + s) for s in range(8)],
                 2_500_096: [row(2_500_096, s, 85 + s) for s in range(8)]}
        self.run_dir("A0-D1-s9", "A0", "D1", reads)
        t = self.B.tripwire("A0-D1-s9")
        self.assertEqual(t["tripped"], ["A-2"])                                 # above 83.3 but falling, p <= .05

    def test_A2pp_B_rules(self):
        reads = {0: [row(0, s, 30, lp=0.2) for s in range(8)],
                 1_000_448: [row(1_000_448, s, 20, lp=0.2) for s in range(8)]}
        self.run_dir("A2pp-D1-s0", "A2pp", "D1", reads)
        self.assertIn("B-1", self.B.tripwire("A2pp-D1-s0")["tripped"])           # leg progress not up
        reads = {0: [row(0, s, 30, lp=0.2) for s in range(8)],
                 1_000_448: [row(1_000_448, s, 20, lp=0.8 + s / 100) for s in range(8)],
                 2_500_096: [row(2_500_096, s, 20, T=0, lp=0.9) for s in range(8)]}
        self.run_dir("A2pp-D1-s8", "A2pp", "D1", reads)
        t = self.B.tripwire("A2pp-D1-s8")
        self.assertEqual({r["id"]: r["status"] for r in t["tripwires"]}["B-2"], "trip")   # no M1 at 2.5M

    def test_A1_S_rules_and_notify(self):
        mv = lambda r, maps: {**r, "maps_visited": maps}
        reads = {0: [mv(row(0, s, 6.0), ["0x28"]) for s in range(8)],
                 1_000_448: [mv(row(1_000_448, s, 12 + s, T=1 if s < 2 else 0), ["0x28", "0x0"]) for s in range(8)],
                 2_500_096: [mv(row(2_500_096, s, 13 + s, T=1 if s < 3 else 0), ["0x28", "0x0", "0xc"]) for s in range(8)]}
        self.run_dir("r1-A1-D1-s1", "A1", "D1", reads, vm="e1-a")
        t = self.B.tripwire("r1-A1-D1-s1")
        st = {r["id"]: r["status"] for r in t["tripwires"]}
        self.assertEqual(st, {"S-1": "pass", "S-2": "pass", "S-3@1M": "notify", "S-3@2.5M": "pass", "C": "pass"})
        self.assertEqual((t["tripped"], t["notify"]), ([], ["S-3@1M"]))      # S-3 never pauses
        d = {r["id"]: r["detail"] for r in t["tripwires"]}
        self.assertIn("exact p 0.0565", d["S-3@1M"])                          # k = 2: C(8,2)/C(32,2)
        self.assertIn("exact p 0.0113", d["S-3@2.5M"])                        # k = 3
        s = json.loads((self.tmp / "r1-A1-D1-s1" / "summary-marks.json").read_text())
        self.assertAlmostEqual(s["sps_ratio"], round(400 / 389.2, 3))          # I12: batch-2 A0 D1 rate on e1-a
        m1 = next(m for m in s["marks"] if m["step"] == 1_000_448)["eval"]
        self.assertEqual((m1["x_R0"], m1["a1"]["P1"], m1["a1"]["never_left"]), (None, 15.5, 0))

    def test_A1_S1_and_S2_trip(self):
        mv = lambda r, maps: {**r, "maps_visited": maps}
        reads = {0: [mv(row(0, s, 6.0), ["0x28"]) for s in range(8)],
                 1_000_448: [mv(row(1_000_448, s, 7.0), ["0x28"]) for s in range(8)]}
        self.run_dir("r1-A1-D1-s2", "A1", "D1", reads)
        t = self.B.tripwire("r1-A1-D1-s2")
        self.assertEqual(t["tripped"], ["S-1", "S-2"])                         # below R0's P1; parked 8/8 at 1M
        self.assertEqual(t["notify"], ["S-3@1M"])                              # 0/8 reach M1

    def test_p_vs_zero_exact(self):
        p = lambda k: self.B.p_vs_zero([1] * k + [0] * (8 - k), [0] * 24)
        self.assertEqual([round(p(k)[0], 4) for k in (0, 1, 2, 3)], [1.0, 0.25, 0.0565, 0.0113])
        self.assertEqual(p(2)[1], "exact")

    def test_C_canary_and_infra(self):
        reads = {0: [row(0, s, 30, L=(s == 3)) for s in range(8)]}
        self.run_dir("A0-D1-s7", "A0", "D1", reads, rh="000000000000")
        t = self.B.tripwire("A0-D1-s7")
        c = next(r for r in t["tripwires"] if r["id"] == "C")
        self.assertEqual(c["status"], "trip")
        self.assertIn("L > 0", c["detail"])
        self.assertIn("reward hash", c["detail"])
        self.assertEqual({r["id"]: r["status"] for r in t["tripwires"]}["A-1"], "pending")

    def test_closing_read_prints_C(self):
        """H4′ C at every read, closing included: summary (the closing read) evaluates and prints C, without tripwires."""
        d = self.run_dir("A2pp-D2-s3", "A2pp", "D2", {}, vm="e1-a")
        (d / "eval").mkdir()
        (d / "eval" / "episodes.jsonl").write_text("".join(json.dumps(row(st, s, 30, T=1, state="D2")) + "\n"
                                                         for st in (0, 5_000_192) for s in range(8)))
        (d / "metrics.jsonl").write_text("".join(json.dumps({"sps_update": 300.0}) + "\n" for _ in range(130)))
        s = self.B.summary("A2pp-D2-s3")
        self.assertEqual(s["C"]["status"], "trip")                         # 300 / 381 = 0.79 < 0.85
        self.assertIn("sps 300.0 < 85% of 381", s["C"]["detail"])
        self.assertIn("C (H4′, every read incl. closing): TRIP", (d / "summary-marks.md").read_text())
        self.assertIn("; C trip (sps 300.0", self.B.brief("A2pp-D2-s3"))
        (d / "metrics.jsonl").write_text("".join(json.dumps({"sps_update": 400.0}) + "\n" for _ in range(130)))
        s = self.B.summary("A2pp-D2-s3")
        self.assertEqual(s["C"]["status"], "pass")
        self.assertIn("C (H4′, every read incl. closing): PASS: 16 eval rows ok", (d / "summary-marks.md").read_text())

    def test_sps_skips_updates_inside_interim_reads(self):
        d = self.run_dir("A0-D1-s6", "A0", "D1", {}, vm="e1-a")
        rows = [{"sps_update": 400.0, "time": 1000.0 + 10 * i, "t_collect": 8, "t_learn": 1} for i in range(60)]
        rows[40:50] = [{"sps_update": 30.0, "time": 1000.0 + 10 * i, "t_collect": 8, "t_learn": 1} for i in range(40, 50)]
        (d / "metrics.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        (d / "eval-interim").mkdir()
        (d / "eval-interim" / "stops.json").write_text(json.dumps([[1395.0, 1495.0, "ckpt-0.pt", 9]]))
        self.assertEqual(self.B.sps_now(d), 400.0)
        (d / "eval-interim" / "stops.json").write_text(json.dumps([[1000.0, 1600.0, "x", 9]]))
        self.assertIsNone(self.B.sps_now(d))                                   # < 20 clean updates: pending

    def test_resume_is_flagged_as_partial(self):
        d = self.run_dir("A0-D1-s5", "A0", "D1", {0: [row(0, s, 30) for s in range(8)]})
        cfg = json.loads((d / "config.json").read_text())
        cfg["resume_log"] = [{"resume": 1, "kind": "resume", "resumed_from_ckpt": "ckpt-700000.pt",
                              "resumed_from_step": 700000, "stopped_at_step": 712000, "replayed_steps": 12000,
                              "restored": {"emulator_state": False, "full_restore": False}}]
        (d / "config.json").write_text(json.dumps(cfg))
        self.assertIn("RESUMED x1", self.B.brief("A0-D1-s5"))
        self.assertIn("partial", (d / "summary-marks.md").read_text())

    def test_H9_resume_rule(self):
        B = self.B
        cfg = {"total_steps": 5_000_000}
        ok = {"resume": 1, "kind": "resume", "resumed_from_ckpt": "ckpt-1.pt", "replayed_steps": 40_000,
              "stop": {"reason": "signal 15"}}
        self.assertEqual(B.resume_rule({**cfg, "resume_log": [ok]}, 5_000_192)["status"], "keep")
        self.assertTrue(B.resume_rule({**cfg, "resume_log": [ok]}, 3_000_000)["status"].startswith("pending"))
        crash = {**ok, "stop": {"reason": "exception RuntimeError: x"}}
        self.assertEqual(B.resume_rule({**cfg, "resume_log": [crash]}, 5_000_192)["status"], "escalate")
        vm = {**ok, "stop": None, "vm_event": {"type": "preempted"}}
        self.assertEqual(B.resume_rule({**cfg, "resume_log": [vm]}, 5_000_192)["status"], "keep")
        self.assertEqual(B.resume_rule({**cfg, "resume_log": [ok, ok, ok]}, 5_000_192)["status"], "escalate")
        self.assertEqual(B.resume_rule({**cfg, "resume_log": [ok, {**ok, "replayed_steps": 70_000}]}, 5_000_192)["status"],
                         "escalate")                                   # 110k > 100k
        self.assertEqual(B.resume_rule({**cfg, "resume_log": [{**ok, "kind": "restart", "resumed_from_ckpt": None}]},
                                       5_000_192)["status"], "escalate")
        self.assertEqual(B.resume_rule(cfg, 5_000_192)["status"], "none")

    def test_heldout_training_state_not_summarised(self):
        train = [{"global_step": 500_000, "P": 40.0, "T": 0}]
        self.run_dir("A2pp-HA-s1", "A2pp", "HA", {}, train)
        s = self.B.summary("A2pp-HA-s1")
        self.assertFalse(any("train" in m or "eval" in m for m in s["marks"]))


if __name__ == "__main__":
    unittest.main()
