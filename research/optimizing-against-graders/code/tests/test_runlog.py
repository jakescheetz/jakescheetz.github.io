"""pokered.runlog: resume entries, marker rows and dropping discarded branches. No ROM needed."""
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pokered import runlog  # noqa: E402

U = 1024  # one update = rollout x envs


def mrow(step, t=None, eps=0):
    return {"step": step, "time": t if t is not None else 1000.0 + step / U, "sps_update": 400.0, "episodes": eps}


class TestRunlog(unittest.TestCase):
    def test_kept_rows_with_markers(self):
        rows = [mrow(s) for s in range(U, 11 * U, U)]                 # 1..10 updates, stop after 10
        rows.append({"resume": 1, "step": 6 * U})                     # resumed from the ckpt at 6
        rows += [mrow(s) for s in range(7 * U, 13 * U, U)]
        kept = runlog.kept_rows(rows)
        self.assertEqual([r["step"] for r in kept], list(range(U, 13 * U, U)))
        self.assertEqual(len(rows) - 1 - len(kept), 4)                  # 7..10 of the first branch dropped

    def test_episodes_markers(self):
        eps = [{"global_step": s} for s in (100, 200, 300)] + [{"resume": 1, "global_step": 150}] + \
              [{"global_step": s} for s in (250, 400)]
        self.assertEqual([e["global_step"] for e in runlog.kept_rows(eps)], [100, 250, 400])

    def test_old_logs_synth_markers(self):
        """No markers: the metrics step drop gives the resume; the per-update episode counts place the
        boundary in episodes.jsonl, even when episode steps never drop (the A0 case)."""
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            cfg = {"hp": {"rollout": 128}, "envs": 8, "resume_log": []}
            (d / "config.json").write_text(json.dumps(cfg))
            m = [mrow(s, eps=8 if s in (3 * U, 9 * U) else 0) for s in range(U, 11 * U, U)]
            m += [mrow(s, eps=8 if s == 12 * U else 0) for s in range(7 * U, 14 * U, U)]
            (d / "metrics.jsonl").write_text("".join(json.dumps(r) + "\n" for r in m))
            ep = [{"global_step": 3 * U}] * 8 + [{"global_step": 9 * U}] * 8 + [{"global_step": 12 * U}] * 8
            (d / "episodes.jsonl").write_text("".join(json.dumps(r) + "\n" for r in ep))
            self.assertEqual(runlog.synth_markers(cfg, m), [(10, 16, 6 * U)])
            km = runlog.load_kept(d, "metrics.jsonl")
            self.assertEqual([r["step"] for r in km], list(range(U, 14 * U, U)))
            ke = runlog.load_kept(d, "episodes.jsonl")
            self.assertEqual([e["global_step"] for e in ke], [3 * U] * 8 + [12 * U] * 8)   # 9U branch dropped

    def test_resume_entry_bases(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            (d / "metrics.jsonl").write_text(json.dumps(mrow(10 * U, t=1000.0)) + "\n")
            # no stop.json, no VM event: lower bound
            e = runlog.resume_entry(d, d, 6 * U, "ckpt-6144.pt", 1)
            self.assertEqual((e["basis"], e["stopped_at_step"], e["replayed_steps"]), ("metrics", 10 * U, 4 * U))
            # a VM event 10 s after the last log: estimate = last + 400 sps x 10 s
            (d / runlog.VM_EVENTS).write_text(json.dumps({"id": "1", "type": "preempted",
                                                          "time": "1970-01-01T00:16:50+00:00"}) + "\n")
            e = runlog.resume_entry(d, d, 6 * U, "ckpt-6144.pt", 1)
            self.assertEqual((e["basis"], e["stopped_at_step"]), ("metrics+vm_event", 10 * U + 4000))
            # stop.json written by the trainer on the signal: exact, and renamed for the next segment
            runlog.write_stop(d, 10 * U + 1024, "signal 1")
            e = runlog.resume_entry(d, d, 6 * U, "ckpt-6144.pt", 2)
            self.assertEqual((e["basis"], e["stopped_at_step"], e["stop"]["reason"]), ("stop.json", 11 * U, "signal 1"))
            self.assertTrue((d / "stop-2.json").exists() and not (d / "stop.json").exists())

    def test_last_row_skips_markers_and_cut_lines(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "metrics.jsonl"
            p.write_text(json.dumps(mrow(5 * U)) + "\n" + json.dumps({"resume": 1, "step": 2 * U}) + "\n{\"st")
            self.assertEqual(runlog.last_metrics_row(p)["step"], 5 * U)
            self.assertEqual(runlog.max_step(p), 5 * U)


if __name__ == "__main__":
    unittest.main()
