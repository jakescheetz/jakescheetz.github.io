"""Leg-progress metric (§9 "A2″ 10M extension"), extra eval marks, and the run-state logic the VM scripts
use to decide what to resume (gcp/remote/runs.py, gcp/remote/queue.sh).

The runs.py / queue.sh tests reproduce the 2026-11-04 incident: an eval-only R0 dir (a copy of a run's
config with random_policy) was listed as "partial", so `run_plan.sh --resume-all` started it as a trainer.
They run the real scripts against a temporary HOME with a fake trainer; no ROM, no GCP.

Run:  .venv/bin/python -m unittest discover -s tests
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))
from pokered import walkdist as W  # noqa: E402
from test_progress import LineWorld  # noqa: E402


def rows(path):
    """(map, x, T) -> trace rows"""
    return [{"map": mp, "x": x, "y": 0, "T": T} for mp, x, T in path]


class LegProgress(unittest.TestCase):
    def test_partial_leg(self):
        # leg 1 (d = 20 − x): best x = 15 -> d_min 5 -> 0 + (1 − 5/20)
        self.assertAlmostEqual(W.leg_progress(rows([(0, 0, 0), (0, 10, 0), (0, 15, 0), (0, 3, 0)]), LineWorld()),
                               0.75)

    def test_backtracking_keeps_best(self):
        a = W.leg_progress(rows([(0, 0, 0), (0, 15, 0)]), LineWorld())
        b = W.leg_progress(rows([(0, 0, 0), (0, 15, 0), (0, 0, 0)]), LineWorld())
        self.assertEqual(a, b)

    def test_second_leg_after_milestone(self):
        # M1 on entering map 1 (T 0 -> 1); leg 2: d = 10 − x, d_leg 10. Settle: 2 steps after the map change.
        tr = rows([(0, 0, 0), (0, 19, 0), (1, 0, 1), (1, 0, 1), (1, 0, 1), (1, 4, 1), (1, 6, 1)])
        self.assertAlmostEqual(W.leg_progress(tr, LineWorld()), 1 + (1 - 4 / 10))

    def test_cap_at_segment_end(self):
        self.assertEqual(W.leg_progress(rows([(1, 0, 3), (1, 5, 3)]), LineWorld()), 3.0)
        self.assertEqual(W.leg_progress(rows([(1, 0, 4)]), LineWorld()), 3.0)

    def test_no_usable_square_counts_zero(self):
        self.assertEqual(W.leg_progress(rows([(0, -1, 0), (0, -2, 0)]), LineWorld()), 0.0)

    def test_start_beyond_reference_clamps_to_zero(self):
        class Far(LineWorld):
            def d(self, leg, mp, x, y):
                return 40
        self.assertEqual(W.leg_progress(rows([(0, 0, 0), (0, 1, 0)]), Far()), 0.0)


class ExtraEvalMarks(unittest.TestCase):
    def test_marks(self):
        import train
        self.assertEqual(train.eval_marks(5_000_000, [3_000_000]),
                         [250_000, 500_000, 1_000_000, 2_000_000, 3_000_000, 4_000_000, 5_000_000])
        # older configs (no extra_eval_marks) keep their original marks
        self.assertEqual(train.eval_marks(5_000_000), [250_000, 500_000, 1_000_000, 2_000_000, 4_000_000, 5_000_000])
        self.assertNotIn(3_000_000, train.eval_marks(2_000_000, [3_000_000]))


def make_run(runs, name, total, last_step=None, **cfg):
    d = runs / name
    d.mkdir(parents=True)
    (d / "config.json").write_text(json.dumps({"run_id": name, "total_steps": total, **cfg}))
    if last_step is not None:
        (d / "metrics.jsonl").write_text(json.dumps({"step": last_step}) + "\n")
    return d


class RunState(unittest.TestCase):
    def setUp(self):
        self.home = Path(tempfile.mkdtemp(prefix="e1home"))
        self.runs = self.home / "roms" / "runs"
        self.runs.mkdir(parents=True)
        self.env = {**os.environ, "HOME": str(self.home)}

    def tearDown(self):
        shutil.rmtree(self.home, ignore_errors=True)

    def runs_py(self, *args):
        return subprocess.run([sys.executable, str(CODE / "gcp/remote/runs.py"), *args], env=self.env,
                              capture_output=True, text=True, check=True).stdout.split()

    def test_r0_is_eval_only_and_never_partial(self):
        # the incident: R0 = a copy of A0-D1-s0's config + random_policy, eval dirs, no metrics
        make_run(self.runs, "R0", 5_000_000, random_policy=True, like_run="A0-D1-s0")
        make_run(self.runs, "A2pp-D1-s0", 10_000_000, last_step=6_136_832)
        make_run(self.runs, "A0-D1-s0", 5_000_000, last_step=5_000_192)
        make_run(self.runs, "bench-1", 1_000, last_step=10)
        self.assertEqual(self.runs_py("state", "R0"), ["eval-only"])
        self.assertEqual(self.runs_py("state", "A0-D1-s0"), ["done"])
        self.assertEqual(self.runs_py("state", "nope"), ["new"])
        self.assertEqual(self.runs_py("partial"), ["A2pp-D1-s0"])

    @unittest.skipUnless(shutil.which("bash") and shutil.which("xargs"), "needs bash")
    def test_queue_skips_eval_only_and_trains_the_rest(self):
        """queue.sh worker with a fake trainer: an R0 line is skipped, a new run trains, a partial resumes."""
        code = self.home / "e1" / "code"
        (code / "gcp" / "remote").mkdir(parents=True)
        for f in ("runs.py", "queue.sh"):
            shutil.copy(CODE / "gcp/remote" / f, code / "gcp/remote" / f)
        (code / "CODE_SHA256").write_text("test\n")
        (code / ".venv" / "bin").mkdir(parents=True)
        calls = self.home / "calls.txt"
        # fake python: records the train.py args and writes a done metrics row for the run
        fake = code / ".venv" / "bin" / "python"
        fake.write_text(f"""#!/usr/bin/env bash
echo "$*" >> {calls}
args="$*"; id=$(sed -n 's/.*--run-id \\([^ ]*\\).*/\\1/p' <<<"$args"); [ -n "$id" ] || id=$(sed -n 's/.*--resume \\([^ ]*\\).*/\\1/p' <<<"$args")
d={self.runs}/$id; mkdir -p $d
[ -f $d/config.json ] || echo '{{"total_steps": 100}}' > $d/config.json
echo '{{"step": 100000000}}' >> $d/metrics.jsonl
""")
        fake.chmod(0o755)
        make_run(self.runs, "R0", 5_000_000, random_policy=True)
        make_run(self.runs, "A0-D1-s1", 5_000_000, last_step=1_000)
        q = self.home / "e1" / "queue"
        q.mkdir(parents=True)
        (q / "plan.txt").write_text("--arm A0 --random --run-id R0\n--arm A0 --run-id A0-D1-s1\n"
                                    "--arm A2pp --run-id A2pp-D1-s1\n")
        (q / "active").write_text("x")
        subprocess.run(["bash", str(code / "gcp/remote/queue.sh"), "worker"], env=self.env, check=True,
                       capture_output=True, text=True, timeout=60)
        got = calls.read_text().splitlines()
        self.assertEqual([c for c in got if c.startswith("train.py")],
                         ["train.py --resume A0-D1-s1", "train.py --arm A2pp --run-id A2pp-D1-s1"])
        # §9 H4′: every trained entry gets a summary watcher (reads none by default)
        self.assertEqual(sorted(c for c in got if c.startswith("b2_reads.py")),
                         ["b2_reads.py watch A0-D1-s1 --reads none", "b2_reads.py watch A2pp-D1-s1 --reads none"])
        self.assertIn("R0: eval-only baseline dir", (q / "queue.log").read_text())
        self.assertFalse((self.runs / "R0" / "metrics.jsonl").exists())

    @unittest.skipUnless(shutil.which("bash") and shutil.which("xargs"), "needs bash")
    def test_queue_extra_eval_dir_and_eval_only_lines(self):
        """§9 batch-1 plan (A1): a second queue over done runs adds an eval under --out without retraining;
        an eval-only line (R0) runs evaluate.py only; evals whose <out>/summary.json exists are skipped."""
        code = self.home / "e1" / "code"
        (code / "gcp" / "remote").mkdir(parents=True)
        for f in ("runs.py", "queue.sh"):
            shutil.copy(CODE / "gcp/remote" / f, code / "gcp/remote" / f)
        (code / "CODE_SHA256").write_text("test\n")
        (code / ".venv" / "bin").mkdir(parents=True)
        calls = self.home / "calls.txt"
        fake = code / ".venv" / "bin" / "python"
        fake.write_text(f"""#!/usr/bin/env bash
echo "$*" >> {calls}
if [ "$1" = evaluate.py ]; then
  out=$(sed -n 's/.*--out \\([^ ]*\\).*/\\1/p' <<<"$*"); mkdir -p {self.runs}/$2/${{out:-eval}}
  echo '{{}}' > {self.runs}/$2/${{out:-eval}}/summary.json
fi
""")
        fake.chmod(0o755)
        make_run(self.runs, "R0", 5_000_000, random_policy=True)
        for rid in ("A0-D2-s1", "A2pp-D2-s1"):
            make_run(self.runs, rid, 5_000_000, last_step=5_000_192)
            (self.runs / rid / "eval").mkdir()
            (self.runs / rid / "eval" / "summary.json").write_text("{}")
        (self.runs / "A2pp-D2-s1" / "eval-d2").mkdir()
        (self.runs / "A2pp-D2-s1" / "eval-d2" / "summary.json").write_text("{}")   # already done: skipped
        q = self.home / "e1" / "queue"
        q.mkdir(parents=True)
        ev = " ; eval: --states D2 --episodes 8 --out eval-d2"
        (q / "plan.txt").write_text(f"--arm A0 --run-id A0-D2-s1{ev}\n--arm A2pp --run-id A2pp-D2-s1{ev}\n"
                                    "eval-only: R0 --random A0-D1-s0 --states D2 --episodes 24 --out eval-24-d2\n")
        (q / "active").write_text("x")
        subprocess.run(["bash", str(code / "gcp/remote/queue.sh"), "worker"], env=self.env, check=True,
                       capture_output=True, text=True, timeout=60)
        self.assertEqual(calls.read_text().splitlines(),
                         ["evaluate.py A0-D2-s1 --states D2 --episodes 8 --out eval-d2",
                          "evaluate.py R0 --random A0-D1-s0 --states D2 --episodes 24 --out eval-24-d2"])
        st = subprocess.run(["bash", str(code / "gcp/remote/queue.sh"), "status"], env=self.env, check=True,
                            capture_output=True, text=True).stdout
        self.assertIn("eval-d2 done", st)
        self.assertRegex(st, r"R0 +eval-only +eval-24-d2 done")


    @unittest.skipUnless(shutil.which("bash") and shutil.which("xargs"), "needs bash")
    def test_queue_pause_stops_new_entries_and_keeps_the_queue(self):
        """§9 H4′: a tripwire (here: the fake trainer of run 1 calls queue.sh pause) lets the running entry
        finish, starts no new entry, keeps the queue active (not 'finished'), drops nothing; a peer's forced
        command pauses too; resume clears the flag; watch segments reach the watcher."""
        code = self.home / "e1" / "code"
        (code / "gcp" / "remote").mkdir(parents=True)
        for f in ("runs.py", "queue.sh"):
            shutil.copy(CODE / "gcp/remote" / f, code / "gcp/remote" / f)
        (code / "CODE_SHA256").write_text("test\n")
        (code / ".venv" / "bin").mkdir(parents=True)
        calls = self.home / "calls.txt"
        qs = code / "gcp/remote/queue.sh"
        fake = code / ".venv" / "bin" / "python"
        fake.write_text(f"""#!/usr/bin/env bash
echo "$*" >> {calls}
[ "$1" = train.py ] || exit 0
args="$*"; id=$(sed -n 's/.*--run-id \\([^ ]*\\).*/\\1/p' <<<"$args")
d={self.runs}/$id; mkdir -p $d; echo '{{"total_steps": 100}}' > $d/config.json; echo '{{"step": 100}}' >> $d/metrics.jsonl
[ "$id" = A0-D1-s1 ] && bash {qs} pause "A0-D1-s1 @ eval-1000448.pt: tripwire A-1"
exit 0
""")
        fake.chmod(0o755)
        q = self.home / "e1" / "queue"
        q.mkdir(parents=True)
        (q / "plan.txt").write_text("--arm A0 --run-id A0-D1-s1 ; watch: 0 1M 2.5M ; eval: --states D1 --out eval\n"
                                    "--arm A2pp --run-id A2pp-D1-s1\n")
        (q / "active").write_text("x")
        subprocess.run(["bash", str(qs), "worker"], env=self.env, check=True, capture_output=True, text=True, timeout=60)
        got = calls.read_text().splitlines()
        self.assertIn("b2_reads.py watch A0-D1-s1 --reads 0 1M 2.5M", got)
        self.assertIn("train.py --arm A0 --run-id A0-D1-s1", got)
        self.assertIn("evaluate.py A0-D1-s1 --states D1 --out eval", got)          # the running entry finishes
        self.assertFalse(any("A2pp-D1-s1" in c for c in got))                       # no new entry starts
        self.assertTrue((q / "active").exists())                                     # queue not retired
        self.assertIn("tripwire A-1", (q / "paused").read_text())
        log = (q / "queue.log").read_text()
        self.assertIn("PAUSED", log)
        self.assertNotIn("queue finished", log)
        st = subprocess.run(["bash", str(qs), "status"], env=self.env, check=True, capture_output=True, text=True).stdout
        self.assertIn("PAUSED", st)
        # resume (no tmux here: remove 'active' so resume only clears the flag)
        (q / "active").unlink()
        subprocess.run(["bash", str(qs), "resume"], env=self.env, check=True, capture_output=True, text=True)
        self.assertFalse((q / "paused").exists())
        # the peer's restricted key runs pause-from-peer with the reason in SSH_ORIGINAL_COMMAND
        subprocess.run(["bash", str(qs), "pause-from-peer"], env={**self.env, "SSH_ORIGINAL_COMMAND": "e1-b: B-2"},
                       check=True, capture_output=True, text=True)
        self.assertIn("from peer: e1-b: B-2", (q / "paused").read_text())


    @unittest.skipUnless(shutil.which("bash") and shutil.which("xargs"), "needs bash")
    def test_queue_eval2_to_sealed_path(self):
        """§9 H8: the held-out closing eval (eval2:) writes to ~/roms/sealed-b2/RUN/..., never under the runs
        tree; done-ness is checked at that path, so a rerun of the queue skips both evals."""
        code = self.home / "e1" / "code"
        (code / "gcp" / "remote").mkdir(parents=True)
        for f in ("runs.py", "queue.sh"):
            shutil.copy(CODE / "gcp/remote" / f, code / "gcp/remote" / f)
        (code / "CODE_SHA256").write_text("test\n")
        (code / ".venv" / "bin").mkdir(parents=True)
        calls = self.home / "calls.txt"
        fake = code / ".venv" / "bin" / "python"
        fake.write_text(f"""#!/usr/bin/env bash
echo "$*" >> {calls}
if [ "$1" = evaluate.py ]; then
  out=$(sed -n 's/.*--out \\([^ ]*\\).*/\\1/p' <<<"$*"); out=${{out:-eval}}
  case "$out" in "~/"*) dir="$HOME/${{out#\\~/}}" ;; *) dir={self.runs}/$2/$out ;; esac
  mkdir -p "$dir"; echo '{{}}' > "$dir/summary.json"
fi
""")
        fake.chmod(0o755)
        make_run(self.runs, "b2-A0-D1-s1", 5_000_000, last_step=5_000_192)
        q = self.home / "e1" / "queue"
        q.mkdir(parents=True)
        (q / "plan.txt").write_text("--arm A0 --run-id b2-A0-D1-s1 ; eval: --states D1 D2 ; "
                                    "eval2: --states HA HB --out ~/roms/sealed-b2/b2-A0-D1-s1/eval-heldout\n")
        (q / "active").write_text("x")
        for _ in range(2):
            (q / "active").write_text("x")              # the worker retires 'active' when the queue finishes
            subprocess.run(["bash", str(code / "gcp/remote/queue.sh"), "worker"], env=self.env, check=True,
                           capture_output=True, text=True, timeout=60)
        ev = [c for c in calls.read_text().splitlines() if c.startswith("evaluate.py")]
        self.assertEqual(ev, ["evaluate.py b2-A0-D1-s1 --states D1 D2",
                              "evaluate.py b2-A0-D1-s1 --states HA HB --out ~/roms/sealed-b2/b2-A0-D1-s1/eval-heldout"])
        self.assertTrue((self.home / "roms/sealed-b2/b2-A0-D1-s1/eval-heldout/summary.json").exists())
        self.assertFalse(any("HA" in str(p) or "heldout" in str(p) for p in (self.runs / "b2-A0-D1-s1").rglob("*")))


if __name__ == "__main__":
    unittest.main()
