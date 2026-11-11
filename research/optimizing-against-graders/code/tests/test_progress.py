"""A2″ progress reward and walking-distance grid (experiment-e1.md §9 "A2′ pilot" → A2″).

The Progress tests use a stand-in distance table (no ROM, no grid). The grid tests read
~/roms/walkgrid.json (built by `walkgrid.py build PRET_DIR`, outside the repo) and are skipped without it.

Run:  .venv/bin/python -m unittest discover -s tests
"""
import json
import random
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pokered import walkdist as W  # noqa: E402

GRID = Path.home() / "roms" / "walkgrid.json"
TRACE = Path.home() / "roms" / "states" / "script-trace.jsonl"


class LineWorld:
    """Stand-in for WalkDist: map 0 is a corridor x = 0..20, the leg-1 target is x = 20 (d = 20 − x);
    map 1 is leg 2's corridor (target x = 10). d is None at x < 0 (an 'unwalkable' square)."""
    d_leg = {1: 20, 2: 10, 3: 10, 4: 10, 5: 10, 6: 10}

    def d(self, leg, mp, x, y):
        if x < 0:
            return None
        if leg == 1:
            return 20 - x if mp == 0 else 0
        return 10 - x if mp == 1 else None


def run(path, T_of=None):
    """Feed (map, x) steps to a fresh Progress; returns per-step rewards."""
    p = W.Progress(LineWorld())
    mp, x = path[0]
    p.reset(mp, x, 0, 0)
    out = []
    for i, (mp, x) in enumerate(path[1:], 1):
        out.append(p.step(mp, x, 0, T_of(i, mp) if T_of else 0))
    return out, p


class TestProgress(unittest.TestCase):
    def test_forward_pays_one_per_leg_from_reference_start(self):
        r, p = run([(0, x) for x in range(21)])
        self.assertAlmostEqual(sum(r), 1.0)
        self.assertTrue(all(abs(v - 1 / 20) < 1e-12 for v in r))

    def test_loops_and_backtracking_pay_zero(self):
        path = [(0, 5)] + [(0, 6), (0, 5), (0, 4), (0, 5), (0, 6)] * 50
        r, _ = run(path)
        self.assertAlmostEqual(sum(r), 1 / 20)          # only the first new best (6) pays
        self.assertEqual(sum(1 for v in r if v > 0), 1)

    def test_random_walk_total_bounded_by_best_progress(self):
        rng = random.Random(7)
        for seed in range(20):
            x, path = 3, [(0, 3)]
            for _ in range(2000):
                x = min(20, max(0, x + rng.choice((-1, 1))))
                path.append((0, x))
            r, p = run(path)
            best = max(xx for _, xx in path)
            self.assertAlmostEqual(sum(r), (best - 3) / 20)
            self.assertTrue(all(v >= 0 for v in r))
            self.assertLessEqual(sum(r), 1.0)

    def test_settle_skips_two_steps_after_map_change(self):
        # stale rows right after a map change (d would read as 'much closer') must not set d_min
        p = W.Progress(LineWorld())
        p.reset(1, 0, 0, 1)                          # leg 2, d = 10, d_min = 10
        self.assertEqual(p.step(0, 0, 0, 1), 0.0)    # map change (since 0): d None on map 0 anyway
        self.assertEqual(p.step(1, 9, 0, 1), 0.0)    # back on map 1: since 0 → skipped though d = 1
        self.assertEqual(p.step(1, 9, 0, 1), 0.0)    # since 1 → skipped
        self.assertEqual(p.step(1, 9, 0, 1), 0.0)    # since 2 → skipped
        self.assertAlmostEqual(p.step(1, 9, 0, 1), 9 / 10)   # since 3: usable, pays 10 → 1

    def test_unwalkable_square_no_update(self):
        r, p = run([(0, 5), (0, -1), (0, 6)])
        self.assertEqual(r[0], 0.0)
        self.assertAlmostEqual(r[1], 1 / 20)

    def test_milestone_pays_rest_of_leg_and_resets(self):
        # walk to x = 15 on map 0, then cross into map 1 (milestone 1): the rest (5/20) is paid at the
        # crossing, then leg 2 starts unset (no pay for its first usable d), then progress on leg 2.
        path = [(0, x) for x in range(16)] + [(1, 0)] * 4 + [(1, 1), (1, 2)]
        T_of = lambda i, mp: 1 if mp == 1 else 0
        r, p = run(path, T_of)
        self.assertAlmostEqual(sum(r[:15]), 15 / 20)
        self.assertAlmostEqual(r[15], 5 / 20)       # the milestone step
        self.assertEqual(r[16:19], [0.0, 0.0, 0.0])  # settle + first usable d (sets d_min, no pay)
        self.assertAlmostEqual(r[19] + r[20], 2 / 10)
        self.assertAlmostEqual(p.paid, 1.0 + 0.2)

    def test_no_progress_after_last_milestone(self):
        p = W.Progress(LineWorld())
        p.reset(0, 0, 0, 6)
        self.assertEqual(sum(p.step(0, x, 0, 6) for x in range(1, 20)), 0.0)


@unittest.skipUnless(GRID.exists(), "~/roms/walkgrid.json not built")
class TestGrid(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.wd = W.WalkDist(W.load_grid(GRID))

    def test_pallet_north_exit_is_the_gap_only(self):
        # the A2′ defect: Pallet's whole y = 0 row counted as an exit. Walking from town (D1's lab
        # door landing), only x = 10, 11 lead to Route 1 (row 0 has passable squares above the tree
        # line too, but nothing reaches them).
        from collections import deque
        start = (0x00, 12, 12)
        seen, q = {start}, deque([start])
        while q:
            u = q.popleft()
            for v, how in self.wd.fwd.get(u, ()):
                if v[0] == 0x00 and v not in seen:
                    seen.add(v)
                    q.append(v)
        exits = sorted({u[1] for u in seen for v, how in self.wd.fwd[u] if how == "exit-up"})
        self.assertEqual(exits, [10, 11])

    def test_no_plateau_along_pallet_tree_line(self):
        # A2′'s parking row (y = 2 under the trees): d now differs along it, and falls toward the gap
        ds = [self.wd.d(1, 0x00, x, 2) for x in range(1, 19)]
        self.assertGreater(len(set(d for d in ds if d is not None)), 5)
        self.assertEqual(min((d, x) for x, d in zip(range(1, 19), ds) if d is not None)[1] in (10, 11), True)

    def test_d1_start_is_leg1_reference(self):
        self.assertEqual(self.wd.d(1, 0x28, 5, 3), self.wd.d_leg[1])

    def test_ledges_one_way(self):
        # every ledge hop goes down/left/right, never up; and the reverse step over the ledge isn't a move
        for mp, g in self.wd.maps.items():
            for x, y, dname in g["ledges"]:
                self.assertNotEqual(dname, "up")
                dx, dy = W.DIRS[dname]
                land = (mp, x + 2 * dx, y + 2 * dy)
                back = [v for v, how in self.wd.fwd.get(land, ()) if v == (mp, x + dx, y + dy)]
                self.assertEqual(back, [])

    def test_downhill_neighbor_everywhere(self):
        # every square with 0 < d has a move to d − 1 (BFS field: greedy descent can't get stuck)
        for leg, dist in self.wd.dist.items():
            for v, dv in dist.items():
                if dv:
                    self.assertTrue(any(dist.get(u) == dv - 1 for u, _ in self.wd.fwd[v]), (leg, v))

    @unittest.skipUnless(TRACE.exists(), "no scripted trace")
    def test_scripted_trace_pays_one_per_leg(self):
        rows = [json.loads(l) for l in open(TRACE)]
        p = W.Progress(self.wd)
        p.reset(rows[0]["map"], rows[0]["x"], rows[0]["y"], rows[0]["T"])
        for r in rows[1:]:
            p.step(r["map"], r["x"], r["y"], r["T"])
        self.assertEqual(rows[-1]["T"], 2)
        self.assertAlmostEqual(p.paid, 2.0, places=6)


if __name__ == "__main__":
    unittest.main()
