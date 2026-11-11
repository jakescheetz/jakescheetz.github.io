"""A2′ shaping (experiment-e1.md §4) and detector O / U-novelty (§6, §9 U review). No ROM needed; the
scripted-trace test reads ~/roms/states/script-trace.jsonl if it exists and is skipped otherwise.

Run:  .venv/bin/python -m unittest discover -s tests
"""
import json
import random
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pokered import detectors as det  # noqa: E402
from pokered import ram, shaping  # noqa: E402
from pokered import shaping as S  # noqa: E402


def shaped(states, gamma):
    """Shaping terms γΦ(s') − Φ(s) along a list of (map, x, y, T) states."""
    ph = [S.phi(*s) for s in states]
    return [gamma * b - a for a, b in zip(ph, ph[1:])], ph


def random_walk(n, seed, start=S.D1_START, T=0):
    """Random tile moves over the region's maps, jumping through a random exit now and then: a stand-in
    for an agent's (map, x, y) path. T rises when the walk lands on the next milestone's map."""
    rng = random.Random(seed)
    mp, x, y = start
    out = [(mp, x, y, T)]
    for _ in range(n):
        gm = S.node(mp, y)
        if rng.random() < 0.05 and S.EXITS.get(gm):
            ex, ey, dm, dx, dy = rng.choice(S.EXITS[gm])
            mp, x, y = S.raw(dm), dx, dy
        else:
            w, h = S.SIZE[mp]
            dx, dy = rng.choice([(0, 1), (0, -1), (1, 0), (-1, 0)])
            x, y = min(max(x + dx, 0), w - 1), min(max(y + dy, 0), h - 1)
        if T < S.N_LEGS and T < 5 and mp == S.LEG_TARGET[T + 1]:
            T += 1
        out.append((mp, x, y, T))
    return out


class TestPhi(unittest.TestCase):
    def test_leg_lengths_by_hand(self):
        # leg 1: lab (5,3) -> door (5,11) 8 + 1; Pallet (12,11) -> north edge 11 + 1; Route 1 (12,35) -> 35 + 1
        self.assertEqual(S.D_LEG[1], 9 + 12 + 36)
        # leg 2: Viridian (22,35) -> north edge (x 10..29 overlaps Route 2) 35 + 1
        self.assertEqual(S.D_LEG[2], 36)
        # leg 5: Pewter (13,35) -> gym door (16,17): 3 + 18 + 1; leg 6: gym door (4,13) -> below Brock (4,2)
        self.assertEqual(S.D_LEG[5], 22)
        self.assertEqual(S.D_LEG[6], 11)

    def test_bounded_and_zero_when_done(self):
        for s in random_walk(20000, 1):
            self.assertTrue(-(5 + S.D_CAP) <= S.phi(*s) <= 0.0)
        self.assertEqual(S.phi(S.GYM, 4, 2, 6), 0.0)
        self.assertEqual(S.phi(0x99, 3, 3, 6), 0.0)
        self.assertEqual(S.phi(0x99, 3, 3, 0), -(5 + S.D_CAP))      # outside the region: the floor

    def test_continuous_across_a_milestone(self):
        # one step before Viridian (Route 1 top row, T=0) vs one step after (Viridian bottom row, T=1)
        before, after = S.phi(S.ROUTE1, 11, 0, 0), S.phi(S.VIRIDIAN, 21, 35, 1)
        self.assertAlmostEqual(before, -(5 + 1 / S.D_LEG[1]), places=6)
        self.assertLess(abs(after - before), 0.1)

    def test_telescopes_over_an_episode(self):
        gamma = S.GAMMA
        states = random_walk(5000, 2)
        terms, ph = shaped(states, gamma)
        disc = sum(gamma ** t * f for t, f in enumerate(terms))
        self.assertAlmostEqual(disc, gamma ** len(terms) * ph[-1] - ph[0], places=8)

    def test_closed_loop(self):
        """γ = 1: a closed loop nets exactly 0. γ < 1: it nets Φ(s0)(γ^N − 1), which is >= 0 because Φ <= 0
        (a stipend of at most (1 − γ)·|Φ| per step, 0.014 here). That is not "<= 0" as §4's text says (that
        holds for Φ >= 0); it is still policy-invariant (Ng et al.): the stipend is exactly what the
        discounting of everything after the loop costs. Flagged for the researcher."""
        for seed in range(5):
            walk = random_walk(800, 10 + seed)
            loop = walk + walk[-2::-1]                 # there and back: a closed loop in state space
            terms, _ = shaped(loop, 1.0)
            self.assertAlmostEqual(sum(terms), 0.0, places=9)
            terms, ph = shaped(loop, S.GAMMA)
            disc = sum(S.GAMMA ** t * f for t, f in enumerate(terms))
            self.assertAlmostEqual(disc, ph[0] * (S.GAMMA ** len(terms) - 1), places=8)
            self.assertGreaterEqual(disc, 0.0)
            self.assertLessEqual(max(S.GAMMA * b - a for a, b in zip(ph, ph[1:]) if a == b), (1 - S.GAMMA) * 7 + 1e-12)

    def test_pacing_two_tiles_pays_nothing(self):
        a, b = (S.ROUTE1, 10, 20, 0), (S.ROUTE1, 10, 19, 0)
        terms, _ = shaped([a, b] * 500 + [a], 1.0)
        self.assertAlmostEqual(sum(terms), 0.0, places=9)

    def test_route2_split(self):
        # T=3 (in the Forest -> Pewter): the south half of Route 2 routes back through the Forest
        south, north = S.distance(4, S.ROUTE2, 3, 44), S.distance(4, S.ROUTE2, 3, 11)
        self.assertGreater(south, 60)
        self.assertEqual(north, 12)                    # (3, 11) = the north gate's door: 11 rows + 1 step
        # T=2 (Viridian -> Forest): Route 2's south half is on the way to the south gate
        self.assertEqual(S.distance(3, S.ROUTE2, 3, 44), 1 + 1 + S.distance(3, S.SOUTH_GATE, 4, 7))

    def test_stale_coordinates_after_a_warp_are_clamped(self):
        # wCurMap updates 1-2 steps before x/y: Viridian's door coords on the Center map
        self.assertEqual(S.phi(S.V_CENTER, 23, 25, 1), S.phi(S.V_CENTER, 13, 7, 1))

    @unittest.skipUnless((Path.home() / "roms/states/script-trace.jsonl").exists(), "no scripted trace")
    def test_rises_along_scripted_trace(self):
        with open(Path.home() / "roms/states/script-trace.jsonl") as f:
            rows = [json.loads(l) for l in f]
        ph = [S.phi(r["map"], r["x"], r["y"], r["T"]) for r in rows]
        first = {n: next(i for i, r in enumerate(rows) if r["T"] >= n) for n in (1, 2)}
        self.assertAlmostEqual(ph[0], -6.0, places=6)                 # D1 start = leg 1's reference point
        self.assertGreater(ph[first[1]], ph[0] + 0.9)                 # start -> M1: about one leg
        self.assertGreater(ph[first[2]], ph[first[1]] + 0.9)          # M1 -> M2
        # every map transition on the trace lands where pret says (after the 1-2 stale rows)
        lands = {}
        for gm, es in S.EXITS.items():
            for _, _, dm, dx, dy in es:
                lands.setdefault((S.raw(gm), S.raw(dm)), set()).add((dx, dy))
        for i in range(1, len(rows) - 3):
            a, b = rows[i - 1], rows[i]
            if a["map"] != b["map"]:
                settled = {(r["x"], r["y"]) for r in rows[i:i + 3]}
                self.assertTrue(settled & lands[(a["map"], b["map"])], (a, b))


def row(step, **kw):
    r = dict(step=step, action=0, map=ram.PALLET_TOWN, x=5, y=5, hp=[20], levels=[5], in_battle=0,
             blackout_map=0, menu_item=0, p=0.0, r_heal=0.0, r_explore=0.0, T=1)
    r.update(kw)
    return r


def exploring(n, mp, T, start=0):
    """n rows on map mp, each on a new tile paying 0.1 explore."""
    return [row(start + i, map=mp, x=i % 40, y=i // 40, T=T, p=0.1, r_explore=0.1) for i in range(n)]


class TestOffPath(unittest.TestCase):
    def test_off_path_maps_by_T(self):
        self.assertIsNone(det.off_path(row(0, map=ram.PALLET_TOWN, T=0)))      # start of the segment
        self.assertEqual(det.off_path(row(0, map=ram.PALLET_TOWN, T=1)), "behind")
        self.assertEqual(det.off_path(row(0, map=0x25, T=0)), "side")          # Red's house
        self.assertIsNone(det.off_path(row(0, map=ram.VIRIDIAN_POKECENTER, T=3)))   # neutral
        self.assertIsNone(det.off_path(row(0, map=ram.ROUTE_22, T=1)))         # D's, not O's
        self.assertIsNone(det.off_path(row(0, map=ram.VIRIDIAN_CITY, T=1)))
        self.assertEqual(det.off_path(row(0, map=ram.VIRIDIAN_CITY, T=2)), "behind")
        self.assertEqual(det.off_path(row(0, map=ram.ROUTE_2, y=50, T=3)), "behind")
        self.assertIsNone(det.off_path(row(0, map=ram.ROUTE_2, y=5, T=3)))

    def test_O_fires_on_backtracking_novelty(self):
        win = exploring(1100, ram.ROUTE_1, 1) + exploring(948, ram.VIRIDIAN_CITY, 1, 1100)
        ev = det.detect_off_path(det.mark_revisits(win))
        self.assertIsNotNone(ev)
        self.assertEqual(ev["kind"], "behind")

    def test_O_needs_explore_and_majority_and_no_milestone(self):
        low = [dict(r, p=0.0, r_explore=0.0) for r in exploring(det.W, ram.ROUTE_1, 1)]
        low[:49] = exploring(49, ram.ROUTE_1, 1)                       # 4.9 explore: below the floor
        self.assertIsNone(det.detect_off_path(low))
        half = exploring(1024, ram.ROUTE_1, 1) + exploring(1024, ram.VIRIDIAN_CITY, 1, 1024)
        self.assertIsNone(det.detect_off_path(half))                    # exactly 50%: not > 50%
        ms = exploring(det.W, ram.ROUTE_1, 1)
        ms[-1]["T"] = 2
        self.assertIsNone(det.detect_off_path(ms))

    def test_O_covers_U_and_U_novelty_label(self):
        T = 1
        on = exploring(det.W, ram.VIRIDIAN_CITY, T)
        off = exploring(det.W, ram.ROUTE_1, T)
        trace = []
        for i in range(5):                                     # 4W of flat T before the last window
            trace += [dict(r, step=len(trace) + j, x=(r["x"] + 40 * i) % 40, y=r["y"] + 52 * i)
                      for j, r in enumerate(on)]
        trace += [dict(r, step=len(trace) + j) for j, r in enumerate(off)]
        recs = det.window_records(trace)
        det.apply_u(recs, 0.0)
        last = recs[-1]
        self.assertIn("O", last["fired"])
        self.assertNotIn("U", last["fired"])
        u = [r for r in recs if "U" in r["fired"]]
        self.assertTrue(u)
        whole = [r for r in u if r["offpath_fraction"] == 0.0]   # windows entirely on Viridian: novelty
        self.assertTrue(whole and all(r["u_kind"] == "novelty" for r in whole))
        mixed = [r for r in u if 0 < r["offpath_fraction"] <= 0.5]  # half off-path, O not yet: review
        self.assertTrue(all(r["u_kind"] == "review" for r in mixed))

    def test_U_review_when_not_novelty(self):
        # P from level reward, not explore: U fires (if no hack) with kind review
        trace = [row(i, map=ram.VIRIDIAN_CITY, x=i % 4, y=0, p=(1.0 if i % 300 == 0 else 0.0),
                     r_level=(1.0 if i % 300 == 0 else 0.0)) for i in range(5 * det.W)]
        recs = det.window_records(trace)
        det.apply_u(recs, 0.0)
        u = [r for r in recs if "U" in r["fired"]]
        self.assertTrue(u)
        self.assertTrue(all(r["u_kind"] == "review" for r in u))

    def test_window_stats_have_per_term_P(self):
        st = det.loop_stats(det.mark_revisits(exploring(det.W, ram.ROUTE_1, 1)))
        self.assertAlmostEqual(st["p_explore"], 0.1 * det.W, places=4)
        self.assertEqual(st["offpath_fraction"], 1.0)
        self.assertAlmostEqual(st["p_new_explore_route"], 0.0)


class TestShapedOracleReward(unittest.TestCase):
    def test_reward_is_T_gain_plus_shaping(self):
        from pokered.env import ShapedOracle
        m = bytearray(0x10000)
        m[ram.W_CUR_MAP], m[ram.W_X_COORD], m[ram.W_Y_COORD] = S.ROUTE1, 11, 1
        r = ShapedOracle()
        r.reset(m)
        tot = 0.0
        for y in (0, ):                                   # one step north on Route 1
            m[ram.W_Y_COORD] = y
            v, parts = r(m, 0)
            tot += v
        self.assertAlmostEqual(tot, S.GAMMA * S.phi(S.ROUTE1, 11, 0, 0) - S.phi(S.ROUTE1, 11, 1, 0), places=9)
        m[ram.W_CUR_MAP], m[ram.W_X_COORD], m[ram.W_Y_COORD] = S.VIRIDIAN, 21, 35
        v, parts = r(m, 1)                               # M1: +1 plus continuous shaping
        self.assertAlmostEqual(v, 1 + S.GAMMA * S.phi(S.VIRIDIAN, 21, 35, 1) - S.phi(S.ROUTE1, 11, 0, 0), places=9)
        self.assertGreater(v, 0.9)
        self.assertNotIn("shaping", parts)               # P (the proxy) never sees the shaping term


if __name__ == "__main__":
    unittest.main()


class TestHolm(unittest.TestCase):
    def test_holm_step_down(self):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        import r0_compare
        self.assertEqual([round(x, 6) for x in r0_compare.holm([0.01, 0.04, 0.03, 0.005])], [0.03, 0.06, 0.06, 0.02])
        self.assertEqual(r0_compare.holm([0.5, 1.0]), [1.0, 1.0])            # capped at 1, monotone
        self.assertEqual(r0_compare.holm([]), [])


class TestPermutation(unittest.TestCase):
    def test_exact_permutation_p(self):
        import random as _r
        import r0_compare
        rng = _r.Random(0)
        p, method, n = r0_compare.perm_p([1.0] * 8, [0.0] * 8, exact=True)   # complete separation: 1 of 12,870
        self.assertEqual((method, n), ("exact", 12870))
        self.assertAlmostEqual(p, 1 / 12870)
        p, _, _ = r0_compare.perm_p([0.0] * 8, [0.0] * 8, exact=True)       # all tied: every relabeling >= observed
        self.assertEqual(p, 1.0)
        # small case by hand: a = [2, 3], b = [0, 1]; sums of 2 of {2,3,0,1}: 6 relabelings, sums >= 5: {2,3} only
        p, _, n = r0_compare.perm_p([2.0, 3.0], [0.0, 1.0], exact=True)
        self.assertEqual((p, n), (1 / 6, 6))
        p, _, _ = r0_compare.perm_p([0.0, 1.0], [2.0, 3.0], exact=True)     # wrong direction: one-sided p = 1
        self.assertEqual(p, 1.0)

    def test_monte_carlo_permutation_p(self):
        import r0_compare
        # 40 vs 24 (5 seeds x 8 vs R0 at 24): complete separation is far below 1/(B+1), so p = 1/(B+1)
        p, method, n = r0_compare.perm_p([1.0] * 40, [0.0] * 24)
        self.assertEqual((method, n), ("monte_carlo", r0_compare.P_B))
        self.assertAlmostEqual(p, 1 / (r0_compare.P_B + 1))
        self.assertEqual(r0_compare.perm_p([0.0] * 40, [0.0] * 24)[0], 1.0)
        # MC agrees with the exact p where both are possible (8 vs 8, partial overlap)
        a, b = [0.0, 0.27, 0.53, 0.73, 0.0, 0.67, 0.93, 0.73], [0.0] * 7 + [0.07]
        pe = r0_compare.perm_p(a, b, exact=True)[0]
        pm = r0_compare.perm_p(a, b)[0]
        self.assertLess(abs(pe - pm), 4 * (pe * (1 - pe) / r0_compare.P_B) ** 0.5 + 2e-5)
