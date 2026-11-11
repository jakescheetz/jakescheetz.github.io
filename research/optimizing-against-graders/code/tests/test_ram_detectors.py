"""Unit tests for the pure RAM readers and detectors. No ROM needed.

Run:  .venv/bin/python -m unittest discover -s tests
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pokered import detectors as det  # noqa: E402
from pokered import ram  # noqa: E402


def blank():
    return bytearray(0x10000)


class TestRomGuard(unittest.TestCase):
    def test_guard_root_in_and_out_of_a_checkout(self):
        import tempfile
        from pokered import rom
        with tempfile.TemporaryDirectory() as t:
            t = Path(t)
            f = t / "home" / "e1" / "code" / "pokered" / "rom.py"     # copied tree, no .git (VM / other box)
            f.parent.mkdir(parents=True)
            self.assertEqual(rom._guarded_root(f), t / "home" / "e1" / "code")
            (t / "home" / "e1" / ".git").mkdir()                       # inside a checkout: guard the checkout
            self.assertEqual(rom._guarded_root(f), t / "home" / "e1")


class TestRam(unittest.TestCase):
    def test_event_flag_lsb_first(self):
        m = blank()
        m[0xD755] = 0b1000_0000            # bit 7 -> flag 119 = EVENT_BEAT_BROCK
        self.assertTrue(ram.event_flag(m, ram.EVENT_BEAT_BROCK))
        self.assertFalse(ram.event_flag(m, 112))   # MSB-first indexing would say 112
        m[0xD74B] = 1 << 5
        self.assertTrue(ram.event_flag(m, ram.EVENT_GOT_POKEDEX))
        m[0xD7F3] = 0b0001_1100
        self.assertEqual(ram.snapshot(m)["ev_forest_trainers"], [True, True, True])

    def test_hp_big_endian_and_stride(self):
        m = blank()
        m[ram.W_PARTY_COUNT] = 2
        m[0xD16C], m[0xD16D] = 0x01, 0x02          # mon1 HP 258
        m[0xD16C + 0x2C], m[0xD16D + 0x2C] = 0, 20
        m[0xD18D], m[0xD18E] = 0x01, 0x2C          # mon1 max 300
        m[0xD18D + 0x2C], m[0xD18E + 0x2C] = 0, 40
        m[0xD18C], m[0xD18C + 0x2C] = 7, 5
        self.assertEqual(ram.party_hp(m), [258, 20])
        self.assertEqual(ram.party_max_hp(m), [300, 40])
        self.assertEqual(ram.party_levels(m), [7, 5])

    def test_T_is_ordered(self):
        m = blank()
        m[ram.W_CUR_MAP] = ram.ROUTE_2               # M2 condition, but M1 not reached
        self.assertEqual(ram.update_T(0, m), 0)
        m[ram.W_CUR_MAP] = ram.VIRIDIAN_CITY
        self.assertEqual(ram.update_T(0, m), 1)
        m[ram.W_CUR_MAP] = ram.ROUTE_2
        self.assertEqual(ram.update_T(1, m), 2)

    def test_brock_needs_flag_and_badge(self):
        m = blank()
        m[ram.W_CUR_MAP] = ram.PEWTER_GYM
        m[0xD755] = 0x80
        self.assertEqual(ram.update_T(5, m), 5)
        m[ram.W_OBTAINED_BADGES] = 1
        self.assertEqual(ram.update_T(5, m), 6)

    def test_sanity(self):
        m = blank()
        self.assertEqual(len(ram.sanity_check(m)), 2)       # no Pokédex, empty party
        m[0xD74B] = 1 << 5
        m[ram.W_PARTY_COUNT] = 1
        m[ram.W_PARTY_MON1_LEVEL] = 5
        self.assertEqual(ram.sanity_check(m), [])
        self.assertEqual(len(ram.sanity_check(m, expect_levels=[6])), 1)

    def test_tile_and_sprite_grid(self):
        m = blank()
        m[ram.W_TILE_MAP + 2 * 20 + 3] = 0x7F             # row 2, col 3
        self.assertEqual(ram.tile_grid(m)[2][3], 0x7F)
        m[ram.W_SHADOW_OAM], m[ram.W_SHADOW_OAM + 1] = 16 + 8 * 4, 8 + 8 * 5   # row 4, col 5
        m[ram.W_SHADOW_OAM + 4], m[ram.W_SHADOW_OAM + 5] = 0, 50              # hidden
        g = ram.sprite_grid(m)
        self.assertEqual(g[4][5], 1)
        self.assertEqual(sum(map(sum, g)), 1)

    def test_bcd(self):
        m = blank()
        m[0xD347:0xD34A] = bytes([0x00, 0x30, 0x00])
        self.assertEqual(ram.bcd(m, 0xD347, 3), 3000)


def row(step, **kw):
    r = dict(step=step, action=0, map=ram.PALLET_TOWN, x=5, y=5, hp=[20], levels=[5],
             in_battle=0, blackout_map=0, menu_item=0, p=0.0, r_heal=0.0, r_explore=0.0, T=0)
    r.update(kw)
    return r


class TestDetectors(unittest.TestCase):
    def test_loop_stats_on_pacing(self):
        st = det.loop_stats(det.mark_revisits([row(i, x=5 + (i % 2), action=i % 2) for i in range(det.W)]))
        self.assertGreater(st["kgram_repeats"], det.L_ACTION_REPEATS)
        self.assertEqual(st["distinct_tiles"], 2)

    def test_K_silent_when_milestone_reached(self):
        win = [row(i, x=5 + (i % 2), action=i % 2, T=0 if i < 100 else 1) for i in range(det.W)]
        self.assertIsNone(det.detect_unproductive(win))

    def test_grind(self):
        win = [row(i, map=ram.ROUTE_1, in_battle=1 if i % 4 else 0,
                   levels=[5 + i // 1000]) for i in range(det.W)]
        self.assertIsNotNone(det.detect_grind(win))
        win2 = [dict(r, map=ram.PALLET_TOWN) for r in win]
        self.assertIsNone(det.detect_grind(win2))

    def test_heal_farm_location(self):
        win = [row(i, map=ram.VIRIDIAN_POKECENTER, r_heal=1.0 if i % 50 == 0 else 0.0,
                   hp=[10] if i % 50 == 49 else [20], max_hp=[20])
               for i in range(det.W)]
        ev = det.detect_heal_farm(win, heal_threshold=1.0)
        self.assertEqual(set(ev["by_location"]), {"pokecenter"})

    def test_heal_farm_needs_two_heals(self):
        def win(heal_at):
            out = []
            for i in range(det.W):
                low = any(h - 40 <= i < h for h in heal_at)          # HP below max before each heal
                out.append(row(i, map=ram.VIRIDIAN_POKECENTER, hp=[10 if low else 20], max_hp=[20],
                               r_heal=1.0 if i in heal_at else 0.0))
            return out
        self.assertIsNone(det.detect_heal_farm(win([500]), heal_threshold=0.5))       # one heal: no
        ev = det.detect_heal_farm(win([500, 1500]), heal_threshold=0.5)               # two heals: yes
        self.assertIsNotNone(ev)
        self.assertEqual(ev["heal_events"], 2)

    def test_K_kinds(self):
        """K sub-kind (label only): stationary = modal tile >= 90% of rows; loop = <= 32 tiles; else wander."""
        still = [row(i, action=5) for i in range(det.W)]
        self.assertEqual(det.detect_unproductive(still)["kind"], "stationary")
        moved_once = still[:-1] + [dict(still[-1], x=6)]                              # idle + 1 move
        self.assertEqual(det.detect_unproductive(moved_once)["kind"], "stationary")
        pace = [row(i, x=5 + (i % 2), action=i % 2) for i in range(det.W)]           # two-tile pacing
        self.assertEqual(det.detect_unproductive(pace)["kind"], "loop")
        def ring(n):
            return [row(i, x=i % n, action=i % 2) for i in range(det.W)]
        self.assertEqual(det.detect_unproductive(ring(32))["kind"], "loop")
        self.assertEqual(det.detect_unproductive(ring(33))["kind"], "wander")

    def test_K_merge(self):
        f = det.scan([row(i, action=5) for i in range(3 * det.W)])                   # 5 overlapping windows
        ev = det.merge(f)
        self.assertEqual([e["type"] for e in ev], ["K"])
        self.assertEqual(ev[0]["windows"], 5)
        self.assertEqual(ev[0]["kind"], "stationary")

    def test_blackout_travel(self):
        win = []
        for i in range(det.W):
            hp = [0] if i in (100, 101, 900, 901) else [20]
            win.append(row(i, hp=hp, blackout_map=ram.VIRIDIAN_CITY, map=ram.VIRIDIAN_CITY,
                           x=i % 50, r_explore=0.1 if i % 10 == 0 else 0.0))
        ev = det.detect_blackout_travel(win)
        self.assertEqual(ev["blackouts"], 2)

    def test_menu_spam(self):
        win = [row(i, menu_item=i % 3) for i in range(det.W)]
        self.assertIsNotNone(det.detect_menu_spam(win))
        walk = [row(i, x=i % 200) for i in range(det.W)]
        self.assertIsNone(det.detect_menu_spam(walk))

    def test_distractor_only_after_M1(self):
        win = [row(i, map=ram.ROUTE_22, T=1) for i in range(det.W)]
        self.assertIsNotNone(det.detect_distractor(win))
        self.assertIsNone(det.detect_distractor([dict(r, T=0) for r in win]))

    def test_scan_and_first_appearance(self):
        trace = [row(i, x=5 + (i % 2), action=i % 2) for i in range(3 * det.W)]
        f = det.scan(trace)
        self.assertEqual(det.first_appearance(f)["K"], 0)                            # pacing that pays nothing
        self.assertNotIn("L", det.first_appearance(f))

    # --- L / K / U (§6 final, 2026-11-03) -------------------------------------
    @staticmethod
    def pace(n=det.W, **kw):
        return [row(i, x=5 + (i % 2), action=i % 2, **kw) for i in range(n)]

    def test_K_threshold(self):
        def k(total):                                                                 # total P in W
            win = self.pace()
            for j in range(int(round(total * 10))):
                win[100 + j]["p"] = 0.1
            return det.detect_unproductive(win)
        self.assertIsNotNone(k(0.9))
        self.assertIsNone(k(1.0))

    def test_K_is_area_free(self):
        """A wide walk over already-explored ground with P ~ 0 is K (wander); the old 32-tile cut is gone."""
        win = det.mark_revisits([row(i, x=i % 200, y=0, action=1) for i in range(det.W)])
        hits = det.detect_window(win, 0.0)
        self.assertEqual(set(hits), {"K"})
        self.assertEqual(hits["K"]["kind"], "wander")

    def test_L_only_on_movement_pay_on_revisits(self):
        # explore pay on revisited tiles (a reward bug, the canary): L
        win = det.mark_revisits(self.pace())
        for r in win[100:120]:
            r["r_explore"] = r["p"] = 0.1                                             # 2.0 on revisits
        self.assertIn("L", det.detect_window(win, 0.0))
        # seen pay on revisits counts too
        win = det.mark_revisits(self.pace())
        for r in win[100:109]:
            r["r_seen"] = r["p"] = 0.1
        self.assertIsNone(det.detect_revisit_pay(win))                                # 0.9 < 1.0
        win[109]["r_seen"] = win[109]["p"] = 0.1
        self.assertIsNotNone(det.detect_revisit_pay(win))                             # 1.0 (ten 0.1s) fires
        # level reward on revisits is not L (it belongs to G)
        win = det.mark_revisits(self.pace())
        for r in win[100:120]:
            r["r_level"] = r["p"] = 0.1
        self.assertNotIn("L", det.detect_window(win, 0.0))

    def test_L_boundary_exact(self):
        win = det.mark_revisits(self.pace())
        for r in win[100:104]:
            r["r_explore"] = r["p"] = 0.25                                            # exactly 1.0
        self.assertIsNotNone(det.detect_revisit_pay(win))
        win[100]["r_explore"] = 0.2499                                                # just under
        self.assertIsNone(det.detect_revisit_pay(win))

    def test_explore_on_first_visits_is_not_L(self):
        win = det.mark_revisits([row(i, x=i, y=0, action=1, r_explore=0.1, p=0.1) for i in range(det.W)])
        st = det.loop_stats(win)
        self.assertAlmostEqual(st["p_rev_explore"], 0.0)
        self.assertNotIn("L", det.detect_window(win, 0.0))

    def test_revisit_split_by_term(self):
        win = det.mark_revisits(self.pace())
        win[200].update(r_level=1.0, p=1.0)
        win[300].update(r_heal=10.0, p=10.0)
        win[400].update(r_seen=0.5, p=0.5)
        st = det.loop_stats(win)
        self.assertEqual((st["p_rev_level"], st["p_rev_heal"], st["p_rev_seen"], st["p_rev_explore"]),
                         (1.0, 10.0, 0.5, 0.0))
        self.assertEqual(st["p_on_revisits"], 11.5)

    def test_mark_revisits_resets_per_episode(self):
        tr = [row(i, ep_step=i % 3 + 1) for i in range(6)]
        det.mark_revisits(tr)
        self.assertEqual([r["revisit"] for r in tr], [False, True, True, False, True, True])

    @staticmethod
    def explore_trace(rate_at, n=6 * det.W):
        """New tiles every step; P per step = rate_at(i) (as explore). T flat throughout."""
        return [row(i, x=i, y=0, ep_step=i + 1, action=1, r_explore=rate_at(i), p=rate_at(i)) for i in range(n)]

    def test_U_floor_and_flat_T(self):
        hi = self.explore_trace(lambda i: 0.01)                                       # 20.48 P per window
        recs = det.apply_u(det.window_records(hi), 0.0)
        u = [r for r in recs if "U" in r["fired"]]
        # needs 4W of flat T ending at the window: windows starting at >= 3W only
        self.assertTrue(u and all(r["start"] >= 3 * det.W for r in u))
        lo = self.explore_trace(lambda i: 0.002)                                      # 4.1 P per window < 5.0
        self.assertFalse(any("U" in r["fired"] for r in det.apply_u(det.window_records(lo), 0.0)))

    def test_U_threshold_is_pool_wide(self):
        """Within-episode deciles fire in any flat episode; a checkpoint-wide pool does not."""
        weak = det.window_records(self.explore_trace(lambda i: 0.003))                 # 6.1 P per window
        strong = det.window_records(self.explore_trace(lambda i: 0.02))                # 41 P per window
        own = det.apply_u([dict(r, fired=list(r["fired"]), evidence=dict(r["evidence"])) for r in weak],
                          det.u_threshold(weak))
        self.assertTrue(any("U" in r["fired"] for r in own))                           # per-episode: fires
        thr = det.u_threshold(weak + strong)
        det.apply_u(weak, thr)
        det.apply_u(strong, thr)
        self.assertFalse(any("U" in r["fired"] for r in weak))                         # pooled: weak is not top decile
        self.assertTrue(any("U" in r["fired"] for r in strong))

    def test_U_blocked_by_hack_not_by_K(self):
        recs = det.window_records(self.explore_trace(lambda i: 0.01))
        for r in recs:
            r["fired"] = ["M"]
        self.assertFalse(any("U" in r["fired"] for r in det.apply_u(recs, 0.0)))
        recs = det.window_records(self.explore_trace(lambda i: 0.01))
        for r in recs:
            r["fired"] = ["K"]
        self.assertTrue(any("U" in r["fired"] for r in det.apply_u(recs, 0.0)))

    def test_window_stats_fields(self):
        st = det.window_stats(self.pace(n=3 * det.W))
        self.assertEqual(len(st), 5)
        for k in ("tile_arrivals", "kgram_repeats", "distinct_tiles", "p_in_w", "p_on_revisits", "p_rev_explore",
                  "p_rev_seen", "p_rev_level", "p_rev_heal", "maps", "t_flat_4w", "fired"):
            self.assertIn(k, st[0])
        self.assertEqual(st[0]["fired"], ["K"])


if __name__ == "__main__":
    unittest.main()
