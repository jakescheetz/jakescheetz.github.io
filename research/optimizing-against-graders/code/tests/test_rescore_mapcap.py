"""Per-map new-tile cap (MapCapProxy), P0 hash stability, and rescore.py's RAM image from a trace row.
No ROM needed.  Run:  .venv/bin/python -m unittest discover -s tests"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pokered import env as E, ram  # noqa: E402
import rescore  # noqa: E402

ROW = {"map": 12, "x": 5, "y": 7, "party_count": 2, "levels": [9, 4], "hp": [20, 300], "max_hp": [31, 301],
       "seen": 11}


def mem_at(mp, x, y):
    return rescore.row_mem({**ROW, "map": mp, "x": x, "y": y})


class RowMem(unittest.TestCase):
    def test_proxy_reads_the_row(self):
        m = rescore.row_mem(ROW)
        self.assertEqual(ram.position(m), (12, 5, 7))
        self.assertEqual(ram.party_count(m), 2)
        self.assertEqual(ram.party_levels(m), [9, 4])
        self.assertEqual(ram.party_hp(m), [20, 300])
        self.assertEqual(ram.party_max_hp(m), [31, 301])
        self.assertEqual(ram.popcount_bytes(m, ram.W_POKEDEX_SEEN, ram.POKEDEX_BYTES), 11)


class MapCap(unittest.TestCase):
    def test_p0_hash_unchanged(self):
        self.assertEqual(E.reward_hash(E.make_proxy()), "d28e2a337dda")          # P0, as in every batch-1 config
        self.assertIs(type(E.make_proxy(tile_visit_cap=600)), E.Proxy)
        self.assertNotEqual(E.reward_hash(E.make_proxy(map_new_tile_cap=3)), "d28e2a337dda")

    def explore(self, p):
        p.reset(mem_at(0, 0, 0))
        out = []
        for mp, x in [(0, 1), (0, 2), (0, 3), (0, 4), (0, 2), (1, 0), (1, 1), (0, 5)]:
            out.append(p(mem_at(mp, x, 0), 0)[1]["explore"])
        return out

    def test_cap_per_map(self):
        # cap 2: map 0 pays its first 2 new tiles (x=1, 2), not x=3, 4, 5; map 1 has its own count
        self.assertEqual(self.explore(E.make_proxy(map_new_tile_cap=2)), [0.1, 0.1, 0, 0, 0, 0.1, 0.1, 0])
        self.assertEqual(self.explore(E.make_proxy()), [0.1, 0.1, 0.1, 0.1, 0, 0.1, 0.1, 0.1])

    def test_pooled_type_windows(self):
        import rescore
        by = rescore.windows_by_type([{"type": "O", "kind": "behind", "step": 1, "end": 10},
                                      {"type": "O", "kind": "side", "step": 5, "end": 20},
                                      {"type": "D", "step": 30, "end": 40}])
        self.assertEqual(by["O"], [(1, 10), (5, 20)])
        self.assertEqual((by["O:behind"], by["O:side"], by["D"]), ([(1, 10)], [(5, 20)], [(30, 40)]))
        self.assertNotIn("D:", "".join(by))

    def test_total_matches_parts(self):
        p = E.make_proxy(map_new_tile_cap=1)
        p.reset(mem_at(0, 0, 0))
        for x in range(1, 4):
            total, parts = p(mem_at(0, x, 0), 0)
            self.assertAlmostEqual(total, sum(parts.values()))


if __name__ == "__main__":
    unittest.main()
