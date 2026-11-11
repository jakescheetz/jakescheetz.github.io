"""g5_planner.py: BFS to the nearest unpaid square, blocked squares, move-menu parsing. No ROM needed."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import g5_planner as G  # noqa: E402


def line(n, m=1):
    """A corridor of n squares on map m, both directions."""
    fwd = {}
    for x in range(n):
        out = fwd.setdefault((m, x, 0), [])
        if x + 1 < n:
            out.append(((m, x + 1, 0), "right"))
        if x > 0:
            out.append(((m, x - 1, 0), "left"))
    return fwd


class BFS(unittest.TestCase):
    def test_nearest_unpaid(self):
        fwd = line(6)
        visited = {(1, x, 0) for x in range(4)}
        path = G.bfs_target(fwd, (1, 1, 0), visited, set(), set())
        self.assertEqual([n for _, n in path], [(1, 2, 0), (1, 3, 0), (1, 4, 0)])
        self.assertEqual([h for h, _ in path], ["right"] * 3)

    def test_blocked_and_dropped(self):
        fwd = line(6)
        fwd[(1, 0, 0)].append(((2, 0, 0), "warp"))
        fwd[(2, 0, 0)] = [((1, 0, 0), "warp")]
        visited = {(1, x, 0) for x in range(4)}
        self.assertEqual(G.bfs_target(fwd, (1, 1, 0), visited, set(), set())[-1][1], (2, 0, 0))   # 2 moves < 3
        self.assertEqual(G.bfs_target(fwd, (1, 1, 0), visited, {(1, 0, 0)}, set())[-1][1], (1, 4, 0))
        self.assertIsNone(G.bfs_target(fwd, (1, 1, 0), visited, {(1, 0, 0)}, {(1, 4, 0), (1, 5, 0)}))


class Menu(unittest.TestCase):
    def test_cursor_move_and_pp(self):
        t = "RATTATA 3      SQUIRTLE 7 TYPE/ NORMAL 24/ 24 30/30  TACKLE >TAIL WHIP"
        self.assertEqual(G.cursor_move("RATTATA 3      SQUIRTLE 7 TYPE/ NORMAL 24/ 24 35/35  >TACKLE TAIL WHIP"),
                         ("TACKLE TAIL WHIP", 35))
        name, pp = G.cursor_move(t)
        self.assertTrue(name.startswith(G.NON_DAMAGING))
        self.assertEqual(pp, 30)
        self.assertEqual(G.cursor_move("WEEDLE 4  CHARMANDER 13 TYPE/ NORMAL 26/ 37 0/35  >SCRATCH GROWL")[1], 0)


if __name__ == "__main__":
    unittest.main()
