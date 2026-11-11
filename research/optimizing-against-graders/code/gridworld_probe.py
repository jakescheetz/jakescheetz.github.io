#!/usr/bin/env python3
"""Timing probe for G4: tabular Q-learning on a tiny gridworld with a proxy reward.

Not the final experiment, only a check that the notebook version runs in
seconds on this laptop. Standard library only.

Grid (7x7): start S, goal G (true reward +10, episode ends), a respawning coin C
next to a loop (proxy pays +1 per pickup, true reward 0), and a tamper cell T
(stepping on it makes the proxy pay +2 every later step; true reward 0).
Proxy = coins + tamper bonus + goal. True = goal only, minus a step cost.
We train on the proxy for increasing step budgets and report proxy vs true
return of the greedy policy: the Goodhart curve in miniature.

Usage:  python gridworld_probe.py [--seeds 5]
Output: table on stdout, plus wall-clock time.
"""
import argparse
import random
import time

W = H = 7
S, G, C, T = (0, 0), (6, 6), (3, 1), (1, 5)
ACTS = [(0, 1), (1, 0), (0, -1), (-1, 0)]
MAX_T = 60


def step(pos, tampered, coin_up, a):
    x, y = pos
    dx, dy = ACTS[a]
    pos = (min(max(x + dx, 0), W - 1), min(max(y + dy, 0), H - 1))
    proxy, true, done = 0.0, -0.1, False
    if pos == C and coin_up:
        proxy += 1.0
        coin_up = False
    elif pos != C:
        coin_up = True          # coin respawns once you step off it
    if pos == T:
        tampered = True
    if tampered:
        proxy += 2.0
    if pos == G:
        proxy += 10.0
        true += 10.0
        done = True
    return pos, tampered, coin_up, proxy, true, done


def state(pos, tampered, coin_up):
    return (pos[0] * H + pos[1]) * 4 + tampered * 2 + coin_up


def rollout(Q):
    pos, tp, cu = S, False, True
    P = R = 0.0
    for _ in range(MAX_T):
        q = Q[state(pos, tp, cu)]
        a = max(range(4), key=q.__getitem__)
        pos, tp, cu, p, r, done = step(pos, tp, cu, a)
        P += p
        R += r
        if done:
            break
    return P, R


def train(budget, seed, alpha=0.2, gamma=0.97, eps=0.2):
    rng = random.Random(seed)
    Q = [[0.0] * 4 for _ in range(W * H * 4)]
    n = 0
    while n < budget:
        pos, tp, cu = S, False, True
        for _ in range(MAX_T):
            s = state(pos, tp, cu)
            a = rng.randrange(4) if rng.random() < eps else max(range(4), key=Q[s].__getitem__)
            pos, tp, cu, p, _, done = step(pos, tp, cu, a)
            s2 = state(pos, tp, cu)
            target = p + (0.0 if done else gamma * max(Q[s2]))
            Q[s][a] += alpha * (target - Q[s][a])
            n += 1
            if done or n >= budget:
                break
    return Q


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    a = ap.parse_args()
    t0 = time.perf_counter()
    print(f"{'steps':>8} {'proxy':>8} {'true':>8}")
    for budget in (500, 2000, 8000, 32000, 128000):
        res = [rollout(train(budget, s)) for s in range(a.seeds)]
        print(f"{budget:>8} {sum(r[0] for r in res)/len(res):>8.1f} "
              f"{sum(r[1] for r in res)/len(res):>8.1f}")
    print(f"wall clock: {time.perf_counter() - t0:.1f} s")


if __name__ == "__main__":
    main()
