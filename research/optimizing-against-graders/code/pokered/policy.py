"""Actor-critic policies for E1 (experiment-e1.md §5a).

Every policy takes one flat float32 observation vector (see `flatten_obs`) and
returns (logits over env.ACTIONS, value). Encoders:

  tilegrid  (main arms) slim tiles-CNN: wTileMap ids -> 8-dim embedding, plus the
            sprite-count channel, conv 16/32/32, concat a RAM-vector branch, 128 hidden.
  screen    (R1 replicate) Nature-style CNN on the 72x80 screen, same RAM branch and head.
  tilegrid_visited  (batch 2, §9 "Batch 2") the same, plus the binary visited channel as a 10th conv input.
  ram       (smoke runs only) MLP on the 74-dim RAM vector.

The RAM branch and the head are shared code, so R1 differs from the main arms
only in the encoder, as §5a requires.
"""
import numpy as np
import torch
import torch.nn as nn

from .env import ACTIONS

N_ACT = len(ACTIONS)
RAM_DIM = 74
TILE_CELLS = 18 * 20
SCREEN_CELLS = 72 * 80
OBS_DIM = {"tilegrid": 2 * TILE_CELLS + RAM_DIM, "tilegrid_visited": 3 * TILE_CELLS + RAM_DIM,
           "screen": SCREEN_CELLS + RAM_DIM, "ram": RAM_DIM}
ENV_OBS_MODE = {"tilegrid": "tilegrid", "tilegrid_visited": "tilegrid_visited", "screen": "screen", "ram": "ram"}
OBS_VERSION = {"tilegrid_visited": 2}      # batch 2 (§9 "Batch 2" H2); everything else is version 1


def flatten_obs(kind, o):
    if kind == "tilegrid":
        return np.concatenate([o["tiles"].ravel(), o["sprites"].ravel(), o["ram"]]).astype(np.float32)
    if kind == "tilegrid_visited":
        return np.concatenate([o["tiles"].ravel(), o["sprites"].ravel(), o["visited"].ravel(),
                               o["ram"]]).astype(np.float32)
    if kind == "screen":
        return np.concatenate([o["screen"].ravel() / 255.0, o["ram"]]).astype(np.float32)
    return o["ram"].astype(np.float32)


def _init(layer, gain=np.sqrt(2)):
    if isinstance(layer, (nn.Linear, nn.Conv2d)):
        nn.init.orthogonal_(layer.weight, gain)
        nn.init.zeros_(layer.bias)
    return layer


class TileEncoder(nn.Module):
    def __init__(self, visited=False):
        super().__init__()
        self.visited = visited                 # batch 2: + the visited channel (first conv 9 -> 10 inputs)
        self.emb = nn.Embedding(256, 8)
        self.conv = nn.Sequential(
            _init(nn.Conv2d(9 + visited, 16, 3, 1, 1)), nn.ReLU(),
            _init(nn.Conv2d(16, 32, 3, 2)), nn.ReLU(),
            _init(nn.Conv2d(32, 32, 3, 1)), nn.ReLU(), nn.Flatten())
        self.out_dim = 32 * 6 * 7

    def forward(self, x):
        t = x[:, :TILE_CELLS].long().clamp(0, 255).view(-1, 18, 20)
        sp = x[:, TILE_CELLS:2 * TILE_CELLS].view(-1, 1, 18, 20)
        parts = [self.emb(t).permute(0, 3, 1, 2), sp]
        if self.visited:
            parts.append(x[:, 2 * TILE_CELLS:3 * TILE_CELLS].view(-1, 1, 18, 20))
        return self.conv(torch.cat(parts, 1))


class ScreenEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            _init(nn.Conv2d(1, 32, 8, 4)), nn.ReLU(),
            _init(nn.Conv2d(32, 64, 4, 2)), nn.ReLU(),
            _init(nn.Conv2d(64, 64, 3, 1)), nn.ReLU(), nn.Flatten())
        self.out_dim = 64 * 5 * 6

    def forward(self, x):
        return self.conv(x[:, :SCREEN_CELLS].view(-1, 1, 72, 80))


class ActorCritic(nn.Module):
    def __init__(self, kind="tilegrid", hidden=128):
        super().__init__()
        self.kind = kind
        if kind == "ram":
            self.enc, enc_dim = None, 0
            self.ram = nn.Sequential(_init(nn.Linear(RAM_DIM, 256)), nn.Tanh(), _init(nn.Linear(256, 256)), nn.Tanh())
            ram_dim = 256
        else:
            self.enc = (TileEncoder(visited=kind == "tilegrid_visited") if kind.startswith("tilegrid")
                        else ScreenEncoder())
            enc_dim = self.enc.out_dim
            self.ram = nn.Sequential(_init(nn.Linear(RAM_DIM, 64)), nn.ReLU())
            ram_dim = 64
        self.trunk = nn.Sequential(_init(nn.Linear(enc_dim + ram_dim, hidden)), nn.ReLU())
        self.pi = _init(nn.Linear(hidden, N_ACT), 0.01)
        self.v = _init(nn.Linear(hidden, 1), 1.0)

    def forward(self, x):
        r = self.ram(x[:, -RAM_DIM:])
        h = r if self.enc is None else torch.cat([self.enc(x), r], 1)
        h = self.trunk(h)
        return self.pi(h), self.v(h).squeeze(-1)
