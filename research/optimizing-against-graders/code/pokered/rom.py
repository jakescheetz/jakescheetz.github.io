"""ROM location, SHA-1 check, and the out-of-repo directories (experiment-e1.md §3, §12).

The ROM path comes only from $POKEMON_RED_ROM. Save states default to
<rom dir>/states/ and run outputs (screenshots, clips) to <rom dir>/runs/.
Both can be overridden with $POKEMON_STATES_DIR / $POKEMON_RUNS_DIR, but we
refuse any path inside this git checkout, so nothing ROM-derived lands in it.
"""
import hashlib
import os
from pathlib import Path

# English Pokémon Red, per Whidden's README (experiment-e1.md §3).
RED_SHA1 = "ea9bcae617fdf159b045185467ae58b2e4a48b9a"


def _guarded_root(here=None) -> Path:
    """The git checkout this code sits in. If there is none (code copied to a VM or
    another box, e.g. ~/e1/code), guard the code directory itself; a fixed
    parents[n] would there resolve to /home and wrongly refuse ~/roms."""
    here = Path(here or __file__).resolve()
    for d in here.parents:
        if (d / ".git").exists():
            return d
    return here.parents[1]


REPO_ROOT = _guarded_root()


class RomError(RuntimeError):
    pass


def _outside_repo(p: Path, what: str) -> Path:
    p = p.expanduser().resolve()
    if p == REPO_ROOT or REPO_ROOT in p.parents:
        raise RomError(f"{what} {p} is inside the repo; keep ROM-derived files outside it")
    return p


def sha1(path) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def rom_path(check=True) -> Path:
    env = os.environ.get("POKEMON_RED_ROM")
    if not env:
        raise RomError("set POKEMON_RED_ROM to your own Pokémon Red dump, e.g. ~/roms/pokemon_red.gb")
    p = _outside_repo(Path(env), "POKEMON_RED_ROM")
    if not p.is_file():
        raise RomError(f"no ROM at {p}")
    if check:
        got = sha1(p)
        if got != RED_SHA1:
            raise RomError(f"ROM SHA-1 {got} != English Red {RED_SHA1}; refusing to run")
    return p


def states_dir() -> Path:
    d = os.environ.get("POKEMON_STATES_DIR")
    p = _outside_repo(Path(d) if d else rom_path(check=False).parent / "states", "states dir")
    p.mkdir(parents=True, exist_ok=True)
    return p


def runs_dir() -> Path:
    d = os.environ.get("POKEMON_RUNS_DIR")
    p = _outside_repo(Path(d) if d else rom_path(check=False).parent / "runs", "runs dir")
    p.mkdir(parents=True, exist_ok=True)
    return p
