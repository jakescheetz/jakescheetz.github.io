#!/usr/bin/env bash
# On the VM: system packages, pinned CPython via uv, the pinned venv, unit tests, ROM check.
set -euo pipefail
UV=${E1_UV_VERSION:-0.13.1}; PY=${E1_PYTHON:-3.14.8}
cd ~/e1/code
sudo apt-get update -qq
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq python3-venv tmux rsync >/dev/null
# uv is used only to fetch the exact CPython the laptop uses; packages are pip + requirements.txt as on the laptop
[ -x ~/.e1-uv/bin/uv ] || { python3 -m venv ~/.e1-uv && ~/.e1-uv/bin/pip install -q "uv==$UV"; }
~/.e1-uv/bin/uv python install "$PY"
[ -x .venv/bin/python ] || ~/.e1-uv/bin/uv venv --seed --python "$PY" .venv
.venv/bin/python -m pip install -q -r requirements.txt
.venv/bin/python -c 'import sys, pyboy, torch, numpy; print("python", sys.version.split()[0], "torch", torch.__version__, "numpy", numpy.__version__)'
.venv/bin/python -m pip freeze > ~/e1/pip-freeze.txt
diff <(grep -v '^#\|^--' requirements.txt | LC_ALL=C sort -f) <(LC_ALL=C sort -f ~/e1/pip-freeze.txt) \
  && echo "venv matches requirements.txt" || echo "WARNING: venv differs from requirements.txt (see diff above)"
.venv/bin/python -m unittest discover -s tests 2>&1 | tail -3
export POKEMON_RED_ROM=~/roms/pokemon_red.gb
.venv/bin/python -c 'from pokered import rom; print("ROM OK", rom.rom_path(), "runs ->", rom.runs_dir())'
grep -q POKEMON_RED_ROM ~/.bashrc || echo 'export POKEMON_RED_ROM=~/roms/pokemon_red.gb' >> ~/.bashrc
nproc; free -g | head -2
