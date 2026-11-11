#!/usr/bin/env bash
# On the VM: the 1-hour bench (README "Before any long run", step 4). Replaces the unverified cloud
# throughput in Lean compute §3 with numbers measured on this machine type.
#   part 1  bench_lean.py: env-only speeds, fast_mode WRAM+HRAM fidelity, policy cost   (~10 min)
#   part 2  train.py, tiles-CNN, NORMAL core (§5b), at env counts N/2, N-1, N; plus one fast_mode point at N
#           (for the guarded-fast_mode option, §5b ruling), MIN minutes each
#   summary sps_update median/p10 per env count (first 2 min dropped as warm-up) -> summary.txt
# Outputs: ~/roms/runs/bench-<stamp>/ (fetch.sh brings them back). Uses D1 if it exists, else power-on.
set -uo pipefail
TOTAL=${1:-60}
cd ~/e1/code
export POKEMON_RED_ROM=~/roms/pokemon_red.gb
STAMP=$(date +%Y%m%d-%H%M); OUT=~/roms/runs/bench-$STAMP; mkdir -p "$OUT"
N=$(nproc)
if [ -f ~/roms/states/D1.state ]; then STATE=(--state D1); LEAN=(--state D1); TAG=D1
else STATE=(--smoke --policy tilegrid); LEAN=(); TAG=boot; echo "no D1 yet: benching from power-on (smoke mode)"; fi
{
  echo "machine $(curl -sf -H 'Metadata-Flavor: Google' http://metadata.google.internal/computeMetadata/v1/instance/machine-type | sed 's#.*/##')"
  echo "provisioning $(curl -sf -H 'Metadata-Flavor: Google' http://metadata.google.internal/computeMetadata/v1/instance/scheduling/provisioning-model)"
  echo "vcpus $N   $(lscpu | grep 'Model name' | sed 's/  */ /g')"
  echo "code $(cat CODE_SHA256 2>/dev/null)   state $TAG   started $(date -Is)"
} | tee "$OUT/machine.txt"

T0=$(date +%s)
echo "== part 1: bench_lean.py"
.venv/bin/python bench_lean.py "${LEAN[@]}" 2>&1 | tee "$OUT/bench_lean.txt"
USED=$(( ($(date +%s) - T0) / 60 ))

POINTS=$(printf '%s\n' $((N / 2)) $((N - 1)) $N | awk '$1>=2' | sort -nu)
NPTS=$(( $(echo "$POINTS" | wc -l) + 1 ))
MIN=$(( (TOTAL - USED - 2) / NPTS )); [ "$MIN" -lt 5 ] && MIN=5
echo "== part 2: train.py sweep, envs = $(echo $POINTS), $MIN min each"
for k in $POINTS; do
  echo "-- envs $k (normal core)"
  timeout -s TERM "${MIN}m" .venv/bin/python train.py --arm A0 "${STATE[@]}" --seed 0 \
    --run-id "bench-$STAMP-e$k" --envs "$k" --steps 1000000000 --no-final-eval --ckpt-minutes 1000 \
    2>&1 | tail -n 4 | tee -a "$OUT/train_sweep.txt"
done
echo "-- envs $N (fast_mode, for comparison only)"
timeout -s TERM "${MIN}m" .venv/bin/python train.py --arm A0 "${STATE[@]}" --seed 0 --fast-mode \
  --run-id "bench-$STAMP-fast-e$N" --envs "$N" --steps 1000000000 --no-final-eval --ckpt-minutes 1000 \
  2>&1 | tail -n 4 | tee -a "$OUT/train_sweep.txt"

.venv/bin/python - "$STAMP" "$OUT" $POINTS "fast-e$N" <<'PY' | tee "$OUT/summary.txt"
import json, statistics as st, sys
from pathlib import Path
stamp, out, pts = sys.argv[1], Path(sys.argv[2]), sys.argv[3:]
print(open(out / "machine.txt").read().strip())
print(f"{'envs':>5} {'sps med':>8} {'p10':>6} {'p90':>6} {'collect s':>9} {'learn s':>8} {'load1':>6} {'MHz':>5} {'updates':>7}")
best = None
for k in pts:
    f = out.parent / (f"bench-{stamp}-{k}" if not k.isdigit() else f"bench-{stamp}-e{k}") / "metrics.jsonl"
    rows = [json.loads(l) for l in open(f)] if f.exists() else []
    rows = [r for r in rows if r["time"] - rows[0]["time"] >= 120] if rows else []
    if not rows:
        print(f"{k:>5}  no rows"); continue
    s = sorted(r["sps_update"] for r in rows)
    q = lambda p: s[min(len(s) - 1, int(p * len(s)))]
    mean = lambda key: st.mean(r[key] for r in rows if r.get(key) is not None)
    print(f"{k:>5} {st.median(s):>8.0f} {q(.1):>6.0f} {q(.9):>6.0f} {mean('t_collect'):>9.2f} "
          f"{mean('t_learn'):>8.2f} {mean('load1'):>6.1f} {mean('cpu_mhz'):>5.0f} {len(rows):>7}")
    if k.isdigit() and (best is None or st.median(s) > best[1]):
        best = (k, st.median(s))
if best:
    k, sps = best
    print(f"\nbest (normal core): --envs {k} at {sps:.0f} training steps/s (tiles-CNN, serial PPO, one run)")
    for name, steps in (("pilot 5M", 5e6), ("base plan + R1 ~207M", 207e6)):
        print(f"  {name:<22} ~{steps / sps / 3600:6.1f} h of VM time at this rate")
PY
echo "done $(date -Is); outputs in $OUT"
