#!/usr/bin/env bash
# The 1-hour bench on the VM (it must exist: gcp/launch.sh). Runs inside tmux on the VM, so a dropped
# ssh does not stop it. See remote/bench.sh for what it measures.
#
#   gcp/bench.sh [--minutes 60] [--no-sync] [--dry-run]   start it
#   gcp/bench.sh --status                                 tail the log
#   gcp/bench.sh --attach                                 watch live (detach: Ctrl-b d)
# Then gcp/fetch.sh; the summary is ~/roms/runs/bench-<stamp>/summary.txt.
set -euo pipefail
. "$(dirname "$0")/config.sh"
MIN=60; SYNC=1; MODE=start
while [ $# -gt 0 ]; do
  case "$1" in
    --minutes) MIN=$2; shift ;; --no-sync) SYNC=0 ;; --dry-run) DRY_RUN=1 ;;
    --status) MODE=status ;; --attach) MODE=attach ;;
    --vm) shift ;;                          # handled in config.sh
    -h|--help) sed -n '2,8p' "$0"; exit 0 ;; *) die "unknown argument $1" ;;
  esac; shift
done
need_gcloud; ensure_running
case "$MODE" in
  status) gssh "tail -n 25 ~/e1/bench.log"; exit ;;
  attach) act g compute ssh "$E1_VM" --zone "$E1_ZONE" -- -t tmux attach -t bench; exit ;;
esac
gssh "! tmux has-session -t bench 2>/dev/null" || die "a bench is already running (gcp/bench.sh --status)"
gssh "! pgrep -f '[t]rain.py' >/dev/null" || die "training is running on $E1_VM; a bench now would measure contention"
[ "$SYNC" = 1 ] && sync_code
gssh "tmux new -d -s bench 'bash ~/e1/code/gcp/remote/bench.sh $MIN 2>&1 | tee ~/e1/bench.log'"
say "bench started (~$MIN min). gcp/bench.sh --status | --attach; then gcp/fetch.sh"
