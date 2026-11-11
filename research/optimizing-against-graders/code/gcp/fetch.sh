#!/usr/bin/env bash
# Copy ~/roms/runs/ (runs, checkpoints, logs, bench results, run screenshots) from the VM back to the
# laptop's ~/roms/runs/, outside the repo. Never deletes anything locally. Safe to run while training.
#
#   gcp/fetch.sh [--no-ckpt] [--dry-run]     --no-ckpt skips rolling ckpt-*.pt (eval-*.pt always come)
# Moving RAM-derived logs into the repo for figures is a separate, manual step (README "Run outputs").
set -euo pipefail
. "$(dirname "$0")/config.sh"
EXTRA=()
while [ $# -gt 0 ]; do
  case "$1" in
    --no-ckpt) EXTRA+=(--exclude 'ckpt-*.pt') ;; --dry-run) DRY_RUN=1 ;;
    --vm) shift ;;                          # handled in config.sh
    -h|--help) sed -n '2,6p' "$0"; exit 0 ;; *) die "unknown argument $1" ;;
  esac; shift
done
need_gcloud; ensure_running
mkdir -p "$E1_LOCAL_ROMS/runs"
grsync -az --partial "${EXTRA[@]}" --exclude '*.tmp' "$E1_VM:roms/runs/" "$E1_LOCAL_ROMS/runs/"
grsync -az "$E1_VM:e1/pip-freeze.txt" "$E1_LOCAL_ROMS/runs/gcp-pip-freeze.txt" || true
say "fetched into $E1_LOCAL_ROMS/runs/"
ls -1t "$E1_LOCAL_ROMS/runs" | head -n 10
