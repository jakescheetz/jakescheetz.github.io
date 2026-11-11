#!/usr/bin/env bash
# Delete the E1 VM and its disk (stops all billing for it). Fetches results first unless --no-fetch.
#
#   gcp/teardown.sh [--no-fetch] [--yes] [--dry-run]
# Afterwards it lists anything still labelled purpose=e1 (instances, disks, snapshots) so nothing is left.
set -euo pipefail
. "$(dirname "$0")/config.sh"
FETCH=1; YES=0
while [ $# -gt 0 ]; do
  case "$1" in
    --no-fetch) FETCH=0 ;; --yes|-y) YES=1 ;; --dry-run) DRY_RUN=1 ;;
    --vm) shift ;;                          # handled in config.sh
    -h|--help) sed -n '2,5p' "$0"; exit 0 ;; *) die "unknown argument $1" ;;
  esac; shift
done
need_gcloud
S=$(vm_status || true)
if [ -n "$S" ]; then
  if [ "$FETCH" = 1 ]; then
    say "fetching results first (the disk is about to be deleted)"
    "$GCP_DIR/fetch.sh" $([ "$DRY_RUN" = 1 ] && echo --dry-run) || die "fetch failed; not deleting. Fix it, or --no-fetch to delete anyway"
    [ "$DRY_RUN" = 1 ] || gssh "pgrep -f '[t]rain.py' >/dev/null && echo 'WARNING: training is still running on the VM'" || true
  fi
  echo "About to DELETE $E1_VM ($S) in $E1_ZONE with all its disks. Unfetched results on it are lost."
  if [ "$YES" != 1 ] && [ "$DRY_RUN" != 1 ]; then
    read -r -p "Type the VM name to confirm: " ok; [ "$ok" = "$E1_VM" ] || die "aborted"
  fi
  act g compute instances delete "$E1_VM" --zone "$E1_ZONE" --delete-disks=all --quiet
else
  say "no VM $E1_VM in $E1_ZONE"
fi
say "anything left with label purpose=e1 (should be empty):"
[ "$DRY_RUN" = 1 ] && exit 0
g compute instances list --filter='labels.purpose=e1' --format='table(name,zone,status)'
g compute disks list --filter="labels.purpose=e1 OR name=$E1_VM" --format='table(name,zone,sizeGb)'
g compute snapshots list --filter='labels.purpose=e1' --format='table(name)'
