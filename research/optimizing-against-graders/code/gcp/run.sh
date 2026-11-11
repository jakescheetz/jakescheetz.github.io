#!/usr/bin/env bash
# Train arms on the VM, each in its own tmux window (session "e1"), checkpointing every 15 min and on
# SIGTERM (spot preemption, cost-guard stop). Starting a plan again resumes partial runs and skips done ones.
#
#   gcp/run.sh PLAN [--envs N] [--no-sync] [--dry-run]   e.g. gcp/run.sh gcp/plans/pilot.txt --envs 30
#   gcp/run.sh --resume-all                              after a preemption / guard stop (starts the VM)
#   gcp/run.sh --status                                  step, sps, T per run; tmux windows
#   gcp/run.sh --attach                                  live view (Ctrl-b n next window, Ctrl-b d detach)
#   gcp/run.sh --guard HOURS                             re-arm the in-guest uptime guard from now
set -euo pipefail
. "$(dirname "$0")/config.sh"
MODE=plan; PLAN=; SYNC=1; ENVS=; HOURS=
while [ $# -gt 0 ]; do
  case "$1" in
    --resume-all) MODE=resume ;; --status) MODE=status ;; --attach) MODE=attach ;;
    --guard) MODE=guard; HOURS=$2; shift ;;
    --envs) ENVS=$2; shift ;; --no-sync) SYNC=0 ;; --dry-run) DRY_RUN=1 ;;
    --vm) shift ;;                          # handled in config.sh
    -h|--help) sed -n '2,11p' "$0"; exit 0 ;;
    -*) die "unknown argument $1" ;; *) PLAN=$1 ;;
  esac; shift
done
need_gcloud; ensure_running
record_vm_events      # date any stop since the last run, for the resume log (pokered/runlog.py)
case "$MODE" in
  status)
    gssh "python3 ~/e1/code/gcp/remote/runs.py table; tmux list-windows -t e1 -F '#{window_name} dead=#{pane_dead}' 2>/dev/null; uptime"
    exit ;;
  attach) act g compute ssh "$E1_VM" --zone "$E1_ZONE" -- -t tmux attach -t e1; exit ;;
  guard)
    case "$HOURS" in ''|*[!0-9]*) die "--guard takes whole hours" ;; esac
    act g compute instances add-metadata "$E1_VM" --zone "$E1_ZONE" --metadata "e1-guard-hours=$HOURS"
    gssh "sudo shutdown -h +$((HOURS * 60)) 'e1 cost guard: $HOURS h'"
    say "in-guest guard re-armed: power-off in $HOURS h. The GCP hard cap (--termination-time) still applies;"
    say "to move it, use gcp/arm.sh (sizes both to a queue and checks the credit ceiling)"
    exit ;;
  resume) gssh "bash ~/e1/code/gcp/remote/run_plan.sh --resume-all"; exit ;;
esac
[ -n "$PLAN" ] && [ -f "$PLAN" ] || die "give a plan file, e.g. gcp/plans/pilot.txt"
if [ "$SYNC" = 1 ]; then
  # never swap code under a running trainer: resumed runs must use the code they started with
  gssh "! pgrep -f '[t]rain.py' >/dev/null" || die "training is running; rerun with --no-sync to add runs on the same code"
  sync_code
fi
name=$(basename "$PLAN")
gssh "mkdir -p ~/e1/plans"
grsync -a "$PLAN" "$E1_VM:e1/plans/$name"
gssh "E1_ENVS=$ENVS bash ~/e1/code/gcp/remote/run_plan.sh ~/e1/plans/$name"
say "gcp/run.sh --status to check; gcp/fetch.sh to copy results back"
