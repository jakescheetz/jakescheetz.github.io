#!/usr/bin/env bash
# Arm one VM for one batch of the main arms on ON-DEMAND, then hand the batch to the in-VM queue, so the
# laptop can sleep. (Needs Jake's go-ahead per batch: it starts billable on-demand time.)
#
#   gcp/arm.sh --ceiling --M N [--dry-run]           once: write the plan's credit ceiling (plan_cost.py) to the ledger
#   gcp/arm.sh --ceiling-b2 USD HOURS_A HOURS_B [--dry-run]
#                                                    batch 2 (gcp/b2_cost.py): new $ ceiling for batch-2 spend only
#                                                    (earlier windows move to "history"); per-VM hours are on top of
#                                                    each VM's lifetime counter at its first batch-2 arming
#   gcp/arm.sh QUEUE --vm e1-a --M N [--envs K] [--hours H] [--dry-run] [--yes]
#                                                    --hours: window = H whole hours instead of plan_cost.py's
#                                                    (batch 2: what is left of this VM's batch-2 hours ceiling)
#                                                    size guards to QUEUE, set on-demand + deadline, start the queue
#   gcp/arm.sh --window HOURS --vm e1-a [--dry-run]  a short window with no queue (fetch, inspect), charged the same way
#   gcp/arm.sh --show                                ledger: ceiling, used, armed windows
#
# What arming sets (the VM must be stopped for scheduling changes; arm.sh stops it only if nothing trains):
#   scheduling   --provisioning-model STANDARD --maintenance-policy MIGRATE --restart-on-failure
#                (host maintenance live-migrates, processes keep running; a host error restarts the VM and
#                the startup script resumes the queue: remote/guard-startup.sh step 4, remote/queue.sh)
#   GCP cap      --termination-time = now + window, absolute: a restart does not extend it (unlike
#                --max-run-duration), and once it passes the VM cannot be started until re-armed
#   in-guest     e1-guard-hours = window (per boot), e1-budget-hours = this VM's whole-plan ceiling hours
#                (cumulative over all boots), e1-idle-minutes = 20: the VM stops when the queue is done
#   window       = plan_cost.py hours QUEUE x 1.3 + 1 h, whole hours
#   ledger       ~/roms/runs/gcp-budget.json (outside the repo). Refuses to arm if this VM's used hours
#                (in-guest counter) + the window would pass its ceiling hours, or the plan's total
#                (both VMs, at the plan_cost.py rates) would pass ceiling_usd.
set -euo pipefail
. "$(dirname "$0")/config.sh"
MODE=queue; QUEUE=; M=; ENVS=; YES=0; WIN=; HOURS=
while [ $# -gt 0 ]; do
  case "$1" in
    --ceiling) MODE=ceiling ;; --show) MODE=show ;;
    --ceiling-b2) MODE=ceiling_b2; B2=("$2" "$3" "$4"); shift 3 ;;
    --window) MODE=window; WIN=$2; shift ;;
    --M) M=$2; shift ;; --envs) ENVS=$2; shift ;; --hours) HOURS=$2; shift ;;
    --vm) shift ;;                          # handled in config.sh
    --dry-run) DRY_RUN=1 ;; --yes|-y) YES=1 ;;
    -h|--help) sed -n '2,26p' "$0"; exit 0 ;;
    -*) die "unknown argument $1" ;; *) QUEUE=$1 ;;
  esac; shift
done
LEDGER=$E1_LOCAL_ROMS/runs/gcp-budget.json
IDLE=20
PC="python3 $GCP_DIR/plan_cost.py"

ledger() {  # ledger ACTION [ARGS] — tiny JSON helper (stdlib)
  DRY_RUN=$DRY_RUN python3 - "$LEDGER" "$@" <<'EOF'
import json, sys, os, datetime
path, act, *args = sys.argv[1:]
L = json.load(open(path)) if os.path.exists(path) else {}
if act == "init":
    M, usd, ha, hb = args
    L.update({"M": int(M), "ceiling_usd": float(usd), "ceiling_hours": {"e1-a": float(ha), "e1-b": float(hb)},
              "windows": L.get("windows", []), "set": datetime.datetime.now().astimezone().isoformat()})
elif act == "init_b2":   # batch 2: USD HOURS_A HOURS_B (hours relative to each VM's counter at first arming)
    usd, ha, hb = args
    hist = L.get("history", []) + [{k: v for k, v in L.items() if k != "history"}]
    L = {"batch": "b2", "ceiling_usd": float(usd), "rel_hours": {"e1-a": float(ha), "e1-b": float(hb)},
         "base_hours": {}, "ceiling_hours": {}, "windows": [], "history": hist,
         "set": datetime.datetime.now().astimezone().isoformat(), "source": "gcp/b2_cost.py"}
elif act == "check":     # check VM USED_H WIN_H RATE  -> prints budget hours for the guest, or exits 1
    vm, used, win, rate = args[0], float(args[1]), float(args[2]), float(args[3])
    if "ceiling_usd" not in L:
        sys.exit("no ceiling yet: gcp/arm.sh --ceiling --M N")
    if "rel_hours" in L:   # batch 2: base = this VM's lifetime counter at its first batch-2 arming
        base = L["base_hours"].get(vm, used)
        L["ceiling_hours"][vm] = base + L["rel_hours"][vm]
        if os.environ.get("DRY_RUN") != "1":        # a dry run reads no counter (used = 0): never store it
            L["base_hours"][vm] = base
            json.dump(L, open(path + ".tmp", "w"), indent=1); os.replace(path + ".tmp", path)
    cap = L["ceiling_hours"][vm]
    # other VM: its counter at its last arming + that whole window (an upper bound until it is armed again)
    base = L.get("base_hours", {})
    other = 0.0
    for o in {w["vm"] for w in L["windows"]} - {vm}:
        w = [w for w in L["windows"] if w["vm"] == o][-1]
        other += (w["used_before_h"] - base.get(o, 0.0) + w["hours"]) * w["rate"]
    if used + win > cap:
        sys.exit(f"{vm}: {used:.1f} h used + {win:.0f} h window > ceiling {cap:.0f} h")
    mine = (used - base.get(vm, 0.0) + win) * rate
    if other + mine > L["ceiling_usd"]:
        sys.exit(f"plan total would pass ${L['ceiling_usd']:.0f}: other VM ${other:.2f} + {vm} ${mine:.2f}")
    print(int(cap))
    sys.exit(0)
elif act == "add":       # add VM HOURS RATE DEADLINE WHAT USED
    vm, h, rate, dl, what, used = args
    L["windows"].append({"vm": vm, "hours": float(h), "rate": float(rate), "deadline": dl, "what": what,
                         "used_before_h": float(used), "armed": datetime.datetime.now().astimezone().isoformat()})
elif act == "show":
    print(json.dumps(L, indent=1)); sys.exit(0)
json.dump(L, open(path + ".tmp", "w"), indent=1); os.replace(path + ".tmp", path)
EOF
}

case "$MODE" in
  show) ledger show; exit ;;
  ceiling_b2)
    say "batch-2 ceiling: \$${B2[0]}; e1-a <= ${B2[1]} h, e1-b <= ${B2[2]} h on top of each VM's counter at its first batch-2 arming"
    [ "$DRY_RUN" = 1 ] && exit 0
    ledger init_b2 "${B2[@]}"; say "wrote $LEDGER (batch-1 entries kept under history)"; exit ;;
  ceiling)
    [ -n "$M" ] || die "--ceiling needs --M"
    read -r USD HA HB < <(python3 - "$GCP_DIR" "$M" <<'EOF'
import sys; sys.path.insert(0, sys.argv[1]); import plan_cost as P
r = P.estimate(int(sys.argv[2]), 14)
print(round(r["ceiling_usd"], 2), round(r["ceiling_vm_hours"]["e1-a"]), round(r["ceiling_vm_hours"]["e1-b"]))
EOF
)
    $PC estimate --M "$M" | tail -n 4
    say "ceiling: \$$USD; e1-a <= $HA h, e1-b <= $HB h (in-guest lifetime budget; arm.sh refuses past it)"
    [ "$DRY_RUN" = 1 ] && exit 0
    mkdir -p "$(dirname "$LEDGER")"; ledger init "$M" "$USD" "$HA" "$HB"; say "wrote $LEDGER"; exit ;;
esac

need_gcloud
TYPE=$(g compute instances describe "$E1_VM" --zone "$E1_ZONE" --format='value(machineType.basename())' 2>/dev/null) \
  || die "no VM $E1_VM (gcp/launch.sh first)"
VCPUS=${TYPE##*-}
if [ "$MODE" = queue ]; then
  [ -f "$QUEUE" ] || die "give a queue file (gcp/plans/main-*.txt)"
  grep -q '{M}' "$QUEUE" && [ -z "$M" ] && die "$QUEUE uses {M}; give --M"
  read -r WIN EST RATE < <($PC hours "$QUEUE" --vcpus "$VCPUS" ${M:+--M "$M"})
  if [ -n "$HOURS" ]; then
    case "$HOURS" in *[!0-9]*) die "--hours takes whole hours" ;; esac
    WIN=$HOURS
  fi
  WHAT="queue $(basename "$QUEUE") M=$M"
else
  case "$WIN" in ''|*[!0-9]*) die "--window takes whole hours" ;; esac
  read -r _ _ RATE < <($PC hours /dev/null --vcpus "$VCPUS")
  EST=$WIN; WHAT="window"
fi

# hours this VM has already run under the in-guest lifetime counter (/var/lib/e1/uptime-s). It can only be
# read from a running guest, so a stopped VM is armed, started, read, and stopped again if over the ceiling.
ST=$(vm_status)
read_used() {
  [ "$DRY_RUN" = 1 ] && { echo 0; return; }
  local s; s=$(gssh "sudo cat /var/lib/e1/uptime-s 2>/dev/null || echo 0" 2>/dev/null | tail -1)
  case "$s" in ''|*[!0-9]*) s=0 ;; esac
  python3 -c "print(round($s / 3600, 2))"
}
check() {  # check USED -> sets BUDGET or dies (stopping the VM if this script started it)
  BUDGET=$(ledger check "$E1_VM" "$1" "$WIN" "$RATE") && return 0
  [ "${STARTED:-0}" = 1 ] && act g compute instances stop "$E1_VM" --zone "$E1_ZONE" --quiet
  die "ceiling check failed (above); not arming"
}
USED="(read after start)"
BUDGET=$(python3 -c "
import json, sys
L = json.load(open('$LEDGER'))
if '$E1_VM' in L.get('ceiling_hours', {}):
    print(int(L['ceiling_hours']['$E1_VM']))
elif '$E1_VM' in L.get('base_hours', {}):
    print(int(L['base_hours']['$E1_VM'] + L['rel_hours']['$E1_VM']))
else:   # a relative ceiling with no base counter: the guest would get the relative hours as its LIFETIME budget
    sys.exit(1)                          # and power off at boot (2026-11-10 r1, e1-a)" 2>/dev/null) \
  || die "no lifetime ceiling for $E1_VM in $LEDGER (a relative ceiling needs base_hours.$E1_VM = its uptime counter)"
if [ "$ST" = RUNNING ]; then USED=$(read_used); check "$USED"; fi
DEADLINE=$(date -u -d "+$WIN hours" +%Y-%m-%dT%H:%M:%SZ)
cat <<EOF
Arm $E1_VM ($TYPE) on ON-DEMAND (billable, trial credit):
  what        $WHAT   (estimate $EST h on this VM)
  window      $WIN h: in-guest per-boot guard $WIN h, GCP --termination-time $DEADLINE (absolute)
  lifetime    $USED h used of $BUDGET h ceiling for $E1_VM; idle stop after $IDLE min
  max spend   ≈ \$$(python3 -c "print(round($WIN * $RATE, 2))") for this window at \$$RATE/h (UNVERIFIED third-party rate)
EOF
if [ "$YES" != 1 ] && [ "$DRY_RUN" != 1 ]; then
  read -r -p "Arm it? [y/N] " ok; [ "$ok" = y ] || [ "$ok" = Y ] || die "aborted"
fi

# scheduling changes need a stopped VM; never stop one that is training
if [ "$ST" = RUNNING ]; then
  [ "$DRY_RUN" = 1 ] || gssh "! pgrep -f '[t]rain.py|[e]valuate.py' >/dev/null" \
    || die "$E1_VM is training/evaluating; arm it after that finishes"
  act g compute instances stop "$E1_VM" --zone "$E1_ZONE" --quiet
fi
act g compute instances set-scheduling "$E1_VM" --zone "$E1_ZONE" \
  --provisioning-model STANDARD --maintenance-policy MIGRATE --restart-on-failure \
  --clear-max-run-duration --termination-time "$DEADLINE" --instance-termination-action STOP
REMOTE_USER=${E1_REMOTE_USER:-$USER}
act g compute instances add-metadata "$E1_VM" --zone "$E1_ZONE" \
  --metadata "e1-guard-hours=$WIN,e1-budget-hours=$BUDGET,e1-idle-minutes=$IDLE,e1-user=$REMOTE_USER" \
  --metadata-from-file "startup-script=$GCP_DIR/remote/guard-startup.sh"
if [ "$ST" != RUNNING ]; then
  # start with the GCP deadline already in place, then read the counter and check before anything runs
  STARTED=1; ensure_running; USED=$(read_used); check "$USED"
else
  ensure_running
fi
[ "$DRY_RUN" = 1 ] || ledger add "$E1_VM" "$WIN" "$RATE" "$DEADLINE" "$WHAT" "$USED"
if [ "$MODE" = queue ]; then
  sync_code                                  # the VM is idle here (checked above), so no trainer sees a swap
  gssh "mkdir -p ~/e1/plans"
  grsync -a "$QUEUE" "$E1_VM:e1/plans/$(basename "$QUEUE")"
  gssh "E1_M=$M E1_ENVS=${ENVS:-} bash ~/e1/code/gcp/remote/queue.sh start ~/e1/plans/$(basename "$QUEUE")"
  say "armed. The VM runs the queue and stops itself; the laptop can sleep."
  say "check: gcp/run.sh --vm $E1_VM --status; gssh 'bash ~/e1/code/gcp/remote/queue.sh status'"
else
  say "window open until $DEADLINE (or $IDLE min idle)"
fi
