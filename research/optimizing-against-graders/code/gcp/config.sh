# Shared settings for the E1 GCP scripts. Sourced, not run.
# Override any of these in the environment, e.g.  E1_ZONE=us-east1-b gcp/launch.sh
# Nothing here creates resources; only launch.sh / teardown.sh do, and only when Jake runs them.

E1_VM=${E1_VM:-e1-box}                       # instance name
# zone given in the environment (not one this file resolved for another VM earlier in the same shell): use it
E1_ZONE_SET=; [ -n "${E1_ZONE:-}" ] && [ "${E1_ZONE_AUTO:-0}" != 1 ] && E1_ZONE_SET=1
[ "${E1_ZONE_AUTO:-0}" = 1 ] && E1_ZONE=
E1_ZONE=${E1_ZONE:-us-central1-a}             # default for new VMs; an existing VM's own zone wins (below)
# The trial project (pokemon-e1). Set explicitly: the gcloud default project is unset and Jake has others.
E1_PROJECT=${E1_PROJECT:-sylvan-storm-513522-q9}
# T2D: one vCPU = one physical core (no SMT), so the most emulator throughput per unit of the trial's
# binding quota (CPUS_ALL_REGIONS = 12). Type = <family>-<class>-<vcpus>; t2d only has "standard".
E1_FAMILY=${E1_FAMILY:-t2d}
E1_DISK_GB=${E1_DISK_GB:-30}
E1_IMAGE_FAMILY=${E1_IMAGE_FAMILY:-debian-12}
E1_GUARD_HOURS=${E1_GUARD_HOURS:-12}         # cost guard: VM powers off after this many hours of uptime
E1_IDLE_MINUTES=${E1_IDLE_MINUTES:-45}       # cost guard: ... or after this long with no train/bench process
E1_UV_VERSION=${E1_UV_VERSION:-0.13.1}       # uv only installs the pinned CPython; packages come from requirements.txt
E1_PYTHON=${E1_PYTHON:-3.14.8}               # same as the laptop venv
E1_LOCAL_ROMS=${E1_LOCAL_ROMS:-$HOME/roms}   # Jake's ROM + states + runs, outside the repo
E1_RED_SHA1=ea9bcae617fdf159b045185467ae58b2e4a48b9a   # experiment-e1.md §3

GCP_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
CODE_DIR=$(dirname "$GCP_DIR")
export E1_VM E1_ZONE E1_PROJECT

DRY_RUN=${DRY_RUN:-0}

family_class() { case "$1" in t2d|t2a) echo standard ;; *) echo highcpu ;; esac; }

# --vm NAME (any script): pick which VM (two VMs fit the 12-vCPU cap, e.g. e1-a 8 + e1-b 4)
for ((i = 1; i <= $#; i++)); do
  if [ "${!i}" = --vm ]; then j=$((i + 1)); E1_VM=${!j}; export E1_VM; fi
done

die() { echo "error: $*" >&2; exit 1; }
say() { echo "== $*"; }

# Every call that creates, changes or deletes something goes through `act`; with --dry-run it only prints.
act() {
  if [ "$DRY_RUN" = 1 ]; then printf '[dry-run]'; printf ' %q' "$@"; echo; else "$@"; fi
}

g() { gcloud ${E1_PROJECT:+--project "$E1_PROJECT"} "$@"; }

# An existing VM keeps its own zone (e1-b may be created elsewhere when a zone has no capacity), so every
# script finds it by name. One list call; skipped when E1_ZONE was set explicitly or gcloud is missing.
if [ -z "$E1_ZONE_SET" ] && command -v gcloud >/dev/null; then
  _z=$(g compute instances list --filter="name=($E1_VM)" --format='value(zone.basename())' 2>/dev/null | head -1)
  [ -n "$_z" ] && E1_ZONE=$_z
  E1_ZONE=${E1_ZONE:-us-central1-a}; E1_ZONE_AUTO=1
  export E1_ZONE E1_ZONE_AUTO
fi

need_gcloud() {
  command -v gcloud >/dev/null || die "gcloud not found (install the Google Cloud CLI, then: gcloud auth login; gcloud config set project <id>)"
  [ -n "$E1_PROJECT" ] || die "no project: gcloud config set project <id>, or E1_PROJECT=<id>"
}

# ssh to the VM (gcloud manages keys and host keys). Extra args are the remote command.
gssh() { act g compute ssh "$E1_VM" --zone "$E1_ZONE" --quiet -- "$@"; }

# rsync over gcloud ssh; remote side is written as  :path  (host is filled in).
grsync() { act rsync -e "$GCP_DIR/_gssh" "$@"; }

vm_status() { g compute instances describe "$E1_VM" --zone "$E1_ZONE" --format='value(status)' 2>/dev/null; }

ensure_running() {
  [ "$DRY_RUN" = 1 ] && return 0
  local s; s=$(vm_status)
  case "$s" in
    RUNNING) ;;
    TERMINATED|STOPPED|SUSPENDED)
      say "$E1_VM is $s (cost guard, spot preemption, or manual stop); starting it"
      g compute instances start "$E1_VM" --zone "$E1_ZONE" --quiet \
        || die "could not start $E1_VM; if its --termination-time has passed, re-arm it: gcp/arm.sh --window 1 --vm $E1_VM"
      # sshd comes up some seconds after RUNNING; wait for it (up to 5 min)
      local i; for i in $(seq 1 30); do sleep 10; gssh true 2>/dev/null && break
        [ "$i" = 30 ] && die "$E1_VM started but no ssh after 5 min"; done ;;
    "") die "no VM $E1_VM in $E1_ZONE; run gcp/launch.sh first" ;;
    *) die "$E1_VM is $s; wait and retry" ;;
  esac
}

# Copy code/ to ~/e1/code on the VM: no venv, caches, local runs, or anything ROM-derived.
sync_code() {
  say "rsync code/ -> $E1_VM:~/e1/code/"
  # the Debian image has no rsync; install it before the first copy (setup.sh installs the rest)
  gssh "command -v rsync >/dev/null || { sudo apt-get update -qq && sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq rsync >/dev/null; }; mkdir -p ~/e1/code"
  grsync -az --delete \
    --exclude .venv --exclude __pycache__ --exclude 'runs/' --exclude 'frames/' \
    --exclude '*.gb' --exclude '*.gbc' --exclude '*.state' --exclude '*.sav' \
    --exclude '*.png' --exclude '*.mp4' --exclude '*.gif' --exclude CODE_SHA256 --exclude 'results/sealed-*' \
    "$CODE_DIR/" "$E1_VM:e1/code/"
  # record exactly which code the runs used (config.json does not hash the code)
  local h; h=$(cd "$CODE_DIR" && find . -path ./.venv -prune -o -name '*.py' -print -o -name '*.sh' -print \
            -o -name requirements.txt -print | LC_ALL=C sort | xargs sha256sum | sha256sum | cut -c1-16)
  gssh "echo $h > ~/e1/code/CODE_SHA256"
  say "code hash $h"
}

# Append this VM's GCP stop events (Spot preemption, guest shutdown, manual stop) to ~/roms/runs/vm-events.jsonl
# on the VM, so a resumed train.py can date the stop (pokered/runlog.py). Deduplicated by operation id.
record_vm_events() {
  [ "$DRY_RUN" = 1 ] && return 0
  local ev
  ev=$(g compute operations list --zones "$E1_ZONE" --format=json 2>/dev/null | python3 -c '
import json, sys
vm = sys.argv[1]
kinds = {"compute.instances.preempted": "preempted", "compute.instances.guestTerminate": "guest_shutdown",
         "stop": "stop", "compute.instances.hostError": "host_error"}
for o in sorted(json.load(sys.stdin), key=lambda o: o["insertTime"]):
    if o.get("targetLink", "").rsplit("/", 1)[-1] == vm and o.get("operationType") in kinds:
        print(json.dumps({"id": o["id"], "type": kinds[o["operationType"]], "time": o["insertTime"], "vm": vm}))' "$E1_VM")
  [ -n "$ev" ] || return 0
  printf '%s\n' "$ev" | gssh "python3 ~/e1/code/gcp/remote/vm_events.py" 2>/dev/null || true
}
