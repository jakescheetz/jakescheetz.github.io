#!/usr/bin/env bash
# Create the E1 VM, copy code + ROM, build the pinned venv.   (needs Jake's go-ahead: creates billable resources)
#
#   gcp/launch.sh [VCPUS] [--vm NAME] [--spot] [--hours H] [--family t2d|n2|n2d|e2|c2d] [--zone Z] [--dry-run] [--yes]
#
# VCPUS defaults to 8 and is clamped to what the project's quota allows (regional family quota, plus
# the all-regions CPU quota, which is 12 on this trial and counts VMs already running). The machine is
# <family>-<class>-<n> (t2d-standard-<n> by default; highcpu for the other families), the largest size
# in the zone that fits. For the trial's 12 vCPUs: gcp/launch.sh 8 --vm e1-a, then gcp/launch.sh 4 --vm e1-b.
# Debian 12, no service account (the VM holds no GCP credentials), Secure Boot on.
#
# Cost guards, all of which STOP the VM (the disk and its results survive; teardown.sh deletes it):
#   1. in-guest timer: power off after --hours of uptime (default 12), re-armed on every boot
#   2. idle guard: power off after 45 min with no train.py / bench process running
#   3. GCP-side hard cap: --termination-time = now + hours + 1 (absolute; a restart doesn't extend it, and
#      after it the VM can't start until gcp/arm.sh re-arms it), so the VM stops even if the guest hangs
# On-demand (the default, no --spot): host maintenance live-migrates (MIGRATE) and a host error restarts
# the VM (--restart-on-failure); the startup script then resumes an active queue (gcp/arm.sh, remote/queue.sh).
# For the main arms, create the VM with this script, then gcp/arm.sh sizes the guards to each batch.
# The ROM goes over ssh (gcloud compute scp) to ~/roms on the VM: never a bucket, never public.
set -euo pipefail
. "$(dirname "$0")/config.sh"

VCPUS=8; SPOT=0; HOURS=$E1_GUARD_HOURS; YES=0; CONTINUE=0
while [ $# -gt 0 ]; do
  case "$1" in
    --spot) SPOT=1 ;;
    --hours) HOURS=$2; shift ;;
    --family) E1_FAMILY=$2; shift ;;
    --zone) E1_ZONE=$2; shift ;;
    --vm) shift ;;                          # handled in config.sh
    --dry-run) DRY_RUN=1 ;;
    --yes|-y) YES=1 ;;
    --continue) CONTINUE=1 ;;              # VM already created: redo only the copy + setup steps
    -h|--help) sed -n '2,18p' "$0"; exit 0 ;;
    [0-9]*) VCPUS=$1 ;;
    *) die "unknown argument $1" ;;
  esac
  shift
done
case "$HOURS" in ''|*[!0-9]*) die "--hours takes whole hours" ;; esac
REGION=${E1_ZONE%-*}
need_gcloud
[ -f "$E1_LOCAL_ROMS/pokemon_red.gb" ] || die "no ROM at $E1_LOCAL_ROMS/pokemon_red.gb"
LOCAL_SHA=$(sha1sum "$E1_LOCAL_ROMS/pokemon_red.gb" | cut -d' ' -f1)
[ "$LOCAL_SHA" = "$E1_RED_SHA1" ] || die "local ROM SHA-1 $LOCAL_SHA does not match §3; not copying it anywhere"
if [ "$CONTINUE" = 1 ]; then
  [ "$(vm_status)" = RUNNING ] || die "--continue: $E1_VM is not RUNNING"
  say "continuing on existing $E1_VM (no create)"
else
[ -z "$(vm_status)" ] || die "$E1_VM already exists in $E1_ZONE (gcp/teardown.sh, or E1_VM=<other name>)"

# --- quota: how many vCPUs can this project actually run? -----------------------------------
case "$E1_FAMILY" in
  e2) METRIC=E2_CPUS ;;                    # this project lists E2_CPUS (24); older projects pool E2 into CPUS
  t2d) METRIC=T2D_CPUS ;; n2) METRIC=N2_CPUS ;; n2d) METRIC=N2D_CPUS ;; c2d) METRIC=C2D_CPUS ;;
  *) die "family $E1_FAMILY not handled (t2d, e2, n2, n2d, c2d)" ;;
esac
CLASS=$(family_class "$E1_FAMILY")
free_of() {  # free_of METRIC  < quotas JSON   -> limit-usage, or empty if the metric is absent
  python3 -c '
import json, sys
m = sys.argv[1]
for q in json.load(sys.stdin).get("quotas", []):
    if q.get("metric") == m:
        print(int(q["limit"] - q.get("usage", 0))); break' "$1"
}
RQ=$(g compute regions describe "$REGION" --format=json)
PQ=$(g compute project-info describe --format=json)
ALLOW=$VCPUS; WHY="requested"
[ "$E1_FAMILY" = e2 ] && [ -z "$(free_of E2_CPUS <<<"$RQ")" ] && METRIC=CPUS
for pair in "$METRIC:$(free_of "$METRIC" <<<"$RQ")" \
            "CPUS_ALL_REGIONS:$(free_of CPUS_ALL_REGIONS <<<"$PQ")"; do
  m=${pair%%:*}; f=${pair#*:}
  [ -n "$f" ] && [ "$f" -lt "$ALLOW" ] && { ALLOW=$f; WHY="$m quota"; }
done
if [ "$SPOT" = 1 ]; then  # Spot uses PREEMPTIBLE_CPUS only once that quota has been requested (limit > 0)
  f=$(free_of PREEMPTIBLE_CPUS <<<"$RQ"); lim=$(python3 -c '
import json,sys
print(next((int(q["limit"]) for q in json.load(sys.stdin).get("quotas",[]) if q["metric"]=="PREEMPTIBLE_CPUS"),0))' <<<"$RQ")
  [ "$lim" -gt 0 ] && [ -n "$f" ] && [ "$f" -lt "$ALLOW" ] && { ALLOW=$f; WHY="PREEMPTIBLE_CPUS quota"; }
fi
[ "$ALLOW" -ge 2 ] || die "quota allows $ALLOW vCPUs for $E1_FAMILY in $REGION ($WHY); try --family e2 or another zone"

# largest <family>-<class>-<n> in the zone with n <= ALLOW. (gcloud 591 crashes on machine-types list
# with --filter or a table/value format, so list everything as JSON and filter here.)
read -r N TYPE < <(g compute machine-types list --zones "$E1_ZONE" --format=json \
  | python3 -c '
import json, re, sys
pat, allow = re.compile(r"^%s-%s-(\d+)$" % (sys.argv[1], sys.argv[2])), int(sys.argv[3])
fit = sorted((m["guestCpus"], m["name"]) for m in json.load(sys.stdin) if pat.match(m["name"]) and m["guestCpus"] <= allow)
print(*fit[-1]) if fit else print()' "$E1_FAMILY" "$CLASS" "$ALLOW")
[ -n "${TYPE:-}" ] || die "no ${E1_FAMILY}-${CLASS} type with <= $ALLOW vCPUs in $E1_ZONE"
[ "$N" = "$VCPUS" ] || say "asked for $VCPUS vCPUs; using $N ($WHY allows $ALLOW, nearest size $TYPE)"

PROV=STANDARD; [ "$SPOT" = 1 ] && PROV=SPOT
MAINT=(--maintenance-policy MIGRATE --restart-on-failure)
[ "$SPOT" = 1 ] && MAINT=(--maintenance-policy TERMINATE --no-restart-on-failure)
DEADLINE=$(date -u -d "+$((HOURS + 1)) hours" +%Y-%m-%dT%H:%M:%SZ)
cat <<EOF
About to create (billable; on the trial it draws from the \$300 credit):
  VM         $E1_VM   $TYPE ($N vCPU)   $PROV   zone $E1_ZONE   project $E1_PROJECT
  disk       ${E1_DISK_GB} GB pd-balanced, $E1_IMAGE_FAMILY
  guards     stop after ${HOURS} h uptime or ${E1_IDLE_MINUTES} min idle; GCP hard stop at $DEADLINE
  ROM        $E1_LOCAL_ROMS/pokemon_red.gb (+ states/) -> VM ~/roms over ssh
EOF
if [ "$YES" != 1 ] && [ "$DRY_RUN" != 1 ]; then
  read -r -p "Create it? [y/N] " ok; [ "$ok" = y ] || [ "$ok" = Y ] || die "aborted"
fi

# --- create --------------------------------------------------------------------------------
act g compute instances create "$E1_VM" --zone "$E1_ZONE" \
  --machine-type "$TYPE" --provisioning-model "$PROV" \
  "${MAINT[@]}" --termination-time "$DEADLINE" --instance-termination-action STOP \
  --image-family "$E1_IMAGE_FAMILY" --image-project debian-cloud \
  --boot-disk-size "${E1_DISK_GB}GB" --boot-disk-type pd-balanced \
  --no-service-account --no-scopes --shielded-secure-boot \
  --labels purpose=e1 \
  --metadata "e1-guard-hours=$HOURS,e1-idle-minutes=$E1_IDLE_MINUTES,e1-user=${E1_REMOTE_USER:-$USER}" \
  --metadata-from-file "startup-script=$GCP_DIR/remote/guard-startup.sh"

fi   # end of create (skipped with --continue)

say "waiting for ssh"
if [ "$DRY_RUN" != 1 ]; then
  for i in $(seq 1 30); do gssh true 2>/dev/null && break; sleep 10; [ "$i" = 30 ] && die "no ssh after 5 min"; done
fi

sync_code

# --- ROM + states, over ssh only -------------------------------------------------------------
say "copying ROM + states to $E1_VM:~/roms (ssh; no bucket)"
gssh "mkdir -p ~/roms/states ~/roms/runs && chmod 700 ~/roms"
act g compute scp --zone "$E1_ZONE" --quiet "$E1_LOCAL_ROMS/pokemon_red.gb" "$E1_VM:roms/pokemon_red.gb"
# only the start states and their metadata (not the review frames and check outputs next to them)
mapfile -t STATES < <(ls "$E1_LOCAL_ROMS"/states/*.state "$E1_LOCAL_ROMS"/states/*.json 2>/dev/null)
if [ "${#STATES[@]}" -gt 0 ]; then
  act g compute scp --zone "$E1_ZONE" --quiet "${STATES[@]}" "$E1_VM:roms/states/"
else
  say "no local states yet (D1 etc.); bench falls back to a boot state, training needs D1"
fi
# the A2″ walking-distance grid (built from pret data, ROM-derived: outside the repo; A2″ training and the
# eval's leg-progress metric read it)
if [ -f "$E1_LOCAL_ROMS/walkgrid.json" ]; then
  act g compute scp --zone "$E1_ZONE" --quiet "$E1_LOCAL_ROMS/walkgrid.json" "$E1_VM:roms/walkgrid.json"
else
  say "WARNING: no $E1_LOCAL_ROMS/walkgrid.json; A2″ runs and evaluate.py's leg progress need it (walkgrid.py build)"
fi
gssh "echo '$E1_RED_SHA1  roms/pokemon_red.gb' | sha1sum -c -"

# --- venv on the VM ------------------------------------------------------------------------
gssh "E1_UV_VERSION=$E1_UV_VERSION E1_PYTHON=$E1_PYTHON bash ~/e1/code/gcp/remote/setup.sh"

say "ready. next: gcp/bench.sh (1 h), then gcp/run.sh gcp/plans/pilot.txt; gcp/fetch.sh; gcp/teardown.sh"
