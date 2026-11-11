#!/bin/bash
# GCE startup script (runs as root on every boot): the in-guest cost guards for the E1 VM, and the
# in-VM resume. All guards only STOP the VM; the disk and ~/roms/runs survive until gcp/teardown.sh.
#   1. per-boot window: power off after e1-guard-hours of uptime (sized by gcp/arm.sh to the queue + margin)
#   2. lifetime budget: power off once this VM's CUMULATIVE uptime over all boots reaches e1-budget-hours
#      (counted every minute into /var/lib/e1/uptime-s, which survives reboots and restarts)
#   3. idle guard: power off after e1-idle-minutes with no train.py / evaluate.py / bench process
#   4. resume: if a queue is active (~/e1/queue/active, gcp/remote/queue.sh), restart it as e1-user, so a
#      host-error restart or a manual start carries on with no laptop involved
# The GCP-side cap (--termination-time, set by gcp/arm.sh / launch.sh) does not depend on any of this.
# (bracketed pgrep patterns so this guard never matches its own command line.)
md() { curl -sf -H 'Metadata-Flavor: Google' \
  "http://metadata.google.internal/computeMetadata/v1/instance/attributes/$1"; }
H=$(md e1-guard-hours);   case "$H" in ''|*[!0-9]*) H=12 ;; esac
B=$(md e1-budget-hours);  case "$B" in ''|*[!0-9]*) B=0 ;; esac       # 0 = no lifetime budget set
I=$(md e1-idle-minutes);  case "$I" in ''|*[!0-9]*) I=45 ;; esac
U=$(md e1-user);          case "$U" in ''|*[!a-z0-9_-]*) U=jake ;; esac
mkdir -p /var/lib/e1
L=/var/lib/e1/uptime-s
[ -f $L ] || echo 0 > $L
used=$(cat $L); case "$used" in ''|*[!0-9]*) used=0 ;; esac
logger "e1 guard: boot; window ${H} h, lifetime ${used}s used of ${B} h budget, idle ${I} min"
if [ "$B" -gt 0 ] && [ "$used" -ge $((B * 3600)) ]; then
  logger "e1 guard: lifetime budget ${B} h already used; powering off"; poweroff; exit 0
fi
shutdown -h "+$((H * 60))" "e1 cost guard: ${H} h uptime"
systemctl stop e1-idle-guard 2>/dev/null || true
systemd-run --unit=e1-idle-guard --description="e1 idle + lifetime guard" /bin/bash -c "
  idle=0
  while sleep 60; do
    u=\$(( \$(cat $L 2>/dev/null || echo 0) + 60 )); echo \$u > $L.tmp && mv $L.tmp $L
    if [ $B -gt 0 ] && [ \$u -ge $((B * 3600)) ]; then
      logger 'e1 guard: lifetime budget ${B} h reached, powering off'; poweroff; fi
    if pgrep -f '[t]rain.py|[e]valuate.py|[b]ench_lean.py|remote/[b]ench.sh|[p]ip install' >/dev/null; then idle=0
    else idle=\$((idle + 1)); fi
    if [ \$idle -ge $I ]; then logger 'e1 idle guard: ${I} min idle, powering off'; poweroff; fi
  done"
# 4. resume an active queue (after the network is up; the user's tmux is independent of this script)
if [ -f "/home/$U/e1/queue/active" ]; then
  logger "e1 guard: active queue for $U, resuming"
  su - "$U" -c 'bash ~/e1/code/gcp/remote/queue.sh --boot' || logger "e1 guard: queue resume failed"
fi
