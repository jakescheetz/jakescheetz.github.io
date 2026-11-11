#!/usr/bin/env bash
# §9 H4′ from the laptop: pause (or resume) BOTH queues.
#   gcp/pause.sh "REASON"     queue.sh pause --no-peer REASON on e1-a and e1-b (running entries finish)
#   gcp/pause.sh --resume     queue.sh resume on both (only after the decision is logged in §9)
set -euo pipefail
. "$(dirname "$0")/config.sh"
for v in e1-a e1-b; do
  [ "$(E1_VM=$v vm_status)" = RUNNING ] || { say "$v not running (a stopped VM starts nothing)"; continue; }
  if [ "${1:-}" = --resume ]; then E1_VM=$v gssh "bash ~/e1/code/gcp/remote/queue.sh resume"
  else E1_VM=$v gssh "bash ~/e1/code/gcp/remote/queue.sh pause --no-peer 'laptop: ${1:-manual}'"; fi
done
