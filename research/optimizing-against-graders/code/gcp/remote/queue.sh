#!/usr/bin/env bash
# On the VM: run a queue of training runs ONE AT A TIME, each followed by its eval, with no laptop needed.
#
#   queue.sh start QUEUE_FILE     install the queue (~/e1/queue/plan.txt) and run it in tmux window e1:queue
#   queue.sh --boot               called by the startup script on every boot: if a queue is active, resume it
#   queue.sh status               one line per queue entry: done / partial / new, eval done or not
#   queue.sh pause REASON         (§9 H4′) no new queue entry starts here AND on the peer VM (the running
#                                 entry finishes); used by b2_reads.py on a tripwire. --no-peer: here only
#   queue.sh resume               clear the pause here and restart the worker (each VM resumes separately)
#   queue.sh pause-from-peer      forced command of the peer's restricted ssh key (~/.ssh/authorized_keys)
#
# A queue line is train.py arguments (must include --run-id), optionally followed by
#   " ; eval: <evaluate.py arguments after the run id>"
# e.g.   --arm A0 --state D1 --seed 1 --steps {M} --run-id A0-D1-s1 --envs {envs} ; eval: --states D1 HA HB --episodes 8 --window-stats --save-traces
# and optionally " ; watch: <b2_reads.py --reads args>" (all | none | 0 1M 2.5M; default none = summaries only):
# every trained entry gets a b2_reads.py watcher next to train.py (per-mark summary-marks.json; interim evals
# and tripwires on the reads it names, which pause both queues on a trip). Segments may come in any order.
# {envs} -> $E1_ENVS (default nproc), {M} -> $E1_M (saved with the queue, so a reboot uses the same value).
# The eval counts as done when <run>/<out>/summary.json exists (<out> = the eval's --out, default eval), so a
# second queue over the same (done) runs with  ; eval: ... --out eval-d2  adds an eval without retraining.
# " ; eval2: ..." is a second closing eval after the first (batch 2: --states HA HB --out ~/roms/sealed-b2/RUN/...,
# a path outside the runs tree, so held-out output is sealed from the start); an --out that starts with / or ~/
# is used as is.
# An eval-only line runs evaluate.py and nothing else (e.g. R0, which is never trained):
#   eval-only: R0 --random A0-D1-s0 --states D2 --episodes 24 --window-stats --save-traces --out eval-24-d2
#
# Restart rules (experiment-e1.md §8 resume rule): a run with a config but short of its steps is resumed
# (train.py --resume, newest checkpoint); a done run is skipped; an eval whose summary.json is missing is
# rerun from scratch (evaluate.py overwrites its output dir). A train.py that exits non-zero while the
# machine is NOT shutting down is retried twice, then marked failed and skipped (~/e1/queue/failed).
# During a shutdown the queue just stops; the next boot picks it up where it was.
set -uo pipefail
Q=~/e1/queue
R=~/e1/code/gcp/remote/runs.py
export POKEMON_RED_ROM=~/roms/pokemon_red.gb
mkdir -p "$Q" ~/roms/runs/logs

log() { echo "$(date -Is) $*" | tee -a "$Q/queue.log"; }
stopping() { case $(systemctl is-system-running 2>/dev/null) in stopping) return 0 ;; esac; return 1; }

expand() {  # expand LINE -> train args with {envs}/{M} filled in
  local l=$1
  l=${l//\{envs\}/${E1_ENVS:-$(nproc)}}
  l=${l//\{M\}/${E1_M:-}}
  echo "$l"
}

entries() {  # print non-empty, comment-stripped queue lines
  while IFS= read -r line || [ -n "$line" ]; do
    line=${line%%#*}                                   # trim with bash, not xargs: xargs strips quotes (the
    line=${line#"${line%%[![:space:]]*}"}; line=${line%"${line##*[![:space:]]}"}   # r1 --proxy JSON, 2026-11-10)
    [ -n "$line" ] && echo "$line"
  done < "$Q/plan.txt"
}

run_id_of() { sed -n 's/.*--run-id[ =]\([^ ]*\).*/\1/p' <<<"$1"; }
seg() {  # seg LINE NAME -> the text of " ; NAME: ..." up to the next " ; " (empty if absent)
  local l=$1 n=$2; [[ $l == *" ; $n:"* ]] || return 0; l=${l#* ; $n:}; echo "${l%% ; *}"; }
paused() { [ -f "$Q/paused" ]; }
out_of() { local o; o=$(sed -n 's/.*--out[ =]\([^ ]*\).*/\1/p' <<<"$1"); echo "${o:-eval}"; }  # evaluate.py --out
outdir() {  # outdir RUN OUT -> the eval's output dir: <run>/<out>, or OUT itself if absolute or ~/ (sealed paths)
  case "$2" in /*) echo "$2" ;; "~/"*) echo "$HOME/${2#\~/}" ;; *) echo "$HOME/roms/runs/$1/$2" ;; esac; }
logname() { basename "$1"; }

worker() {  # the loop itself; runs inside tmux window e1:queue
  # shellcheck disable=SC1091
  [ -f "$Q/env" ] && . "$Q/env"
  cd ~/e1/code || exit 1
  log "queue start (M=${E1_M:-} envs=${E1_ENVS:-$(nproc)} code $(cat CODE_SHA256 2>/dev/null))"
  entries | while IFS= read -r line; do
    if paused; then log "PAUSED: $(cat "$Q/paused"); no new entry starts (queue.sh resume)"; exit 0; fi
    if [[ $line == eval-only:* ]]; then
      ev=${line#eval-only:}; id=$(awk '{print $1}' <<<"$ev"); out=$(out_of "$ev")
      if [ ! -f ~/roms/runs/$id/$out/summary.json ]; then
        log "$id: evaluate.py$ev"
        PYTHONUNBUFFERED=1 .venv/bin/python evaluate.py $ev </dev/null >> ~/roms/runs/logs/$id-$out.log 2>&1
        rc=$?
        if stopping; then log "$id: shutting down during eval; next boot reruns it"; exit 0; fi
        [ $rc = 0 ] || log "$id: evaluate.py rc $rc (left for a manual rerun)"
      fi
      log "$id: $out done"; continue
    fi
    train=$(expand "${line%% ; *}")
    ev=$(seg "$line" eval); wr=$(seg "$line" watch); wr=${wr:-none}
    id=$(run_id_of "$train")
    [ -n "$id" ] || { log "skip (no --run-id): $line"; continue; }
    grep -qx "$id" "$Q/failed" 2>/dev/null && { log "$id: failed earlier, skipped"; continue; }
    [ "$(python3 "$R" state "$id")" = eval-only ] && { log "$id: eval-only baseline dir (random_policy), not trained; skipped"; continue; }
    wpid=""
    if [ "$(python3 "$R" state "$id")" != done ]; then
      rm -f ~/roms/runs/$id/.train_done
      PYTHONUNBUFFERED=1 .venv/bin/python b2_reads.py watch "$id" --reads $wr </dev/null >> ~/roms/runs/logs/$id-watch.log 2>&1 &
      wpid=$!; log "$id: watcher pid $wpid (reads: $wr)"
    fi
    endwatch() { [ -n "$wpid" ] || return 0; mkdir -p ~/roms/runs/$id; touch ~/roms/runs/$id/.train_done; wait "$wpid"; wpid=""; }
    tries=0
    while [ "$(python3 "$R" state "$id")" != done ]; do
      if [ "$(python3 "$R" state "$id")" = partial ]; then args="--resume $id"; else args=$train; fi
      log "$id: train.py $args"
      echo "code $(cat CODE_SHA256)  $(date -Is)  queue" >> ~/roms/runs/logs/$id.log
      PYTHONUNBUFFERED=1 .venv/bin/python train.py $args </dev/null >> ~/roms/runs/logs/$id.log 2>&1
      rc=$?
      if stopping; then log "$id: machine shutting down (rc $rc); queue stops, next boot resumes"; exit 0; fi
      [ "$(python3 "$R" state "$id")" = done ] && break
      tries=$((tries + 1)); log "$id: train.py exited rc $rc before its steps (try $tries)"
      if [ $tries -ge 3 ]; then echo "$id" >> "$Q/failed"; log "$id: FAILED, skipped"; endwatch; continue 2; fi
      sleep 30
    done
    endwatch                     # the watcher's last interim read + summary finish before the closing eval
    for ev in "$ev" "$(seg "$line" eval2)"; do   # eval2: a second closing eval (batch 2: HA HB -> a sealed path)
      [ -n "$ev" ] || continue
      out=$(out_of "$ev")
      [ -f "$(outdir "$id" "$out")/summary.json" ] && continue
      log "$id: evaluate.py $id $ev"
      PYTHONUNBUFFERED=1 .venv/bin/python evaluate.py "$id" $ev </dev/null >> ~/roms/runs/logs/$id-$(logname "$out").log 2>&1
      rc=$?
      if stopping; then log "$id: shutting down during eval; next boot reruns it"; exit 0; fi
      [ $rc = 0 ] || log "$id: evaluate.py rc $rc (left for a manual rerun)"
    done
    log "$id: done"
  done
  stopping && exit 0
  if paused; then log "queue paused, still active; queue.sh resume continues it"; exit 0; fi
  log "queue finished"
  mv "$Q/active" "$Q/finished.$(date +%Y%m%dT%H%M%S)" 2>/dev/null
  # nothing runs now: the idle guard powers the VM off after e1-idle-minutes
}

launch() {  # start the worker in tmux window e1:queue (no-op if it is already running)
  tmux has-session -t e1 2>/dev/null || tmux new -d -s e1 -n shell
  tmux set -wg remain-on-exit on
  if tmux list-windows -t e1 -F '#{window_name} #{pane_dead}' | grep -qx "queue 0"; then
    echo "queue already running"; return; fi
  tmux kill-window -t e1:queue 2>/dev/null || true
  tmux new-window -d -t e1 -n queue "bash ~/e1/code/gcp/remote/queue.sh worker" \
    || { log "could not start the tmux window"; exit 1; }
  echo "queue running in tmux window e1:queue; log $Q/queue.log"
}

case "${1:-}" in
  start)
    [ -f "${2:-}" ] || { echo "usage: queue.sh start QUEUE_FILE" >&2; exit 2; }
    [ -f "$Q/active" ] && tmux list-windows -t e1 -F '#{window_name} #{pane_dead}' 2>/dev/null | grep -qx "queue 0" \
      && { echo "a queue is already running; wait for it or stop it first" >&2; exit 1; }
    cp "$2" "$Q/plan.txt"; rm -f "$Q/failed"
    printf 'E1_M=%q\nE1_ENVS=%q\n' "${E1_M:-}" "${E1_ENVS:-$(nproc)}" > "$Q/env"
    grep -q '{M}' "$Q/plan.txt" && [ -z "${E1_M:-}" ] && { echo "queue has {M}; set E1_M" >&2; exit 2; }
    date -Is > "$Q/active"
    launch ;;
  --boot)
    [ -f "$Q/active" ] || exit 0
    log "boot: active queue found, resuming"
    launch ;;
  worker) worker ;;
  pause)
    shift; peer=1; [ "${1:-}" = --no-peer ] && { peer=0; shift; }
    reason=$(printf '%s' "$*" | tr -cd '[:print:]' | cut -c1-300)
    if ! paused; then echo "$(date -Is) $reason" > "$Q/paused"; log "PAUSE: $reason"; fi
    if [ $peer = 1 ] && [ -f "$Q/peer" ]; then       # §9 H4′: both queues pause (restricted key, forced command)
      ssh -i ~/.ssh/e1_peer -o BatchMode=yes -o ConnectTimeout=15 -o StrictHostKeyChecking=accept-new \
        "$(cat "$Q/peer")" "$(hostname): $reason" </dev/null && log "PAUSE sent to peer $(cat "$Q/peer")" \
        || log "PAUSE: could not reach peer $(cat "$Q/peer") (the laptop must pause it: gcp/pause.sh)"
    fi ;;
  pause-from-peer)
    bash "$0" pause --no-peer "from peer: ${SSH_ORIGINAL_COMMAND:-?}" ;;
  resume)
    paused || { echo "not paused"; exit 0; }
    log "RESUME (was: $(cat "$Q/paused"))"; mv "$Q/paused" "$Q/paused.cleared.$(date +%Y%m%dT%H%M%S)"
    if [ -f "$Q/active" ]; then launch; fi ;;
  status)
    [ -f "$Q/plan.txt" ] || { echo "no queue"; exit 0; }
    # shellcheck disable=SC1091
    [ -f "$Q/env" ] && . "$Q/env"
    echo "queue: $([ -f "$Q/active" ] && echo "active since $(cat "$Q/active")" || echo inactive)  M=${E1_M:-}"
    paused && echo "PAUSED: $(cat "$Q/paused")"
    entries | while IFS= read -r line; do
      if [[ $line == eval-only:* ]]; then
        ev=${line#eval-only:}; id=$(awk '{print $1}' <<<"$ev"); out=$(out_of "$ev")
        e="$out pending"; [ -f ~/roms/runs/$id/$out/summary.json ] && e="$out done"
        printf '  %-24s %-8s %s\n' "$id" "eval-only" "$e"; continue
      fi
      id=$(run_id_of "$(expand "${line%% ; *}")")
      out=eval; [[ $line == *" ; eval:"* ]] && out=$(out_of "$(seg "$line" eval)")
      e=-; [[ $line == *" ; eval:"* ]] && { e="$out pending"; [ -f ~/roms/runs/$id/$out/summary.json ] && e="$out done"; }
      grep -qx "$id" "$Q/failed" 2>/dev/null && e="FAILED"
      printf '  %-24s %-8s %s\n' "$id" "$(python3 "$R" state "$id")" "$e"
    done
    tail -n 3 "$Q/queue.log" 2>/dev/null ;;
  *) sed -n '2,12p' "$0"; exit 2 ;;
esac
