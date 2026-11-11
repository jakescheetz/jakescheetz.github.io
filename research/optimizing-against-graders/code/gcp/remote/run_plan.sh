#!/usr/bin/env bash
# On the VM: start each run of a plan file in its own window of tmux session "e1".
#   run_plan.sh PLAN_FILE        new runs start, partial runs resume, done runs are skipped
#   run_plan.sh --resume-all     resume every partial run (after a spot preemption or cost-guard stop)
# A plan line is train.py arguments and must include --run-id; "{envs}" becomes $E1_ENVS (default nproc-2).
set -uo pipefail
cd ~/e1/code
export POKEMON_RED_ROM=~/roms/pokemon_red.gb
mkdir -p ~/roms/runs/logs
ENVS=${E1_ENVS:-$(( $(nproc) - 2 ))}
R=~/e1/code/gcp/remote/runs.py

tmux has-session -t e1 2>/dev/null || tmux new -d -s e1 -n shell
# keep a finished or crashed trainer's window (pane_dead=1) so status checks can see it; a window option in tmux 3.x
tmux set -wg remain-on-exit on
alive() { tmux list-windows -t e1 -F '#{window_name} #{pane_dead}' | grep -qx "$1 0"; }
start() {  # start RUN_ID ARGS...
  local id=$1; shift
  if alive "$id"; then echo "  $id: already running"; return; fi
  tmux kill-window -t "e1:$id" 2>/dev/null || true
  # tmux windows get the tmux server's environment, not this script's: pass the ROM path explicitly (a
  # server started from a plain ssh shell has no POKEMON_RED_ROM, and train.py exits at once without it)
  local cmd; cmd="export POKEMON_RED_ROM=$POKEMON_RED_ROM; cd ~/e1/code && echo \"code \$(cat CODE_SHA256)  \$(date -Is)\" >> ~/roms/runs/logs/$id.log && \
PYTHONUNBUFFERED=1 .venv/bin/python train.py $* 2>&1 | tee -a ~/roms/runs/logs/$id.log"
  tmux new-window -d -t e1 -n "$id" "$cmd"
  echo "  $id: started: train.py $*"
}

if [ "${1:-}" = --resume-all ]; then
  ids=$(python3 "$R" partial)
  [ -n "$ids" ] || { echo "no partial runs"; exit 0; }
  for id in $ids; do start "$id" --resume "$id"; done
  exit 0
fi

PLAN=$1; total=0
while IFS= read -r line || [ -n "$line" ]; do
  line=${line%%#*}; line=$(echo "$line" | xargs)
  [ -z "$line" ] && continue
  line=${line//\{envs\}/$ENVS}
  id=$(sed -n 's/.*--run-id[ =]\([^ ]*\).*/\1/p' <<<"$line")
  [ -n "$id" ] || { echo "  skipped (no --run-id): $line"; continue; }
  e=$(sed -n 's/.*--envs[ =]\([0-9]*\).*/\1/p' <<<"$line"); total=$(( total + ${e:-3} ))
  case $(python3 "$R" state "$id") in
    new) start "$id" $line ;;
    partial) start "$id" --resume "$id" ;;
    done) echo "  $id: done, skipped" ;;
  esac
done < "$PLAN"
[ "$total" -le "$(nproc)" ] || echo "WARNING: plan uses $total envs on $(nproc) vCPUs; runs will contend (and steps/s drops)"
