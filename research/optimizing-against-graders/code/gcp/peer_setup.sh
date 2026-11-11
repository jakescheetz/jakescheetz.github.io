#!/usr/bin/env bash
# §9 H4′: let a tripwire on one VM pause the other VM's queue without the laptop.
#   gcp/peer_setup.sh            (both VMs running) per VM: an ed25519 key ~/.ssh/e1_peer; the PEER's
#                                ~/.ssh/authorized_keys accepts it for one forced command only
#                                (queue.sh pause-from-peer: no shell, pty, forwarding or agent); ~/e1/queue/peer
#                                holds the peer's internal name. Then a test: pause e1-a -> e1-b paused, resume both.
# Internal network only (default-allow-internal); nothing leaves the project; no new accounts.
set -euo pipefail
. "$(dirname "$0")/config.sh"
FC='command="bash ~/e1/code/gcp/remote/queue.sh pause-from-peer",no-pty,no-port-forwarding,no-agent-forwarding,no-X11-forwarding'
on() { E1_VM=$1 gssh "${@:2}"; }
for v in e1-a e1-b; do
  on "$v" "mkdir -p ~/.ssh ~/e1/queue && chmod 700 ~/.ssh && [ -f ~/.ssh/e1_peer ] || ssh-keygen -q -t ed25519 -N '' -C e1-peer-$v -f ~/.ssh/e1_peer"
done
for pair in "e1-a e1-b" "e1-b e1-a"; do
  set -- $pair; src=$1 dst=$2
  pub=$(on "$src" "cat ~/.ssh/e1_peer.pub" 2>/dev/null | tail -1)
  on "$dst" "grep -qF '${pub}' ~/.ssh/authorized_keys 2>/dev/null || echo '$FC ${pub}' >> ~/.ssh/authorized_keys"
  on "$src" "echo \$(whoami)@$dst > ~/e1/queue/peer"
done
say "test: pause on e1-a must reach e1-b"
on e1-a "bash ~/e1/code/gcp/remote/queue.sh pause 'peer_setup test'"
on e1-b "cat ~/e1/queue/paused" | tail -1
for v in e1-a e1-b; do on "$v" "bash ~/e1/code/gcp/remote/queue.sh resume >/dev/null; ls ~/e1/queue/paused 2>/dev/null || echo '$v: not paused'"; done
