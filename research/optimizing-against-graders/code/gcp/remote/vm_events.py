#!/usr/bin/env python3
"""On the VM: append GCP stop events (JSON lines on stdin) to ~/roms/runs/vm-events.jsonl, skipping
operation ids already there. Fed by record_vm_events in gcp/config.sh; read by pokered/runlog.py."""
import json
import os
import sys

path = os.path.expanduser("~/roms/runs/vm-events.jsonl")
os.makedirs(os.path.dirname(path), exist_ok=True)
seen = set()
if os.path.exists(path):
    for line in open(path):
        try:
            seen.add(json.loads(line)["id"])
        except (ValueError, KeyError):
            pass
with open(path, "a") as f:
    for line in sys.stdin:
        try:
            e = json.loads(line)
        except ValueError:
            continue
        if e.get("id") not in seen:
            f.write(json.dumps(e) + "\n")
            seen.add(e["id"])
