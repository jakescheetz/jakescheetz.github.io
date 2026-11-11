# §9 H11 checks (developer, 2026-11-10): dev only, no rule change

**Verdict:** (1) reproduces bit for bit, and (2) and (3) are clean. So the three parked 5M endings are a property of the learned weights, not of the host or of eval.

## (1) Re-evaluate b2-A2pp-D1-s2 at 5M with the same seeds

`evaluate.py b2-A2pp-D1-s2 --states D1 --episodes 8 --ckpts eval-5000192.pt eval-4000768.pt --window-stats --save-traces`
(seeds 1000–1007; one torch thread per episode, as in the closing eval). There were two re-runs:

| where | CPU | 5M: identical to the closing eval | 5M T / P | 5M never left the start map | 4M T / P (contrast) |
|---|---|---|---|---|---|
| laptop, `eval-h11-local/` | Intel i5-8210Y (AVX2) | 8 of 8 (P, T, final_pos, maps); trace steps byte-identical for the seeds diffed (1000, 1004) | 0.0 / 0.78 | 8 of 8 (map 0x28, Oak's Lab) | 3.0 / 40.8; 0 of 8 parked; 8 of 8 identical too |
| e1-a, booted 11-10 11:15, `eval-h11-vm/` | AMD EPYC 7B13 | 8 of 8 (P, T, final_pos); trace steps identical for seed 1003 (diffed) | 0.0 / 0.78 | 8 of 8 | – |

Only the trace header differs (no `--kl-init` in the re-runs). Every step matches.

## (2) The 5M eval checkpoint against the final training weights

I hashed the model state_dict (15 tensors: names, dtypes, shapes and bytes) in `eval-5000192.pt` and in the learner's last rolling `ckpt-5000192.pt`:

| run | host | eval-5000192 | ckpt-5000192 | identical |
|---|---|---|---|---|
| b2-A2pp-D1-s2 | new | 3cec5231855e5ad0 | 3cec5231855e5ad0 | yes |
| b2-A2pp-D2-s2 | new | 53b521f1f39946d5 | 53b521f1f39946d5 | yes |
| b2-A2pp-D1-s3 | new | d42418c4b988bd5e | d42418c4b988bd5e | yes |
| b2-A2pp-D2-s3 | new | 9385256a3f4ba96c | 9385256a3f4ba96c | yes |
| b2-A2pp-D1-s1 (control) | old | 8bc7289a27db6673 | 8bc7289a27db6673 | yes |
| b2-A2pp-D2-s1 (control) | old | 515e277dbc3c33c4 | 515e277dbc3c33c4 | yes |

## (3) Old host vs new host (e1-a, boot ea070d49 ending 11-08 17:05 vs boot 6a5458ec from 17:06)

- **Hardware:** both are AMD EPYC 7B13 (family 0x19, model 0x1, stepping 0x0), with TSC 2450 MHz, the same xstate features (0x7, AVX, 832-byte context) and KVM. Torch reports CPU capability AVX2 and 8 threads. No avx512 flags on either.
- **Software:** the same disk, so the same Debian 6.1.0-54-cloud-amd64 kernel. No dpkg installs, upgrades or removals since 11-07. No venv package newer than 11-07. Every run config records `versions` {python 3.14.8, torch 2.15.0+cpu, pyboy 2.9.0, x86_64}, `threads` 0, `train_threads` 8 and `envs` 8, the same before and after.
- **Code:** train.py, evaluate.py and pokered/*.py were last modified on 11-07 at 21:51 or earlier, before the main batch was armed (11-08 00:11). They are hash-identical to the laptop copy. The run configs' `code_sha256` changed (b441ab52 → 8a092352 → 7e5133f4) only through read and report tooling (b2_reads.py, gap_*.py, train_series.py, b1_report.py, tests, README). Every config diff against b2-A2pp-D1-s1, except per-run fields, is `code_sha256` alone. The reward hash is 1f739117e21a throughout.
