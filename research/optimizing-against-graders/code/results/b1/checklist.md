# Batch-1 pipeline checklist (23 runs)

| check | result | detail |
|---|---|---|
| dev view exists for every run | PASS | 23/23 runs have eval-dev rows |
| both dev states (D1, D2) evaluated | PASS | all runs have D1 and D2 |
| every (run, mark, eval state) cell has n = 8 | PASS | cell sizes {8: 322} |
| 7 marks per run, including 3M | PASS | all runs: 7 marks, e.g. [250880, 500736, 1000448, 2000896, 3000320, 4000768, 5000192] |
| A0/A2″ paired by seed × training state (same VM) | PASS | 10 pairs; unpaired none; pairs on one VM: True |
| leg progress present in every dev row | PASS | 2576/2576 rows |
| T in eval rows matches milestone_steps | PASS | 0 mismatches |
| VM name in every row | PASS | 2576/2576 rows |
| runlog.load_kept applied to training episodes (resume markers / discarded branches dropped) | PASS | A0-D1-s1: 304 rows -> 304 kept, 0 resumes; A2pp-D1-s1: 304 rows -> 304 kept, 0 resumes; A0-D2-s1: 304 rows -> 304 kept, 0 resumes; A2pp-D2-s1: 304 rows -> 304 kept, 0 resumes; A0-D1-s2: 304 rows -> 304 kept, 0 resumes; A2pp-D1-s2: 304 rows -> 304 kept, 0 resumes; A0-D2-s2: 304 rows -> 304 kept, 0 resumes; A2pp-D2-s2: 304 rows -> 304 kept, 0 resumes; A0-D1-s3: 304 rows -> 304 kept, 0 resumes; A2pp-D1-s3: 304 rows -> 304 kept, 0 resumes; A0-D2-s3: 304 rows -> 304 kept, 0 resumes; A2pp-D2-s3: 304 rows -> 304 kept, 0 resumes; A0-D1-s4: 304 rows -> 304 kept, 0 resumes; A2pp-D1-s4: 304 rows -> 304 kept, 0 resumes; A0-D2-s4: 304 rows -> 304 kept, 0 resumes; A2pp-D2-s4: 304 rows -> 304 kept, 0 resumes; A0-D1-s5: 304 rows -> 304 kept, 0 resumes; A2pp-D1-s5: 304 rows -> 304 kept, 0 resumes; A0-D2-s5: 304 rows -> 304 kept, 0 resumes; A2pp-D2-s5: 304 rows -> 304 kept, 0 resumes; A2pp-HA-s1: 304 rows -> 304 kept, 0 resumes; A2pp-HA-s2: 304 rows -> 304 kept, 0 resumes; A2pp-HA-s3: 304 rows -> 304 kept, 0 resumes |
| KL vs init logged (D2-trained runs on D2; D1-trained need the D1 KL pass) | PASS | 1120 rows with kl_init; 5 distinct init hashes |
| KL init rebuilt with each run's training thread count (e1-a 8, e1-b 4), one hash per run | PASS | runs to leave out of the KL panel: none |
| R0 dev base rates (24 episodes) for D1 and D2 | PASS | R0 dev states ['D1', 'D2'] |
| attribution labels come out (primary) | PASS | proxy_attributed: 0 types; learner_artifact: 0 types; undetermined: 0 types; undetermined_reference_collapsed: 0 types |
| attribution labels come out (cross, descriptive) | PASS | proxy_attributed: 0 types; learner_artifact: 0 types; undetermined: 0 types; undetermined_reference_collapsed: 0 types |
| Holm family = detector × arm over the 14 primary dev cells (§9 A′); cross cells a separate family | PASS | A0 primary family sizes [14]; A2″ [14] (14 minus 0 cells with every A2″ seed collapsed); cross family sizes [14] |
| collapsed A2″ seed-ckpts dropped from the A2″ reference in attribution and A2″-vs-R0, every cell of that checkpoint (§9 (1)–(2), B′ (b)) | PASS | 16 seed-ckpts dropped (A2pp-D1-s1@5,000,192, A2pp-D1-s2@3,000,320, A2pp-D1-s2@4,000,768, A2pp-D1-s3@2,000,896, A2pp-D1-s3@3,000,320, A2pp-D1-s3@4,000,768, A2pp-D1-s4@3,000,320, A2pp-D2-s1@5,000,192, A2pp-D2-s2@3,000,320, A2pp-D2-s2@4,000,768, A2pp-D2-s2@5,000,192, A2pp-D2-s3@5,000,192, A2pp-D2-s4@3,000,320, A2pp-D2-s4@4,000,768, A2pp-D2-s5@4,000,768, A2pp-D2-s5@5,000,192); reference seeds kept per primary cell [(1, 1), (2, 2), (3, 2), (4, 2), (5, 7)] (seeds: cells); cells with none left: primary 0, cross 0 (labelled 'undetermined (reference collapsed)' where A0 is present); kept counts match the dev rows: True |
| U decile over dev windows only (D1 + D2 per checkpoint, §9 A′) | PASS | U pools in dev rows: [('D1', 'D2')] |
| O on HB excluded from the family | PASS | HB is held-out: not in these dev tables (r0_compare DESCRIPTIVE = {(O, HB)} applies in the sealed tables) |
| Fig. 1 (§9 C): P, T, leg progress vs step and vs KL, A0 + A2″, per training state | PASS | panels ['D1→D1', 'D2→D2']; 28 pooled points; collapsed A2″ seed-ckpts 16 (A2pp-D1-s1@5,000,192, A2pp-D1-s2@3,000,320, A2pp-D1-s2@4,000,768, A2pp-D1-s3@2,000,896, A2pp-D1-s3@3,000,320, A2pp-D1-s3@4,000,768, A2pp-D1-s4@3,000,320, A2pp-D2-s1@5,000,192, A2pp-D2-s2@3,000,320, A2pp-D2-s2@4,000,768, A2pp-D2-s2@5,000,192, A2pp-D2-s3@5,000,192, A2pp-D2-s4@3,000,320, A2pp-D2-s4@4,000,768, A2pp-D2-s5@4,000,768, A2pp-D2-s5@5,000,192); first-appearance ticks pooled 0, per seed 7 |
| H0 (i)–(iii) per training state (§9 D), MC permutation B = 100k, Holm over 4 marks ≥ 2M | PASS | states ['D1', 'D2']; marks ≥ 2M [2000896, 3000320, 4000768, 5000192]; verdicts computed (values in h0.md) |
| catalogue fields (§9 E): prevalence + share of P per type × mark, first appearance pooled + per seed | PASS | 560 A0 own-state episodes; traces missing 0; trace ΣP ≠ P 0; per-seed first appearance from 7 (run, type) hits |
| sealed part (HA/HB): held-out Fig. 1 panels, vs-R0 families, catalogue column, under results/sealed-b1/ | PASS | 20 runs, 2240 held-out rows, 4 panels, traces missing 0; files vs-R0-A0-heldout.md, vs-R0-A2pp-heldout.md, fig1-heldout.{svg,md,json}, catalogue-heldout.{md,json} in results/sealed-b1/b1 (counts only) |
