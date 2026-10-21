# Per-workload merge for round r11

Projected portal us per workload (rented change in the same container in brackets; median of 4 passes each). A candidate's rented time is scaled by the portal/rented ratio its own lineage shows in r10-merge's portal result; a workload moves when the projection beats the best's portal time by 3% (10% for a lineage with no portal result yet, marked new).

| workload | r10-merge portal us (rented) | r11-d-e1-g2-mp2-bulk | r11-d-e2-w1bulk | r11-j-e3-gate-killed | r11-k-e4-overlap-gate | routed to |
|---|---|---|---|---|---|---|
| 1,192,28,28 | 23.5 (18.3) | 23.5 (+0.2%) | 24.0 (+2.2%) | 114.7 (+388.1%) new | 114.4 (+386.8%) new | r10-merge |
| 1,96,56,56 | 23.7 (18.4) | 23.5 (-0.9%) | 23.5 (-0.9%) | 83.1 (+250.6%) new | 82.9 (+249.6%) new | r10-merge |
| 2,768,7,7 | 30.8 (24.2) | 29.7 (-3.7%) | 31.3 (+1.7%) | 158.1 (+413.3%) new | 158.4 (+414.3%) new | r11-d-e1-g2-mp2-bulk |
| 2,96,56,56 | 32.5 (25.1) | 32.5 (+0.1%) | 32.5 (+0.1%) | 130.1 (+300.4%) new | 130.2 (+300.6%) new | r10-merge |
| 1,160,449,7 | 33.4 (25.9) | 32.9 (-1.6%) | 32.9 (-1.5%) | 143.0 (+328.3%) new | 142.4 (+326.2%) new | r10-merge |
| 4,192,28,28 | 37.4 (29.7) | 36.4 (-2.7%) | 37.4 (+0.1%) | 175.6 (+369.4%) new | 175.5 (+369.2%) new | r10-merge |
| 8,768,7,7 | 38.7 (30.4) | 38.9 (+0.5%) | 38.7 (+0.1%) | 337.2 (+771.2%) new | 337.1 (+771.0%) new | r10-merge |
| 8,384,14,14 | 43.9 (34.2) | 44.6 (+1.5%) | 43.6 (-0.6%) | 349.0 (+695.1%) new | 349.3 (+695.6%) new | r10-merge |
| 4,96,56,56 | 46.8 (37.1) | 48.7 (+2.3%) | 48.7 (+2.2%) | 179.2 (+283.0%) new | 179.6 (+283.7%) new | r10-merge |
| 16,768,7,7 | 52.5 (40.8) | 53.4 (+1.7%) | 53.3 (+1.4%) | 757.5 (+1342.8%) new | 757.3 (+1342.5%) new | r10-merge |
| 8,192,28,28 | 54.1 (42.5) | 54.0 (-0.2%) | 54.5 (+0.8%) | 275.8 (+409.8%) new | 275.8 (+409.9%) new | r10-merge |
| 16,384,14,14 | 65.4 (51.9) | 63.5 (-2.9%) | 63.5 (-2.9%) | 444.2 (+579.2%) new | 443.7 (+578.4%) new | r10-merge |
| 8,320,32,32 | 74.6 (58.4) | 787.2 (+964.4%) | 788.8 (+966.6%) | 795.0 (+965.7%) new | 794.8 (+965.5%) new | r10-merge |
| 32,768,7,7 | 78.1 (61.6) | 75.3 (-3.6%) | 77.9 (-0.2%) | 1547.9 (+1881.9%) new | 1548.8 (+1883.1%) new | r11-d-e1-g2-mp2-bulk |
| 32,128,64,64 | 286.8 (225.8) | 2091.5 (+631.3%) | 2090.9 (+631.1%) | 2093.4 (+629.9%) new | 2097.3 (+631.3%) new | r10-merge |
| 64,96,112,112 | 1071.4 (858.2) | 7085.2 (+551.7%) | 7086.7 (+551.8%) | 6976.5 (+551.2%) new | 6970.7 (+550.6%) new | r10-merge |

Merged kernel `problems/035_convnextv2_block_with_grn/rounds/r11/candidates/r11-merge.json`: 16/16 workloads passed its test (timing flakes count as failures here; see the trace).
Projected portal score 0.7402 against the best's 0.7391 (#71901).

Operator follow-up: the 2% merge routed 16,384 and 4,192 on gains that were partly noise; rerun at 3% (r11-merge, 0.7402), then r11-merge-b = r11-merge with 16,384 on r10 D E1's tc2 path (r10-merge ran r8's tc_kernel there). ab_container vs r10-merge: projected 0.7408.
