# Per-workload merge for round r12

Projected portal us per workload (rented change in the same container in brackets; median of 4 passes each). A candidate's rented time is scaled by the portal/rented ratio its own lineage shows in r11-merge-b's portal result; a workload moves when the projection beats the best's portal time by 3% (10% for a lineage with no portal result yet, marked new).

| workload | r11-merge-b portal us (rented) | r12-d-e1-h48-gate-parent | r12-d-e2-grn-epi-counters | r12-i-e3-tc05-pass1-547 | r12-l-e4-toeplitz-gate | routed to |
|---|---|---|---|---|---|---|
| 1,192,28,28 | 23.5 (18.2) | 23.5 (+0.2%) | 114.3 (+386.4%) | 114.5 (+387.3%) new | 114.7 (+388.0%) new | r11-merge-b |
| 1,96,56,56 | 23.7 (18.4) | 23.5 (-0.7%) | 83.3 (+251.6%) | 83.2 (+250.9%) new | 83.2 (+251.0%) new | r11-merge-b |
| 2,768,7,7 | 28.3 (22.9) | 28.7 (+1.4%) | 29.9 (+5.7%) | 153.7 (+443.2%) new | 153.5 (+442.5%) new | r11-merge-b |
| 2,96,56,56 | 32.8 (25.1) | 32.7 (-0.2%) | 131.4 (+300.8%) | 131.6 (+301.2%) new | 131.5 (+300.8%) new | r11-merge-b |
| 1,160,449,7 | 33.3 (25.8) | 32.7 (-1.7%) | 141.8 (+325.7%) | 142.3 (+327.4%) new | 142.1 (+326.8%) new | r11-merge-b |
| 4,192,28,28 | 37.3 (29.6) | 36.3 (-2.6%) | 175.7 (+371.1%) | 175.8 (+371.3%) new | 175.9 (+371.5%) new | r11-merge-b |
| 8,768,7,7 | 38.6 (30.7) | 38.4 (-0.5%) | 38.1 (-1.3%) | 331.9 (+759.8%) new | 331.9 (+759.9%) new | r11-merge-b |
| 8,384,14,14 | 44.2 (34.0) | 45.2 (+2.2%) | 43.4 (-1.8%) | 354.1 (+701.0%) new | 354.2 (+701.4%) new | r11-merge-b |
| 4,96,56,56 | 47.0 (37.1) | 48.1 (+2.1%) | 184.9 (+284.5%) | 180.3 (+283.7%) new | 180.7 (+284.5%) new | r11-merge-b |
| 16,768,7,7 | 51.9 (40.8) | 52.8 (+1.8%) | 50.4 (-2.9%) | 750.6 (+1346.3%) new | 750.3 (+1345.6%) new | r11-merge-b |
| 8,192,28,28 | 54.2 (42.7) | 53.8 (-0.8%) | 276.5 (+410.1%) | 276.5 (+410.1%) new | 276.4 (+409.9%) new | r11-merge-b |
| 16,384,14,14 | 64.5 (50.5) | 64.4 (-0.2%) | 64.7 (+0.3%) | 449.1 (+596.3%) new | 448.8 (+595.8%) new | r11-merge-b |
| 8,320,32,32 | 74.9 (58.5) | 806.6 (+963.6%) | 806.5 (+963.6%) | 797.1 (+964.2%) new | 796.4 (+963.3%) new | r11-merge-b |
| 32,768,7,7 | 77.4 (59.7) | 77.6 (+0.3%) | 71.4 (-7.8%) | 1576.4 (+1936.7%) new | 1578.1 (+1938.9%) new | r12-d-e2-grn-epi-counters |
| 32,128,64,64 | 288.3 (226.5) | 2137.6 (+627.6%) | 2137.7 (+627.6%) | 2097.6 (+627.6%) new | 2100.9 (+628.7%) new | r11-merge-b |
| 64,96,112,112 | 1077.0 (862.1) | 7238.4 (+547.3%) | 7240.0 (+547.4%) | 6972.5 (+547.4%) new | 6972.9 (+547.4%) new | r11-merge-b |

Merged kernel `problems/035_convnextv2_block_with_grn/rounds/r12/candidates/r12-merge.json`: 16/16 workloads passed its test (timing flakes count as failures here; see the trace).
Projected portal score 0.7415 against the best's 0.7404 (#73053).
