# Per-workload merge for round r10

Projected portal us per workload (rented change in the same container in brackets; median of 4 passes each). A candidate's rented time is scaled by the portal/rented ratio its own lineage shows in r9-merge's portal result; a workload moves when the projection beats the best's portal time by 2% (10% for a lineage with no portal result yet, marked new).

| workload | r9-merge portal us (rented) | r10-d-e1-balanced-nostack | r10-g-e2-g2-coop | r10-i-e3-gate-unrun | routed to |
|---|---|---|---|---|---|
| 1,192,28,28 | 23.3 (18.7) | 22.7 (-2.7%) | 114.5 (+376.1%) | 110.3 (+373.5%) new | r10-d-e1-balanced-nostack |
| 1,96,56,56 | 23.8 (18.4) | 23.6 (-1.0%) | 48.0 (+106.6%) | 83.6 (+251.1%) new | r9-merge |
| 2,768,7,7 | 30.7 (24.0) | 30.7 (-0.1%) | 158.8 (+417.8%) | 157.8 (+414.1%) new | r9-merge |
| 1,160,449,7 | 33.2 (25.8) | 32.6 (-1.7%) | 142.2 (+328.8%) | 141.5 (+326.4%) new | r9-merge |
| 2,96,56,56 | 33.3 (25.6) | 32.5 (-2.4%) | 48.7 (+50.7%) | 130.0 (+290.4%) new | r10-d-e1-balanced-nostack |
| 4,192,28,28 | 37.1 (29.5) | 36.3 (-2.1%) | 177.9 (+371.9%) | 174.8 (+371.1%) new | r10-d-e1-balanced-nostack |
| 8,768,7,7 | 42.0 (33.0) | 38.7 (-7.8%) | 334.5 (+693.6%) | 333.7 (+694.5%) new | r10-d-e1-balanced-nostack |
| 8,384,14,14 | 44.4 (33.6) | 44.5 (+0.3%) | 345.4 (+704.7%) | 357.7 (+705.7%) new | r9-merge |
| 4,96,56,56 | 46.7 (37.0) | 49.8 (+1.9%) | 50.1 (+7.2%) | 179.6 (+284.5%) new | r9-merge |
| 8,192,28,28 | 54.7 (42.5) | 54.3 (-0.7%) | 276.7 (+409.8%) | 278.9 (+409.9%) new | r9-merge |
| 16,768,7,7 | 56.7 (43.6) | 52.8 (-7.0%) | 740.9 (+1246.8%) | 763.1 (+1245.9%) new | r10-d-e1-balanced-nostack |
| 16,384,14,14 | 67.5 (51.4) | 66.1 (-2.0%) | 447.4 (+581.7%) | 459.6 (+580.9%) new | r10-d-e1-balanced-nostack |
| 8,320,32,32 | 74.5 (58.0) | 802.5 (+968.6%) | 81.8 (+9.8%) | 795.4 (+967.6%) new | r9-merge |
| 32,768,7,7 | 79.0 (61.1) | 78.7 (-0.4%) | 1531.8 (+1890.5%) | 1575.4 (+1894.2%) new | r9-merge |
| 32,128,64,64 | 288.0 (225.6) | 2134.1 (+631.2%) | 288.5 (+0.2%) | 2106.3 (+631.4%) new | r9-merge |
| 64,96,112,112 | 1079.0 (856.2) | 7226.9 (+552.5%) | 1083.7 (+0.4%) | 7040.5 (+552.5%) new | r9-merge |

Merged kernel `problems/035_convnextv2_block_with_grn/rounds/r10/candidates/r10-merge.json`: 16/16 workloads passed its test (timing flakes count as failures here; see the trace).
Projected portal score 0.7392 against the best's 0.7359 (#66880).
