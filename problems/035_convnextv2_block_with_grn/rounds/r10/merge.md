# Per-workload merge for round r10

Projected portal us per workload (rented change in the same container in brackets; median of 4 passes each). A candidate's rented time is scaled by the portal/rented ratio its own lineage shows in r9-merge's portal result; a workload moves when the projection beats the best's portal time by 2% (10% for a lineage with no portal result yet, marked new).

| workload | r9-merge portal us (rented) | r10-d-e1-balanced-nostack | r10-g-e2-g2-coop | r10-g-e2-g2-coop-v3 | r10-i-e3-gate-unrun | r10-i-e3-tmaconv-gate | routed to |
|---|---|---|---|---|---|---|---|
| 1,192,28,28 | 23.3 (18.7) | 22.7 (-2.7%) | 114.5 (+377.4%) | 114.9 (+379.0%) | 110.8 (+375.5%) new | 110.9 (+375.8%) new | r10-d-e1-balanced-nostack |
| 1,96,56,56 | 23.8 (18.3) | 23.6 (-0.7%) | 48.5 (+109.5%) | 44.9 (+93.9%) | 83.7 (+251.9%) new | 83.9 (+252.6%) new | r9-merge |
| 2,768,7,7 | 30.7 (24.1) | 30.6 (-0.5%) | 158.9 (+416.0%) | 157.8 (+412.2%) | 157.7 (+413.6%) new | 157.4 (+412.5%) new | r9-merge |
| 1,160,449,7 | 33.2 (25.7) | 32.7 (-1.6%) | 142.1 (+330.8%) | 141.8 (+330.0%) | 142.1 (+328.1%) new | 142.1 (+327.9%) new | r9-merge |
| 2,96,56,56 | 33.3 (25.5) | 32.5 (-2.4%) | 48.8 (+51.7%) | 46.0 (+42.8%) | 130.5 (+292.0%) new | 130.6 (+292.2%) new | r10-d-e1-balanced-nostack |
| 4,192,28,28 | 37.1 (29.8) | 35.9 (-3.2%) | 177.4 (+365.0%) | 177.6 (+365.4%) | 172.5 (+365.0%) new | 172.4 (+364.7%) new | r10-d-e1-balanced-nostack |
| 8,768,7,7 | 42.0 (33.1) | 38.4 (-8.5%) | 335.3 (+692.8%) | 335.8 (+694.1%) | 333.6 (+694.3%) new | 333.6 (+694.3%) new | r10-d-e1-balanced-nostack |
| 8,384,14,14 | 44.4 (33.6) | 44.5 (+0.3%) | 346.4 (+704.8%) | 345.1 (+701.8%) | 356.9 (+703.9%) new | 357.2 (+704.5%) new | r9-merge |
| 4,96,56,56 | 46.7 (37.0) | 49.8 (+2.0%) | 50.0 (+7.2%) | 46.7 (-0.1%) | 179.3 (+284.0%) new | 179.6 (+284.7%) new | r9-merge |
| 8,192,28,28 | 54.7 (42.5) | 54.2 (-0.9%) | 276.7 (+408.6%) | 277.0 (+409.2%) | 279.0 (+410.0%) new | 278.6 (+409.4%) new | r9-merge |
| 16,768,7,7 | 56.7 (43.6) | 52.9 (-6.8%) | 742.2 (+1245.8%) | 742.6 (+1246.4%) | 762.8 (+1245.3%) new | 762.2 (+1244.3%) new | r10-d-e1-balanced-nostack |
| 16,384,14,14 | 67.5 (51.2) | 66.4 (-1.6%) | 448.2 (+584.0%) | 447.7 (+583.3%) | 461.2 (+583.2%) new | 461.0 (+583.0%) new | r9-merge |
| 8,320,32,32 | 74.5 (58.1) | 803.3 (+967.2%) | 82.4 (+10.6%) | 78.5 (+5.3%) | 793.9 (+965.7%) new | 794.5 (+966.5%) new | r9-merge |
| 32,768,7,7 | 79.0 (61.0) | 78.5 (-0.6%) | 1538.3 (+1894.6%) | 1537.9 (+1894.1%) | 1576.1 (+1895.0%) new | 1576.3 (+1895.4%) new | r9-merge |
| 32,128,64,64 | 288.0 (225.2) | 2131.5 (+631.1%) | 288.7 (+0.2%) | 288.8 (+0.3%) | 2107.4 (+631.7%) new | 2107.4 (+631.7%) new | r9-merge |
| 64,96,112,112 | 1079.0 (853.6) | 7236.7 (+554.7%) | 1084.2 (+0.5%) | 1083.9 (+0.5%) | 7061.4 (+554.4%) new | 7060.6 (+554.4%) new | r9-merge |

Merged kernel `problems/035_convnextv2_block_with_grn/rounds/r10/candidates/r10-merge.json`: 16/16 workloads passed its test (timing flakes count as failures here; see the trace).
Projected portal score 0.7392 against the best's 0.7359 (#66880).
