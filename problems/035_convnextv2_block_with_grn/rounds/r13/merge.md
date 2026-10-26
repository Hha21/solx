# Per-workload merge for round r13

Projected portal us per workload (rented change in the same container in brackets; median of 4 passes each). A candidate's rented time is scaled by the portal/rented ratio its own lineage shows in r12-merge's portal result; a workload moves when the projection beats the best's portal time by 3% (10% for a lineage with no portal result yet, marked new).

| workload | r12-merge portal us (rented) | r13-d-e1-balanced-ranges-h49 | r13-d-e2-sc-stripconv-512 | r13-i-e3-rows-tail | r13-m-e4-token-stationary | routed to |
|---|---|---|---|---|---|---|
| 1,192,28,28 | 23.2 (18.8) | 110.2 (+374.9%) | 21.1 (-9.0%) | 110.5 (+376.2%) new | 111.4 (+380.3%) new | r13-d-e2-sc-stripconv-512 |
| 1,96,56,56 | 23.7 (18.4) | 83.0 (+250.1%) | 21.7 (-8.5%) | 83.2 (+251.0%) new | 26.3 (+11.0%) new | r13-d-e2-sc-stripconv-512 |
| 2,768,7,7 | 28.3 (22.8) | 31.0 (+9.6%) | 29.2 (+3.3%) | 153.5 (+442.3%) new | 156.0 (+451.4%) new | r12-merge |
| 2,96,56,56 | 32.5 (25.0) | 129.9 (+299.8%) | 29.9 (-8.1%) | 129.8 (+299.3%) new | 26.8 (-17.7%) new | r13-m-e4-token-stationary |
| 1,160,449,7 | 33.1 (25.8) | 142.0 (+329.0%) | 30.4 (-8.2%) | 141.5 (+327.5%) new | 142.3 (+329.9%) new | r13-d-e2-sc-stripconv-512 |
| 4,192,28,28 | 37.1 (29.6) | 174.6 (+370.7%) | 34.3 (-7.5%) | 174.2 (+369.6%) new | 174.9 (+371.5%) new | r13-d-e2-sc-stripconv-512 |
| 8,768,7,7 | 38.7 (30.9) | 40.2 (+3.9%) | 38.8 (+0.2%) | 332.0 (+758.0%) new | 331.6 (+756.8%) new | r12-merge |
| 8,384,14,14 | 43.2 (33.9) | 42.8 (-0.9%) | 44.4 (+2.8%) | 347.4 (+704.1%) new | 346.9 (+703.1%) new | r12-merge |
| 4,96,56,56 | 46.5 (37.0) | 180.8 (+282.8%) | 41.6 (-11.9%) | 177.9 (+282.6%) new | 42.5 (-8.5%) new | r13-d-e2-sc-stripconv-512 |
| 16,768,7,7 | 52.1 (40.8) | 52.2 (+0.3%) | 53.6 (+2.9%) | 755.3 (+1349.6%) new | 755.0 (+1349.2%) new | r12-merge |
| 8,192,28,28 | 54.9 (42.6) | 279.2 (+408.6%) | 51.1 (-7.0%) | 279.4 (+408.9%) new | 280.1 (+410.1%) new | r13-d-e2-sc-stripconv-512 |
| 16,384,14,14 | 65.1 (50.6) | 59.7 (-8.3%) | 65.1 (+0.0%) | 452.9 (+595.8%) new | 453.3 (+596.3%) new | r13-d-e1-balanced-ranges-h49 |
| 32,768,7,7 | 69.8 (54.8) | 72.6 (+4.1%) | 76.2 (+9.2%) | 1556.3 (+2129.7%) new | 1555.7 (+2128.8%) new | r12-merge |
| 8,320,32,32 | 74.6 (58.5) | 795.2 (+965.4%) | 794.6 (+964.6%) | 795.2 (+965.9%) new | 795.3 (+966.0%) new | r12-merge |
| 32,128,64,64 | 286.9 (226.3) | 2106.5 (+630.1%) | 2105.8 (+629.8%) | 230.3 (-19.7%) new | 2093.5 (+629.7%) new | r13-i-e3-rows-tail |
| 64,96,112,112 | 1066.2 (862.2) | 7127.7 (+548.5%) | 7130.1 (+548.7%) | 921.7 (-13.6%) new | 6921.7 (+549.2%) new | r13-i-e3-rows-tail |

Merged kernel `problems/035_convnextv2_block_with_grn/rounds/r13/candidates/r13-merge.json`: 16/16 workloads passed its test (timing flakes count as failures here; see the trace).
Projected portal score 0.7563 against the best's 0.7425 (#75393).
