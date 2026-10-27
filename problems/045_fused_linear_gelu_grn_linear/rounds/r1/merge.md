# Per-workload merge for round r1

Projected portal us per workload (rented change in the same container in brackets; median of 4 passes each). A candidate's rented time is scaled by the portal/rented ratio its own lineage shows in r0-simt-3k's portal result; a workload moves when the projection beats the best's portal time by 3% (10% for a lineage with no portal result yet, marked new).

| workload | r0-simt-3k portal us (rented) | r1-a-e1-ts-fp16x3 | r1-a-e1-ts-fp16x3-m | r1-a-e4-rounds-tmem | r1-a-e4-rounds-tmem-v2 | r1-b-e2-fanout | r1-b-e2-fanout-v3 | r1-c-e3-l2ring | r1-c-e3-l2ring-b | routed to |
|---|---|---|---|---|---|---|---|---|---|---|
| 8,14,14 | 175.4 (138.1) | 28.4 (-83.8%) new | 28.4 (-83.8%) new | 39.0 (-77.8%) new | 39.1 (-77.7%) new | 17.2 (-90.2%) new | 17.3 (-90.1%) new | 56.8 (-67.6%) new | 56.8 (-67.6%) new | r1-b-e2-fanout |
| 4,28,28 | 176.1 (138.8) | 29.1 (-83.5%) new | 29.4 (-83.3%) new | 37.0 (-79.0%) new | 37.1 (-79.0%) new | 21.0 (-88.1%) new | 21.0 (-88.1%) new | 55.3 (-68.6%) new | 55.2 (-68.7%) new | r1-b-e2-fanout |
| 16,14,14 | 176.2 (138.5) | 31.6 (-82.1%) new | 31.5 (-82.1%) new | 39.0 (-77.9%) new | 39.0 (-77.8%) new | 21.4 (-87.8%) new | 21.4 (-87.8%) new | 55.6 (-68.4%) new | 55.6 (-68.5%) new | r1-b-e2-fanout |
| 32,14,14 | 177.4 (140.8) | 35.6 (-79.9%) new | 35.6 (-79.9%) new | 41.3 (-76.7%) new | 41.3 (-76.7%) new | 177.2 (-0.1%) new | 177.3 (-0.0%) new | 55.6 (-68.7%) new | 55.6 (-68.7%) new | r1-a-e1-ts-fp16x3 |
| 1,28,28 | 178.3 (140.3) | 26.5 (-85.1%) new | 26.5 (-85.1%) new | 41.2 (-76.9%) new | 41.2 (-76.9%) new | 14.6 (-91.8%) new | 14.7 (-91.8%) new | 57.7 (-67.6%) new | 57.8 (-67.6%) new | r1-b-e2-fanout |
| 2,56,56 | 183.2 (145.5) | 32.1 (-82.5%) new | 31.7 (-82.7%) new | 36.3 (-80.2%) new | 36.3 (-80.2%) new | 183.0 (-0.1%) new | 183.4 (+0.1%) new | 54.6 (-70.2%) new | 54.5 (-70.3%) new | r1-a-e1-ts-fp16x3-m |
| 64,14,14 | 284.9 (223.0) | 46.5 (-83.7%) new | 46.4 (-83.7%) new | 48.1 (-83.1%) new | 48.1 (-83.1%) new | 284.9 (-0.0%) new | 284.8 (-0.0%) new | 58.6 (-79.4%) new | 58.7 (-79.4%) new | r1-a-e1-ts-fp16x3-m |
| 16,28,28 | 285.4 (222.1) | 41.0 (-85.6%) new | 40.9 (-85.7%) new | 40.6 (-85.8%) new | 40.7 (-85.8%) new | 285.4 (+0.0%) new | 285.4 (-0.0%) new | 56.8 (-80.1%) new | 56.8 (-80.1%) new | r1-a-e4-rounds-tmem |
| 4,56,56 | 286.4 (222.8) | 38.6 (-86.5%) new | 38.6 (-86.5%) new | 38.4 (-86.6%) new | 38.4 (-86.6%) new | 286.6 (+0.1%) new | 286.3 (-0.0%) new | 56.3 (-80.4%) new | 56.2 (-80.4%) new | r1-a-e4-rounds-tmem-v2 |
| 32,28,28 | 420.1 (328.2) | 419.9 (-0.0%) new | 419.9 (-0.1%) new | 70.6 (-83.2%) new | 70.6 (-83.2%) new | 419.9 (-0.0%) new | 420.2 (+0.0%) new | 99.4 (-76.3%) new | 99.6 (-76.3%) new | r1-a-e4-rounds-tmem |
| 16,56,56 | 798.9 (620.0) | 798.8 (-0.0%) new | 799.2 (+0.0%) new | 99.0 (-87.6%) new | 99.0 (-87.6%) new | 799.0 (+0.0%) new | 799.2 (+0.0%) new | 151.8 (-81.0%) new | 152.0 (-81.0%) new | r1-a-e4-rounds-tmem |
| 64,28,28 | 799.0 (620.0) | 798.8 (-0.0%) new | 798.6 (-0.0%) new | 106.5 (-86.7%) new | 106.5 (-86.7%) new | 799.0 (+0.0%) new | 799.2 (+0.0%) new | 156.4 (-80.4%) new | 156.4 (-80.4%) new | r1-a-e4-rounds-tmem |
| 32,56,56 | 1449.5 (1119.6) | 1449.9 (+0.0%) new | 1449.6 (+0.0%) new | 185.5 (-87.2%) new | 185.4 (-87.2%) new | 1449.4 (-0.0%) new | 1450.1 (+0.0%) new | 303.5 (-79.1%) new | 303.0 (-79.1%) new | r1-a-e4-rounds-tmem-v2 |
| 64,56,56 | 2859.7 (2200.0) | 2859.8 (+0.0%) new | 2859.3 (-0.0%) new | 333.2 (-88.3%) new | 333.5 (-88.3%) new | 2859.7 (+0.0%) new | 2860.5 (+0.0%) new | 558.3 (-80.5%) new | 557.9 (-80.5%) new | r1-a-e4-rounds-tmem |

Merged kernel `problems/045_fused_linear_gelu_grn_linear/rounds/r1/candidates/r1-merge.json`: 14/14 workloads passed its test (timing flakes count as failures here; see the trace).
Projected portal score 0.7214 against the best's 0.2442 (#75960).
