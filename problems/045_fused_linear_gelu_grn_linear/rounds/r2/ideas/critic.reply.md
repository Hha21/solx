### Audit

I ran six B200 probes. The seventh, the weight-ring hot-spot test for I21/I30/I4, hit the probe budget and did not run. All numbers are rented and unlocked-clock.

**Wrong:**
- **Cluster capacity.** Every cluster design assumed far more clusters than fit when each CTA uses ≥180 KB of shared memory. `cuOccupancyMaxActiveClusters` reports:
  - cluster 4: 33 clusters, 132 SMs. I3 and I13 assumed 37.
  - cluster 8: 15 clusters, 120 SMs. I30 assumed 18, I19 assumed 18 slots.
  - cluster 16: 7 clusters, 112 SMs.

  So 11–24% of the SMs sit idle before any work starts. Every cluster lineage's effect estimate shrinks by that much.
- **I26 arithmetic.** I26 divides the critic's "136 issue-clk/token" by 4 SMSPs a second time. That figure is already per SM (it is compared with the 96-clk/SM MMA). The exact-GELU pass is about 123 µs at 64,56,56, not 31, so the 150–190 µs recompute projection does not hold.
- **I1, bulk reduce-add.** `cp.reduce.async.bulk.add.f32` is slower than plain `red.add`. Measured under the harness, 4-way at 1.6 MB out: store 2.91 µs, `red.add` 3.17, bulk 3.87. At 6.2 MB out: 5.94 / 7.8 / 8.48 µs. The premise that "red.add costs ~2 µs" is also wrong: it adds only about 0.3 µs over plain stores at the 3136-token size.

**Overstated:**
- **I24, packed GELU.** `f32x2` halves issue slots but not the FMA pipe. Measured: FFMA2 about 120 lanes/clk/SM against FFMA about 100, so the pipe ceiling stays near 128. The ≤1e-6 accuracy target needs a degree-9 polynomial (max error 9.3e-7; degree 8 gives 2.8e-6), so about 15–17 FMA-pipe ops per element. That is roughly 65–70 clk/token, against the claimed ~40 and the current gelu2's MUFU bound of 64. The real gain comes from overlapping the epilogue with the MMA, not from packing.
- **I19, global barrier cost.** I19 assumes a global barrier costs 2–4 µs. Measured: 1.2 µs per counter barrier at 128–148 CTAs, against 0.45 µs for a cluster-8 barrier plus a 2 KB DSMEM exchange. That saves about 0.75 µs per crossing.
- **I12.** The CTA start spread in the barrier probe was only 0.03–0.06 µs, and the barrier cost is mostly the atomic round trip. I12's skew-based saving is therefore likely small. This probe had balanced work, though; real kernels with uneven work may have more skew.
- **H10, L2 ring.** Data written after the 2×L2 zero-fill does stay in L2: a 76 MB region re-reads at 7.0–7.3 TB/s against 3.6 TB/s cold with the same kernel, and at 100 MB the hit rate drops (4.8 TB/s). But the `evict_last` hint gives nothing over the default (EL = EN), and the ">10 TB/s" claim is not demonstrated.
- **Leader stories.** Several cite "0.72 ns/token, a bandwidth signature" as evidence. The ledger itself says that number is an artefact of the equal-S formula.

**Unsupported:**
- **r1-merge precision.** The shipped r1-merge never logged the per-row max err/tol over 10 rounds that the ship gate requires (≤ 0.8).
- **The 19 µs chain.** The claim that the e1 chain of about 19 µs is "the weight ring", and every estimate of e4's per-round breakdown, rest on no phase stamps.

**Holds:**
- **MUFU rate.** It is 16/clk/SM (ex2 and rcp each 15.8), so H8/I20's pipe arithmetic is right.
- **A&S GELU accuracy.** The A&S 7.1.26 GELU already has 3.3e-7 max error, so precision is not a reason to replace it.

```json premises
[
{"claim":"37 clusters of 4 (148 SMs) / 18 clusters of 8 / 16-CTA clusters at 6-8 fit with ~180-216 KB smem per CTA","where":"pool I3, I13, I30, I19, I2","verdict":"wrong","why":"cuOccupancyMaxActiveClusters on B200: cluster4 = 33 (132 SMs), cluster8 = 15 (120 SMs), cluster16 = 7 (112 SMs) at 180-216 KB; cluster2 = 74 (148 SMs)","if_wrong":"cluster lineages lose 11-24% of SMs at L/M; per-SM estimates scale by 148/132 or 148/120; cluster-2 pairs are the only lossless cluster shape","test":"done (probe); recheck with exact smem and 1-CTA/SM TMEM before building"},
{"claim":"Exact GELU at 136 issue slots/token over 4 SMSPs = 34 SM-clk/token, so ~31 us per pass at 64,56,56, hidden under the MMA","where":"pool I26 (recompute revival of H5)","verdict":"wrong","why":"H5's 136 issue-clk/token is already per SM (it is compared with the 96-clk/SM MMA); dividing by 4 again is a 4x error. 200704 tok x 136 clk / 148 / 1.5 GHz = 123 us per GELU pass","if_wrong":"two-launch recompute pays 2 x 123 us of GELU issue unless the GELU is also cut; 150-190 us projection does not hold","test":"arithmetic; already settled"},
{"claim":"Packed f32x2 epilogue drops GELU to ~10-12 issue slots per element, ~40 clk/token, -45% at L","where":"pool I24, leader story 'MMA-efficient in-TMEM design'","verdict":"overstated","why":"FFMA2 halves issue but measured FMA-pipe throughput is ~120 lanes/clk/SM (FFMA ~100): no 2x. The <=1e-6 fit needs degree 9 (9.3e-7; deg 8 = 2.8e-6), so ~15-17 FMA-pipe lane-ops/elem + bias/sumsq/split = ~65-70 clk/token, about equal to the current gelu2 MUFU bound (64)","if_wrong":"I24 alone moves the binding pipe from MUFU to FMA with little net gain; the L lever is overlapping the epilogue with the MMA (I22/I27/I10), not packing","test":"compile the full epilogue (TMEM ld, bias, GELU, sumsq, split, st) both ways for sm_100a and time it in cycles over TMEM-sized data"},
{"claim":"Split-K red.add costs ~2 us at 3136 tokens; cp.reduce.async.bulk cuts it below 0.5 us","where":"pool I1, I28, I5(4)","verdict":"wrong","why":"harness_time, 4-way: store 2.91, red.add v1 3.17, red.add v4 3.23, bulk 3.87 us at 1.6 MB out; 5.94 / 7.81 / 7.78 / 8.48 at 6.2 MB out. red.add adds about 0.3 us over plain stores at the S size; bulk is slower","if_wrong":"keep red.add (v4) in fan-out designs; drop bulk-reduce variants; the M-row items design (I28) remains viable with red.add (~+1.9 us at 6.2 MB)","test":"done (probe)"},
{"claim":"Global counter chain costs 2-4 us vs ~0.5 us cluster barrier, worth -30..-45% on HW 196/784 rows","where":"pool I19, I2","verdict":"overstated","why":"measured 1.20-1.22 us per global red.release/ld.acquire barrier at 128-148 CTAs vs 0.45 us for cluster-8 barrier + 2 KB DSMEM; saves ~0.75 us per crossing. I19's bigger gains must come from removing weight ingest and phase 0, not from the barrier","if_wrong":"re-attribute I19's projected 9 us saving; the barrier part is <1 us","test":"done (probe); stamps on B fan-out would split the rest"},
{"claim":"evict_last workspace stays in L2 and re-reads >10 TB/s; evict_last is what keeps it resident","where":"ledger H10, lineage C, I16","verdict":"overstated","why":"after a 2xL2 zero_: re-read of a just-written 76 MB region 7.0 (EL) / 7.3 (EN) TB/s vs 3.6 cold with the same naive kernel; 100 MB drops to 4.8. Residency is real up to ~76 MB, but the hint is irrelevant and >10 TB/s is unshown","if_wrong":"stored-h ring designs work at the ~76 MB scale without hints; budget L2 read at ~7 TB/s until a better reader shows more","test":"same probe with a deeper-MLP reader (8 float4/thread, 1 wave) to find the L2 hit ceiling"},
{"claim":"Leader stores h (or uses bandwidth) because implied L times are linear at 0.72 ns/token","where":"leader stories (stored-h, two-launch)","verdict":"unsupported","why":"ledger note: implied per-workload times are the equal-S formula's output; the linearity is an artefact","if_wrong":"stories rest only on mean score; no design is favoured by it","test":"none needed; drop the 'fits' line"},
{"claim":"r1-merge meets the precision ship gate (err/tol <= 0.8 per row)","where":"ledger notes ship gate; r1 findings (log NOT collected)","verdict":"unsupported","why":"only harness pass/fail at ~3e-3 atol exists; critic emulation had single-accumulator GEMM2 at 0.78, near the gate","if_wrong":"a seed could fail a row (scores 0) on the portal","test":"run r1-merge per row over 10 seeds against an fp64 reference and log max err/tol"},
{"claim":"e1's ~19 us fixed chain is the serial weight ring / L2 hot-spot","where":"r1 finding r1-a-e1-ts-fp16x3-m, pool I4/I21/I30","verdict":"unsupported","why":"no phase stamps; hot-spot probe not run (budget)","if_wrong":"replica/multicast work is wasted if the chain is phase 0 + grid counter","test":"globaltimer stamps in e1 at 1,28,28; the 148-CTA shared vs replica vs private bulk-ring probe (written, not run)"}
]
```

```json probes
[
{"idea":"I3/I13/I30/I19/I2 cluster capacity","test":"cuOccupancyMaxActiveClusters, 512 threads, 100/180/216 KB dyn smem, cluster 2/4/8/16","result":"c2: 74 (148 SMs) at >=180K; c4: 33 (132); c8: 15 (120); c16: 7 (112); at 100K c4 71, c8 33, c16 14","verdict":"fails","note":"1-CTA/SM cluster designs idle 16-36 SMs; only cluster-2 pairs are lossless"},
{"idea":"I24 tail-form GELU accuracy","test":"fp64 weighted minimax fit of g on [0,5.75], fp32 Horner x exp2 with 2^-22 error, u in [-60,60]","result":"max |err|: deg7 9.8e-6, deg8 2.8e-6, deg9 9.3e-7, deg10 4.2e-7; A&S 7.1.26 GELU 3.3e-7; fp32 rounding floor 2.4e-7","verdict":"holds","note":"accurate at degree 9-10, so ~15-17 FMA-pipe ops per element, not ~9"},
{"idea":"I24/I20/H8 pipe rates","test":"all SMs, wall-time-based lanes/clk/SM","result":"FFMA ~100, FFMA2 ~120 lanes/clk/SM; MUFU.EX2 15.8, MUFU.RCP 15.8 per clk per SM","verdict":"unclear","note":"MUFU = 16/clk/SM confirmed; f32x2 does not double FMA throughput, so packing only relieves issue"},
{"idea":"I1/I28 bulk reduce-add","test":"harness_time, 4-way split-K into 1.6 MB and 6.2 MB outputs, 512-thread CTAs","result":"1.6 MB: store 2.91, red.add 3.17, red.v4 3.23, bulk 3.87 us; 6.2 MB: 5.94/7.81/7.78/8.48 us","verdict":"fails","note":"red.add is cheap; bulk reduce adds smem staging and is slower"},
{"idea":"I19/I2/I12 barrier cost","test":"20 back-to-back global counter barriers (red.release + ld.acquire spin) vs cluster-8 barrier with 2 KB DSMEM stores, globaltimer","result":"global 1.20 us (128 CTAs), 1.22 us (148); cluster-8 0.45 us; start spread 0.03-0.06 us","verdict":"fails","note":"global crossing is ~1.2 us not 2-4; cluster saves ~0.75 us per crossing; I12's skew saving likely small"},
{"idea":"H10 / lineage C / I16 L2-resident h ring","test":"2xL2 zero_, write N MB with EL/EN/EF policy, re-read; globaltimer span","result":"76 MB: EL 7.0, EN 7.3, EF 3.7, cold 3.6 TB/s; 38 MB: 5.5/5.7/3.3/3.1; 100 MB: 4.8/4.8/3.5/3.5","verdict":"unclear","note":"residency holds to ~76 MB regardless of hint; >10 TB/s not shown with this reader"},
{"idea":"I21/I30/I4 weight-ring hot-spot","test":"148 CTAs x 64 serial 32 KB bulk copies, S=1/3/6, shared vs 16 replicas vs private","result":"not run: probe budget exhausted","verdict":"unclear","note":"script is ready; run it first next round before building replicas or multicast"}
]
```