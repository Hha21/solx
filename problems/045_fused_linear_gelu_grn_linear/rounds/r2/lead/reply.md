### Assessment

The score we can still gain is mostly in the five L rows. They sit at S = 0.65–0.69, and cutting them by 45% predicts 0.775 (+0.053). The M rows come next (−20% on M and −5% on S predicts 0.741), and S gives the least.

**What we learned at L.** e4 runs 64,56,56 in 11 batch-aligned rounds of about 29.7 µs each on the portal, against 8.2 µs of MMA per round. H13 is refuted. The kernel runs its steps one after another (GEMM1 → GELU tail → batch wait → pass 2 → GEMM2 → store) because fp32 h fills all 512 TMEM columns. The cuts need two things:
- **Overlap.** The critic measured that packing the GELU cannot get the epilogue down to ~10 slots: MUFU runs at 16/clk, FFMA2 at only 1.2× FFMA, and the tail form needs degree 9. So the epilogue has to run under the MMA.
- **Cheaper weight path.** The weight stream also needs to get cheaper, or move out of the way.

This round tests two independent structural answers at L:
- **E2 (lineage A):** cluster-2 hidden-split pairs. Each SM keeps half the hidden channels, so two 128-token tiles of h are resident at once. Cluster-2 is the only cluster shape that loses no SMs (74 pairs).
- **E4 (new lineage D):** the stored-h, two-launch design. Each step is a plain pipelined GEMM with weights resident in smem (halves per CTA), and the time is budgeted at the DRAM floor of about 5 KB/token (~150 µs at 64,56,56).

**M rows.** e1's ~19 µs fixed chain has not been attributed. E1 stamps it and runs the hot-spot probe the critic already wrote, then fixes the weight path in both e1 and e4.

**B.** B's fan-out wins the four smallest rows. E3 extends it to the two 6,272-token rows (32,14,14 is our worst M row) and stamps the 3,136-token rows.

**Lineage C is retired** under the measured-hardware exception. After the flush, L2 re-reads run at 7.0–7.3 TB/s whatever the eviction hint, and at 4.8 TB/s at 100 MB. That is not the >10 TB/s its thesis needed, and its kernel measured 1.5× slower than e4 at L. D carries the "h through memory" idea forward with a DRAM-rate budget.

**Answers to the audit:**
- **Corrected in the ledger:** cluster capacity, the GELU arithmetic in I26, f32x2 packing, bulk reduce, barrier cost and H10.
- **Scheduled:** the precision log is owed for every shipped row, and each session logs err/tol for its own rows. The attribution of e1's chain is E1's first step.
- **Already settled:** the leader-linearity argument was dropped in r1.

```yaml lineages
lineages:
  - id: A
    thesis: The hidden h never leaves the SM. A cooperative tcgen05 FP16x3 launch, token-stationary, keeps fp32 h in TMEM across per-batch GRN counters; to scale at L it must overlap tiles and stop streaming 512 KB of weights per tile.
    status: active
    started: r1
    best: r1-a-e4-rounds-tmem
    conclusions:
      - "e1 wins 2,56,56 / 32,14,14 / 64,14,14 (25-36 us rented); e4 wins every L row plus 4,56,56 and 16,28,28 in r1-merge (portal 0.7216)"
      - "e1 has a ~19 us fixed chain that does not depend on work (784 tokens take 21 us with 0.3 us of MMA); not yet attributed between phase 0, grid counter and the weight ring"
      - "e4 at 64,56,56 runs 11 rounds of ~29.7 us portal against 8.2 us of MMA: its steps run one after another because fp32 h fills the TMEM (H13 refuted)"
      - "Precision: FP16x3 with a separate GEMM2 cross accumulator passes every row in 10 harness rounds; the err/tol log is still owed"
      - "Only cluster-2 is free of SM loss at >=180 KB smem (c4 33, c8 15, c16 7 clusters): overlap designs should use pairs"
  - id: B
    thesis: S rows are a latency chain, not a FLOP problem. Fan each token tile out over hidden slices so every SM ingests at most 64 KB of weights, cross one grid barrier, and reduce the split-K GEMM2 with red.add into an output prefilled with the bias.
    status: active
    started: r1
    best: r1-b-e2-fanout
    conclusions:
      - "Fan-out wins the 4 smallest rows: 11.6/13.7/16.5/16.7 us rented (-88..-92% vs r0); 1,28,28 misses the 9 us line"
      - "3136-token rows run ~5 us above 784: the cost grows with the token tile TT, not with the fixed chain (unstamped)"
      - "red.add v4 costs ~0.3 us over plain stores at 1.6 MB; cp.reduce.async.bulk is slower (critic probe): keep red.add"
      - "Global counter barrier costs 1.2 us at 128-148 CTAs, cluster-8 0.45 us: the barrier is worth at most ~0.75 us"
      - "256-wide slices do not fit (smem/TMEM); 6272-token rows need TT=196 x 4 slices with GEMM2 in token halves"
  - id: C
    thesis: At L the GRN dependency crosses through L2, not TMEM, via a per-CTA evict_last h ring re-read one round later in a single persistent launch.
    status: retired
    started: r1
    best: r1-c-e3-l2ring-b
    conclusions:
      - "Retired at r2 under the measured-hardware exception: after the 2xL2 flush an L2 re-read runs at 7.0-7.3 TB/s whatever the hint (4.8 at 100 MB), not the >10 TB/s the thesis assumed"
      - "r1-c-e3-l2ring-b passed but runs 159.8 us L geomean rented vs e4 104.3 (1.5x slower); its 4 KB/token per-tile weight re-stream is unmeasured"
      - "The idea of crossing GRN through memory continues in lineage D, budgeted at the DRAM rate with two launches"
  - id: D
    thesis: At L, cross the GRN through memory with two plain pipelined tcgen05 launches (GEMM1+GELU writes h at 2 KB/token plus per-tile sumsq partials; a PDL launch runs GEMM2), with weight halves resident in smem, so time sits at the stored-h DRAM floor (~5 KB/token, ~150 us at 64,56,56) instead of at serial on-chip steps.
    status: active
    started: r2
    best: null
    conclusions:
      - "Seeded r2 from the stored-h leader stories and I16/I6; no grid barrier, no TMEM retention, no weight ring if weights are split hidden-half (K1) / out-half (K2) per CTA"
      - "Budget: DRAM ~7 TB/s for x + h write + h read + out (1.03 GB at 64,56,56); L2 residency up to ~76 MB is a bonus for 32,28,28 and below"
```

```yaml ledger-changes
hypotheses:
  - id: H5
    statement: At L, recomputing GEMM1 beats storing h once GEMMs run on tensor cores.
    status: parked
    evidence:
      - "arithmetic at 64,56,56: recompute ~+44-88 us of MMA vs store ~+100 us at 8 TB/s"
      - "critic r1 SASS count: exact erff GELU is 1 MUFU + ~28 ALU per element, ~136 issue-clk/token with bias, sumsq and split, above the 96-clk FP16x3 MMA; recompute repeats this GELU"
      - "critic r2: the 136 clk/token is already per SM; I26's division by 4 SMSPs is a 4x error, so a second GELU pass costs ~123 us at 64,56,56 (gelu2 form: ~64 clk/token MUFU-bound, ~58 us). Recompute stays parked"
  - id: H8
    statement: In on-chip and stored-h designs the CUDA-core epilogue (bias, GELU, sumsq, hi/lo split, TMEM round trip) binds at L rather than the MMA, unless it is overlapped with the MMA; packing alone cannot cut it to ~10 issue slots per element.
    status: open
    evidence:
      - "critic r1: erff 1 MUFU + ~28 ALU; normcdff 3 MUFU + ~60; erfcf 3 MUFU + ~45 (sm_100a SASS)"
      - "#129 K: co-running epilogue and MMA warps keep 54-70% of their solo rates"
      - "critic r2 pipe probe (rented B200, all SMs): MUFU.EX2/RCP 15.8/clk/SM; FFMA ~100, FFMA2 ~120 lanes/clk/SM, so f32x2 does not double FMA throughput"
      - "critic r2 fit: tail-form GELU needs degree 9 (max err 9.3e-7; degree 8 2.8e-6), ~15-17 FMA-pipe ops per element, ~65-70 clk/token, about equal to gelu2's MUFU bound of 64 clk/token"
  - id: H10
    statement: A per-CTA workspace ring of h (2 KB/token, at most 2 rounds, about 76 MB) written with evict_last and re-read by the same CTA one round later stays in L2 under the harness's dirty-L2 flush, so read-back runs above 10 TB/s aggregate.
    status: refuted
    evidence:
      - "critic r2 probe (2xL2 zero_, write then re-read, globaltimer): 76 MB re-read 7.0 TB/s (evict_last) / 7.3 (normal) / 3.7 (evict_first) vs 3.6 cold; 38 MB 5.5/5.7; 100 MB 4.8/4.8. Residency holds to ~76 MB but the hint does not matter and >10 TB/s is not shown"
      - "r1-c-e3-l2ring-b: 159.8 us L geomean rented vs e4 104.3"
  - id: H13
    statement: At L, persistent rounds aligned to whole batches, with fp32 h in TMEM and per-batch counters, run at <= 15 us per 128-token round (about 165 us at 64,56,56) despite the full TMEM preventing overlap.
    status: refuted
    evidence:
      - "arithmetic: 8.2 us of MMA per round"
      - "r1-merge portal: e4 at 64,56,56 = 326.4 us over 11 rounds = ~29.7 us per round (~23 rented); 16,56,56 97.4, 64,28,28 104.3. Steps run one after another; the split between weight ring, epilogue, wait and pass 2 is unmeasured (E2 stamps it)"
  - id: H11
    statement: For N <= 6272, a hidden-slice fan-out (token tiles x hidden slices, at most 64 KB of weights per CTA, one grid barrier, red.add split-K) reaches 1,28,28 in <= 9 us rented and 3136-token rows in <= 11 us.
    status: open
    evidence:
      - "critic r1 skeleton at 1,28,28, 128 CTAs, rented: empty 1.63 us, loads and stores 4.80, plus barrier 6.27, plus 16-way red.add 6.82"
      - "r1-b-e2-fanout-v3 run_tests: 1,28,28 11.6, 8,14,14 13.7, 4,28,28 16.5, 16,14,14 16.7 us rented; both success lines missed, the 12 us refutation line not reached; portal 14.7/17.0/21.0/21.4"
      - "critic r2: red.add v4 adds ~0.3 us over plain stores at 1.6 MB; global barrier 1.2 us"
  - id: H15
    statement: "Split-K output reduction: per-thread red.global.add (v1 or v4) costs only ~0.3 us over plain stores at 1.6 MB out (4-way); cp.reduce.async.bulk.add.f32 from smem is slower and should not be used."
    status: supported
    evidence:
      - "critic r2 probe, harness_time, 512-thread CTAs: 1.6 MB store 2.91, red.add 3.17, red.v4 3.23, bulk 3.87 us; 6.2 MB 5.94 / 7.81 / 7.78 / 8.48 us"
  - id: H16
    statement: "Cross-CTA sync costs: a global red.release + ld.acquire barrier over 128-148 CTAs costs ~1.2 us; a cluster-8 barrier with a 2 KB DSMEM exchange ~0.45 us. Moving a GRN crossing into a cluster saves <= ~0.75 us per crossing."
    status: supported
    evidence:
      - "critic r2 probe: 20 back-to-back barriers, globaltimer: 1.20 us (128 CTAs), 1.22 us (148), cluster-8 0.45 us; start spread 0.03-0.06 us"
  - id: H17
    statement: "Cluster capacity on B200 at 512 threads and >=180 KB dynamic smem: cluster-2 74 (148 SMs), cluster-4 33 (132), cluster-8 15 (120), cluster-16 7 (112). Only pairs are free of SM loss for 1-CTA/SM designs."
    status: supported
    evidence:
      - "critic r2 probe, cuOccupancyMaxActiveClusters at 100/180/216 KB; at 100 KB c4 71, c8 33, c16 14"
  - id: H18
    statement: "e1's ~19 us work-independent chain is mostly the weight path (phase-0 image build + grid counter + 16 serial 32 KB ring units, each refill gated by the previous unit's commit, possibly an L2 hot-spot when 148 SMs read the same lines); fixing it (replicas, deeper or finer ring, entry L2 prefetch, or no phase-0 barrier) cuts the e1 M rows by >= 20% rented."
    status: open
    evidence:
      - "r1-a-e1-ts-fp16x3-m: 784 tokens 21.0 us with ~0.3 us of MMA; t=16 -> t=96 adds only 9-15 us"
      - "hot-spot probe (148 CTAs x 64 serial 32 KB bulk copies, shared vs 16 replicas vs private, S=1/3/6) written by the critic, not run"
      - "r2 E1 tests it"
  - id: H19
    statement: "A cluster-2 pair that splits the 512 hidden channels 256/256 holds two 128-token tiles of fp32 h per SM (2 x 256 TMEM columns), so GEMM1+GELU of tile i+1 overlaps the GRN wait, pass 2 and GEMM2 of tile i; partial GEMM2 outputs are summed through DSMEM; L rows drop >= 35% vs e4 rented."
    status: open
    evidence:
      - "arithmetic: per SM per tile 6.1k clk MMA (half the hidden), GELU 32768 elements = 4.1k MUFU clk; 64,56,56 = 21.2 tiles per pair -> ~86 us of MMA"
      - "H17: 74 pairs fit at >=180 KB smem"
      - "r2 E2 tests it"
  - id: H20
    statement: "A two-launch stored-h design (K1: GEMM1 FP16x3 + GELU, writes h at 2 KB/token plus per-tile sumsq partials with plain stores; K2 via PDL: GRN scale, x2 split or per-batch folded W2, GEMM2 FP16x3) with weight halves resident in smem runs near the stored-h DRAM floor: 64,56,56 <= 170 us rented, and beats e4 by >= 10% on every L row."
    status: open
    evidence:
      - "arithmetic: x 0.5 + h 2 + h 2 + out 0.5 KB/token = 1.03 GB at 64,56,56, ~146 us at 7.05 TB/s; MMA 6 passes 87 us and epilogue ~58-65 us overlap the stream"
      - "H10 refuted: budget h at DRAM rate; L2 residency up to ~76 MB helps rows up to 32,28,28"
      - "r2 E4 tests it"
  - id: H21
    statement: "The B fan-out extends to 6272-token rows with TT=196 x 4 hidden slices (128 CTAs, one wave): W2 slice bulk-copied over W1 after GEMM1 commits, GEMM2 in two token halves of 104; 2,56,56 and 32,14,14 <= 19 us rented (vs e1 25.3/28.3)."
    status: open
    evidence:
      - "196-token tiles never straddle a batch at HW 196/784/3136; ingest 164 KB per CTA; 48 MMAs at N=208 ~3.3 us"
      - "r2 E3 tests it"
notes:
  - "Leader-gap per-workload times are the equal-S formula's output (t = Tsol + 0.171 (Tb - Tsol)); the HW=3136 \"penalty\" and the 0.723 ns/token slope are artefacts and say nothing about the leader's design (critic r1 algebra)."
  - "leader_gap.py has GFLOP = 0 for #45, so its \"memory-bound, precision does not matter, ceiling 0.994\" output is wrong. Tsol recovered from the portal anchors is 0.334 ns/token at L (67.45 us at 64,56,56)."
  - "Precision ship gate: per row, max over 10 rounds of max |err| / (atol + rtol |ref|) <= 0.8 with the row's own tolerance, logged in findings; never rely on the 99% matched ratio. Prefer a separate cross-term accumulator on GEMM2. r1-merge has NO err/tol log yet: every r2 session logs err/tol for the rows its kernels serve (E1 also for every e1/e4 row in r1-merge)."
  - "Engine template: #129 lineage-M kernel (read_example solx_129_token_stationary_tcgen05.cu) - SW128 descriptors, idesc kind::f16, elect.sync issue, release/acquire counters with departure reset, packed A&S gelu2, cooperative launch."
  - "Portal noise: rows under ~40 us moved about +-1 us between identical submissions on #129. Merge thresholds: 3% projected gain for a lineage with portal history, 10% for a new one."
  - "Cluster shapes: only cluster-2 keeps all 148 SMs at >=180 KB smem (H17); designs on c4/c8/c16 lose 11-24% of SMs (I2/I3/I13/I19/I30 parked on this)."
  - "Modal B200 supply is intermittent (about two at a time; calls not scheduled within 15 min return an error): run the short probes first, retry unscheduled calls later, never block a session on one call."
```

```json tasks
[
  {"id": "E1", "lineage": "A", "hypothesis": "H18",
   "title": "Attribute and remove e1's ~19 us weight-path chain (M rows), carry the fix into e4",
   "operation": "structural_mutation",
   "parents": ["r1-a-e1-ts-fp16x3-m"], "band": "M",
   "instructions": "Step 0 (precision gate, owed from r1): run r1-merge's e1 and e4 paths on every row they serve (32,14,14 64,14,14 2,56,56 4,56,56 16,28,28 and the 5 L rows) over 10 seeds against an fp64 reference. Log max err/(atol+rtol|ref|) per row; the gate is 0.8. Step 1: add %globaltimer stamps to e1 (lane 0 of the producer, MMA and one epilogue warp) at kernel entry, phase-0 image done, grid-counter release, each weight unit landing (bar_full), GEMM1 done, batch-counter release, GEMM2 done and last store. Print the CTA-median and max at 1,28,28 and 64,14,14. Step 2: run the critic's ready hot-spot probe: 148 CTAs x 64 serial 32 KB bulk copies through S=1/3/6 stages, from one shared image vs 16 replicas vs private copies, after a 2xL2 zero_. Step 3: run_tests the already-written ring variant r1-a-e1-ts-fp16x3-ring (ring depth from t). Step 4: fix the largest term the stamps show: (a) replicas: phase 0 writes R=8-16 image copies and CTA i reads copy i%R, if the probe shows shared minus replica > 0.3 us/unit; (b) a deeper or finer ring (16 KB stages) if MMA waits on bar_full; (c) cp.async.bulk.prefetch.L2 of the image at entry; (d) if phase 0 plus the grid counter is > 3 us, convert-on-ingest weights so the grid barrier goes away. Step 5: apply the same producer change to e4 (r1-a-e4-rounds-tmem) and run_tests the L rows in the same container. Hand in both kernels.",
   "success": "e1 M rows -20% rented in the same container (64,14,14 <= 29 us, 2,56,56 <= 20 us), with the stamps showing which term moved; err/tol <= 0.8 on every row logged",
   "refuted_if": "the stamps show weight units plus phase 0 make up < 30% of e1's chain at 1,28,28, and no weight-path variant moves any M row by more than 5%",
   "model": "opus", "fresh": false, "continue": "r1-a-e1-ts-fp16x3-m"},
  {"id": "E2", "lineage": "A", "hypothesis": "H19",
   "title": "Cluster-2 hidden-split pairs: two 128-token tiles of h per SM so GEMM1/GELU of tile i+1 overlaps the wait, pass 2 and GEMM2 of tile i (L)",
   "operation": "structural_mutation",
   "parents": ["r1-a-e4-rounds-tmem-v2"], "band": "L",
   "instructions": "Step 0 (5 min): stamp e4 at 64,56,56 and 16,56,56 (globaltimer per round: x converted, each GEMM1 block committed, GELU done, batch counter released, pass 2 done, GEMM2 committed, out stored, and the MMA warp's total wait on bar_full). Record the fraction of the round the MMA is idle. Step 1: build the pair variant. Use a persistent one-wave grid of 74 clusters of 2 (check cudaOccupancyMaxActiveClusters >= 74 with your exact smem and the cooperative + cluster launch attributes; H17 says 74 at >=180 KB). CTA rank r owns hidden [256r, 256r+256). Per 128-token tile each SM runs 1SM swap-AB GEMM1 (2 hidden blocks x 128 token columns = 256 TMEM columns), using e4's GELU/sumsq epilogue on its 256 channels. Two tiles are resident (columns 0-255 and 256-511). Use rounds of 9408 tokens (3x3136 = 12x784 = 48x196, about 73.5 tiles, batch-aligned) so that each pair runs GEMM1 of round k+1 before waiting on round k's per-batch counters; this is deadlock-free because GEMM1 never waits on a counter. After the wait, pass 2 converts the tile's own 256 channels; the GEMM2 partial (out^T: M=128 outs, N=128 tokens, K=256 own hidden), with main and cross accumulators placed in that tile's freed h columns as e4 does. Then exchange halves through DSMEM (each SM sends 64 tokens x 128 outs = 32 KB, mbarrier complete_tx), add b2' and store its 64 tokens. Each SM streams only its half of the weights (256 KB per tile); keep W1 resident instead if smem allows. Step 2: run_tests on all L rows plus 16,28,28 and 4,56,56, interleaved with e4 in one container, and log err/tol over 10 seeds. Fallback if the pair build does not pass in time: cross-round overlap in e4 (I22: next round's x into R0 and GEMM1 blocks 2/3 into columns 256-511 during GEMM2 hb3 and the output drain), measured the same way.",
   "success": "pair kernel passes all served rows with err/tol <= 0.8 and runs L rows >= 35% faster than e4 rented (64,56,56 round-equivalent <= 1.5x its own MMA+exchange projection)",
   "refuted_if": "step-0 stamps show the MMA idle < 25% of an e4 round (then overlap cannot pay), or the pair kernel at 64,56,56 is not >= 15% faster than e4 in the same container",
   "model": "opus", "fresh": false, "continue": "r1-a-e4-rounds-tmem-v2"},
  {"id": "E3", "lineage": "B", "hypothesis": "H21",
   "title": "Fan-out for the 6272-token rows (TT=196 x 4 slices) plus stamps of the 3136-token rows",
   "operation": "specialisation",
   "parents": ["r1-b-e2-fanout-v3"], "band": "M",
   "instructions": "Step 0 (precision gate): log err/tol per row over 10 seeds (fp64 reference) for the 4 rows B serves. Step 1: stamp the 4,28,28 path with globaltimer (x and weights landed, GEMM1 committed, GELU+sumsq done, barrier released, x2 split done, GEMM2 committed, red.add done). Attribute the ~5 us gap between 784 and 3136 tokens. Step 2: add a TT=196 instance for 6272 tokens (2,56,56 and 32,14,14): 32 token tiles x 4 hidden slices = 128 CTAs, one wave. smem: x hi/lo 100 KB + W1 slice hi/lo 64 KB; after GEMM1 commits, bulk-copy the W2 slice hi/lo over W1; the x2 split reuses the x buffer. TMEM: D1 = 208 columns; keep h in TMEM if registers do not fit (the TT=98 instance is already at 125 regs); run GEMM2 in two token halves of 104 with main and cross accumulators (208 columns) beside D1. Keep red.add v4 into the b2-prefilled output (H15: bulk reduce is slower). Check sm_100a regs and spills before running. Step 3: run_tests on all 14 rows against r1-merge in one container; route only the rows this kernel wins. Step 4, if time remains and step 1 shows GELU+split > 3 us at 3136 tokens: vectorise the x2 split (transposed TMEM read, one 32-bit STS per 2 tokens) and re-measure the 3136 rows.",
   "success": "2,56,56 and 32,14,14 <= 19 us rented (vs e1 25.3 / 28.3 in the same container), all served rows err/tol <= 0.8",
   "refuted_if": "TT=196 cannot fit without spills or a second wave, or both 6272 rows run > 23 us rented (under 10% better than e1)",
   "model": "opus", "fresh": false, "continue": "r1-b-e2-fanout-v3"},
  {"id": "E4", "lineage": "D", "hypothesis": "H20",
   "title": "New design: two-launch stored-h FP16x3 pipeline with resident weight halves (L rows)",
   "operation": "new_design",
   "parents": [], "band": "L",
   "instructions": "You may read the r0 baseline and read_example's #129 token-stationary tcgen05 kernel (SW128 descriptors, idesc kind::f16, elect.sync issue, packed A&S gelu2). Step 0 (hardware premise, 10 min): a memory-only skeleton under the harness flush (2xL2 zero_). K1 reads x and writes 2 KB/token; K2 is launched with PDL (griddepcontrol, same stream), reads that back and writes out; no maths. Time it at 64,56,56, 32,56,56, 16,56,56 and 32,28,28. Expect about 150 / 78 / 40 / 20 us rented. Step 1, K1 (persistent, 148 CTAs): each CTA owns a hidden half (CTA parity) and keeps its W1 half as SW128 hi/lo in smem (128 KB, built in the prologue from fp32 W1 while the first x tile loads). Per 128-token tile: x hi/lo, GEMM1 FP16x3 swap-AB into one of two TMEM accumulators (2 x 256 columns, double-buffered so the epilogue of tile i overlaps the MMA of tile i+1), then a gelu2 epilogue. Write per-tile per-channel sumsq partials with plain stores (two slots when a tile straddles a batch: no atomics, no zeroing, no state across calls), and store h (fp32, or fp16 hi/lo of unscaled h) at 2 KB/token through smem plus bulk stores. Step 2, K2 (PDL): its prologue reduces the partials of its batch(es) into G, nf and s. Per 128-token tile, choose between (a) folding s into a per-batch W2'_b out-half held resident in smem (rebuilt only when the CTA's batch changes; gb W2 + b2 as an fp32 bias), so K2 is a pure TMA->MMA GEMM on h hi/lo, and (b) loading fp32 h by TMA, with converter warps applying x2 = h*s + gb and the split. Use FP16x3 with a separate GEMM2 cross accumulator. A crude first version may use 1SM N=64 out-halves; note in findings whether cta_group::2 pairs (each SM loads half the operand) are the next step. Allocate the h workspace with torch.empty in the binding (no fills). The kernel must be correct for every shape; route only the rows where it wins. Step 3: run_tests on all rows interleaved with r1-merge, and log err/tol over 10 seeds for the L rows.",
   "success": "64,56,56 <= 170 us rented and every L row >= 10% faster than e4 in the same container, err/tol <= 0.8; first-session gate: the full pipeline within 1.4x of its own step-0 skeleton time at 64,56,56",
   "refuted_if": "kill line (hardware premise): the step-0 memory-only skeleton at 64,56,56 takes > 200 us rented, i.e. the stored-h stream cannot beat e4 by 20% even with free maths; otherwise build and measure the best configuration reached",
   "model": "opus", "fresh": false}
]
```