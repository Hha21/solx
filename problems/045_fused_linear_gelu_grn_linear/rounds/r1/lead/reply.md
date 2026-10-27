### Assessment
Every row scores 0.15–0.36. The only kernel we have runs exact fp32 SIMT with 13–49 CTAs, so all 14 rows are open. The model says cutting each band by about 93% gives 0.84. Round 1 should therefore get a working FP16x3 tcgen05 engine onto every row, with each band covered by more than one design so the per-workload merge has choices. The #129 lineage-M kernel already has the descriptors, counters and packed GELU, so we start from it.

What the audit changes:
- The leader stories are not evidence. The leader-gap per-row times come straight out of the equal-S formula, so the "HW=3136 penalty" and the "5 KB/token slope" are artefacts.
- leader_gap.py has a bug (GFLOP = 0). Tsol is about 0.334 ns/token.
- FP16x3 on real tensor cores passes, but with less margin than emulated: worst err/tol 0.78. A separate cross-term accumulator on GEMM2 brings it to 0.44, so every design must log err/tol and should use it.
- The epilogue binds. Exact GELU plus split is about 136 issue clocks per token against 96 MMA clocks. Recompute runs the GELU twice, so I park the recompute family.

Lineages and this round's experiments:
- **A: the hidden h stays on the SM.** It gets two experiments: E1, one round for the 9 rows with N ≤ 18,944, and E4, batch-aligned rounds for L. A has the most rows and they need different structures.
- **B: S-row fan-out over hidden slices (E2).** The critic's skeleton measured 6.8 µs without MMAs, which projects about 8–9 µs rented against the leader's ~10.5. It avoids the 512 KB per-CTA weight ingest that hurts token-stationary designs at S.
- **C: stored h at L (E3).** Each CTA keeps its own h in an L2-resident workspace ring and re-reads it one round later, so only the 512-float statistics cross CTAs. E3 starts with the L2-readback probe the critic did not get to run. Even if that probe fails, the DRAM fallback (about 5 KB/token) lands near 150 µs.

E3 and E4 put stored h and on-chip h against each other at L. E1 and E2 compete on the S rows.

```yaml lineages
lineages:
  - id: A
    thesis: The hidden h never leaves the SM. One cooperative tcgen05 FP16x3 launch, token-stationary, keeps fp32 h in TMEM across per-batch GRN counters; one round at S/M, rounds aligned to whole batches at L.
    status: active
    started: r1
    best: null
    conclusions:
      - Seeded r1 from pool I1/I2/I5/I24; engine template is #129 lineage M (SW128 descriptors, elect.sync issue, release/acquire counters with departure reset, packed A&S gelu2).
      - Precision (critic, hardware tl.dot) - single accumulator err/tol 0.78; a separate GEMM2 cross-term accumulator gives 0.44; GEMM2 dominates.
      - Known costs - 512 KB of hi/lo weights per CTA (about 4 us at 120 GB/s/SM bulk) hurts the S rows; TMEM is full at 128 tokens, so GEMM2's accumulators must reuse freed h columns.
  - id: B
    thesis: S rows are a latency chain, not a FLOP problem. Fan each token tile out over hidden slices so every SM ingests at most 64 KB of weights, cross one grid barrier, and reduce the split-K GEMM2 with red.add into an output prefilled with the bias.
    status: active
    started: r1
    best: null
    conclusions:
      - Seeded r1 from pool I23/I29/I18.
      - Critic skeleton at 1,28,28 (128 CTAs, rented) - empty 1.63 us, loads and stores 4.80, plus grid barrier 6.27, plus 16-way red.add 6.82; projects about 8-9 us rented with MMAs against the leader's ~10.5.
  - id: C
    thesis: At L the GRN dependency crosses through L2, not TMEM. A persistent single launch stores each CTA's h tile in a per-CTA workspace ring with evict_last and re-reads it one round later; only the 512-float statistics cross CTAs, and the wait hides behind the next round's GEMM1.
    status: active
    started: r1
    best: null
    conclusions:
      - Seeded r1 from pool I26/I12/I3 (stored h with L2 reuse); recompute (I25/I30) parked because it doubles the GELU, which already costs more issue than the MMA (H5, H8).
      - Even with h fully in DRAM the traffic is about 5 KB/token, about 0.72 ns/token or 145 us at 64,56,56, which is a leader-class floor.
```

```yaml ledger-changes
hypotheses:
  - id: H2
    statement: Split-precision tensor-core GEMMs (FP16x3 with fp32 accumulation) meet the tolerance with zero failing elements on real tcgen05 hardware; any 1- or 2-term scheme and BF16x3 fail.
    status: supported
    evidence:
      - 'probe_b200 emulation: 3xTF32 and FP16x3 max err 1.8-2.3e-3, 0% fail; BF16x3 0.8-2.2% fail; TF32 96% fail'
      - 'critic r1, hardware Triton tl.dot FP16x3 on both GEMMs, 7 shapes x 1-3 seeds: worst err/tol 0.78 single accumulator (cross-first ordering the same), 0.44 with a separate GEMM2 cross-term accumulator, 0.29 with a per-k-block fp32 add; GEMM2 alone 0.70, GEMM1 alone 0.25; 0 failing elements in 19 runs'
      - 'raw max err 5.7e-3 is above atol but err/tol is what counts, because rtol*|ref| grows with |ref|'
  - id: H3
    statement: FP16x3 on kind::f16 beats 3xTF32 on kind::tf32; FP16x3 is the engine for every lineage.
    status: supported
    evidence:
      - 'critic r1 hardware probe: TF32x3 max err 9.2e-3 (rounded hi) and 1.1e-2 (raw hi), against 5.7e-3 for FP16x3, so TF32 is also worse for accuracy'
      - 'rate (2x) not measured; it follows from the spec'
  - id: H4
    statement: For N <= 18944 tokens (all S and M rows) one cooperative launch that keeps fp32 h in TMEM across per-batch counters beats recompute and multi-launch designs; target 12544 tokens <= 21 us rented.
    status: open
    evidence:
      - 'arithmetic: 148 x (256 KB TMEM) ~ 37.9 MB vs h = N x 2 KB'
      - 'r1 E1 tests it'
  - id: H5
    statement: At L, recomputing GEMM1 beats storing h once GEMMs run on tensor cores.
    status: parked
    evidence:
      - 'arithmetic at 64,56,56: recompute ~+44-88 us of MMA vs store ~+100 us at 8 TB/s'
      - 'critic r1 SASS count: exact erff GELU is 1 MUFU + ~28 ALU per element, ~136 issue-clk/token with bias, sumsq and split, above the 96-clk FP16x3 MMA; recompute repeats this GELU, so it pays ~136 clk/token more, not +50% of MMA'
  - id: H8
    statement: In on-chip and stored-h designs the CUDA-core epilogue (bias, exact GELU, sumsq, hi/lo split, TMEM round trip) binds at L rather than the MMA, unless it is cut to about 10 issue slots per element.
    status: open
    evidence:
      - 'critic r1: erff 1 MUFU + ~28 ALU; normcdff 3 MUFU + ~60; erfcf 3 MUFU + ~45 (sm_100a SASS)'
      - '#129 K: co-running epilogue and MMA warps keep 54-70% of their solo rates'
      - 'r1 E3/E4 stamps (epilogue vs MMA per tile) test it'
  - id: H10
    statement: A per-CTA workspace ring of h (2 KB/token, at most 2 rounds, about 76 MB) written with evict_last and re-read by the same CTA one round later stays in L2 under the harness's dirty-L2 flush, so read-back runs above 10 TB/s aggregate and DRAM sees mostly x, out and the flush write-backs.
    status: open
    evidence:
      - 'critic r1: probe not run (budget)'
      - 'r1 E3 step 0 tests it'
  - id: H11
    statement: For N <= 6272, a hidden-slice fan-out (token tiles x hidden slices, at most 64 KB of weights per CTA, one grid barrier, red.add split-K) reaches 1,28,28 in <= 9 us rented and 3136-token rows in <= 11 us.
    status: open
    evidence:
      - 'critic r1 skeleton at 1,28,28, 128 CTAs, rented: empty 1.63 us, loads and stores 4.80, plus barrier 6.27, plus 16-way red.add 6.82'
      - 'r1 E2 tests it'
  - id: H12
    statement: For token-stationary designs at S, ingesting the 512 KB of hi/lo weight images per CTA (about 4 us at ~120 GB/s/SM) is the largest term on the S critical path.
    status: open
    evidence:
      - '#129 C: one-shot cp.async.bulk into smem after the flush runs at about 120 GB/s/SM'
      - 'r1 E1 phase stamps at 1,28,28 test it'
  - id: H13
    statement: At L, persistent rounds aligned to whole batches, with fp32 h in TMEM and per-batch counters, run at <= 15 us per 128-token round (about 165 us at 64,56,56) despite the full TMEM preventing overlap.
    status: open
    evidence:
      - 'arithmetic: 6 passes x 128x512x128 MAC / 4096 MAC/clk = 12.3k clk = 8.2 us of MMA per round, plus epilogue, wait and store'
      - 'r1 E4 tests it'
notes:
  - 'Leader-gap per-workload times are the equal-S formula''s output (t = Tsol + 0.171 (Tb - Tsol)); the HW=3136 "penalty" and the 0.723 ns/token slope are artefacts and say nothing about the leader''s design (critic r1 algebra).'
  - 'leader_gap.py has GFLOP = 0 for #45, so its "memory-bound, precision does not matter, ceiling 0.994" output is wrong. Tsol recovered from the portal anchors is 0.334 ns/token at L (67.45 us at 64,56,56).'
  - 'Precision ship gate: per row, max over 10 rounds of max |err| / (atol + rtol |ref|) <= 0.8 with the row''s own tolerance, logged in findings; never rely on the 99% matched ratio. Prefer a separate cross-term accumulator on GEMM2.'
  - 'Engine template: #129 lineage-M kernel (read_example solx_129_token_stationary_tcgen05.cu) - SW128 descriptors, idesc kind::f16, elect.sync issue, release/acquire counters with departure reset, packed A&S gelu2, cooperative launch.'
  - 'Portal noise: rows under ~40 us moved about +-1 us between identical submissions on #129. Merge thresholds: 3% projected gain for a lineage with portal history, 10% for a new one.'
```

```json tasks
[
  {"id": "E1", "lineage": "A", "hypothesis": "H4", "title": "Token-stationary one-launch FP16x3 tcgen05 kernel, fp32 h in TMEM, per-batch counters, for N <= 18944 (S+M rows)", "operation": "new_design",
   "parents": [], "band": "M",
   "instructions": "1) Start from the #129 lineage-M engine (read_example solx_129_token_stationary_tcgen05.cu): SW128 descriptors, kind::f16 idesc, elect.sync issue, release/acquire counters with departure reset, gelu2, cooperative launch at 1 CTA/SM, 512 threads. 2) Tokens per CTA t = ceil16(N/148) capped at 96 (12544 tokens -> 96 x 131 CTAs; 6272 -> 48; 3136 -> 32; 784/1568 -> 16). A CTA's tokens may straddle 2 batches (HW=196): keep two batch slots as in #129. 3) Phase 0: every CTA converts 1/G of W1 and W2 into exact fp16 hi/lo SW128 images in a torch.empty workspace, computes part of b2' = b2 + W2.gb, then bumps one counter. The x tile lands by bulk copy and is split hi/lo in registers into the B tile. 4) GEMM1 swap-AB D1[hidden 128-block][t]: 4 blocks, with the 3 products as K-concatenated passes; W1 hi/lo streamed in 64 KB blocks through a 2-stage ring. Epilogue: + b1, gelu2, per-channel sumsq, fp32 h back into TMEM with tcgen05.st, red.add into gsum[b][512], per-batch counter. 5) After the batch wait: s_b = 1 + gw*G/(mean G + eps) with exact sqrt and div; h' = h*s_b split hi/lo into smem per 128-k chunk; GEMM2 D2[out 128][t] with hi*hi in one accumulator and the cross terms in a second one placed in freed h columns (TMEM <= 512); W2 hi/lo streamed per 64 KB k-chunk. out = A + B + b2', stored coalesced [token][128] with evict_last. 6) Rows with N > 18944 route to the r0-simt-3k path unchanged. One launch per call; torch.empty is the only torch call in run(). 7) Add %globaltimer phase stamps (weights ready, x landed, GEMM1, epilogue, batch wait, GEMM2, store) at 1,28,28, 4,28,28, 2,56,56, 4,56,56 and 16,28,28, and a SASS count of epilogue instructions per element. 8) run_tests all 14 rows; log err/tol per row over 10 rounds. If the 12544-token rows miss the gate, measure the fallback configuration t = 64 with more CTAs (2 rounds) and ship the better one.",
   "success": "All 9 rows with N <= 18944 pass with err/tol <= 0.8; 12544-token rows <= 21 us rented, 6272 <= 15 us, 784 <= 13 us.",
   "refuted_if": "After the fallback configuration the 12544-token rows exceed 30 us rented, and the stamps show weight landing plus batch wait above 50% of the kernel time (a structural cost, not a bug).",
   "model": "opus", "fresh": false},
  {"id": "E2", "lineage": "B", "hypothesis": "H11", "title": "S-row hidden-slice fan-out: token tiles x hidden slices, one grid barrier, red.add split-K GEMM2, for N <= 6272", "operation": "new_design",
   "parents": [], "band": "S",
   "instructions": "1) Grid = token tiles x hidden slices, about 112-148 CTAs, one wave. Start from 784 = 7 tiles of 112 x 16 slices of 32; 1568 = 14 x 8 of 64 (or 16 x 8 at 98); 3136 = 28 x 4 of 128; 6272 = 56 x 2 of 256. Tune the slice count per shape (dispatch on shape only). 2) Each CTA bulk-loads its own fp32 slices W1[slice,:] and W2[:,slice] plus x, b1, gw, gb at entry and converts them to fp16 hi/lo SW128 smem images itself: no workspace and no weight counter. 3) GEMM1 FP16x3 (non-swapped M = 128 tokens with masked rows, N = slice, K = 128 as 3 K-concatenated products, or swap-AB if the slice is 128), + b1, gelu2 from #129, sumsq red.add into gsum[b][512]. Before the barrier, the slice-0 CTA of each tile stores b2' = b2 + W2.gb into its out rows: do not rely on the harness zero-fill. 4) One monotonic grid counter (the #129 form; departure reset). 5) Then s_b for its slice, x2' = h*s_b split hi/lo, GEMM2 partial D2[tokens][128 out] with K = slice (separate cross-term accumulator), red.global.add.v4.f32 into out. If the red.add tail exceeds 1.5 us at 16 slices, try 8 slices or a cluster DSMEM reduction (allowed: small one-wave grid). 6) Other rows route to r0-simt-3k. 7) Phase stamps at 1,28,28, 8,14,14, 4,28,28, 16,14,14, 2,56,56 and 32,14,14; err/tol log per row over 10 rounds; run_tests all 14.",
   "success": "The 6 rows with N <= 6272 pass with err/tol <= 0.8; 1,28,28 <= 9 us rented, 3136-token rows <= 11 us, 6272-token rows <= 14 us.",
   "refuted_if": "1,28,28 exceeds 12 us rented (1.5x the 8 us projection) after tuning the slice count, and the stamps attribute the excess to the chain (ingest, barrier, red.add), not to a bug.",
   "model": "opus", "fresh": false},
  {"id": "E3", "lineage": "C", "hypothesis": "H10", "title": "L: persistent single launch, h stored to a per-CTA L2 workspace ring with evict_last, re-read one round later; correct for all 14 rows", "operation": "new_design",
   "parents": [], "band": "L",
   "instructions": "0) Probe first, <= 3 minutes: after a harness-style 2xL2 zero-fill, 148 CTAs write 38 and then 76 MB with st.global.L2::cache_hint evict_last, cross a grid counter, then each CTA re-reads its own bytes (cp.async.bulk). Report read-back TB/s and total time against a DRAM-only control. 1) Kernel: 148 CTAs at 1 CTA/SM, cooperative launch, static batch-major schedule of 128-token tiles. Round r: GEMM1 FP16x3 of this CTA's tile (x by bulk copy and register split; W1 hi/lo images from a phase-0 conversion, as in #129) -> + b1, gelu2, sumsq red.add per batch, per-batch counter -> store h (fp32, or fp16 hi/lo, your choice; 2 KB/token) into this CTA's workspace slot (r mod 2) with evict_last. Then GEMM2 of the tile from round r-1: wait for its batches' counters (they should already be complete), re-land the CTA's own h tile from L2, h' = h*s_b split hi/lo into smem (or fold s_b into a per-batch W2'_b if you store hi/lo), GEMM2 with a separate cross-term accumulator, out + b2' stored with evict_last. Prefetch the next x tile behind GEMM2. Workspace from torch.empty; counters self-reset. 2) The same kernel must be correct for all 14 rows (S/M = one round plus a drain). 3) Stamps per round at 32,28,28, 16,56,56 and 64,56,56: GEMM1, epilogue, h store, wait, h re-land, GEMM2. SASS count per epilogue element. 4) run_tests all 14; err/tol per row over 10 rounds. If the probe refutes L2 residency, still ship the same structure (DRAM fallback, about 5 KB/token) and report its time.",
   "success": "All 14 rows pass with err/tol <= 0.8; 64,56,56 <= 165 us rented, 16,56,56 <= 45 us, 32,28,28 <= 30 us.",
   "refuted_if": "The probe's read-back runs under 10 TB/s aggregate (H10 refuted) and the kernel exceeds 200 us at 64,56,56 (1.5x its ~130 us projection).",
   "model": "opus", "fresh": false},
  {"id": "E4", "lineage": "A", "hypothesis": "H13", "title": "L: persistent batch-aligned rounds with fp32 h in TMEM and per-batch counters (on-chip h, no h in memory)", "operation": "new_design",
   "parents": [], "band": "L",
   "instructions": "1) Same engine family as E1 (#129 lineage-M template), but persistent: 148 CTAs at 1 CTA/SM, cooperative launch. A round holds whole batches only: HW=3136 -> 6 batches (18816 tokens, 128-token tiles with masking, ~147 CTAs busy); HW=784 -> 24 batches. Tiles that straddle batches keep two batch slots. Choose tile 128 (512 TMEM columns of h, GEMM2 accumulators in columns freed by converting h blocks to smem) or tile 96 (128 columns free); measure both if time allows. 2) Per round: GEMM1 FP16x3 (W1 hi/lo images from a phase-0 conversion and counter; 64 KB ring) -> + b1, gelu2, sumsq red.add into gsum[b][512] (each batch's slot zeroed before its round by a designated CTA, with ordering argued) -> per-batch counter -> s_b -> h' chunks split hi/lo into smem -> GEMM2 with a separate cross-term accumulator -> out + b2' stored with evict_last. Prefetch the next round's x tile during the wait and GEMM2. Counters monotonic per call with departure reset. 3) Rows with N <= 18944 may run as one round (a competitor to E1); all 14 rows must be correct. 4) Stamps per round at 32,28,28, 16,56,56 and 64,56,56 (GEMM1, epilogue, wait, GEMM2, store) and the MMA-warp idle fraction. 5) run_tests all 14; err/tol per row over 10 rounds.",
   "success": "All 14 rows pass with err/tol <= 0.8; <= 15 us per 128-token round, i.e. 64,56,56 <= 170 us rented and 32,28,28 <= 30 us.",
   "refuted_if": "After the fallback tile choice a round takes more than 22 us at 128 tokens (64,56,56 > 240 us rented), with the stamps showing serial epilogue plus wait as the cause.",
   "model": "opus", "fresh": false}
]
```