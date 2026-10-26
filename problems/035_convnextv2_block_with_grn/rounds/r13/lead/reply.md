### Assessment
We are 0.0084 behind first place. predict_score puts S −6 % / M −5 % / L −3 % at 0.7519, which would pass the leader, so this round spreads its work over the rows rather than chasing one. What r12 taught:

1. **H49 removed g but paid only in GEMM2** (GEMM2 phase −30..−55 %). In GEMM1 the image wait (~2 us) plus a second pass replaced the g pass at about equal cost. What is left on the wide-C rows is per-tile fixed cost on top of the 2-tile tails. These are the rows furthest from the leader (16,384 −23 %, 8,384 −19 %, 16,768 −16 %, 8,768 −15 %).
2. **H48 is refuted.** The sc weight chain was already hidden: removing it entirely changed the rows by +0.05 us on average. The sc rows' phase-A term is the depthwise conv, at 5–10× its FMA bound because too few warps run it.
3. **Lineage L hit a measured hardware limit.** Its tensor-core conv saturates the shared-memory port (98–138 clk per MMA in the kernel against 60 isolated), its tile cannot feed a fused GEMM1, and its two rows are worth at most +0.0015. L is retired under the incubation exception.
4. **Lineage I** has pass 1 at 547 us at 64,96, but the tail is not built, so no row ships yet.

Answers to the audit:
- **Launch mode (I10/I15):** the critic's probe shows cooperative launch costs nothing, so both are dropped.
- **Barrier layouts (I14):** a separate flag line and private flags are slower than one counter polled directly (1.21 us), so I14 is dropped. The skewed-arrival case is measured inside E2, once the new conv evens out the arrivals.
- **MMA pricing:** the critic's port-cap rule for small-N tiles goes into H31 and into E1's tile pricing.
- **W1′ in TMEM (I7):** parked as H57.
- **Polynomial and half2 erf (I27/I1):** they would each need a fresh 10-round precision log, so they get no slot.

This round:
- **E1 (D):** token-range GEMM1 tiles that may cross image boundaries, under the H49 counters, so each CTA holds one balanced tile. It starts with a 10-minute cache-hint test (I25, the critic's top untested idea).
- **E2 (D):** an all-warp strip conv on the sc rows, with coalesced stores and 512 threads, extended to the 14×14 tc conv, then the barrier A/B test under skewed arrival.
- **E3 (I, incubation):** ship the 64,96 and 32,128 rows.
- **E4 (new lineage M):** a token-stationary C=96 kernel with weights resident in smem, the hidden kept in TMEM and per-image counters, testing the S-row-specialist story.

```yaml lineages
lineages:
  - id: A
    thesis: Phase-separated pipeline of plain launches (bulk/TMA-staged dwconv+LN, cuBLASLt FP16 GEMMs, lean one-round elementwise kernels, fp16 materialised hidden) that wins by per-phase efficiency; the control lineage and carrier of conv/epilogue techniques.
    status: retired
    started: r1
    best: r5-a-e2-ringk2
    conclusions:
      - Retired r9. A 5-6 launch pipeline runs at portal ratio 1.40-1.55 on S/M rows against 1.20-1.32 for single launches.
      - Keep. Each custom launch is a 2-3 dependent-round chain (6-17 us), not a 2 us launch; launch count is a first-order axis on S/M rows.
      - Keep. K2 is FMA/INT issue of the exact erf, not stream-bound; only a GEMM-epilogue GELU removes it.
      - Keep. cuBLASLt FP16 1.4-2.1x faster than TF32; its GELU epilogue is the tanh form (rejected); swap-AB batched NCHW GEMM2 right at HW >= 784 only.
      - Keep. Bulk-plane conv and the TMA 3-D box ring are the conv carriers; every pick needs an interleaved same-container control.
  - id: B
    thesis: One cooperative launch with every phase fused (FP16 mma.sync, fp16 hidden in L2), aiming at S rows <= 24 us rented by cutting the per-phase chain between the two grid barriers.
    status: retired
    started: r1
    best: r3-coop-flat-bulkrows
    conclusions:
      - Retired r8; D's sc_kernel runs the same structure faster on the tcgen05 engine.
      - Keep. Single-launch coop beats a launch pipeline on the portal even where the bench says otherwise.
      - Keep. Barrier microbench r4 (flat counter 0.86-1.07 us); r13 critic - one monotonic counter polled directly is the cheapest form (1.21 us marginal), separate flag lines and private flags are slower.
      - Keep. Per-tile (H27) and image-level (H38) dataflows lost for per-item serial-chain reasons.
      - Pitfalls. runtime guards around mma.sync ~470 clk per k-block; register-direct erff epilogues 16 us per tile; a release arrive with loads in flight stalls.
  - id: C
    thesis: On-chip hidden with no global intermediates (hidden tile in shared memory across one grid barrier, weight slices ingested from L2, GEMM2 split-K reduced with red.add into a prefilled output).
    status: retired
    started: r1
    best: r2-e3-onchip-hidden-v2
    conclusions:
      - Retired r3. H13 (16-CTA clusters, DSMEM ~12 B/clk/SM) and H20 (mma.sync one-barrier on-chip hidden, 30.8 us at 1,192) refuted.
      - Keep. One-shot cp.async.bulk into smem after the flush ~120 GB/s/SM; converting inside the ring is 2-2.5x slower.
      - Keep. TFLOAT32 TMA maps round to nearest even (H14); red.global.add.f32 reaches 4.1-4.4 TB/s from 256-512 CTAs.
      - Keep. Cold __ldg of small vectors at point of use adds a DRAM round each; bulk-load them at entry.
      - r12 arithmetic - a tcgen05 on-chip-hidden revival for the wide-C rows dies on split-K reduction volume; the C=96 rows are re-tested by lineage M with resident weights.
  - id: D
    thesis: Cooperative single-launch specialist (bulk-plane conv, per-tile GRN statistics, split-K GEMM2) on a hand-written tcgen05 kind::f16 swap-AB engine; the six wide-C rows (tc_kernel) and the C <= 192 rows, 1,160 and 4,96 (sc_kernel).
    status: active
    started: r3
    best: r12-d-e2-grn-epi-counters
    conclusions:
      - Holds 12 rows in r12-merge (0.7425); H49 took 32,768 (-9 % rented, portal 77.4 -> 69.8); portal ratios 1.20-1.32.
      - r12 H49 - no-g pays in GEMM2 (phase -30..-55 %), not GEMM1 (pass 1 2-3.5 + wait ~2 + pass 2 1-2.5 us per tile); a perfect wait saves only 0.9-1.4 us; fp32 h parked in TMEM by tcgen05.st keeps STACK 0; counters stress-tested 400 calls.
      - r12 H48 refuted - the sc weight chain is already hidden; the sc phase-A term is the conv (2.8-10.5 us after landing, 5-10x FMA bound, 6 warps/SM); coalesced conv stores drop regs 168 -> 107 and gave -0.5..-1.4 us on 4 rows; more NS strips lose.
      - Remaining - wide-C per-tile fixed costs and 2-tile tails (r13 E1), sc conv occupancy and post-sync chain (r13 E2), phase-A W1 stream (no cheaper form found).
      - Recipes and dead ends - SW128 descriptors, all-warp 512-thr epilogue, staged coalesced stores, release/acquire + fence.proxy.async counters with departure reset, [B200] only, check STACK on every build; dead - lighter epilogue arithmetic, 2-deep tcgen05.ld, smem g, W2 prefetch, pairs at B <= 8, bulk weight pass, in-ring W1 conversion, W2 work inside the image wait.
  - id: E
    thesis: Library tcgen05 with fused epilogues (CUTLASS 4.4 sm100 builder GEMMs with EVT epilogues; TF32 operands straight from fp32 memory).
    status: retired
    started: r5
    best: null
    conclusions:
      - Retired r8 (H30 refuted). Per-image batching re-reads W1; TF32 doubles the bytes.
      - Keep. EVT nodes that instantiate on sm100 1Sm - ColBroadcast, RowBroadcast, multiply_add, GELU (erf), RowReduction non-final.
      - Keep. CUTLASS tfloat32_t on fp32 memory rounds to nearest; split-K as batch mode works on sm100.
      - Unresolved. CUPTI sequence-matching errors on multi-kernel library paths (14/16 twice).
  - id: F
    thesis: Library-lean four-launch pipeline for the HW<=784 rows (cluster-of-8 LN statistics, cuBLASLt FP16 GEMM1, one-launch GRN kernel, cuBLASLt swap-AB GEMM2).
    status: retired
    started: r6
    best: r6-f-e1-libpipe
    conclusions:
      - Retired r8 (H32 refuted); D beat F on every row it targeted.
      - Keep. cuBLASLt GELU epilogue is tanh (fails own-atol); folding gb into the fp16 hidden fails; exact erf on fp32 pre-activation and b2' = W2.gb + b2 required.
      - Corrected r13 - the '1.8 us plain-vs-cooperative' saving came from K3's multi-wave configuration; the critic's probe shows launch mode costs nothing (148x512x227 KB, 1.92-2.06 vs 1.90-2.08 us).
      - Keep. Cluster-of-8 LN exchange 0.4-2.4 us; L2 weight prefetch at entry no effect; conv and weight conversion serialise inside a CTA.
      - Precision log for all 9 F paths (0 beyond atol, 59-76 %).
  - id: G
    thesis: Fused pass-1 persistent kernel for the HW >= 1024 rows (CUDA-core conv + LN stats into an smem fp16 d tile as GEMM1 B, epilogue warps with LN fold, exact GELU, sumsq, fp16 hidden), then K3 + cuBLASLt GEMM2.
    status: paused
    started: r7
    best: r9-g-e3-tc05-p1s
    conclusions:
      - Holds 4 rows - 64,96 (1066 portal) and 32,128 (287) from r7 g_pass1; 8,320 (74.6) and 4,96 (46.5) from r9 g_tc; 4-launch pipeline at portal ratio 1.26-1.27.
      - r10 H42 refuted - custom GEMM2 serial chain before the first MMA; K3 + cuBLASLt G2 stay.
      - Remaining at 8,320 rented - K0 ~4 / P1 ~39 / K3 8.9 / G2 11.3 (3.8 TB/s); the epilogue is ~2x its issue bound.
      - Unbuilt revival - fold K3 into g_tc behind a counter barrier (pool I23, about +0.0023); top rows handed to lineage I.
      - Precision logged for all 4 shipped rows (61-82 % of atol).
  - id: H
    thesis: Image-counter dataflow in one persistent tcgen05 kernel (static work queue, per-image readiness counters instead of grid barriers).
    status: retired
    started: r8
    best: null
    conclusions:
      - Retired r10. H38 refuted (+58-98 % vs D); summed item CTA-time exceeds D's rows.
      - Keep. Queue mechanics cheap; counters self-reset through a departure counter; lint rejects __nanosleep.
      - Keep. CUTLASS-style release (bar.sync then red.release.gpu) + consumer ld.acquire + fence.proxy.async is correct for TMA readers.
      - Keep. Fully unrolled predicated load arrays; load JB=8 float4 before shuffles; scalar red.add volume scales with split-K.
      - Counter mechanics live on in D's H49 per-image wait.
  - id: I
    thesis: Top-row specialist (64,96 and 32,128) - pass-1 kernel with fixed warp roles, conv on the FMA pipe from TMA-staged x bands, tcgen05 GEMM1 against a streamed W1', epilogue warps for LN fold / A&S GELU / sumsq / h stores; GRN scale and GEMM2 by the notebook recipe.
    status: active
    started: r12
    best: r12-i-e3-tc05-pass1-547
    conclusions:
      - Ceiling about +0.003-0.005 of mean (dMean/dT 0.10 and 0.41 x1e-4 per us at 64,96 / 32,128).
      - r10 gate - one 3-D TMA box per stage overlaps copies; conv-only 233.8 us at 64,96 (2.2x FMA bound), 55.8 at 32,128.
      - r12 STEP 2 - tcgen05 pass 1 547 us at 64,96 (G r7 668), on a ~255 us skeleton bound by x-ring latency (3 x 30 KB in flight); the K0/K3/cuBLASLt tail is not built, so no row ships yet.
      - MUFU-heavy epilogue warps slow producer/MMA warps sharing their SMSP by 30-40 % (K r11); lineage L closed the tensor-pipe conv, so the conv stays on the FMA pipe.
      - r13 (incubation continuation) ships the tail on both rows with a 10-round precision log, then tries a deeper x ring.
  - id: J
    thesis: Cluster split-K cells for the low-M wide-C rows (2,768 and 8,768).
    status: paused
    started: r11
    best: null
    conclusions:
      - r11 gate failed on configuration (50 KB exchange at fan-in 4-6, 1.63 us); KS2 cells 0.54-0.66 us.
      - Value shrank - 2,768 S 0.737; KS2 fills only 48 CTAs at 2,768; H49 removed g from GEMM2.
      - r13 E1's token-range tiles fill the SMs at 2,768/8,768 without clusters; revisit only if they fail there.
  - id: K
    thesis: Persistent warp-specialised tcgen05 engine with two TMEM accumulators overlapping epilogue and mainloop on the multi-tile wide-C rows.
    status: paused
    started: r11
    best: null
    conclusions:
      - H47 refuted - co-running keeps 60-70 % (epilogue) and 54-65 % (MMA) of solo rates.
      - Limiter is issue/dispatch sharing inside each SMSP (TMEM lane-quadrant rule).
      - STEP 0a - straight-line 1SM issue 50 / 66 / 114 / 130 clk at N = 64 / 128 / 224 / 256; critic r13 - the small-N excess is the ~123 B/clk smem operand port.
      - Most of K's projected gain was balance, now tested directly by D r13 E1.
  - id: L
    thesis: Tensor-core Toeplitz depthwise conv for the HW >= 1024 rows.
    status: retired
    started: r12
    best: null
    conclusions:
      - Retired r13 in incubation on a measured hardware fact - MMA operand reads saturate the ~123-128 B/clk smem port (N=112 MMA 60 clk isolated, 98-138 in-kernel), so the pass ran 155 / 57.8 us against the <= 100 / 25 premise.
      - Keep. Toeplitz B without materialising T (reversed columns, LBO = SBO = 128 B), 16 KB for 7 tap rows; d16 within 1 fp16 ulp.
      - Keep. A Toeplitz tile is 128 rows of one channel, so it cannot feed a fused [C][token] GEMM1 tile; a separate pass costs a 154 MB d16 round trip.
      - Keep. Uncoalesced 56-B row stores cost ~50 us at 64,96; staged tile + cp.async.bulk S2G fixes it.
      - Value cap - pairs or A in TMEM at 32,128 are worth <= +0.0015 of mean.
  - id: M
    thesis: Token-stationary one-launch tcgen05 kernel for the C=96 rows (1,96, 2,96, 4,96; stretch 1,160) - each CTA owns 64-128 tokens, converts the whole fp32 W1/W2 into resident fp16 smem (no scratch, no sync 1), keeps the fp32 hidden in TMEM, waits only on per-image GRN counters, and writes NCHW out once.
    status: active
    started: r13
    best: null
    conclusions:
      - Seeded r13 from pool I22/I30 (S-row-specialist leader story) as the round's new design.
      - Differs from refuted H20 (r2 C - mma.sync, smem hidden, grid barrier, 30.8 us) by the tcgen05 engine, a TMEM-resident hidden, resident C=96 weights (144 KB fp16) and per-image counters instead of two grid barriers.
      - Ceiling about +0.011 of mean if 1,96 / 2,96 / 4,96 reach 13 / 14 / 26 us rented (D sc 18.2 / 25.0 / 37.6).
```

```yaml ledger-changes
hypotheses:
  - id: H31
    statement: 'On B200 a CTA fed from L2 through a TMA or 1-D bulk ring lands at most ~68-92 B/clk/SM in 1SM mode (SM-side cap), so designs must halve per-SM operand bytes (pairs, or operands produced in smem by the CTA) rather than multicast. A straight-line kind::f16 issue reaches the MMA ideal at N >= 128 in 1SM mode; below that the tensor core is bound by its ~123 B/clk shared-memory operand port, so MMA time per k16 is about max(N/2, (4096 + 32N)/123) clk with A in smem.'
    status: supported
    evidence:
      - 'r4 D gate: per 64-k stage 683 clk at N=224 (ideal 448), 587 at N=112, 524 at N=128 with K=3072; stage depth 2-3 vs 8 unchanged'
      - 'r6 D probe: 1SM 479 / 532 / 546 clk at N = 128 / 224 / 256; B multicast no gain; cta_group::2 pair 345 / 455 / 518'
      - 'r9 G: ~105 clk per instruction at N <= 128 in its loop (later shown to be loop overhead); 10-stage bulk ring cold 94 GB/s/SM from L2, 40 GB/s/SM distinct from DRAM'
      - 'r9 D E2: pair GEMM2 stages 658-875 ns vs ~270 ns ideal while producers gated; one producer warp per stage recovered it (-3 % at 32,768)'
      - 'r11 K STEP 0a: straight-line 1SM 50.2 / 66.4 / 114.4 / 130.4 clk per instruction at N = 64 / 128 / 224 / 256 at 1-148 CTAs; pair M=256 45.5 / 66.7 / 114.7 / 130.7'
      - 'r12 L: isolated N=112 60.4 clk, N=64 48.4; in-kernel 98-138 clk when staging/epilogue smem traffic shares the port'
      - 'r13 critic: r11 N=64 (6 KB per 50 clk), N=128 (8 KB per 66.4) and L N=112 (7.5 KB per 60.4) all read ~123 B/clk of operands, a port cap; small-N tile plans (r13 E1 N=24-136, lineage M N=64) must be priced with max(N/2, (4096+32N)/123)'
  - id: H43
    statement: 'A pass-1 kernel for 64,96 / 32,128 with fixed warp roles (TMA x bands, register-blocked FMA conv into the MN-major SW128 d tile, tcgen05 GEMM1 against a streamed W1'', epilogue warps with LN fold, A&S erfc GELU, per-(image,k) sumsq, staged fp16 h stores) plus the K0/K3/cuBLASLt swap-AB GEMM2 tail runs 64,96 <= 780 us and 32,128 <= 215 us rented (G r7 rows ~860 / 228), about +0.002-0.003 of mean.'
    status: open
    evidence:
      - 'r7 G ablations at 64,96: conv side alone 472 us, GEMM side 522; phases add at 1 CTA of 16 warps per SM'
      - 'r10 I gate: one 3-D TMA box per stage, conv-only 233.8 us at 64,96 (2.2x FMA bound), 55.8 at 32,128; edges correct within 1 fp16 ulp'
      - 'r11 K: MUFU-heavy epilogue warps cost the producer/MMA warps sharing their SMSP 30-40 %'
      - 'r12 I (r12-i-e3-tc05-pass1-547, as summarised in the r13 pool; the card exceeds the lead tool limit): tcgen05 pass 1 547 us at 64,96 on a ~255 us skeleton bound by x-ring latency (3 x 30 KB stages); tail not built, no row shipped. Projection pass 1 + K3 8 + G2 186 + K0 3 = ~744 us vs G ~860. Target lines reset to the tail-inclusive row (statement updated r13); r12 L closed the tensor-pipe conv route (H50)'
  - id: H48
    statement: 'In D''s sc_kernel the 2.6-3.1 us weight chain before sync 1 can leave the critical path through owner-converted resident weights: every sc row -1.5..-2.5 us rented.'
    status: refuted
    evidence:
      - 'r7 D stamps at 1,192: phase A 5.4 us = weights 2.6-3.1 then conv 2.5-3 (pre-r8 load order)'
      - 'r12 D E1 STEP 0 gate, interleaved x3: stub without any weight pass moved 1,192 / 2,96 / 4,192 / 1,160 / 8,192 by -0.47 / -0.06 / +0.42 / +0.02 / +0.32 us (mean +0.05 vs the -1.2 gate); stamps show the weight loads already overlap the plane copies and the conv (2.8-10.5 us after landing) is the phase-A term; the exposed conversion is 0.25-0.5 us'
  - id: H49
    statement: 'In D''s tc_kernel the g term can be removed by applying the GRN scale inside the GEMM1 epilogue behind per-image counters (fp32 h kept until the image''s gf is known, h'' stored once), so GEMM2 runs one accumulator over h''.'
    status: supported
    evidence:
      - 'r9 E2 v2 stamps: GEMM1 post-accumulator without the g pass -30..-35 % per tile'
      - 'r12 D E2 (r12-d-e2-grn-epi-counters) 16/16, rented vs r11-merge-b same GPU: 32,768 54.4 vs 59.6 (-8.8 %), 16,768 39.4 vs 40.6 (-3.1 %), 8,384 -0.6 %, 8,768 +0.7 %, 16,384 +0.7..+1.7 % (vs tc2), 2,768 +1.8..+3.0 %; GEMM2 phase 1.28 / 2.30 / 5.12 / 6.66 / 2.30 / 4.86 us vs parent 1.5 / 3.3 / 7.3 / 11.3 / 3.0; per-tile pass 1 / wait / pass 2 e.g. 16,384 3.55 / 1.95 / 2.32, 32,768 3.47 / 2.18 / 2.51 (max wait 10.7 us); a no-wait stub saves only 0.9-1.4 us. Row targets (>= 10 % at 32,768 and 16,768) missed; mechanism holds in GEMM2 only'
      - 'r12-merge portal 0.7425: 32,768 77.4 -> 69.8 us (S 0.715 -> 0.738). Precision log 57-67 % of atol, 0 beyond; 400-call counter stress clean; STACK 0 at REG 128 / 114'
  - id: H50
    statement: 'A Toeplitz tcgen05 formulation of the 7x7 depthwise conv runs <= 100 us rented at 64,96 and <= 25 us at 32,128 and can feed a fused pass 1.'
    status: refuted
    evidence:
      - 'r12 L gate: 155.3 us at 64,96 and 57.8 at 32,128 (60.6 with hi/lo taps); isolated N=112 MMA 60.4 clk but 98-138 clk in-kernel because operand reads (~125 B/clk) saturate the smem port and every staging/epilogue smem access comes out of MMA time; operand traffic ~28 B per output'
      - 'r12 L: the tile is 128 rows of one channel and cannot produce the [C][token] d tile a fused GEMM1 needs; a separate pass needs a 154 MB d16 round trip. Remaining levers (pairs -23 % bytes, A in TMEM at 32,128) are worth <= +0.0015 of mean; lineage L retired r13'
  - id: H51
    statement: 'A plain launch (with counter barriers) instead of cudaLaunchCooperativeKernel saves 1.5-2 us per D row.'
    status: refuted
    evidence:
      - 'r6 F note: plain launch saved 1.8 us for K3 (multi-wave 512-thread configuration)'
      - 'r13 critic probe, harness_time 148 x 512 thr x 227 KB, 5 interleaved rounds: empty body cooperative 1.92-2.06 vs plain 1.90-2.08 us; two barriers 5.60-5.66 vs 5.62-5.66 us; CUPTI shows only the kernel. Pool I10 worth 0; I15 keeps only its barrier part'
  - id: H52
    statement: 'On the sc rows, once phase A is evened out (H54), a single monotonic-counter grid barrier polled directly (target computed from the fetch_add return, no reset) beats cg grid.sync by >= 0.5 us per sync, reversing the r9 flat-barrier loss caused by skewed arrivals.'
    status: open
    evidence:
      - 'r9 D: flat counter barrier -0.1..-1.3 us on tc rows, +1.0..+2.1 us on all seven sc rows (skewed arrivals)'
      - 'r13 critic, simultaneous arrival, marginal cost per barrier: monotonic counter polled directly 1.21 us; relaxed atomic plus fences 1.29; separate flag line 1.80; count and generation with reset 1.94-2.02; private flags one fence 2.12; private release stores 4.0 (pool I14 refuted as a speed-up). Skewed arrival not measured; tested in r13 E2 after the conv rebuild'
  - id: H53
    statement: 'In D''s H49 tc_kernel, balanced token-range GEMM1 tiles that may straddle image boundaries (per-(image,k) sumsq split at boundaries and red.add-ed before the per-image publish; each tile waits on every image it touches) give each CTA about one tile on 144-148 CTAs: 16,384 12x12 ranges of ~261 tokens (N=264 as 136+128), 8,384 12x12 of ~131, 8,768 24x6 of ~65, 16,768 24x6 of ~131, 32,768 24x6 of ~261 or 24x12 of ~131. This removes the 2-tile tails and shrinks the per-tile epilogue where N falls: 16,384 -3..-5, 8,384 -1.5..-2.5, 32,768 -2..-3.5, 16,768 -1.5..-2, 8,768 -1 us rented, about +0.003-0.004 of mean.'
    status: open
    evidence:
      - 'r12 H49 stamps: per-tile pass 1 / wait / pass 2 3.55 / 1.95 / 2.32 us at 16,384 (192 tiles on 148 CTAs, 44 CTAs run 2), 3.46 / 1.79 / 2.43 at 8,384 (96 tiles, 52 CTAs idle), 32,768 max wait 10.7 us on CTAs whose group needs second tiles'
      - 'H41 (r10) balanced straddling tiles with a GEMM2 B-scale tied H49 at 16,384 (tc2 50.1 vs 50.5 rented): balance and no-g were never combined; the B-scale cost disappears under H49. Pool I18 / I20 / I13; leader story 1. Built as r13 E1'
  - id: H54
    statement: 'sc_kernel''s phase-A conv (2.8 / 5.9 / 6.2 / 4.1 / 10.5 us after the planes land at 1,192 / 2,96 / 4,192 / 1,160 / 8,192) is latency-bound at 6 warps per SM: row-strip items dealt over all warps of all 148 CTAs, taps in registers, register-blocked output segments, the coalesced d16/out store path and 512 threads (107 regs) bring it within ~2x of its per-SM LDS bound, -1.5..-5 us rented on the sc rows (about +0.0035-0.007 of mean); the same split applies to tc_kernel<384,196> if its conv exceeds 2 us after landing.'
    status: open
    evidence:
      - 'r12 D E1 stamps and the coalesced-store lead: regs 168 -> 107 at C=96/192, -0.5..-1.4 us on 4,96 / 1,160 / 2,96 / 8,192; NS sweep shows more halo copies lose, so the gain must come from more warps per plane copy. Pool I6 / I24 / I21 / I11. Built as r13 E2'
  - id: H55
    statement: 'A token-stationary one-launch tcgen05 kernel for the C=96 rows (64-128 tokens per CTA, whole fp32 W1/W2 converted per CTA into resident fp16 smem, conv straight into the d tile, fp32 hidden kept in TMEM, per-image GRN counters, GEMM2 in the same CTA, one NCHW store) runs 2,96 <= 17, 4,96 <= 28 and 1,96 <= 14 us rented, about +0.006-0.011 of mean.'
    status: open
    evidence:
      - 'D sc rented 1,96 18.2 / 2,96 25.0 / 4,96 37.6 with two grid syncs and an fp16 scratch round trip; W1+W2 at C=96 are 147 KB fp32 / 72 KB fp16 each, ~1.3-1.5 us of per-SM ingest at 94-120 GB/s/SM; H49 counters validated; TMEM 384 (GEMM1) + 128 (GEMM2) columns fit at 128 tokens. H20 (r2, mma.sync, smem hidden, grid barrier) refuted at 30.8 us - the engine and structure differ. Pool I22 / I30. Built as r13 E4 (lineage M)'
  - id: H56
    statement: 'L2 policy hints on D''s tc rows (createpolicy evict_first on one-pass reads - x plane bulk copies, fp32 W1/W2 loads; evict_last on in-call re-read writes - fp16 weight scratch, h'' stores, out red.add/bulk reduce) cut 0.5-1.5 us per wide-C row by steering evictions onto our clean lines instead of dirty flush lines.'
    status: open
    evidence:
      - 'tricks s3: supported on #84, #85, #88 (5-13 % at M/L), loads evict_first refuted on #38; on #035 hints were only tried inside the refuted H44 bulk pass. r13 critic: best expected value of the untested ideas (pool I2 / I17 / I25). Tested as r13 E1 STEP 0'
  - id: H57
    statement: 'Loading the tile''s W1'' slice into TMEM as the GEMM1 A operand cuts D''s 7x7 GEMM1 mainloop from ~3.4 to ~2 us.'
    status: parked
    evidence:
      - 'r13 critic arithmetic: the port bound for N=112 is ~244 clk per 64-k block, the measured ~550 clk equals 30 KB landed at 55 B/clk (the H31 TMA landing cap), and tcgen05.cp still lands A through smem; pool I7 worth about the 9 % port share. Not scheduled'
```

```json tasks
[{"id": "E1", "lineage": "D", "hypothesis": "H53", "title": "Balanced token-range GEMM1 tiles under the H49 image counters (one tile per CTA) on the wide-C rows, with a 10-minute cache-hint test first", "operation": "structural_mutation", "parents": ["r12-d-e2-grn-epi-counters"], "band": "M", "instructions": "You built H49 last round; continue from that tc-only source and the < 2 s str.replace build loop. STEP 0 (H56, cap 10 B200 minutes): compile-time HINTS variants of the shipped tc_kernel: (a) none, (b) createpolicy.fractional evict_first on the one-pass reads (x plane cp.async.bulk .L2::cache_hint, fp32 W1/W2 LDG via ld.global.L2::cache_hint), (c) b plus evict_last on the fp16 weight scratch stores, the h' stores and the out red.add / bulk reduce. Only our own buffers, no applypriority, no discard. harness_time interleaved x3 at 16,384, 8,768 and 32,768. Keep a variant only if it is >= 0.7 us faster on two rows and no row is > 0.3 us slower; record the numbers either way. STEP 1 (H53): replace the whole-image GEMM1 tile tables by static token-range tables where a tile = (hidden block, contiguous token range) that may straddle images. In the epilogue, split each thread's per-(image,k) sumsq at the static image boundaries of its range (the r10 tc2 code has this split), red.add the partials into the per-image gsum, publish on each image's counter (target = number of tiles touching that image, static per row), then wait on every image the tile touches before pass 2. Keep fp32 h in its TMEM columns as in r12. Tables to start from: 16,384 12 hidden x 12 ranges of 261-262 tokens (N=264 as two MMAs of 136+128, 144 CTAs, 1 tile each); 8,384 12 x 12 of 130-131 (N=136); 16,768 24 x 6 of 130-131; 8,768 24 x 6 of 65-66 (N=72); 32,768 24 x 6 of 261-262 (N=264) vs 24 x 12 of 130-131 (2 tiles per CTA, image-major so wave-1 groups complete in wave 1); 2,768 24 x 6 of 16-17 (N=24) as a stretch. Price each table with the H31 port rule max(N/2, (4096+32N)/123) clk per k16 before you build it. Deadlock freedom: every tile publishes before it waits, and with 2 tiles per CTA the first tiles of all CTAs cover whole image groups. STEP 2: stamp pass 1 / wait / pass 2 / GEMM1 end per row against the parent. Then re-tune the GEMM2 unit tables for 148 busy CTAs (KS/MP, the r12 sweep method). STEP 3: run_tests 16/16 including the portal-order re-runs; a fresh 10-round fp64 precision log on every changed row (the gf summation order changes); a 400-call NaN-prefilled counter stress at 16,384 and 32,768; cuobjdump STACK 0 on every instantiation. Compare each row against r12-merge's path for that row (16,384 is tc2, 32,768 is H49, the rest are r11 tc), interleaved x3 in one container. If the gate is missed, ship the best per-row table you measured with its numbers, never a placeholder.", "success": "Same container, interleaved x3 vs r12-merge: 16,384 <= -3 us AND 8,384 <= -1.5 us, AND at least one of 32,768 / 16,768 / 8,768 <= -1 us; 16/16 pass, 0 beyond-atol elements in the 10-round log, STACK 0", "refuted_if": "16,384 and 8,384 are both within 0.5 us of r12-merge with one-tile-per-CTA tables, i.e. the per-tile fixed costs (accumulator wait, image wait, publish) cancel the balance", "model": "opus", "fresh": false, "continue": "r12-d-e2-grn-epi-counters"},
 {"id": "E2", "lineage": "D", "hypothesis": "H54", "title": "All-warp strip conv with coalesced stores and 512 threads for sc_kernel (and the 14x14 tc conv), then the skewed-arrival barrier A/B", "operation": "structural_mutation", "parents": ["r12-d-e1-h48-gate-parent"], "band": "S", "instructions": "Continue from your r12 sc-only fast loop (7 variants in 5.6 s through ctypes). STEP 1: rebuild sc_kernel's phase-A conv. Split each zero-padded smem plane into row strips; a warp owns one strip of one plane and each lane computes a register-blocked output segment (1x4 or 1x7), with the 49 taps and dwb in registers. Deal the items round-robin over ALL warps of all 148 CTAs (28 CTAs are idle at 1,192 today). Keep the plane copies as they are: the r12 NS sweep showed more copies lose, so the gain must come from more warps per landed plane. Keep the per-output tap summation order unchanged so the roundings do not change. Combine with your coalesced store path (d16 staged in smem, lane-contiguous 16-B stores of d16 and of out = x + b2' from the landed plane). Stamp conv end minus planes landed at 1,192, 2,96, 4,192 and 8,192, interleaved x3 against the parent. STEP 2: with regs at ~107, build sc_kernel with __launch_bounds__(512,1) and run the GEMM1 epilogue and GEMM2 drain on all 16 warps (the tc_kernel all-warp pattern, H36). Check STACK 0 on <192,28,16>, <96,56,4> and <160,7,4>, and keep 512 threads only where it is faster. STEP 3: stamp tc_kernel<384,196>'s conv (8,384 / 16,384) after its planes land. If it is > 2 us, apply the same item split there and time it. STEP 4 (H52): on the rebuilt sc_kernel, A/B cg grid.sync against a single monotonic-counter barrier. All CTAs red.add one counter; the target is computed from the returned value as (old / N + 1) * N, there is no reset cell, and arrivals poll with ld.acquire on the same line; the counter must stay correct for every call (no other cross-call state). Interleave x3 on all 7 sc rows and record the release maxima. STEP 5: run_tests 16/16 with portal-order re-runs. Precision: if the conv arithmetic is unchanged, cite the r8 log plus a single-seed fp64 check per row; any change in rounding or summation means a fresh 10-round log (sc rows sit at 79-89 % of atol). Rows under 40 us move about 1 us between portal submissions, so judge by the interleaved rented deltas, and note that the merge needs 3 % projected gain on a row.", "success": "Conv after landing <= 1.6 us at 1,192 and <= 6 us at 8,192; at least 3 of {2,96, 4,192, 8,192, 4,96, 1,160} faster than r12-merge by >= 1.5 us rented in the same container (interleaved x3); 16/16 pass; STACK 0", "refuted_if": "The conv-after-landing segment shrinks by < 25 % at both 1,192 and 8,192 with all warps busy (then the conv is LDS/halo-bound, not occupancy-bound)", "model": "opus", "fresh": false, "continue": "r12-d-e1-h48-gate-parent"},
 {"id": "E3", "lineage": "I", "hypothesis": "H43", "title": "Ship lineage I's tcgen05 pass 1 as full 64,96 and 32,128 rows (K0 + pass 1 + K3 + cuBLASLt swap-AB G2) with the precision log", "operation": "structural_mutation", "parents": ["r12-i-e3-tc05-pass1-547"], "band": "L", "instructions": "Incubation continuation; build on your 547 us pass 1. STEP 1: attach the notebook tail that G r7/r8 shipped: K0 (weight prep and gsum zeroing in the same call), K3 (per-image fp16 W2_b = s_b * W2 plus b2' hi/lo columns, from red.add per-tile sums, __launch_bounds__(512,2), W2 row loaded up front) and the cuBLASLt batched swap-AB GEMM2 (beta 1 onto x, heuristic top-1). Make sure your pass 1 writes the K-augmentation rows that GEMM2 expects. STEP 2: instantiate the 32,128 configuration (4C=512). STEP 3: run_tests 16/16 with portal-order re-runs (only 64,96 and 32,128 routed to the new path, everything else on the r0 anchor as merge-only), plus a 10-round fp64 precision log on both rows (harness generators, seed 200; count the max |err| and the elements beyond each row's own atol). Wrap harness_time in a retry, because the CUPTI first-kernel drop happens on multi-launch paths. STEP 4, only if time remains (pool I28 skeleton test): a timing-only variant with W1' out of smem (garbage A) and NS = 5/6 x-ring stages against NS = 3 at 64,96. Report the skeleton time; go further only if it is <= 180 us. Do not hand in a placeholder: if a row loses, ship the other row and report both.", "success": "Same container vs r12-merge: 64,96 <= 780 us and 32,128 <= 215 us rented (G ~860 / ~228), 0 beyond-atol in the 10-round log on both rows, 16/16 pass", "refuted_if": "Neither row beats r12-merge by >= 5 % rented with the full tail (pass 1 gain eaten by the tail or the cuBLASLt G2 fallback)", "model": "opus", "fresh": false, "continue": "r12-i-e3-tc05-pass1-547"},
 {"id": "E4", "lineage": "M", "hypothesis": "H55", "title": "New design: token-stationary one-launch tcgen05 kernel for the C=96 rows with resident weights, TMEM-resident hidden and per-image GRN counters", "operation": "new_design", "parents": [], "band": "S", "instructions": "Build from scratch for 1,96,56,56, 2,96,56,56 and 4,96,56,56 (route only these rows; the r0 anchor serves the other rows as merge-only). Use the notebook recipes: H's engine recipe, D's SW128 descriptor values, G's writer address functions for the MN-major B tile, and H49's counters with departure reset. Validate the tcgen05 engine against torch before anything else. Structure: one cooperative launch. Each CTA owns a 64-token range (1,96: 49 CTAs; 2,96: 98) or a 128-token range (4,96: 98 CTAs); ranges may straddle images at 4,96. Entry: issue the x strip copies with their 3-row halos (3-D TMA box with OOB zero fill, start w = -4) and the fp32 W1 copy. Convert W1 to fp16 with the LN-weight fold, u and v row sums into resident K-major SW128 A (72 KB), with no scratch and no sync. Run the conv on all warps straight into the MN-major SW128 d tile, with the LN statistics per token. GEMM1 swap-AB: three 128-hidden M blocks x N tokens into TMEM (384 columns). Epilogue: LN fold, A&S erfc GELU, fp32 h back into TMEM (tcgen05.st), sumsq per (image,k) red.add, publish, and wait on the image counter(s). While waiting, land and convert fp32 W2 into resident fp16 A (72 KB, reusing W1's space). After the wait: h' = fp16(h * (1 + gw*gf*inv_b)) into an smem B tile (or into TMEM as the A operand, if you choose the token-major form of pool I30), then GEMM2 into the remaining 128 TMEM columns, and out = x + acc + b2' (b2' = W2.gb + b2 in fp32) stored NCHW with lanes along hw. The hidden is rounded once. STEP 0 gate: a stub with all data movement, MMAs and the counter wait but no epilogue maths, at 2,96 and 1,96 after the 252 MB flush, stamped per phase. Your projection is ~10-11 us at 2,96, and the gate is the stub within 1.5x of the projection for its part. Hardware premise to measure first: per-CTA ingest plus conversion of the 295 KB of fp32 weights at 98 co-resident CTAs <= 4 us. If it is above, switch to the fallback (grid-wide fp16 weight scratch plus one counter sync) instead of stopping. Price the N=64 MMAs with the H31 port rule. If the gate is missed, build and measure the best fallback configuration, never a placeholder. Finish with run_tests 16/16 (portal-order re-runs), a 10-round fp64 precision log on each shipped row, a 400-call counter stress, STACK 0, and an interleaved x3 timing against r12-merge (D sc). New lineages need a 10 % projected gain for the merge to take a row. Stretch only if everything above is done: 1,160,449,7 (410 KB of weights) behind the same gate.", "success": "2,96 <= 17 us and 4,96 <= 28 us rented in the same container (D sc 25.0 / 37.6), 1,96 <= 14 (18.2); 16/16 pass, 0 beyond-atol in the 10-round log, STACK 0", "refuted_if": "The full kernel is not faster than D's sc_kernel at both 2,96 and 4,96 after the gate stub met its projection (the chain, not the ingest, then binds a token-stationary design)", "model": "opus", "fresh": false}]
```