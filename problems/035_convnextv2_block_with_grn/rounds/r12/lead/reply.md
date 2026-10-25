### Assessment

Our best is r11-merge-b at 0.7404. First place is 0.7509, so we need about +0.0125. predict_score values a 10% cut on any single row at about +0.0012 to +0.0014. Rounds on the six wide-C rows now return only +0.002 to +0.005 each.

r11 settled two premises:
- **The ~105 clk per-instruction floor is not hardware.** When issued straight-line, 1SM tcgen05 reaches the MMA ideal at N ≥ 128 (K STEP 0a). This kills I4's rationale. E2 STEP 0 checks whether D's issue loop pays that overhead.
- **Overlapping epilogue and MMA does not pay.** When they co-run on one SM, each keeps only 60–70% of its rate, so H47 is refuted. J failed on its configuration, not on the idea, but what it can add is now small. H49 also changes the GEMM2 that J would reduce. J and K both pause in incubation.

Where the remaining score is, one experiment each:
1. **E1: the seven sc rows (H48).** Their 2.6–3.1 µs weight chain before sync 1 was deferred by directive, not because it was worth little. Cutting 2.5 µs (portal) from each is worth about +0.006, and a stub prices it within minutes.
2. **E2: the g term (H49).** g doubles GEMM2's B bytes and MMA width, and costs about a third of each GEMM1 tile epilogue. H41's B-scale replacement did not pay at 7x7. H49 instead applies the GRN scale in the GEMM1 epilogue behind per-image counters and stores h' once. The target is ≥10% on 32,768 and 16,768, the two largest cuts in the cheapest allocation, with gains also expected on 8,768 and the 14x14 rows.
3. **E3: the top rows.** They sit 4–5x above their pipe floors, and a 30–40% cut there is worth about +0.008. Lineage I resumes at STEP 2.
4. **E4: a new lineage L (new design).** It tests whether a tcgen05 Toeplitz conv can take the conv off the FMA pipe. If the conv gets to ≤100 µs at 64,96, pass 1 can approach its MUFU bound, and lineage I can use the result.

I tried to read D's r10/r11 kernel cards through get_kernel to see which r11 changes shipped, but the output was too large to open, so E2 STEP 0 reads them first.

This plan departs on purpose from the operator's preference for the five wide-C rows. The cheapest-allocation table is a cost model, and those rows now return the least per round. Only E2 aims at them, using the one structural lever left there that has not been tried. I also checked an on-chip hidden variant for those rows, and the arithmetic rules it out: 12–24-way split-K means 29–116 MB of red.add.

```yaml lineages
lineages:
  - id: A
    thesis: Phase-separated pipeline of plain launches (bulk/TMA-staged dwconv+LN, cuBLASLt FP16 GEMMs, lean one-round elementwise kernels, fp16 materialised hidden) that wins by per-phase efficiency; the control lineage and carrier of conv/epilogue techniques.
    status: retired
    started: r1
    best: r5-a-e2-ringk2
    conclusions:
      - Retired r9. Holds no row after r8-merge; a 5-6 launch pipeline runs at portal ratio 1.40-1.55 on S/M rows against 1.20-1.32 for single launches, so its ceiling is below D on every row.
      - Keep. Each custom launch is a 2-3 dependent-round chain (6-17 us), not a 2 us launch; launch count is a first-order axis on S/M rows.
      - Keep. K2 is not stream-bound (59 % of copy rate at 64,96); the ~100 us left is FMA/INT issue of the exact erf, so only a GEMM-epilogue GELU removes it.
      - Keep. cuBLASLt FP16 is 1.4-2.1x faster than TF32 everywhere; its GELU epilogue is the tanh form (rejected); swap-AB batched NCHW GEMM2 is right at HW >= 784 and 20-190 us slower at HW = 49/196.
      - Keep. Bulk-plane conv and the TMA 3-D box ring for W%4==0 rows are the conv carriers every lineage reuses; cross-host rented drift is 5-12 us, so every pick needs an interleaved same-container control.
  - id: B
    thesis: One cooperative launch with every phase fused (FP16 mma.sync, fp16 hidden in L2), aiming at S rows <= 24 us rented by cutting the per-phase chain between the two grid barriers.
    status: retired
    started: r1
    best: r3-coop-flat-bulkrows
    conclusions:
      - Retired r8. D's sc_kernel (same single-launch structure, tcgen05 engine) runs 1,192 at 19.4 and 1,96 at 19.1 us rented against B's 29.5-30.6.
      - Keep. Single-launch coop beats a launch pipeline on the portal even where the bench says otherwise (ratio 1.25-1.31 vs 1.40-1.55).
      - Keep. Barrier microbench (r4) flat atomic-counter barrier 0.86-1.07 us vs 1.6-1.9 us for the cg flip barrier; a two-level tree is worse (3.9 us); one flag polled by 295 CTAs is observed in 0.64 us.
      - Keep. Per-tile dataflow (H27) lost 1.4-2x for skeleton reasons, and the image-level version (H38) lost on per-item cost; counters are cheap but do not pay as a whole-kernel structure.
      - Pitfalls. runtime guards around mma.sync cost ~470 clk per k-block; register-direct erff epilogues take 16 us per tile; a release arrive with loads in flight stalls.
  - id: C
    thesis: On-chip hidden with no global intermediates (hidden tile in shared memory across one grid barrier, weight slices ingested from L2, GEMM2 split-K reduced with red.add into a prefilled output).
    status: retired
    started: r1
    best: r2-e3-onchip-hidden-v2
    conclusions:
      - Retired r3. H13 (16-CTA clusters, DSMEM ~12 B/clk/SM) and H20 (one-barrier on-chip hidden, 30.8 us at 1,192 vs 25 kill) refuted; no edge over a materialised fp16 hidden.
      - Keep. One-shot cp.async.bulk into smem after the flush runs at ~120 GB/s/SM; converting inside the ring is 2-2.5x slower.
      - Keep. TFLOAT32 TMA maps round to nearest even bit-exactly (H14); red.global.add.f32 into L2 reaches 4.1-4.4 TB/s from 256-512 CTAs.
      - Keep. Cold __ldg of small vectors at the point of use adds a DRAM round each; bulk-load them at entry.
      - r12 lead arithmetic - a tcgen05 on-chip-hidden revival for the wide-C rows dies on split-K reduction volume (12-24 x out bytes = 29-116 MB of red.add at 8,384 .. 32,768).
  - id: D
    thesis: Cooperative single-launch specialist (bulk-plane conv, per-tile GRN statistics, split-K GEMM2 with coalesced red.add) whose GEMM engine is hand-written tcgen05 kind::f16 in the swap-AB orientation (weights as M, tokens as N), so the output tile is NCHW; the six wide-C rows (tc_kernel) and the C <= 192 rows, 1,160 and 4,96 (sc_kernel).
    status: active
    started: r3
    best: r11-d-e1-g2-mp2-bulk
    conclusions:
      - Holds 12 of 16 rows in r11-merge-b (0.7404); portal ratios 1.20-1.32; M/L projections within +-2 %; S rows move +-1.2 us between unchanged submissions.
      - r11 - MP=2 (two W2 tiles sharing one [h|g] B stage) wins where a CTA runs many GEMM2 stages (2,768 -6 %, 32,768 -4 % rented); bulk-copy weight pass and in-ring fp32 W1 conversion refuted (H44); r10's balanced tc2 path kept only on 16,384; r10's main win was the STACK-120 fix (check STACK on every build).
      - Remaining terms per wide-C row - GEMM1 epilogue (issue + h/g store bytes, 1-2 serial tiles per CTA), the g term doubling GEMM2 B bytes and MMA N, the phase-A W1 stream (~2.5 us, no cheaper form found), barriers <= 0.5 us per sync; sc rows - 2.6-3.1 us weight chain before sync 1. r12 tests H49 (g removed via per-image counters) and H48 (sc owner-converted weights).
      - Validated and reusable - MN-major SW128 B (LBO 8192, SBO 1024, +2048 per k16), K-major SW128 A (SBO 1024, +32 B per k16), idesc bits, cta_group::2 recipe, one producer warp per stage round-robin, all-warp epilogue at 512 thr, staged coalesced h stores, target_hardware [B200] only; straight-line issue reaches the MMA ideal at N >= 128 (r11 K), so any engine near 105 clk per instruction has loop overhead. Precision tc 59-65 % of atol, sc 79-89 %.
      - Dead ends - weight warps split from conv warps; lighter epilogue arithmetic; 2-deep pipelined tcgen05.ld; g built in smem inside the 48 KB ring; L2 prefetch of W2 in the sync-1 window; pairs at B <= 8; reordering W1 loads vs conv; bulk weight pass; in-ring fp32 W1 conversion.
  - id: E
    thesis: Library tcgen05 with fused epilogues (CUTLASS 4.4 sm100 builder GEMMs whose EVT epilogues carry the LN fold, exact erf GELU, hidden store and per-image GRN sum of squares; TF32 operands straight from fp32 memory where the MMA is cheap).
    status: retired
    started: r5
    best: null
    conclusions:
      - Retired r8 (H30 refuted r5). Per-image batching re-reads W1 once per image and TF32 doubles the bytes; G1 with the full EVT was 17.0 us at 8,768 against D's whole row at 33.
      - Keep. EVT nodes that instantiate on sm100 TmaWarpSpecialized1Sm - Sm90ColBroadcast, Sm90RowBroadcast, Sm90Compute<multiply_add>, Sm90Compute<GELU> (erf form), Sm90RowReduction with FinalReduction=false; the erf GELU costs +2.4-8.5 us when the epilogue does not overlap a one-tile-per-CTA mainloop.
      - Keep. CUTLASS tfloat32_t on fp32 memory rounds to nearest (H14); split-K as the batch mode works on sm100; three .cu files build in 26-50 s wall.
      - Unresolved. run_tests 14/16 twice from CUPTI sequence-matching errors on changing rows; any multi-kernel library path must clear this before a portal slot.
  - id: F
    thesis: Library-lean four-launch pipeline for the HW<=784 rows (cluster-of-8 DSMEM LN statistics in the conv launch, cuBLASLt FP16 GEMM1, one-launch GRN statistics-and-scale kernel, cuBLASLt swap-AB batched GEMM2 writing NCHW) that wins by launch count.
    status: retired
    started: r6
    best: r6-f-e1-libpipe
    conclusions:
      - Retired r8 (H32 refuted) - D r6 beat F by 4-10 us on every wide-C row and D r7 sc_kernel beat F's best S row (1,192 19.4 vs 26.6 rented).
      - Keep. cuBLASLt's GELU epilogue is the tanh form and fails the own-atol bar; folding gb into the fp16 hidden also fails. The exact erf on an fp32 pre-activation and gb kept in b2' = W2.gb + b2 are required.
      - Keep. One-launch GRN kernel with a plain launch when grid <= residency - 64-bit red.add, ld.relaxed poll, departure counter resetting the cell; plain launch saves 1.8 us over cooperative.
      - Keep. Cluster-of-8 LN exchange costs 0.4-2.4 us; L2 weight prefetch at K1 entry had no measurable effect; conv and weight conversion serialise inside a CTA.
      - Precision log delivered for all 9 F paths (0 beyond atol, 59-76 % of atol).
  - id: G
    thesis: Fused pass-1 persistent kernel for the HW >= 1024 rows (64,96; 32,128; 8,320; 4,96) - CUDA-core warps stage x and run the 7x7 conv plus LN statistics into an smem fp16 d tile used as the GEMM1 B operand (swap-AB, W1' as A; tcgen05 at 8,320 / 4,96, mma.sync at the top two rows); epilogue warps do the LN fold, exact erf GELU, per-(image,k) sumsq and fp16 hidden store; GRN scale and GEMM2 follow in as few launches as possible.
    status: paused
    started: r7
    best: r9-g-e3-tc05-p1s
    conclusions:
      - Holds 4 rows in r11-merge-b - 64,96 (portal 1077, S 0.801) and 32,128 (288, 0.771) from r7 g_pass1 (mma.sync); 8,320 (74.9, 0.762) and 4,96 (47.0, 0.743) from r9 g_tc; 4-launch pipeline at portal ratio 1.26-1.27.
      - r10 H42 refuted - the custom tcgen05 GEMM2 g_g2 ran 23 us cold at 8,320 against the 9 us kill line (serial chain before the first MMA); K3 + cuBLASLt G2 stay.
      - r11 K STEP 0a - the ~105 clk per instruction G measured at N <= 128 is issue-loop overhead, not hardware (straight-line N=64 50 clk); g_tc's 200 MMAs per tile at 8,320 could cost about half, but the epilogue (~2x its issue bound) binds there. Not scheduled.
      - Remaining at 8,320 (rented) - K0 ~4 / P1 ~39 / K3 8.9 / cuBLASLt G2 11.3 (3.8 TB/s). Top rows hand over to lineage I (H43).
      - Paused since r11 (no slot); precision logged for all 4 shipped rows (61-82 % of atol).
  - id: H
    thesis: Image-counter dataflow in one persistent tcgen05 kernel for the B >= 8 wide-C rows - a static work queue claimed through an atomic head, each item gated on per-image readiness counters instead of grid barriers.
    status: retired
    started: r8
    best: null
    conclusions:
      - Retired r10 (incubation over). H38 refuted - +58-98 % vs D on all five rows; summed item CTA-time / 148 = 47 / 111 / 72 us at 8,768 / 32,768 / 16,384 against D's whole rows 34 / 64 / 58.
      - Keep. Queue mechanics are cheap (claim-to-start 1-2 us, inter-item gap 1.5-3.3 us, dependency spin 5-15 % of item time); counters self-reset through a departure counter; the lint rejects __nanosleep in spin loops.
      - Keep. Engine recipe validated on the first build and recorded in the notebook; CUTLASS-style release (bar.sync then red.release.gpu) plus consumer ld.acquire + fence.proxy.async is correct for TMA readers in other CTAs.
      - Keep. LN-statistics prestage must be a fully unrolled predicated load array; weight conversion must load JB=8 float4 into registers before any shuffle; scalar red.add volume scales with split-K (16K per item ~4 us).
      - Counter mechanics reused by D in r12 (H49) as a per-image wait inside the GEMM1 epilogue, not as a whole-kernel dataflow.
  - id: I
    thesis: Top-row specialist (64,96 and 32,128) - a pass-1 kernel with fixed warp roles that puts the conv on the FMA pipe from TMA-staged x row bands (register-blocked patches, taps in registers), the GEMM on tcgen05 against a streamed W1', and the LN fold / A&S GELU / sumsq / h stores on epilogue warps, so pass 1 runs within 1.5x of the SM pipe bound; GRN scale and GEMM2 follow the notebook recipe.
    status: active
    started: r10
    best: null
    conclusions:
      - Seeded r10 from pool I18 (I3 fallback). Ceiling about +0.008 of mean if both rows drop 30-40 %; the two rows have the largest absolute S headroom on the board.
      - r10 gate - per-row 1-D bulk copies are issue-bound (copy-only 578 us); one 3-D TMA box per stage (OOB zero fill for the halo) overlaps fully - conv-only 233.8 us at 64,96 (2.2x the 102 us FMA bound), 55.8 us at 32,128; +12 % at the 128-reg cap.
      - The FMA pipe binds at 64,96 (conv + GELU FMA + stats ~200 us at 1.95 GHz) ahead of MUFU (133) and DRAM (~140); r11 K - MUFU-heavy epilogue warps slow the producer/MMA warps sharing their SMSP by 30-40 %.
      - Resumed r12 (incubation clock restarts; paused r11 by directive) - STEP 2 (tcgen05 GEMM1 + epilogue warps) and the K0/K3/cuBLASLt tail as E3; projection 480-650 us rented at 64,96.
      - If lineage L's Toeplitz conv passes its gate, the conv producer of this pass can move to the tensor pipe.
  - id: J
    thesis: Cluster split-K cells for the low-M wide-C rows (2,768 and 8,768) - both GEMMs split over K across a thread-block cluster so 144 CTAs work instead of 24-96, partials reduced inside the cluster before the GELU (GEMM1) and before the residual store (GEMM2).
    status: paused
    started: r11
    best: null
    conclusions:
      - r11 gate failed on configuration, not idea - a 50 KB exchange at fan-in 4-6 took 1.63 us (> 1.5 line); its own probes show KS2 cells (cluster of 2, 148 co-resident) at 0.54-0.66 us and fp16-packed partials under 1.5 us.
      - Value shrank - 2,768 is no longer in the cheapest allocation (S 0.737) and 8,768 needs -9.7 %; KS2 gives 48 GEMM1 CTAs at 2,768 and 192 > 148 at 8,768 with 98-token tiles.
      - Paused r12 in incubation (not retirable); revisit after H49, which removes g and changes the GEMM2 that J would reduce.
  - id: K
    thesis: Persistent warp-specialised tcgen05 engine for the multi-tile wide-C rows (32,768, 16,768, 16,384, stretch 8,384) - fixed TMA / MMA / epilogue warp roles and two TMEM accumulators so the epilogue of tile i runs under the mainloop of tile i+1.
    status: paused
    started: r11
    best: null
    conclusions:
      - H47 refuted - co-running keeps 60-70 % (epilogue) and 54-65 % (MMA) of solo rates in every store variant; overlap saves 16-24 % of serial work; projection 21 us vs the <= 19.6 us line at 32,768.
      - Limiter is issue/dispatch sharing inside each SMSP - the TMEM lane-quadrant rule forces epilogue warps onto all 4 schedulers, so MUFU work slows the MMA warp's per-stage hand-off by 265-355 clk.
      - STEP 0a KEY - straight-line 1SM issue 50 / 66 / 114 / 130 clk at N = 64 / 128 / 224 / 256 (ideal from N=128); pairs add per-SM rate only at N=64.
      - For a 2-tile CTA the overlap hides only one mainloop (~-1.3 us at 32,768); most of K's projected gain was balance (H41 territory).
      - Paused r12 in incubation (not retirable). Revival levers - MUFU-free overlapped epilogue, one epilogue warp per SMSP, fewer heavier MMA stages from resident slices.
  - id: L
    thesis: Tensor-core depthwise conv for the HW >= 1024 rows - a Toeplitz formulation (per channel, 7 tap-row MMAs of shifted x rows against banded weight matrices, x landed by a TFLOAT32 3-D TMA box into kind::tf32 or fp16 hi/lo) so the conv runs on the tensor pipe at ~1.6-1.8x faster than its FMA bound and leaves the FMA pipe to the GELU.
    status: active
    started: r12
    best: null
    conclusions:
      - Seeded r12 from pool I15 (new_design). Arithmetic - ~9-10x FLOP inflation against 16x the FMA rate (kind::tf32) gives an MMA-bound conv of ~60 us at 64,96 rented vs I's 233.8 us FMA conv.
      - Gate risks - row shifts that are not multiples of the 8-row swizzle atom (descriptor base offset), T construction per channel, and the precision of TF32 x and taps on top of the d16 rounding (simulate first).
```

```yaml ledger
# Hypothesis ledger for #035 (035_convnextv2_block_with_grn). Seeded 2026-10-11 (research phase r0); updated r12 2026-10-25.
# status: open | supported | refuted | parked. evidence cites kernels, portal submissions, probes or findings.
constraints:
- 'No discard/invalidate of cache lines (discard.global.L2 etc.), even on our own inputs: it targets the harness''s measurement, not kernel speed. Off-limits unless NVIDIA approves.'
- Never touch memory we do not own; no state carried between calls except a scratch counter that is correct for every call.
- 'No overlap with the harness''s own kernels (no PDL against the flush memset, no other streams): it hides work from the timed window. No programmatic dependent launch at all, even between our own kernels: the lint rejects griddepcontrol and the PDL launch attributes, so an idea that relies on PDL must be built without it.'
- 'Clusters only where a GEMM tile needs them (2SM tcgen05 pairs, split-K cells); inherited from #38: clustered row-wise grids lost 2-5% on the portal and seemed to slow later plain launches.'
- 'Eviction-priority operations (createpolicy, applypriority.L2::evict_normal) only on the current call''s own input/output buffers, inside our kernel. Any candidate that uses applypriority goes to the operator for a legitimacy check before a portal slot.'
- 'Precision: FP16 and TF32 operands with fp32 accumulation pass every workload with ~5 sigma margin (simulated with the harness generators); BF16 fails. Any reduced-precision candidate must be re-simulated or run_tests 16/16 before a slot; the erf GELU must stay exact (no tanh approximation).'
- 'Weights are converted/transposed inside the call (no cross-call cache of fp16 weights or descriptors; pointers shift every call).'
- 'Workspaces come from torch.empty inside run() (no kernel launched); a cooperative launch counts as one kernel. Grid barriers via cg::grid_group or a self-resetting counter only.'
- 'Host state shared across shapes (function attributes such as max dynamic shared memory, cached plans, handles) must be right for every shape in any order: the portal sets up every workload before it times them (#65570: per-shape cudaFuncSetAttribute shrank the limit for earlier shapes, code 101 on 8 workloads). Set such attributes to the maximum any shape needs, or only ever raise them. run_tests re-runs every workload after all have run once and reports a PORTAL-ORDER CHECK failure.'
notes:
- 'Report effects per workload (or at least C=768 rows vs others), not per MB band: the MB bands mix weight-bound (C=768 7x7), latency-bound and stream-bound rows. Card fix pending: the MB cell for 64,96,112,112 is 616.9 MB, not "118.0 G".'
- 'Where the score is (r12; r11-merge-b #73053 = 0.7404, projection 0.7408; live first place 0.750880, target 0.7529 = +0.0125 of mean). Per-row S, lowest first (portal us): 16,384 0.699 (64.5); 8,384 0.705 (44.2); 32,768 0.715 (77.4); 16,768 0.717 (51.9); 8,768 0.720 (38.6); 8,192 0.729 (54.2); 4,192 0.734 (37.3); 2,768 0.737 (28.3); 1,160 0.738 (33.3); 2,96 0.741 (32.8); 4,96 0.743 (47.0); 1,192 0.746 (23.5); 8,320 0.762 (74.9); 32,128 0.771 (288.3); 1,96 0.788 (23.7); 64,96 0.801 (1077.0). A 10 % cut on any one row is worth +0.0011-0.0014 of mean (predict_score bands - S +0.0062, M +0.0068, L +0.0079 per 10 %). SOL TARGETS cheapest allocation - 32,768 -22.6 %, 16,384 -20.7 %, 16,768 -16 %, 8,384 -14.8 %, 8,768 -9.7 %; a uniform 6 % on all 16 rows also reaches the target. Round gains have shrunk (+0.011, +0.005, +0.003, +0.002 for r8-r11), so r12 spreads over the sc rows (H48), the g term on the wide-C rows (H49), the top rows (H43) and a tensor-core conv gate (H50).'
- 'Portal/rented ratio (audit r4, extended r6-r10, replaces the uniform 1.28): per path, per row AND per launch structure. On S/M rows the 5-6 launch A pipeline runs at 1.40-1.55x its rented time, the single-launch cooperative B/D kernels at 1.20-1.32x (D sc_kernel 1.20-1.27, D tc_kernel 1.20-1.32, G''s 4-launch pipeline 1.26-1.27). Rented hosts differ in launch speed and the portal behaves like the slow host. Rule: each launch removed on an S/M row is worth about 1-1.5 us more on the portal than the bench shows; judge S/M designs by launch count as well as rented time; project each path with its own lineage ratio (r7-merge projected 0.7208 vs 0.7207 measured; r8-merge 0.7333 vs 0.7313; r9-merge 0.7351 vs 0.7359; r10-merge 0.7392 vs 0.7392; r11-merge-b 0.7408 vs 0.7404).'
- 'Audit r10 answers. (1) Zero-conversion TF32 for the C=768 rows (I1/I20, H14): the critic''s arithmetic is accepted and the ledger''s r9 "next lever" line is withdrawn: only the 24 / 96 tile owners would read W1, 393 KB fp32 per tile at 40-94 GB/s/SM = 4-10 us against the 2.5 us grid-wide pass, and TF32 doubles the MMA count against a per-instruction floor; parked without a slot (H14 evidence updated). (2) The ~105 clk per-instruction floor at N <= 128 (G r9): the throughput number is measured and stands for 1SM operands in smem; the A-slice-fetch mechanism is unsupported (the pair probe did 128x128x16 of work per SM in 86 clk). (3) Phase A at C=768 as a serial copy chain with the conv at 15x its bound: wrong, corrected in H2, H18, H39 and lineage D. (4) S-row worth: corrected in the "Where the score is" note. (5) Flat self-resetting barrier safety on tc_kernel: scheduled as STEP 0 of r10 E1.'
- 'Audit r11 answers (same five items re-raised). (1) TF32 zero-conversion: stands parked; the same arithmetic (fp32 A landing doubles per-tile A bytes, first-touch DRAM at 40 GB/s/SM distinct) now gates the in-ring W1 variant of H44, which is built only where >= 4 token tiles share a W1 slice. (2) ~105 clk floor mechanism: the number stands; the critic''s straight-line test (64 unrolled MMAs, precomputed descriptors, N = 64/128/224/256, 1 vs 128 CTAs, 1SM M=128 vs pair M=256) is STEP 0a of E4 (H47), because K''s tile-N and pair choices rest on it. (3) Phase-A premise: already corrected in r10 (H2, H18, H39); stands. (4) S-row worth: already corrected in r10; S rows are deferred in r11 by operator directive, not by worth. (5) Flat-barrier safety: whether r10 E1 ran the stress loop is not visible to the lead (r10 card unreadable through the tool limit); E1 r11 STEP 0 runs it (1500 calls at 8,768 and 16,384 against the torch reference, protocol check of release/acquire + fence.proxy.async before TMA reads of other CTAs'' generic writes, cuobjdump STACK on every instantiation) unless the parent card records it done; on any failure tc_kernel reverts to cg grid.sync.'
- 'Audit r12 answers (same five items). (1) TF32 zero-conversion: stands parked for the C=768 rows; at the sc rows the owner-converted fp16 form is built first (H48, E1) because it keeps the logged roundings; TF32 sc (I20) stays in the pool as the alternative; lineage L uses TF32 only for the conv operands, after a fresh simulation. (2) ~105 clk floor: RESOLVED by r11 K STEP 0a - straight-line issue gives 50 / 66 / 114 / 130 clk at N = 64 / 128 / 224 / 256 (1.57 / 1.04 / 1.02 / 1.02x of ideal) at 1-148 CTAs; the floor clause of H31 is withdrawn and G''s number is issue-loop overhead; E2 STEP 0 measures D''s clk per instruction against this table. (3) Phase-A premise: corrected r10; stands. (4) S-row worth: corrected r10; S rows get E1 this round. (5) Flat-barrier safety: the lead still cannot read the r10/r11 D cards (get_kernel output above the tool limit); E2 STEP 0 checks the parent card and runs the 1500-call stress loop at 8,768 and 16,384 if it is not recorded; any failure reverts tc_kernel to cg grid.sync.'
- 'Pool families (dedup r12; the pool is r10''s, reused). GRN scale inside the GEMM1 epilogue behind per-image counters, no g (lead r12, from I9/H41 no-g plus H38 counter mechanics): E2 = H49. Owner-converted sc weights (I17): E1 = H48; TF32 sc_kernel (I20): alternative, unbuilt. Fused top rows at pipe bounds (I18, I3): lineage I, E3 = H43. Tensor-core Toeplitz conv (I15): new lineage L, E4 = H50. Cluster split-K cells (I13, I2): lineage J paused, H46 parked. Persistent overlapped engine: lineage K paused, H47 refuted. GEMM2 rebuild (I11 bulk reduce, I19 one-MMA [h|g]): H45, MP=2 shipped. Weight path off the critical path (I7 variant B, I14, I16): H44 refuted. Balanced image-straddling tiles (I5, I9): H41 parked (kept on 16,384). G fusion / custom GEMM2 (I6, I8): H42 refuted. Zero-conversion TF32 for C=768 (I1): parked. TMEM-sourced A (I4): its premise (the 105 clk floor) refuted by r11 K, parked. Cache hints (I10): only tested inside the refuted bulk weight pass; open in isolation. Right-sized S grids (I12): noise-level. On-chip hidden with tcgen05 for the wide-C rows (leader story): killed by arithmetic (split-K reduction 12-24 x out bytes). Balanced erf (I30): refuted by arithmetic at 64,96. Dataflow/counters as whole-kernel structure (H27, H38): refuted. Library EVT (E), library-lean (F), on-chip hidden (C): retired.'
- 'Precision logging (audit r2, tightened r3, delivered r5-r10 for every shipped path): the margin is about the MAX error over 1e7-1e8 elements x 10 rounds, not the rms. Every run_tests records per workload the max |err| and the count of elements beyond that row''s OWN atol + 1e-5|ref| over the harness''s 10 seeded rounds (seed 200) against an fp64 reference, for every path the lineage ships. A row with any beyond-atol element switches the cheap operand to a hi/lo fp16 split (2 MMAs, free at S/M). Passing through the 98 % ratio is not acceptable. A path whose arithmetic (roundings, summation order of a reduction) is unchanged may cite the previous log plus a single-seed check. A path that rounds the hidden twice (fp16 h then fp16(h*s), the sc_kernel class at 79-89 %) must be logged afresh on every row it ships; the cuBLASLt GELU epilogue is the tanh form and fails this bar; the A&S erfc form (1.2e-6 in h) is accepted as exact. New or resumed lineages (I, L) and new arithmetic in D (H48 u-sum order, H49 h'' rounding) log every shipped row over 10 rounds.'
- 'Cross-session rented timings drift by ~6 us per call and across Modal hosts by 5-20 us on unchanged rows. Every experiment times its parent (or r11-merge-b for new lineages) interleaved x3 in its own container. Rented within-noise picks lose on the portal (r3 merge at 8,768); the merge step times candidates side by side in one container and projects with per-lineage ratios.'
- 'Portal #65570 per-row S for lineage A v3 (8 rows that ran): 1,192 0.61; 1,96 0.63; 2,96 0.59; 4,192 0.60; 4,96 0.65; 8,192 0.63; 32,128 0.62; 64,96 0.65.'
- 'Audit r3/r4 on tcgen05 (H23): the ideal GEMM savings on the four GEMM-heavy rows are worth about +0.010 of mean, +0.005 against cuBLASLt; but tcgen05 is also the enabler for fused epilogues, because the custom mma.sync engine lost 4x to cuBLASLt (H26). Routes: D''s hand-written engine (H28, H34, H36 supported; H33 partly), G''s fused pass (H35 open, H40 supported, H42 refuted), E (H30 refuted), F (H32 refuted), H (H38 refuted), J (H46 parked), K (H47 refuted), I (H43 open), L (H50 open).'
- 'Cross-problem code: a design session may read another problem''s archive kernel as a structure reference (the #35 tc_ws tcgen05 ring, tricks s5); within this problem lineages still share conclusions only. get_kernel on D''s and the merge kernels returns 100-685 KB, above the tool limit; sessions should request the kernel''s card and sources in parts or rely on the notebook recipes (H''s r8 engine recipe, G''s r9 writer address functions and D''s descriptor values are complete). A lineage that builds a tcgen05 engine validates it against torch before anything else.'
- 'Per-SM operand ingest (H31, corrected r6, extended r9, corrected r12): a tcgen05 tile fed from L2 through a TMA ring lands 68-92 B/clk/SM in 1SM mode, the cap is SM-side, and a cta_group::2 pair halves per-SM B bytes. r11 K STEP 0a: a straight-line kind::f16 issue reaches the MMA ideal at N >= 128 in 1SM mode (N=64 1.57x); pairs add instruction rate only at N=64; the ~105 clk per instruction measured in G''s engine (r9) is loop/issue overhead. r7 sc_kernel: the GEMM2 B-tile scale costs 0.3-0.45 us per 64-k block (TMA landing, not arithmetic). r8 D: a 4-stage ring of 48 KB stages at 7x7 is latency-bound (~0.7 us per stage for two N-tile MMAs); work inserted between landing and MMA extends every stage unless the per-stage fence + arrive are pipelined (r9 E2 v3: one producer warp owns a whole stage, stages round-robin over warps).'
- 'r6-r10 engineering lessons: (a) a per-thread array the compiler cannot keep in registers becomes a local stack frame (cuobjdump STACK); a stamp build alone can push a 512-thread epilogue over the 128-reg cap, so stamp numbers are relative; r10: the shipped r9 tc_kernel<768,49> had STACK 120 and removing it was the round''s main win, so every session checks STACK on its shipped build. (b) globaltimer stamps from the warp that does the work. (c) Coalesce 16-B hidden stores through a per-warp smem transpose (D r6, G r9: 8,320 row -4.2 us). (d) target_hardware [B200] only for tcgen05 solutions. (e) torch extension builds define __CUDA_NO_HALF_CONVERSIONS__: use __half2float. (f) Weight conversion first on all warps, then the conv; reordering loads vs conv ties because phase A is bandwidth-bound. (g) Register stats over plane subgroups beat per-element post-conv passes. (h) A token-permuted d layout lets each lane store 8 consecutive tokens per hidden row with one STG.128. (i) Small vectors via __ldg from a K0-padded buffer or by bulk copy with the planes. (j) Fully unrolled predicated load arrays, never runtime-bounded unrolled loops. (k) ptxas forbids mixing cta_group::1 and ::2 tcgen05 ops in one kernel. (l) Pass every file''s content to compile_b200/run_tests (the paths field does not pull archive files). (m) TMA box inner start must be a multiple of 16 B; one 3-D box per stage with OOB zero fill beats per-row 1-D copies (r10 I: copy issue of ~31 ns per small copy binds). (n) r11 K: gate builds must use compile-time variants of the production epilogue; runtime ablation flags made a 36 % slower epilogue that faked a gate pass.'
- 'r4/r8/r9 barrier and dataflow lesson: cheap release-to-observe (0.64 us for one flag polled by 295 CTAs; flat atomic barrier 0.86-1.07 us; cg grid.sync release maxima 1.0-1.5 us in situ) does not translate into kernel time; barriers are worth at most ~0.5 us per sync on any wide-C row, and the flat counter loses 1-2 us on the skewed sc rows. The per-tile and image-level dataflows lost for per-item serial-chain reasons. H49 uses counters only as a per-image wait inside one phase, where the waiting warps would otherwise write g.'
- 'r5 K2 lesson (H29): a register-prefetch or smem-ring K2 reaches only 47-59 % of the plain copy rate because the per-element math is FMA/INT issue, not MUFU; the A&S erf form is 1.16e-6 from erff. SIMT bounds at 64,96 (rented 1.95 GHz): conv 102 us, exact GELU at 20 instr/element 167 us, A&S erfc 2 MUFU per element 133 us; conv + GELU FMA + stats share the FMA pipe (~200 us), which is the binding pipe for a fused pass (H43) unless the conv moves to the tensor pipe (H50).'
- 'Cluster note (r6, r9, r11): the constraint allows clusters where a GEMM tile or split-K cell needs them; cooperative launch combined with the 2x1 cluster attribute works (cudaLaunchKernelExC), cudaOccupancyMaxActiveClusters = 74 at 512 thr / 227 KB, grid.sync costs 1.35 us with or without clusters. r11 J: a 50 KB exchange at fan-in 4-6 costs 1.63 us; KS2 cells 0.54-0.66 us; fp16-packed partials pass 1.5 us.'
- 'Retirements and pauses: r3 C; r8 B, E, F; r9 A; r10 H (incubation over, ceiling below D); r11 G and I paused (operator directive); r12 J and K paused in incubation (J value shrank and H49 changes its GEMM2; K''s H47 refuted), I resumed, L seeded (new_design). Active r12: D (E1 sc rows, E2 wide-C = half), I (E3), L (E4). G still holds 4 rows.'
hypotheses:
- id: H1
  statement: FP16 or TF32 tensor cores (fp32 accumulate) for both GEMMs meet the tolerance and make every workload memory/latency/fixed-cost-bound (GEMMs <= 19 us except 65 us at the largest with FP16; TF32 at twice that is still under 10 us except at the top three sizes).
  status: supported
  evidence:
  - 'probe: fp16/fp16 and tf32/tf32 (round-to-nearest) err rms 4.5e-4 vs tol 2.4e-3..3.8e-3, 0.000% beyond tol on 6 workloads x 2 seeds; bf16 fails 28-50%'
  - 'arithmetic: 118 GFLOP / 1.82 PF/s = 65 us at the largest vs 2.1 ms fp32 SIMT peak; stream floor 99 us'
  - 'r1 A and B: FP16 GEMMs 16/16 pass in two independent kernels; cuBLASLt FP16 vs TF32 cold: FP16 is 1.4-2.1x faster everywhere'
  - 'r5 A precision log (fp64 reference, 10 seeded rounds, 16 rows): 0 beyond-own-atol elements on every row, max |err| 2.427e-3 vs atol 2.7e-3 at 1,96 (tightest); r5 E TF32 paths 1.67-1.94e-3; r6 D 1.70-1.95e-3 (59-67 %); r6 F 1.61-2.07e-3 (59-76 %)'
  - 'r6 F: the tanh GELU (cuBLASLt epilogue) fails, and folding gb into the fp16 hidden fails: the exact erf on an fp32 pre-activation and gb kept in b2'' are required'
  - 'r7-r9: D sc_kernel 79-89 % of atol (hidden rounded twice), D tc 59-65 %, G 61-82 %, H 58-64 %, D r9 pair path single-seed 1.48-1.70e-3. Every shipped path is logged'
- id: H2
  statement: The depthwise 7x7 conv is a first-order cost at the top two rows because the implementations are staging/load-path-bound, not FMA-bound (G's conv side 472 us at 64,96 against a 102 us FMA bound); at the wide-C small-image rows the bulk-plane conv is cheap (ends 0.9 us after its planes land) and is not a lever.
  status: supported
  evidence:
  - 'probe r0: SIMT 4x4 register patches 516-549 us at the largest at 1965 MHz (bound 102 us), latency-bound at 2 CTAs/SM'
  - 'r1 A: removing global->smem staging cut K1 20.7 -> 6.5 us (1,96) and 148 -> 28 us (8,320); ~8 GB/s per SM through LDG/LDGSTS staging'
  - 'r2 A (H18 part): TMA 3-D box ring lifted ingest to 15-34 GB/s/SM; full K1 5.6 / 23 / 81 / 315 us at 1,192 / 8,320 / 32,128 / 64,96. r3 D: bulk-plane conv 1.5 us at 2,768, 6 at 8,768, 10 at 32,768'
  - 'r4 A (E1): bulk-plane K1 interleaved x3: 2,768 18.1 -> 9.7; 8,768 23.2 -> 12.2; 32,768 33.4 -> 14.2; 16,384 28.9 -> 16.7; 1,160 17.9 -> 5.2 us'
  - 'r7/r8 G: inside the fused pass the conv side alone is 472 us at 64,96 (L1/L2 load path, serial 3-row prefetch chain, 4x halo redundancy at R=2); an L2 prefetch of the x halo changed nothing'
  - 'r9 D stamps at 8,768 (audit r10 correction): plane copies issued at entry, landed at 1.5 us without W1 traffic (2.7 us behind the W1 stream), conv end 0.9 us after landing; the earlier "15x its FMA bound" reading measured the wait for the W1 stream, not the conv. At C=768 the conv is not a lever; H43 attacks the top rows'
  - 'r10 I gate: one 3-D TMA box per 16-channel stage gives conv-only 233.8 us at 64,96 (2.2x the FMA bound, latency/issue-bound at 2 conv warps per SMSP) and 55.8 us at 32,128; per-row 1-D copies were copy-issue-bound (578 us copy-only). r12: H50 tests a tensor-pipe conv'
- id: H3
  statement: Recomputing GEMM1 in phase 2 beats materialising the 4C-wide hidden tensor at the two largest sizes; elsewhere a materialised fp16 hidden (<= 42 MB) is a tie.
  status: parked
  evidence:
  - 'r1 H16: the fp16 round trip costs 3.6 us over the two-launch floor at 20 MB and ~1.4x ideal at 134 MB; at the largest row K2 alone took 447 us'
  - 'r7 G materialises the fp16 hidden once from the fused pass and reads it once in GEMM2 (186 us DRAM-bound at 64,96); the pass itself is SIMT-bound at 668 us, so recompute stays parked until the conv and GELU reach their bounds (H43)'
- id: H4
  statement: One cooperative launch with two grid.sync()s costs about the same as three launches on the rented card and the portal difference is within noise.
  status: refuted
  evidence:
  - 'probe r0: empty kernel 1.8 us, +2.0 us per extra launch; 296-CTA atomic barrier ~1.5 us'
  - 'r1 B, interleaved x3: coop vs three plain launches 29.7 vs 36.3 us at 1,192 and 46.7 vs 50.2 at 2,768; stubbed floors 4.6 vs 9.9 us'
  - 'portal r6: the coop kernel''s portal ratio is 1.20-1.32 against 1.40-1.55 for the 5-6 launch pipeline on S/M rows, so the portal difference is larger than the bench difference'
- id: H5
  statement: For small-M / large-K shapes (2,768,7,7; 8,768,7,7) split-K is required to use the SMs; at 1,192,28,28 it is not needed.
  status: supported
  evidence:
  - 'r1 B ablation, split-K auto vs forced 1: 2,768 46.5 vs 64.6 us, 8,768 86.1 vs 120.6; 1,192 29.9 vs 30.6'
  - 'r4 D gate: 768x1568x3072 without split-K 20 us warm (48 CTAs). r8 H: scalar red.add volume scales with split-K (16K per item ~4 us), so KS=2 beat KS=4 by 7-10 us per C=768 row in a design with per-item red'
  - 'r11: GEMM1 split-K reduced before the GELU (cluster cells, H46) failed its exchange gate at fan-in 4-6; GEMM2 MP=2 (shared B over two W2 tiles, H45) won 2,768 -6 % and 32,768 -4 %'
- id: H6
  statement: The fp32 SIMT GEMM core runs at 46-50 TF/s on the rented card, i.e. 35-38 TF/s at the portal lock; the fp32 SIMT ceiling mean (0.680) is below the leader, so SIMT GEMMs cannot reach the top.
  status: supported
  evidence:
  - 'probe: bare 128x128x32 core 46-50 TF/s, 57-60 TF/s without global loads; ablations no-epilogue 34-35 TF/s, no-affine 33-35'
  - 'portal #65377 vs rented: 1.30-1.32'
- id: H7
  statement: Wave quantisation and tile-shape padding cost 20-30% at mid sizes, so per-size tile tables (or image-straddling token ranges) matter as much as the core rate.
  status: supported
  evidence:
  - 'r1 B: 1,160,449,7 P3 ~26 us with N=160 padded to 2x128 tiles. r4 D gate: 3072x1568x768 = 168 CTAs = 1.14 waves (4.3 us per CTA)'
  - 'r6 D stamps: 192 / 264 whole-image GEMM1 tiles on 148 CTAs at 16,384 / 32,768 leave 44 / 116 CTAs with 2 tiles; at 16,384 the GEMM1 epilogue ends at 32.3 us median vs 44.5 max'
  - 'r7 D sc_kernel stamps: 3 GEMM1 chunks per CTA on 40 CTAs at 4,192 / 8,192 (3.9 us each); r8 G: 112-token tiles (112 CTAs) took 4,96. r8 D: the all-warp epilogue cut the per-tile term 16-28 %, so the 2-tile tails shrank with it'
  - 'r9: pairs (H33) did not remove the tails (2 units per CTA at 32,768 remain); H38 dynamic claiming refuted. r10: balanced token-range tables built as H41; targets missed (see H41). r11: K (H47) refuted - overlap of a 2-tile CTA hides only one mainloop; balance remains the tail lever'
- id: H8
  statement: Applying the GRN scale per image to W2 (or scaling the hidden tile in smem while converting to fp16) removes the per-element affine cost that made phase 3 35% slower than its bare GEMM.
  status: open
  evidence:
  - 'r1 B applies the GRN as an in-smem scale pass and A as a separate K3 launch (6-17 us per row); r3 D folds fp16(s_img*W2) per image; r4-r9 D tc uses two accumulators (h and g = gw*gf*h) at the price of doubled GEMM2 B ingest and a g store in the GEMM1 epilogue (~1.2 us of the 2.3 us store term); r6 F''s one-launch K3 costs 7-20 us'
  - 'r7 D sc_kernel scales the fp16 B tile in smem (0.3-0.45 us per 64-k block); r7-r9 G uses per-image fp16 W2_b built in a 5.6-8.9 us K3. r8 D: g produced in smem inside GEMM2 gated every stage of the 4-stage 48 KB ring; r9 E2: pipelined per-stage signalling fixed the gating but the g pass stayed; r10 H41 / H42 built the single-accumulator form (H42 refuted on its serial chain; H41''s tc2 path kept only on 16,384)'
  - 'r12: H49 tests a fourth form - the scale applied to fp32 h inside the GEMM1 epilogue after a per-image counter wait, so neither g nor a B-tile scale exists'
- id: H9
  statement: The reference's hidden baseline Tb is close to eager, so even the fp32 anchor scores > 0.5 on the S/M band.
  status: refuted
  evidence:
  - 'portal #65377: Tb/rented-eager = 0.34-0.59 per workload; the anchor scores 0.10-0.51'
- id: H10
  statement: A per-image dataflow (image counters instead of grid barriers) hides the GRN dependency behind other images' GEMM1 work at B >= 8 and removes two barrier waits.
  status: refuted
  evidence:
  - 'r4 B (E2) microbench: one red.release flag polled by 295 CTAs is observed in 0.64 us median; flat atomic-counter barrier 0.86-1.07 us; the per-tile dataflow kernel (H27) was 1.4-2x slower on every S row'
  - 'r8 H (H38): the image-level version ran +58-98 % vs D on all five B >= 8 wide-C rows; summed item CTA-time / 148 exceeds D''s whole rows. Closed for the wide-C rows; lineage H retired r10'
- id: H11
  statement: The fixed floor F (launch + barriers + ramps) is 4.6 us in a cooperative kernel and 9.9 us for three launches; what sets the S rows is the SM work and dependent rounds inside each phase. Cutting the per-phase chain to ~3-4 us gives S rows of 15-20 us rented (S ~0.78-0.82).
  status: supported
  evidence:
  - 'r1 B stub: all three phases stubbed = 4.2-4.7 us on every shape; full 1,192 = 29.7'
  - 'r2 B audit at 1,192: ~12 dependent steps of 1.5-2.5 us between P1 end and the end at 28.2 us'
  - 'r3/r4 B microbench: barrier release 1.6-1.9 us per 296-CTA flip barrier, flat atomic-counter barrier 0.86-1.07 us, cold rounds 1.2-3.5 us'
  - 'r7 D (E2) SUPPORTED: sc_kernel runs 1,192 at 19.4 us and 1,96 at 19.1 us rented in one launch. Stamped chain at 1,192: phase A 5.4 (weights 2.6-3.1 then conv 2.5-3), sync 1 ~1.2, GEMM1 single tile 0.6 + 1.0 + 3.7 post-accumulator, sync 2, GEMM2 1.4 + 1.0 + 1.6. r9: the flat barrier loses 1-2 us on sc rows (skewed arrivals). Audit r10: the S rows are worth +0.0062 per 10 % and remain a lever; unchanged-code portal spread there is +-1.2 us; deferred in r11 by operator directive; r12 E1 (H48) prices and removes the weight chain'
- id: H12
  statement: Legacy mma.sync.m16n8k16 FP16 (fp32 accumulate) on sm_100 delivers >= 0.3 PF/s at 1.5 GHz.
  status: supported
  evidence:
  - 'r1 probes: 2018-2048 FLOP/clk/SM = 0.455 PF/s at 1.5 GHz, 1/4 of tcgen05 kind::f16'
  - 'r3 A/D: real tiles 160 TF/s; cuBLASLt ~4x faster. r7 G: mma.sync costs ~100-130 us of the 668 us at 64,96 serialised in the epilogue warps. r8 G at 8,320: pipe-bound at 16 warps (9 clk per mma per scheduler); r9 G replaced it with tcgen05 at 8,320 / 4,96 (H40); the top two rows still run it (H43 replaces it)'
- id: H13
  statement: A thread-block cluster per image (8-16 CTAs, hidden slices in smem, statistics and GEMM2 partials reduced through DSMEM) runs the 7x7/14x14 workloads in <= 30 us.
  status: refuted
  evidence:
  - 'r1 C: DSMEM all-gather 13.5-18.5 B/clk/SM, reduce-scatter ~12 B/clk/SM; v1 at 2,768 129 us'
- id: H14
  statement: TFLOAT32 TMA tensor maps round to nearest, so kind::tf32 GEMMs can read fp32 weights and activations straight from the inputs with no conversion pass.
  status: supported
  evidence:
  - 'r1 C probe: bit-exact round-to-nearest-even through a TFLOAT32 map; r5 E: CUTLASS tfloat32_t A/B on fp32 memory gives max |err| 1.1-1.4e-5 vs an fp64 reference with RNE-rounded operands'
  - 'But the zero-conversion route does not pay on the C=768 rows (audit r10, restated r11/r12): only the GEMM1 tile owners would read W1 (24 CTAs at 2,768, 96 at 8,768), 393 KB fp32 per tile at 40-94 GB/s/SM = 4-10 us against 2.5 us for the grid-wide fp16 pass plus 197 KB fp16 per tile; the library route lost on time too (H30). Parked for the C=768 rows; at the S rows the owner-converted fp16 form is built first (H48) and TF32 sc (I20) remains unbuilt; r12 L (H50) uses a TFLOAT32 box for the conv operand x'
- id: H15
  statement: cuBLASLt FP16 GEMM1+GEMM2 cost <= 12 us at the S sizes and pick sane kernels at M=98..784 with K=3072, so the library is not what limits a launch pipeline; the pipeline's floor is its custom launches.
  status: open
  evidence:
  - 'r1 A: GEMM1+GEMM2 5.1+6.8 us at 1,192, 6.0+9.6 at 2,768, 10.1+13.8 at 32,768; NCHW strided-batched GEMM2 (m=HW) is 20-200 us for HW=49/196/odd'
  - 'r6 F: GELU_BIAS epilogue is the tanh form (rejected); swap-AB batched NCHW GEMM2 is 2-4 us faster than channels-last + K4 at HW=784 and 20-190 us slower at HW=49/196. r7-r9 G: the batched swap-AB GEMM2 runs ~6.6 TB/s at HW >= 4096 but 3.8 TB/s at 8,320 (11.1-11.3 us for 42 MB) and 6.6 us at 4,96; r10 H42''s custom replacement lost (23 us cold), so cuBLASLt G2 stays on G''s rows'
- id: H16
  statement: Writing then reading a 10-134 MB fp16 hidden under the harness's dirty-L2 flush costs little, so a materialised fp16 hidden is not a lever below 42 MB.
  status: supported
  evidence:
  - 'r1 A probe, two launches after the 2xL2 zero-fill: 0.01 MB 5.4 us, 20 MB 9.0, 42 MB 14.6, 134 MB 57.4 (ideal 5.7/12/38)'
  - 'r8 D: the h+g stores (~15.7 MB per tile wave at 8,384) cost ~2.3 us of the GEMM1 epilogue at ~6-7 TB/s under the dirty flush; the bytes are cheap but they bind the epilogue, so halving them (no g) is worth ~1.2 us per tile wave'
- id: H17
  statement: The anchor's phase 1 at C=768 is block-level weight transposes plus the 4-tokens-per-lane conv; warp-level transposes and zero-padded smem planes cut it to ~45 us.
  status: supported
  evidence:
  - 'r0 probes: 42 vs 11.5 us transposes; 306 vs 59 us conv; smem planes 35.5 us at 32,768,7,7'
- id: H18
  statement: Staging the dwconv input through TMA boxes (3-D map for W%4==0 rows; 1-D cp.async.bulk plane copies for 7x7, 14x14 and 449x7 rows) in a multi-stage mbarrier ring lifts conv ingest to >= 40 GB/s per SM; at the small-image rows the conv then lands within 1.5x of its FMA bound, at the top rows it does not.
  status: supported
  evidence:
  - 'r2 A (E1): 15-34 GB/s/SM, 11x the old staging; full K1 5.6 / 23 / 81 / 315 us at 1,192 / 8,320 / 32,128 / 64,96 (3x the FMA bound at the largest). Box inner start must be a multiple of 16 B'
  - 'r3 D: bulk-plane copies for W=7/14 (conv phase 1.5-10 us on the six wide-C rows); r4 A: 449x7 with per-lane 1-D bulk copies; r7 D sc_kernel: plane copies of 3136 / 12544 / 12572 B work'
  - 'r9 D (audit r10 correction): at 8,768 the plane copies land at 1.5 us when W1 traffic is absent and the conv ends 0.9 us later (FMA bound 0.4 us): within ~2x of the bound and no longer a lever. r10 I: one 3-D box per stage at the top rows overlaps the copies fully (conv-only 233.8 us at 64,96, compute-bound at 2.2x the FMA bound)'
- id: H19
  statement: In the cooperative mma.sync kernel, the S-row time above F is mostly dependent global rounds that can be overlapped across the grid syncs.
  status: refuted
  evidence:
  - 'r2 B (E2): collapsing every phase to one load round did not move 1,192 (30.5 flat vs 30.4 ring)'
- id: H20
  statement: A one-barrier cooperative kernel that keeps each CTA's hidden tile in shared memory across the grid sync runs the seven C <= 192 rows in <= 15-20 us rented.
  status: refuted
  evidence:
  - 'r2 C (E3) v2: 1,192 30.8 us, 2,96 46.3 (kill line 25 us); r7 D''s sc_kernel reached 19.4 us at 1,192 with a materialised hidden'
- id: H21
  statement: The portal/rented ratio is a per-path, per-row and per-launch-structure quantity, not a constant 1.28. Multi-launch pipelines pay more on the portal than single-launch cooperative kernels on S/M rows; each launch removed on an S/M row is worth 1-1.5 us more on the portal than the bench shows.
  status: supported
  evidence:
  - 'portal #65570 (A v3) vs rented: 1.27-1.28; fp32 anchor 1.30-1.32; #65878/#65879: B paths 1.24-1.26, D paths ~1.32'
  - 'operator r6: moving 1,96 and 1,192 from B to A looked 5-8 % faster rented and was 6-8 % slower on the portal; r6-merge: D at 1.20-1.32, A at 1.40-1.55 on S/M rows'
  - 'r7-r11 merges: projected 0.7208 / 0.7333 / 0.7351 / 0.7392 / 0.7408 vs measured 0.7207 / 0.7313 / 0.7359 / 0.7392 / 0.7404; D sc 1.20-1.27, D tc 1.20-1.32, G 1.26-1.27; M/L rows within +-2 %, S rows +-1.2 us between submissions with unchanged code'
- id: H22
  statement: Single-pass FP16 operands leave a max-error tail that touches the smallest atol on the 77M-element rows; a hi/lo fp16 split of one operand restores fp32-class accuracy at no cost where MMA work is <= 2 us, so per-row error logging decides where to spend it.
  status: supported
  evidence:
  - 'r1 B: err rms 3.9-4.2e-4, max 1.9-2.8e-3; r0 simulation: FP16 hi/lo split err 1e-6'
  - 'r5 A, 10 seeded rounds, 16 rows: 0 beyond-own-atol everywhere; max up to 2.427e-3 (1,96, 90 %). r6-r9: D tc 59-65 %, D sc 79-89 % (tightest 4,192 at 89 %), G 61-82 %, H 58-64 %, F 59-76 %, E TF32 1.67-1.94e-3. No hi/lo split needed on any shipped path; a no-g tc path (H49) rounds h'' once from fp32 and must be logged; the hi/lo B split or scaling the A tile from fp32 W2 are the fallbacks'
- id: H23
  statement: tcgen05 kind::f16 (TMA-fed from fp16 L2 scratch) is worth +0.02-0.03 of mean over mma.sync at the GEMM-heavy rows and is the lever for the L band once the S/M rows are near the fused floor.
  status: supported
  evidence:
  - 'r6: D''s tcgen05 engine took five rows (+0.016); r7: six more (+0.030); r8: the all-warp epilogue took 11 rows (+0.011); r9: G''s tcgen05 g_tc took 8,320 (-17.5 % rented) and 4,96. The top two rows are the last on mma.sync (H43)'
- id: H24
  statement: The 2-4 us cost of a 12-32 KB per-CTA load round right after a grid sync is an issue-rate artefact of 4-byte copies; one cp.async.bulk per operand brings the round to <= 1.2 us.
  status: refuted
  evidence:
  - 'r3 B microbench: just-written 32 KB round cp4+div 2.4 us / cp16 1.36 / ldg128 0.95 / one bulk 0.83 us median; cold DRAM source 2.7-3.5 us whatever the method; run_tests bulk rows 1,192 31.4 vs ring 29.5'
- id: H25
  statement: A weight-stationary cooperative kernel for the six wide-C small-image rows with an mma.sync engine runs 2,768 <= 35 us, 8,768 <= 45, 16,768 <= 55, 32,768 <= 75, 8,384 <= 45, 16,384 <= 60 rented.
  status: refuted
  evidence:
  - 'r3 D v3 run_tests: 2,768 34.1 (met); 8,768 63.9; 16,768 93.6; 32,768 169.6; 8,384 53.5; 16,384 93.7. Binding term: mma.sync 0.64 us per 32-k chunk plus a sync-heavy ring; engine swap in H28'
- id: H26
  statement: Replacing cuBLASLt GEMM1 + K2 and K3 + cuBLASLt GEMM2 + K4 in lineage A with two custom mma.sync FP16 GEMMs with fused epilogues cuts the pipeline to 3 launches and the S/M rows by 10-14 us rented.
  status: refuted
  evidence:
  - 'r3 A (E1): 1,192 36.3 us (target <= 26), 2,768 60.1 (<= 48), 8,384 97.5 (<= 60); at 8,320 the custom GEMMs run at 160 TF/s vs cuBLASLt ~4x faster. The single-launch tcgen05 form (H34) reached 19.4 us'
- id: H27
  statement: In the cooperative kernel, replacing the grid syncs with per-token-tile and per-image counters removes the two barrier releases and the conv tail wait, cutting the S rows by 4-8 us each.
  status: refuted
  evidence:
  - 'r4 B (E2): the dataflow kernel is slower on every S row (1,192 42.8 vs 29.8; 8,192 137.7 vs 66.5); the image-level version (H38) also lost'
- id: H28
  statement: 'A tcgen05 kind::f16 engine in the swap-AB orientation inside D''s cooperative skeleton makes the six wide-C rows MMA- and byte-bound: thread-local LN fold and GRN sumsq per (image, k), hidden stored [k][token] fp16, GRN scale via two accumulators, output tile [c][tokens] = NCHW with the residual in the epilogue.'
  status: supported
  evidence:
  - 'r4 D (E3) first build 16/16: 2,768 32.4 / 8,768 49.1 / 16,768 70.8 / 32,768 99.3 / 8,384 79.8 / 16,384 111.7 us; per 64-k stage 524-683 clk vs 224-448 ideal'
  - 'r6 D (E2): the 2x term was a 128-B local stack frame in the epilogue; fixed: 27.6 / 34.6 / 46.5 / 66.6 / 38.0 / 58.5. r8 (H36): 26.5 / 33.3 / 44.0 / 64.2 / 34.2 / 51.7. r9 (H39 / H33): 24.0 / 32.4 / 43.5 / 62.1 / 33.4 / 50.9. r10: a second stack frame (STACK 120 in tc_kernel<768,49>) removed; portal 2,768 30.8, 8,768 38.7, 16,768 52.5, 32,768 78.1, 8,384 43.9, 16,384 65.4. r11 (MP=2 GEMM2): portal 2,768 28.3, 32,768 77.4, 16,384 64.5. Remaining: the g term (H49), GEMM1 epilogue per tile, phase-A W1 stream'
- id: H29
  statement: Rewriting lineage A's K2 as a stream-rate kernel cuts 447 -> ~200 us at 64,96 and a tcgen05 GEMM1 with the same epilogue would remove K2 entirely.
  status: supported
  evidence:
  - 'r5 A (E2): ring K2 444 -> 319 us at 64,96, 117.8 -> 79.5 at 32,128; the remaining ~100 us is FMA/INT issue of the GELU. r7 G: K2 removed by the fused pass; the GELU is MUFU-bound inside the fused warps (133 us floor at 64,96) and the FMA pipe is shared with the conv (H43)'
- id: H30
  statement: 'A CUTLASS 4.4 sm100 builder GEMM1 (batched per image) with the LN fold, erf GELU, hidden store and per-batch column reduction in its EVT removes K2, K3 and K4 from the wide-C rows.'
  status: refuted
  evidence:
  - 'r5 E (E1): G1 with EVT 17.0 us at 8,768 (kill 14) and 20.0 at 8,384 (kill 12); per-image batching re-reads W1 and TF32 doubles the bytes; CUPTI sequence-matching failures 14/16 twice. Lineage E retired r8'
- id: H31
  statement: 'On B200 a CTA fed from L2 through a TMA or 1-D bulk ring lands at most ~68-92 B/clk/SM in 1SM mode, the cap is SM-side, so designs must halve per-SM operand bytes (cta_group::2 pairs, or operands produced in smem by the CTA itself) rather than multicast. A straight-line kind::f16 issue reaches the MMA ideal at N >= 128 in 1SM mode (N=64 at 1.57x); the ~105 clk per instruction seen in G''s engine is issue-loop overhead, not a hardware floor.'
  status: supported
  evidence:
  - 'r4 D gate: per 64-k stage 683 clk at N=224 (ideal 448), 587 at N=112 (224), 524 at N=128 with K=3072 (256); stage depth 2-3 vs 8 unchanged'
  - 'r6 D probe (148 CTAs, warm L2, 4-stage SW128 ring): 1SM 479 / 532 / 546 clk at N = 128 / 224 / 256; B multicast over a 2-CTA cluster no gain; cta_group::2 pair 345 / 455 / 518 (1.35x / 1.01x / 1.01x of ideal)'
  - 'r9 G: 1SM 128xNx16 with smem-resident operands 106 / 107 / 128 clk per instruction at N = 64 / 128 / 256, independent of accumulator dependence; 10-stage bulk ring cold: 94 GB/s/SM from L2, 40 GB/s/SM distinct-per-CTA from DRAM. Audit r10/r11: the A-slice-fetch mechanism is unsupported (pair probe 86 clk per 128x128x16 per SM); the number stands; the straight-line issue test is E4 STEP 0a (r11)'
  - 'r9 D E2 v2 stamps: pair GEMM2 stages 658-875 ns vs a ~270 ns ideal with the producers gating (fence + cluster arrive in series per stage); v3 (one producer warp per stage) recovered it at 32,768 (-3 %). Pairs pay only where a CTA runs many stages'
  - 'r11 K STEP 0a (64 unrolled MMAs, precomputed SW128 descriptors, elect.sync in a uniform warp): 1SM 50.2 / 66.4 / 114.4 / 130.4 clk per instruction at N = 64 / 128 / 224 / 256 (1.57 / 1.04 / 1.02 / 1.02x of ideal), identical at 1 / 128 / 148 CTAs; pair M=256 45.5 / 66.7 / 114.7 / 130.7. The r9 floor clause is withdrawn; a pair''s only benefit above N=64 is halving B landing. r12 E2 STEP 0 measures D''s clk per instruction against this table'
- id: H32
  statement: 'A library-lean four-launch pipeline (cluster-of-8 LN exchange, cuBLASLt GEMM1 with GELU epilogue, one-launch GRN, cuBLASLt swap-AB GEMM2) removes K2, K4 and the K3 chain from every HW <= 784 row.'
  status: refuted
  evidence:
  - 'r6 F (E1): refutation clause fired on 5 of 6 wide-C rows; D r6/r7 then beat F everywhere. Lineage F retired r8'
- id: H33
  statement: 'In D''s swap-AB tcgen05 engine, GEMM2 as cta_group::2 pairs (M = 256 output channels across the pair, each CTA loading half of the B token tile) halves the per-SM landing and allows a deeper ring in which g is produced in smem by look-ahead warps, so g leaves the GEMM1 epilogue; rows run 16,384 <= 46, 8,384 <= 30, 32,768 <= 57, 16,768 <= 39, 8,768 <= 29, 2,768 <= 24 us rented.'
  status: refuted
  evidence:
  - 'r6 D probe: the pair reaches 1.35x / 1.01x / 1.01x of the MMA ideal at N = 128 / 224 / 256; the pair recipe worked first try'
  - 'r9 D (E2): ptxas forbids mixing cta_group::1 and ::2 in one kernel, so GEMM1 paired too (coupled accumulator wait 2.5-4.2 us per tile). v1/v2 slower than r8 on all four C=768 rows; v3 (one producer warp owns a stage) 32,768 61.1 vs 64.1, 16,768 42.7 vs 43.8, 2/8,768 slower. Shipped v4 pairs only C=768 at B >= 16: 32,768 62.1 vs 64.1 (-3.1 %). Success line missed on every row; the smem-g pass was never re-enabled. Kept: pipelined per-stage signalling, pairs at many stages per CTA'
- id: H34
  statement: 'D''s single-launch tcgen05 skeleton extended to the C <= 192 rows and 1,160 runs 1,192 <= 24 us rented, 4,192 <= 30, 8,192 <= 42, 2,96 <= 32, 1,160 <= 34.'
  status: supported
  evidence:
  - 'r7 D (E2) sc_kernel 16/16: 1,192 19.4; 1,96 19.1; 1,160 26.9; 2,96 27.0; 4,192 30.5; 8,192 44.8; r8: A&S GELU and 4,96 added (1,96 18.4, 2,96 25.5, 4,192 29.4, 8,192 43.4, 1,160 26.0, 1,192 18.6, 4,96 38.8); r9: 18.2 / 25.0 / 29.3 / 41.0 / 25.3 / 18.2 / 37.6. 8,320 is not instantiable (W % 7, 4C <= 768)'
- id: H35
  statement: 'A fused persistent pass-1 kernel for the HW >= 1024 rows replaces K1 + G1 + K2 with one kernel bounded by the SIMT work of conv + GELU (~200-270 us at 64,96 rented): pass 1 <= 300 / 75 / 25 us; with a GRN-scale launch and a cuBLASLt batched swap-AB GEMM2 the rows run <= 550 / 140 / 55 us rented.'
  status: open
  evidence:
  - 'r7 G (E1), mma.sync engine: pass 1 668 us at 64,96 (2.5x the tile bound), 171 at 32,128; rows 857.5 / 229.8 rented (-16.5 % / -10.6 % vs A). Ablations: conv side alone 472 us, GEMM side 522 (mma.sync 100-130 serialised, 2-MUFU GELU floor 133); phases add because 1 CTA of 16 warps fits per SM'
  - 'r9 G (H40): tcgen05 at 8,320 / 4,96 gave -17.5 % / -10 %; the top two rows still run the r7 mma.sync pass. H43 (lineage I, resumed r12) rebuilds pass 1 at the pipe bounds; the direction stands, the targets are open'
- id: H36
  statement: 'In D''s tcgen05 kernels the post-accumulator GEMM1 epilogue is latency- and issue-bound on 8 epilogue warps; running it on all warps cuts the per-tile epilogue by >= 40 %.'
  status: supported
  evidence:
  - 'r8 D (E1): all-warp epilogue at 512 threads: per-tile post-accumulator -16..-28 %; bar missed but 11 rows won (+0.011). Ablation at 8,384: post 7.6 us = pass 1 4.3 + g write 3.0; GELU ~1.4, h+g stores ~2.3; pass 1 is within ~1.3x of its issue estimate (26.6K elements x ~30 instr / 128 lanes), so the remaining levers are fewer store bytes and no g pass (H49); overlap with a mainloop (H47) was refuted'
- id: H37
  statement: 'G''s fused pass-1 pipeline instantiated for 8,320 and 4,96 with mma.sync runs 8,320 <= 50 us and 4,96 <= 34 us rented.'
  status: refuted
  evidence:
  - 'r8 G (E2): 4,96 37.0 vs 41.4 (-10.7 %); 8,320 67.2 vs 73.1 (-8.0 %); mma.sync pipe-bound at 16 warps (16.2 us) with the epilogue serialised behind it; replaced by tcgen05 in H40'
- id: H38
  statement: 'One persistent tcgen05 kernel that claims items from a static queue and gates them on per-image counters runs the B >= 8 wide-C rows without grid barriers.'
  status: refuted
  evidence:
  - 'r8 H (E3): +58-98 % vs D on all five rows; summed item CTA-time / 148 = 47.0 / 111.5 / 71.9 us at 8,768 / 32,768 / 16,384 above D''s whole rows. Lineage H retired r10'
- id: H39
  statement: 'Phase A of D''s kernels is a serial chain, not a bandwidth term: issuing every conv plane copy at kernel entry, converting W1 while they land, and deferring the W2 conversion and the out prefill into the GEMM1 phase cuts phase A to <= 5 us at C=768 and <= 3.5 us at 1,192: -2..-5 us on every D row. Stretch: flat self-resetting counter barriers for another -1..-1.5 us per row.'
  status: refuted
  evidence:
  - 'r9 D (E1): refutation clause fired on 4 of 6 rows after the reorder. Stamps at 8,768: the plane copies were already issued at entry; issuing the 12 LDG.128 per lane of W1 takes 2.2 us (~4 TB/s over the grid), the planes land at 2.7 us instead of 1.5 because they queue behind the W1 stream, conv ends 4.0, W1 conversion 5.6, phase A end 5.7 / 7.1; without W1: 3.2 / 4.1 and sync 1 at 5.3-5.9. Phase A is bound by the W1 fp32 stream (2.5 us of phase A, 3.2 us at sync 1)'
  - 'The deferred W2 pass still wins 1-3 us per row (it leaves the critical path; costs 0.5-0.9 us of GEMM1 end at 8,768, 2.2 us at 32,768); an L2 prefetch of W2 in the sync-1 window only moves the cost. Flat barrier: -0.1..-1.3 us on tc rows, +1..+2.1 us on sc rows (shipped on tc_kernel only)'
- id: H40
  statement: 'Replacing mma.sync by tcgen05 kind::f16 in G''s single-tile kernel removes the 16.2 us pipe-bound mma.sync term at 8,320 and the 3.3 us at 4,96, so pass 1 runs <= 30 us at 8,320 and <= 19 us at 4,96 and the rows <= 55 / <= 33 us rented.'
  status: supported
  evidence:
  - 'r9 G (E3) run_tests 16/16, same container vs r8-merge: 8,320 59.0 vs 71.5 (-17.5 %); 4,96 37.2 vs 41.4 (-10.0 %); taken by r9-merge (portal 74.5 / 46.7). Targets missed (59.0 > 55; 37.2 > 33); refutation clause did not fire. Shipped levers: conv-overlapped MMA in channel rounds and per-warp smem-transposed h stores'
- id: H41
  statement: 'In D''s tc_kernel, balanced image-straddling token-range tiles with the per-(image,k) GRN sumsq split at image boundaries and red.add-ed into a [B][4C] gsum, no g, and a single-accumulator GEMM2 whose landed h tile is scaled in place by the stage-owner producer warp >= 2 stages ahead, remove the 2-tile tails and halve the GEMM1 store bytes and GEMM2 B landing: 16,384 <= 44, 32,768 <= 56, 8,384 <= 30.5, 16,768 <= 39.5, 8,768 <= 30 us rented.'
  status: parked
  evidence:
  - 'arithmetic (r6/r8 stamps): per-tile GEMM1 = ~2.6 mainloop + 8.1 epilogue at 16,384 with 44 of 148 CTAs running 2 tiles; balanced ranges give every CTA ~1.35 images of work'
  - 'r10 E1 shipped as r10-d-e1-balanced-nostack; r10-merge (0.7392) portal 16,384 67.5 -> 65.4, 32,768 79.0 -> 78.1, 8,384 44.4 -> 43.9, 16,768 56.7 -> 52.5, 8,768 42.0 -> 38.7, 2,768 30.7 -> 30.8. Portal-implied rented times (D ratio ~1.25) miss every target by 3-7 us; the operator attributes the round''s main win to the tc_kernel<768,49> STACK-120 fix, not to the tables. r11-merge-b keeps r10''s tc2 path only on 16,384 (64.5 portal). Parked: balance alone does not hide the per-tile epilogue, and the in-smem B-scale costs per stage; H49 removes g without a B-scale'
- id: H42
  statement: 'A custom tcgen05 kind::f16 GEMM2 launch g_g2 for G''s HW >= 1024 rows (swap-AB, B scaled in place by s_b, b2'' in fp32, coalesced NCHW epilogue) replaces K3 and cuBLASLt G2: g_g2 <= 9 us cold at 8,320 and <= 4.5 us at 4,96.'
  status: refuted
  evidence:
  - 'r10 G (E2, resumed): g_g2 ran 23 us cold at 8,320 against the 9 us kill line; the serial chain before the first MMA (s_b prologue from gsum, first landing, scale) binds, not the MMA. Builds r10-g-e2-g2-coop / -v3 are archived; K3 + cuBLASLt G2 stay on G''s rows'
- id: H43
  statement: 'A pass-1 kernel for 64,96 / 32,128 with fixed warp roles (TMA warp streaming x channel-group row bands, 8 conv warps with register-blocked patches writing fp16 d into the MN-major SW128 B tile, one MMA warp against a streamed W1'', epilogue warps with LN fold, A&S erfc GELU, per-(image,k) sumsq and staged fp16 h stores) runs within 1.5x of the SM pipe bound: conv-only <= 180 us and pass 1 <= 350 us rented at 64,96 (G r7: 472 / 668), rows 64,96 <= 560 us and 32,128 <= 150 rented, about +0.006 of mean.'
  status: open
  evidence:
  - 'r7 G ablations at 64,96: conv side alone 472 us, GEMM side 522; phases add at 1 CTA of 16 warps per SM. r9 G: tcgen05 engine validated with d written by st.shared into the MN-major SW128 layout'
  - 'r10 I gate: per-row 1-D copies copy-issue-bound (copy-only 578 us); one 3-D TMA box per stage (OOB zero fill) conv-only 233.8 us at 64,96 (gate <= 250 passed, 180 success missed, refutation > 300 not fired), 55.8 at 32,128; 128-reg cap +12 % / +4 %; edges correct within 1 fp16 ulp. Lineage I paused r11 by operator directive'
  - 'r11 K: MUFU-heavy epilogue warps cost the producer/MMA warps sharing their SMSP 30-40 % of their rate; the pass has only 18 MMAs per tile, so the conv warps'' share of the FMA pipe is what must be protected. r12: resumed as E3 (STEP 2 + K0/K3/cuBLASLt tail)'
- id: H44
  statement: 'In D''s tc_kernel the C=768 phase-A weight cost (2.5 us of phase A, 3.2 us at sync 1) is an LSU-issue term of the 12 LDG.128 per lane W1 stream (~4 TB/s), not DRAM bandwidth: a bulk-copy weight pass (cp.async.bulk fp32 chunks with an L2 evict_first policy into smem, LN fold + cvt.rn.f16x2 by all warps, fp16 rows written by bulk store or STG.128 with evict_last) runs the 9.4 MB W1 read at >= 6.5 TB/s and moves sync 1 >= 1.5-2 us earlier; where >= 4 token tiles share a W1 slice converting W1 inside the GEMM1 ring removes it from phase A entirely if the per-stage clk stays <= 1.25x the fp16 ring.'
  status: refuted
  evidence:
  - 'r9 D stamps: W1 issue 2.2 us at ~4 TB/s, planes queue behind it (land 2.7 vs 1.5 us), sync 1 at 8.1 vs 5.3-5.9 without W1; audit r10/r11 arithmetic: fp32 A landing doubles per-tile A bytes (393 vs 196 KB)'
  - 'r11 D E2 (r11-d-e2-w1bulk): the bulk-copy weight pass and the in-ring fp32 W1 conversion were both refuted (operator summary; the per-row numbers are in the card, which the lead cannot read through the tool limit); rented band geomeans 23.0 / 36.9 / 275.4 vs E1''s 22.6 / 35.8 / 271.1; r11-merge-b took no row from it. The phase-A W1 stream has no cheaper form found'
- id: H45
  statement: 'D''s GEMM2 phase (sync 2 to kernel end, ~8 us at 8,768 and up to ~25 us at 32,768) is bound by per-SM operand ingest times units per CTA on <= 96-120 busy SMs plus a scalar red.add drain of 3.7-8 us at 16/32,768: unit tables that fill 144-148 SMs, a TMA bulk-reduce epilogue and one MMA over an adjacent [h|g] B stage cut the GEMM2 phase by >= 40 %: rows 32,768 -4, 16,768 -3, 16,384 -3, 8,768 -2.5, 8,384 -2, 2,768 -1.5 us rented, about +0.004-0.006 of mean.'
  status: open
  evidence:
  - 'r8 H: 16K scalar red.add per item ~4 of 7.4 us; r9 E2 stamps: red.add drain 3.7-8 us post-sync at 16/32,768; r8 D: ~0.7 us per 48 KB GEMM2 stage, latency/ingest-bound'
  - 'r11 D E1 (r11-d-e1-g2-mp2-bulk): MP=2 (two W2 tiles sharing one [h|g] B stage) wins 2,768 -6 % and 32,768 -4 % rented, where a CTA runs many GEMM2 stages; taken into r11-merge-b (portal 2,768 28.3, 32,768 77.4). Whether the bulk-reduce epilogue shipped and how far the >= 40 % line was missed are in the card, not visible to the lead. The g term itself is attacked by H49'
- id: H46
  statement: 'For the low-M wide-C rows (2,768: 24 GEMM1 tiles; 8,768: 96), cluster split-K cells for both GEMMs (fp32 partials exchanged inside the cluster and reduced before the LN fold and GELU in GEMM1 and before the residual store in GEMM2) fill 144 CTAs and cut both mainloops to 2-4 k-blocks: 2,768 <= 19 us and 8,768 <= 25 us rented, about +0.004-0.006 of mean.'
  status: parked
  evidence:
  - 'r9 D stamps at 8,768: GEMM1 phase 11.7 us (mainloop 3.4-3.6 on 96 CTAs + epilogue 4.8), GEMM2 phase ~8 us; at 2,768 100 of 148 CTAs have no GEMM1 tile; r1 C: DSMEM 12-18 B/clk/SM'
  - 'r11 J (E3) gate killed: a 50 KB exchange at fan-in 4-6 took 1.63 us against the 1.5 us line; the same probes show KS2 cells (cluster of 2, 148 co-resident) at 0.54-0.66 us and fp16-packed partials under 1.5 us, so the idea is not refuted. Parked r12 (lineage J paused): 2,768 left the cheapest allocation, KS2 fills only 48 CTAs at 2,768, and H49 changes GEMM2'
- id: H47
  statement: 'For the multi-tile wide-C rows (32,768, 16,768, 16,384), a persistent warp-specialised tcgen05 engine with fixed roles (TMA warp, MMA warp, 12-14 epilogue warps), two TMEM accumulators and balanced static tile lists of N >= 196 runs each tile''s epilogue under the next tile''s mainloop, so the GEMM1 phase falls from 20-28 us to <= 14 us at 32,768 and <= 11 us at 16,384.'
  status: refuted
  evidence:
  - 'r11 K (E4) STEP 0b gate (148 CTAs x 512 thr, production-efficient epilogue): N=208 direct stores epilogue 69 % / MMA 65 % of solo rates when co-running; staged stores 74 / 62; no stores 73 / 59; N=128 79-80 / 53-55. Gate (>= 80 / 80) failed in every variant; overlap saves 16-24 % of the serial sum'
  - 'Limiter: issue/dispatch sharing inside each SMSP (TMEM lane-quadrant rule puts epilogue warps on all 4 schedulers; MUFU work stretches each MMA stage hand-off by 265-355 clk; MMA smem operand reads contend with epilogue LDS). Ablations (5 stages, 8 or 4 epilogue warps) never clear the gate. Projection 21 us vs the <= 19.6 us line at 32,768; for a 2-tile CTA overlap hides only one mainloop. Lineage K paused r12'
- id: H48
  statement: 'In D''s sc_kernel (1,192 1,96 2,96 1,160 4,96 4,192 8,192) the 2.6-3.1 us weight chain before sync 1 (fp32 W1/W2 -> fp16 L2 scratch, then TMA of A from scratch) can leave the critical path: each GEMM1 CTA bulk-copies its own fp32 W1 slice at entry beside the plane copies and converts it (LN-weight fold, u and v sums) into its resident SW128 A while the conv runs or during the sync-1 wait, and GEMM2 owners do the same for their W2 slice after sync 1; sync 1 then gates only d. Every sc row -1.5..-2.5 us rented (-6..-10 %), about +0.004-0.006 of mean.'
  status: open
  evidence:
  - 'r7 D stamps at 1,192: phase A 5.4 us = weights 2.6-3.1 then conv 2.5-3; W1+W2 fp32 is 1.2 MB at C=192 and 0.3 MB at C=96 (~0.2 us of bandwidth), so the chain is cold rounds plus the scratch round trip, not bytes; r2 C: one-shot cp.async.bulk after the flush ~120 GB/s/SM (98 KB W1 slice ~0.8 us, in the shadow of the plane copies); dMean/dT 2.4-5.2e-4 per us on the seven rows; predict_score S band 10 % = +0.0062. Built as E1 (r12)'
- id: H49
  statement: 'In D''s tc_kernel the g term (second GEMM2 accumulator over g = gw*gf*h) can be removed by applying the GRN scale inside the GEMM1 epilogue: each tile keeps its GELU output h in fp32 registers, publishes its per-(image,k) gf into a [B] sum with a release arrive on a per-image counter, waits for its images (tiles ordered image-major, every wait after the CTA''s earlier tiles, all CTAs co-resident, so no cycle), and stores h'' = fp16(h * (1 + gw*gf*inv_b)) once. The GEMM1 epilogue loses the g pass and g store (-30..-35 % per tile, r9 v2), GEMM2 runs one accumulator over h'' (half the B landing and MMA N of [h|g]), and the hidden is rounded once. Rows 32,768 and 16,768 >= -10 %, 8,768 >= -5 %, 2,768 and the 14x14 rows -3..-8 % rented; about +0.004-0.006 of mean.'
  status: open
  evidence:
  - 'r9 E2 v2 stamps: GEMM1 post-accumulator per tile without the g pass 2.05 / 3.36 / 4.77 / 5.17 us vs 3.04 / 4.77 / 7.07 / 7.92 with it at 2/8/16/32,768 (-30..-35 %); r8: g write ~3.0 us of the 7.6 us post at 8,384; [h|g] doubles GEMM2 B bytes and MMA N; H41''s B-scale replacement paid only on 16,384; r4/r8 counter mechanics: one flag observed by 295 CTAs in 0.64 us, dependency spin 5-15 % of item time; CUTLASS-style release + ld.acquire + fence.proxy.async correct for TMA readers. Built as E2 (r12)'
- id: H50
  statement: 'A Toeplitz formulation of the 7x7 depthwise conv on tcgen05 (per channel, output rows = sum over the 7 tap rows of shifted x rows @ banded T_i built in smem, x landed by a TFLOAT32 3-D TMA box into kind::tf32 or as fp16 hi/lo into kind::f16) runs conv + fp16 d + per-token LN partials at <= 100 us rented for 64,96 and <= 25 us for 32,128 (I''s FMA conv 233.8 / 55.8 us; FMA bound 102 / 22), leaving the FMA pipe to the GELU; with it a fused pass 1 can approach the MUFU bound (~133 us), worth up to +0.004 of mean beyond H43.'
  status: open
  evidence:
  - 'arithmetic: banded T with 7 nonzeros per 64-wide column inflates FLOPs ~9-10x; kind::tf32 runs 4096 FLOP/clk/SM against 256 for FFMA (16x), so an MMA-bound conv is ~1.6-1.8x faster than the FMA bound (~60 us at 64,96 rented) and uses no FMA issue slots; H14: TFLOAT32 maps round to nearest; r10 I: one 3-D box per 16-channel stage overlaps fully. Risks: row shifts that are not multiples of the 8-row swizzle atom, T construction per channel, TF32 rounding of x and taps on top of the d16 rounding. Built as E4 (r12, lineage L)'
```

```json tasks
[
  {
    "id": "E1",
    "lineage": "D",
    "hypothesis": "H48",
    "title": "sc_kernel: owner-converted resident weights, sync 1 gates only d (seven sc rows)",
    "operation": "structural_mutation",
    "parents": ["r11-d-e1-g2-mp2-bulk"],
    "band": "S",
    "instructions": "STEP 0, price the chain (<= 5 B200 min). Read the parent card's sc_kernel phase-A order and smem plan in parts. Build two timing-only stubs of sc_kernel; wrong results are allowed and neither is ever shipped. (a) Skip the weight pass and leave the fp16 scratch as garbage; everything else unchanged. (b) As (a), and also skip the post-sync TMA of A, so GEMM1 reads an A image already in smem. Run harness_time interleaved x3 against the parent at 1,192 / 2,96 / 4,192 / 1,160 / 8,192. Take globaltimer stamps of phase-A end and sync-1 release from the working warps. Gate: stub (a) saves >= 1.2 us mean over those rows. If not, stop, record the prices as findings and ship the parent unchanged. STEP 1, GEMM1 weights. At entry each GEMM1 CTA issues cp.async.bulk of its own fp32 W1 rows (128 hidden x C, 98 KB at C=192), beside the plane copies. After its conv units it converts them into its resident SW128 K-major fp16 A image, using the validated address function. In the same pass it computes u_k (rowsum of the rounded fp16(lnw*W1) values) and v_k = W1.lnb + b1. Issue every load before any shuffle, fully unrolled (lesson j). STEP 2, GEMM2 weights. GEMM2 owners bulk-copy their fp32 W2 slice right after sync 1, or at entry if smem allows, and convert it during GEMM1 or the sync-2 wait. b2' = W2.gb + b2 comes from per-K-slice partial sums of the owners; the output reduction is already additive. Reuse raw fp32 landing areas as unit buffers after conversion. If 4,192 / 8,192 exceed 227 KB, convert in two halves through one staging area, or keep the scratch path on those shapes (shape-only dispatch). Keep cg grid.sync on sc_kernel: the flat barrier lost 1-2 us there in r9. STEP 3, checks. cuobjdump -res-usage must show STACK 0 on every sc instantiation. run_tests 16/16 with portal-order re-runs. The u_k summation order changes, so log precision afresh over 10 seeded rounds (seed 200, fp64 reference, harness generators) on all seven sc rows: max |err| and count beyond own atol. The sc class sits at 79-89 % of atol, and any beyond-atol element blocks shipping (fallback: hi/lo split per H22). STEP 4: harness_time interleaved x3 against r11-merge-b on the seven sc rows; record per-row deltas and merge picks.",
    "success": "-1.5 us rented or better on at least 5 of the 7 sc rows vs r11-merge-b interleaved x3, 0 beyond-atol elements over 10 rounds, STACK 0",
    "refuted_if": "stub (a) saves < 1.2 us mean (the weight chain is already overlapped), or the built version gains < 0.7 us on 4 or more of the 7 rows",
    "model": "opus"
  },
  {
    "id": "E2",
    "lineage": "D",
    "hypothesis": "H49",
    "title": "tc_kernel: GRN scale in the GEMM1 epilogue behind per-image counters, no g, single-accumulator GEMM2",
    "operation": "structural_mutation",
    "parents": ["r11-d-e1-g2-mp2-bulk"],
    "band": "L",
    "instructions": "STEP 0 (<= 5 B200 min). Read the parent card in parts to see which tc paths r11-merge-b runs per row: the 7x7 C=768 MP=2 path, the r10 tc2 path on 16,384, and the pair path at B >= 16. Stamp the parent with globaltimer from the working warps; check STACK on the stamp build too. Rows: 32,768 / 16,768 / 8,768 / 16,384. Stamp the GEMM1 per-tile mainloop and the post-accumulator split (pass 1, g pass, stores), the GEMM2 phase and ns per stage, and clk per MMA instruction in the mainloops. Compare the clk with r11 K's straight-line table (N=64 50, N=128 66, N=224 114 clk). If D issues at >= 90 clk per instruction at N <= 128, precomputed descriptors with an unrolled k16 loop may go into the build as a separately stamped change. If the parent card does not record the flat-barrier stress loop, run it now: 1500 calls at 8,768 and 16,384 against the torch reference. Any failure reverts tc_kernel to cg grid.sync. STEP 1, the 7x7 C=768 path. The GEMM1 epilogue keeps the GELU output h in fp32 registers (thread = hidden row, its share of the columns). It reduces the per-(image,k) sumsq inside the CTA and red.adds sum_k gf into a [B] buffer zeroed in phase A. It then release-arrives on a per-image counter, CUTLASS-style: bar.sync, then red.release.gpu from one thread. The expected count is the number of tiles touching image b. Order tiles image-major, and let a CTA wait only after its earlier tiles are done; in the co-resident cooperative grid the wait graph then has no cycle. Spin with ld.acquire, never __nanosleep. Counters self-reset through a departure counter. Then compute inv_b = 1/(mean_k gf + eps) and store h' = fp16(h * (1 + gw*gf*inv_b)) through the staged coalesced store. gb stays in b2'. While the epilogue warps wait, the MMA warp may already issue the CTA's next tile into a second TMEM accumulator. GEMM2 drops g and runs one accumulator over h' alone. Spend the freed stage bytes on more depth, or on MP=2 over h' only, or on 2x tokens per stage, and pick per row. STEP 2: carry the change to the 14x14 paths (on the tc2 path it replaces the in-smem B-scale), to the pair path, and to 2,768 / 8,768. STEP 3: STACK 0 on every instantiation (holding h in registers at N=208 risks spills; split columns over the 4 warps of a quadrant). run_tests 16/16 with portal-order re-runs. h' is new arithmetic, so log precision afresh over 10 rounds (fp64 reference) on every row shipped. Then run harness_time interleaved x3 against r11-merge-b on the six wide-C rows. Stamp the image wait (median and max per tile) and the GEMM2 phase.",
    "success": ">= 10 % rented on 32,768 and on 16,768 and >= 5 % on 8,768 vs r11-merge-b interleaved x3; image-wait median <= 1.5 us; 0 beyond-atol elements; STACK 0",
    "refuted_if": "< 4 % on both 32,768 and 16,768 after STEP 1, or image-wait median > 2.5 us (per-image skew eats the g saving), or the GEMM2 stage time does not fall without g (stages latency-bound rather than byte- or MMA-bound)",
    "model": "opus"
  },
  {
    "id": "E3",
    "lineage": "I",
    "hypothesis": "H43",
    "title": "Top rows STEP 2: tcgen05 GEMM1 + epilogue warps in the fixed-role pass-1 kernel, then the K0/K3/cuBLASLt tail",
    "operation": "structural_mutation",
    "parents": ["r10-i-e3-tmaconv-gate"],
    "band": "L",
    "instructions": "Resume from r10-i-e3-tmaconv-gate. i_tmaconv_gate.cu holds the validated 3-D TMA box ring: box {W+8, R+6, 16}, start {-4, h0-3, b*C+g*16}, OOB zero fill. STEP 2: one pass-1 kernel for 64,96, then the 32,128 instantiation. 512 threads, 1 CTA/SM, persistent over 224-token tiles. N >= 128 now runs at the MMA ideal in 1SM mode (r11 K), so smaller tiles are allowed where they balance better, e.g. the 32,128 tail. Roles are warp-uniform: (a) 1 TMA warp: x boxes plus a W1' A ring of 2 x 16 KB. (b) 8 conv warps: register-blocked patches from the gate, fp16 d written into the MN-major SW128 B tile with G's r9 address function, LN partials into smem. (c) 1 MMA warp: kind::f16, elect.sync issue, d released per 16-channel group (one k16 step of K=96), 2 x 224-column TMEM ping-pong. (d) Epilogue warps in a multiple of 4 (TMEM lane-quadrant rule): LN fold rs*(acc - mu*u) + v, A&S erfc GELU, per-(image,k) sumsq red.add into [B][4C], coalesced fp16 h stores through a per-warp smem transpose. Smem budget from the gate: x ring 123 KB + d 48 KB + LN 14 KB + A ring 32 KB. r11 K finding: MUFU-heavy epilogue warps cost the warps sharing their SMSP 30-40 %. The FMA pipe binds here, so check that the conv warps keep >= 85 % of their gate rate, and place the producer and MMA warps accordingly. Stamp conv / MMA / epilogue per tile. Validate h for tile 0 and the last tile against torch before timing. Gate: pass 1 <= 400 us rented at 64,96. STEP 3, only if the gate passes: build the tail from the notebook recipes. K0: W1' = fp16(lnw*W1) with u and v, fp16 W2, gsum zeroed. K3: per-image W2_b = fp16(s_b*W2), grid (B, C/16), one row per warp, __launch_bounds__(512,2). GEMM2: cuBLASLt batched swap-AB with beta 1 on C = x, b2' as hi/lo K-augmentation columns paired with constant rows written by pass 1, top-1 heuristic. run_tests 16/16 with portal-order re-runs; STACK 0; a 10-round fp64 precision log on 64,96 and 32,128. Then harness_time interleaved x3 against r11-merge-b on the two rows. If STEP 3 does not finish, archive the pass-1 kernel with its stamps behind the r0 anchor (as r10 did) and record what remains.",
    "success": "pass 1 <= 400 us rented at 64,96; with the tail, rows <= 650 us (64,96) and <= 190 us (32,128) rented vs about 857 / 228 for r11-merge-b in the same container, 0 beyond-atol elements",
    "refuted_if": "pass 1 > 520 us at 64,96 after tuning (less than 22 % better than G's 668 us), or the role layout cannot fit the 128-register cap and 227 KB without dropping below 8 conv warps",
    "model": "opus"
  },
  {
    "id": "E4",
    "lineage": "L",
    "hypothesis": "H50",
    "title": "New design: Toeplitz tensor-core depthwise conv gate for 64,96 and 32,128",
    "operation": "new_design",
    "parents": [],
    "band": "L",
    "instructions": "New lineage, built from scratch around pool idea I15. STEP 0, no GPU. Run a numpy fp64 simulation at 64,96 and 32,128 (harness generators, 2 seeds) of the end-to-end output error when the conv uses (a) TF32-rounded (RNE) x and taps, or (b) fp16 hi/lo x with fp16 taps, optionally hi/lo taps. In both cases d is then rounded to fp16 and the rest follows the fp16 pipeline (fp16(lnw*W1), h16, fp16(s_b*W2)). Report max |err| as a share of each row's atol and pick the cheapest form that stays <= 75 %. STEP 1, single-plane probe on one (b,c) plane of 64,96. Out rows [h][w'] = sum_{i=0..6} X[h+i-3][:] @ T_i, where T_i[w,w'] = k_c[i, w-w'+3] is banded and built in smem per channel. Run it on tcgen05 with x landed by one 3-D TMA box: TFLOAT32 map for kind::tf32, OOB zero fill for the halo, inner start a multiple of 16 B (start at w=-4 and offset the taps). Solve the row shift i without 7 copies of x. Options: the smem descriptor's matrix base-offset field for starts not aligned to the 1024-B swizzle atom; an unswizzled K-major core-matrix layout; or the operand orientation that puts the shift on the side that tolerates it. Validate against torch conv2d to fp32-class error before timing anything. STEP 2, a standalone conv kernel for 64,96 and 32,128, persistent over (image, row band) work. Each CTA covers all C channels of its band, so the per-token LN sum and sumsq stay CTA-local. It writes d fp16 in a padded [c][token] layout (usable as the MN-major B operand of a later GEMM1) plus per-token mean and rstd. Use compile-time variants only for ablations. Report: time after the flush, clk per MMA against the r11 K STEP 0a table, the tensor-pipe share, and STACK. Ship nothing to the merge: the archived solution is the r0 anchor with the probe source behind #if 0 (as r10 I did), and run_tests 16/16.",
    "success": "conv + d write + LN partials <= 100 us rented at 64,96 and <= 25 us at 32,128; d within 1-2 fp16 ulp of fp16(torch conv2d); simulated end-to-end max error <= 75 % of atol",
    "refuted_if": "> 180 us at 64,96 (no clear edge over I's 233.8 us FMA conv), or the row shift cannot be expressed without 3x or more smem or L2 traffic, or every precision form exceeds 85 % of atol in the simulation",
    "model": "opus"
  }
]
```