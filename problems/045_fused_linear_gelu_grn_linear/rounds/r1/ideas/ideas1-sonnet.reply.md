### Notes
- Two arithmetic facts drive everything. FP16x3 on kind::f16 costs 786 kFLOP/token (6 passes), which is 0.43 ns/token at the 1.82 PF/s peak. The leader's slope is 0.72 ns/token at 50k, 100k and 200k tokens, with a fixed cost of about 10 µs.
- Two things fit that slope: 5 KB/token of DRAM traffic at 6.9 TB/s, or FP16x3 with no recompute at 60% of tensor peak.
- SIMT epilogue cost is probably the hidden bound. One exact GELU is about 2 MUFU per element, so about 64 clk/token/SM, or about 58 µs at 200k tokens. Recompute pays it twice.
- The H4 capacity limit does not apply if h stays on chip per batch, not per grid.
- Margin: FP16x3 max error is 2.3e-3 against an atol of 2.8e-3, so numerics need care.

```json ideas
[
{"title":"One-wave token-stationary FP16x3 tcgen05, h in TMEM, grid barrier (S and M, N<=18.9k)",
 "family":"grid-barrier on-chip",
 "design":"One launch with a grid of at most 148 CTAs, all co-resident. Each CTA owns t tokens, with t chosen from N as the multiple of 16 nearest N/148 (96 at 12.5k, 32 at 784). Swap-AB layout, as in the #129 lineage-M kernel: D[hidden 128-block][t tokens], 4 blocks, K=128 as 3 concatenated passes (hi·hi, hi·lo, lo·hi), fp32 accumulators in TMEM. Epilogue warps: +b1, exact-class GELU (A&S erfc), sumsq per channel, fp32 h written back into TMEM with tcgen05.st. Per-CTA sumsq partials go to a scratch [CTA][B_touched][512] with no atomics. A monotonic counter polled directly (about 1.2 µs, measured in #129) gives the barrier. The generation is old/148, so nothing is reset. After the barrier each CTA reduces its batches' partials from L2, computes s_b = 1+gw·nf, and builds h' = hi/lo fp16 of h·s into the smem B tile. GEMM2 is 3 passes with W2 hi/lo streamed in 128-hidden chunks. b2' = W2·gb + b2 comes from a prologue dot product. Weight hi/lo images are converted by 1/G slices per CTA into a workspace (torch.empty is allowed), then bulk-copied in (#129 recipe). Output is written coalesced with evict_last. Resources: 512 TMEM columns, about 200 KB smem. Compile with explicit max-dynamic-smem.",
 "p_expert":0.4,
 "effect":"Arithmetic per CTA: t=96 gives 75 MFLOP per CTA, 6.1 µs at the 12.3 TF/s per-SM peak, about 9.4 µs at 65%. Add about 5 µs for launch, weight ingest, barrier and epilogue. That is about 15 µs at 12544 tokens, against 285 now and a leader of 21-25. At 784-3136 tokens (t=16-32) the estimate is 9-11 µs against the leader's 10.5-12.7. At 6272 tokens it is about 11 µs. Predicted S score about 0.85, M about 0.88, 9 of 14 workloads covered. The larger L sizes are not covered.",
 "risk":"Epilogue SIMT (GELU twice per token on shared SMSPs) and the barrier chain eat the margin. Error is 82% of atol. Weight ingest of 512 KB per CTA (hi/lo) from L2 may take 3 µs before GEMM1 can finish.",
 "test":"Microbench: 148 CTAs, one FP16x3 GEMM1 tile with N=96 in TMEM, timed in SM cycles. Check the issue rate per 48-clk MMA and the epilogue cost per 96x512 tile. Check the error with numpy on 10 random inputs, using fp32 accumulation in the tcgen05 order."},

{"title":"Batch-group persistent dataflow: h never leaves TMEM, for every N including L",
 "family":"batch-group dataflow",
 "design":"The GRN dependency is per batch, so the on-chip capacity limit is per batch, not global. A persistent grid of 148 CTAs is split into groups of g(HW) CTAs: 2 for HW=196, 8-9 for 784, and 27-33 for 3136, with 96-token tiles. A group processes whole batches one after another: GEMM1 + GELU, then a counter wait among the group's CTAs (about 1.2 µs), then the scale and GEMM2. Each CTA keeps two 48-token sub-tiles from different batches and interleaves them. While one waits on its counter the other runs GEMM1, using 2 x 192 + 64 columns of TMEM. Counters are cumulative, so they need no reset. There is no recompute and no h in DRAM, so the work is 6 passes and 1 KB/token of DRAM traffic.",
 "p_expert":0.2,
 "effect":"At 200704 tokens: 0.43 ns/token at peak, about 0.66 at 65%, so about 132 µs. The score at that point is about 0.887, against 0.854 for the leader's implied 154. The largest sizes lose about 10% to idle SMs (a 3136-token batch takes 33 CTAs, so 4 groups use 132 of 148). The barrier chain costs about 1.5 µs per batch, hidden by interleaving. At 25k-100k tokens the estimate is 20, 34, 66 µs, against leader 32, 45, 81. Same S and M numbers as idea 1.",
 "risk":"#129 lineage H (image-counter dataflow, +58-98% vs the grid-barrier design) lost to per-item serial chains. A work-queue stall or a cross-group imbalance would kill it. Epilogue-vs-MMA contention (30-40% slowdown measured in #129) caps the gain.",
 "test":"Single-batch-group probe: one group of 33 CTAs processing 64 batches of 3136 tokens with a dummy epilogue. Measure µs per batch against the 8.2 µs MMA ideal, with and without two-tile interleave."},

{"title":"Cluster-of-8 hidden-split specialist: a batch of 196 tokens per cluster, no global sync",
 "family":"cluster hidden-split",
 "design":"For HW=196 (four of the 14 workloads, N from 1568 to 12544) one cluster of 8 CTAs owns one batch of 196 tokens (N=196 in one MMA, M=128 hidden rows). CTA j holds hidden slice 64j..64j+63, so it ingests only 64 KB of hi/lo weights instead of 512 KB, about 0.4 µs. Channel statistics are local to the CTA. mean_c G is one 8-way DSMEM exchange of 64 floats. GEMM2 is split-K over the slices, giving a partial of 196x128 fp32 per CTA. It is reduced by reduce-scatter through DSMEM, 88 KB received per CTA, which is the term to measure: at the 12 B/clk measured in #129 it is about 4.9 µs, too slow, so use a 2D split (cluster 4 x hidden 128 plus 2 token halves) or red.global.add into a pre-initialised output if DSMEM stays slow. Grid is 8 CTAs per batch with no co-residency requirement: 64, 128, 256 and 512 CTAs. For HW=784 (B=1) use a 49-token-tile cluster plus an 8-wide counter.",
 "p_expert":0.15,
 "effect":"Per-CTA compute 196 tokens x 786 kFLOP / 8 = 19 MFLOP, about 2.5 µs. Estimate 7-8 µs at 8,14,14, 16,14,14 and 32,14,14, against a leader of 11.0, 12.7 and 15.5. A gain of 3-5 µs at the smallest sizes, where dMean/dT is 0.8e-4 per µs, is about +0.003-0.005 of mean.",
 "risk":"DSMEM reduction bandwidth. Clusters of 8 may slow launches on 512-CTA grids (clustered large grids cost 2-5% on the portal).",
 "test":"DSMEM microbench: a cluster of 8 doing an all-to-all reduce-scatter of 100 KB per CTA, timed in SM cycles. Also the same with red.global.add f32 into L2."},

{"title":"Two-kernel recompute pipeline, FP16x3 tcgen05, PDL-linked",
 "family":"recompute 2-kernel",
 "design":"Kernel A (stats): token tile of 128 or 96, GEMM1 as 3 passes, GELU, sumsq; per-tile partials go to a workspace [tile][512] with no atomics. Kernel B launches with PDL and runs its W2/W1 hi/lo conversion prologue before griddepcontrol.wait. After the wait each CTA reduces its batch's partials (tiles x 512 floats from L2, up to 25 x 2 KB), computes s_b, builds W2'_b = W2 ⊙ s_b as hi/lo in smem, recomputes GEMM1, then GELU, hi/lo split of h, GEMM2 and a coalesced store. The b2' = W2·gb + b2 fold is also done in kernel B's prologue. This is the fall-back for every band, and the only one for N>=25k if idea 2 fails.",
 "p_expert":0.5,
 "effect":"9 passes, 1.18 MFLOP/token, 0.65 ns/token at peak, about 0.95 realistic. L: 190 µs at 200704 tokens (S about 0.80), 95 at 100352, 48 at 50176. Two GELUs per token (about 58 µs of MUFU each at 200k) are the extra cost over idea 2. S and M: 12-16 µs.",
 "risk":"The second GELU and the SIMT epilogue bind, not the tensor pipe. The PDL overlap needs the kernel A -> B ordering to be exact.",
 "test":"Measure kernel A's time per token at N=100352 with a null GELU against the real GELU. That splits MMA time from epilogue time."},

{"title":"L2-resident stored h: batch-ordered hi/lo fp16 h in a workspace, not recomputed",
 "family":"stored h",
 "design":"Kernel A computes h = GELU(GEMM1) with 3 passes, stores hi and lo fp16 planes (2 KB/token, the layout GEMM2 wants, so no later conversion) with evict_last, and writes sumsq partials. Kernel B (PDL) reads the planes by TMA straight into smem as the GEMM2 operand. Folding s_b into W2' makes the hi/lo split independent of the scale. Process batches in order, with kernel B starting as soon as the stats of a batch group are ready (a counter), so the reuse distance of h stays below about 30 MB. The working set then stays in the 126 MB L2 and the DRAM cost is only the write-back. DRAM traffic is about 3 KB/token (x in, h written back once, out). The 5 KB/token full round trip costs 145 µs at 7.05 TB/s. Bytes: 411 MB written, and read from L2 not DRAM.",
 "p_expert":0.25,
 "effect":"L: memory floor falls from 145 µs to about 86 µs at 200704 tokens, so compute binds (6 passes, 0.43 ns/token at peak): about 120-135 µs, 25% less than idea 4. At 50176-100352 tokens, 40-70 µs. At M, h fits in L2 outright, so about 14-17 µs.",
 "risk":"The dirty-L2 interaction: our h lines evict the harness's dirty flush lines and their write-back lands inside the window (about 4 µs at M and L in #38). The expected L2 residency may not hold because of the two-partition layout.",
 "test":"Streaming probe at 100 MB: write 2 KB/token with evict_last, then read it back after a counter. Compare time against the same with a full-DRAM round trip."},

{"title":"Compressed stored h: fp16 hi + int8 residual (3 B per element, 4 KB/token total)",
 "family":"stored h",
 "design":"Store h as fp16 hi plus a signed int8 residual q = round(127·(h - hi)/2^(e-11)), where e is hi's exponent. The reader reconstructs lo = q·2^(e-11)/127 with one FMUL and an exponent extract, then feeds hi and lo planes to the FP16x3 GEMM2 as usual. Mantissa precision is 11 + 7 = 18 bits (rms 2^-19.5 relative). That is 1.5 KB/token instead of 2 KB, so 4 KB/token of DRAM traffic instead of 5, and 3 KB/token with L2 reuse. Pack as separate planes (hi plane 1 KB, residual plane 512 B) so both stay TMA-able.",
 "p_expert":0.08,
 "effect":"If the memory floor binds (idea 5 not compute-limited), L falls 20%: 145 to 116 µs at 200704 tokens, +0.03 on that workload only (S 0.854 to 0.89). Decode cost 512 x 3 ops/token is about 3 µs of SIMT.",
 "risk":"18-bit h may fail: error budget is 2^-19, margin already thin (2.3e-3 of 2.8e-3). int8 residual rounding adds about 1.7e-3 at out std 260 if mis-scaled.",
 "test":"numpy: apply the compression to h, run fp64 GEMM2 with the rest exact, and compare max |err| over 10 inputs at all three HW values."},

{"title":"Hybrid: store half of the hidden blocks, recompute the other half",
 "family":"hybrid store/recompute",
 "design":"Two resources are idle at different times: DRAM during MMA phases and MMA during store phases. Pass 1 computes GEMM1 for all 4 hidden blocks (needed for sumsq), but stores h hi/lo only for blocks 0-1 (1 KB/token). Pass 2 reads blocks 0-1 from L2/DRAM and recomputes blocks 2-3 (1.5 extra units of the 6). Fraction f=0.5 recomputed is the model's balance point: compute 0.43+0.21f ns at peak, memory 0.14+0.57(1-f) ns.",
 "p_expert":0.1,
 "effect":"At f=0.36-0.5, 0.51-0.54 ns/token at peak, about 0.65 realistic, so about 130 µs at 200704 tokens. That is no better than idea 2 if idea 2 reaches 0.66, but it beats ideas 4 and 5 by 15-30 µs at 25k-100k tokens, where neither resource is saturated.",
 "risk":"Two code paths in one kernel; the balance depends on SIMT epilogue cost, which is unmeasured.",
 "test":"Run the idea-5 kernel A with 2 vs 4 stored blocks and the idea-4 kernel B at both modes. Read off the crossover where neither unit idles."},

{"title":"Epilogue diet: GELU saturation fast path and MUFU/FMA balanced erf",
 "family":"epilogue SIMT",
 "design":"z ~ N(0, 11.3²), so 72% of elements have |z|>4 and 55% have |z|>6. For |z|>6 GELU(z) = z (z>0) or <1e-8 in magnitude (z<0), exact in fp32 to the 1e-6 we need. Branch per 8-element group with a warp-uniform vote on max |z|<6 over the group; use the full A&S erfc only when needed. Saturated groups cost about 3 ops. Balance MUFU (16/clk/SM) against FMA: where the erf is needed, ex2 on MUFU and the reciprocal by Newton on FMA, packed f32x2 throughout. The hi/lo split uses cvt.rn.f16x2.f32 so two elements cost 3-4 ops. The per-batch s_b scale is folded into W2, so there is no per-element scale. sumsq stays as one packed FMA per pair. The fast path is an exact identity, not a data-dependent skip of work.",
 "p_expert":0.12,
 "effect":"Single GELU at the per-token level: MUFU-bound 64 clk/token/SM, 58 µs of 132 at 200704 tokens. The fast path removes about 60-70% of erf evaluations and so 35-45 µs of epilogue issue time. Under idea 4 it saves up to 90 µs at L (two GELUs); under idea 2, 10-20 µs once overlap is counted. At S and M, 1-2 µs of the epilogue chain per CTA.",
 "risk":"Divergence: a 32-lane vote over 8-element groups succeeds only about 30% of the time at p=0.72 per element. Compact first (ballot+popc) or the saving evaporates. The judge may look at a value-dependent branch.",
 "test":"One-warp microbench: GELU of 512 random N(0,11.3²) elements with and without the fast path, in SM cycles; also max |diff| against erff."},

{"title":"S band: 3xTF32 with the raw fp32 tile as the hi operand (no conversion, no weight-image workspace)",
 "family":"S-band precision",
 "design":"At S the tensor work is negligible (784 tokens is 0.35 µs at peak) and the weight-prep chain is the cost. Use kind::tf32. The hi pass reads the raw fp32 smem tile (hardware ignores the low 13 bits). Only the lo operand is computed, lo = x - (x & 0xFFFFE000), one LOP and one FSUB per element, stored to a second smem tile. Weights then go straight from L2 by bulk copy, without the convert-to-workspace, counter and re-load sequence (about 2.5 µs on the critical path in the #129 recipe). 3 passes at half rate is 6 fp16-equivalents, which is fine for N<=3136. Same single-wave structure as idea 1.",
 "p_expert":0.15,
 "effect":"Removes about 2-3 µs from the S chain, so 8-10 µs at 784-3136 tokens. dMean/dT is about 0.8e-4 per µs, so 2.5 µs on 4 S workloads is about +0.001.",
 "risk":"Truncated hi leaves lo with 13 bits, which TF32 truncates to 10, giving error about 2^-20 of x. The error test (82% of atol already) may fail. Smem for fp32 + lo of 128-hidden chunks is 128 KB.",
 "test":"numpy emulation with truncation hi and truncated lo at all three HW values; need max err <= 2.4e-3."},

{"title":"FA4-style fused MLP with x as A operand in TMEM, h hi/lo through tcgen05.cp (main kernel for L)",
 "family":"TMEM-A pipeline",
 "design":"The swap-AB form with both operands in smem reads about 146 B/clk at N=96 (4 KB weights + 3 KB tokens per 48-clk MMA), which exceeds the ~123 B/clk port #129 measured, so MMA runs at about 80% of ideal (about 65% at N=64). Instead: tokens are M=128; x hi/lo (128 K x 2 x 2 B = 64 KB) is put into TMEM once as the A operand (128 columns). Per 64-hidden chunk, B = W1 chunk from smem (N=64), giving acc1 (64 columns). The epilogue does GELU and the hi/lo split, and writes h hi/lo as the A operand for GEMM2 in TMEM (64+64 columns, via tcgen05.st or smem + tcgen05.cp). GEMM2 uses B = W2'_b chunk from smem (N=128), accumulating into acc2 (128 columns). Double-buffer acc1 and h_A. Budget: 128 + 2x64 + 2x128 + 128 = 640, so single-buffer h_A: 128 + 128 + 128 + 128 = 512 exactly. Use this as kernel B of ideas 4/5 and as the engine of idea 2 at L.",
 "p_expert":0.2,
 "effect":"Removes the smem-port limit on MMA time: A is read from TMEM, only B (about 64 B/clk) from smem. MMA efficiency about 95% vs 70-80%. L: -15 to -25% on the MMA-bound sizes, so 132 to about 105-110 µs at 200704 tokens. Also relieves the fence.proxy.async hand-off to GEMM2.",
 "risk":"TMEM budget is exact with no double buffering. tcgen05.cp and A-from-TMEM layouts are a lot of new code. The MUFU-heavy epilogue shares SMSPs with the issuer warp (30-40% slowdown measured in #129).",
 "test":"Microbench: 1000 MMAs M=128, N=64/128, K=16, A in smem vs A in TMEM. Measure clk per MMA. Prediction: N=64 is 50 clk vs about 33."}
]
```

```json leader-stories
[
{"story":"The leader runs a 3-4 kernel pipeline with FP16x3 tensor-core GEMMs and a stored h (fp32 or hi/lo), so large sizes are DRAM-bound. Per token that is 512 B in, 2 KB h written, 2 KB read back, 512 B out, which is 5 KB. At the leader's slope of 0.723 ns/token (36.3 µs per 50176 tokens, 72.5 per 100352, 153.9 vs 81.4 per 100352 more) it is 6.9 TB/s. The 10 µs intercept is about 3 launches of 3 µs.",
 "consistent_with":"Slope constant to 3 digits over a 4x range of N, i.e. a bandwidth law. 6.9 TB/s is right at the 7.05 read+write ceiling. The intercept matches a multi-launch chain.",
 "test":"Build idea 5 without L2-reuse scheduling (plain two kernels) and time 64,56,56: about 150 µs and an intercept near 10 µs confirms. If our L2-reuse version beats that at the same N by more than 20%, the leader did not do it."},
{"story":"The leader runs one persistent on-chip kernel with no recompute and no h in DRAM, but is compute-bound at about 60% of FP16 tensor peak: 786 kFLOP/token x 0.723 ns = 1.09 PF/s of 1.82. The HW=3136 penalty at equal N (24.7 vs 21.1-21.6 µs at 12544 tokens; 19.9 vs 15.5 at 6272) is a per-batch counter chain, which grows with tiles per batch.",
 "consistent_with":"The HW=3136 workloads are slower than HW=784/196 at equal token count; a stored-h pipeline has no reason to depend on HW. Also fits 10 µs of fixed cost from barrier plus weight ingest.",
 "test":"Idea 1/2 at 12544 tokens: if 4,56,56 comes out 3-4 µs slower than 16,28,28 in our own kernel with per-batch counters, the same effect explains the leader."},
{"story":"The leader does not score the same S everywhere (the leader gap assumes it does). It can reach mean 0.854 with S about 0.97 at the five L sizes (64,56,56 at about 83 µs, which is a near-peak compute-bound on-chip kernel at 0.43 ns/token plus overhead) and a lower S of about 0.70-0.75 at the S sizes (about 25 µs).",
 "consistent_with":"The same mean as the equal-S reading. It would mean the true L times are 40-50% below the tabulated 'leader µs' column, and the small sizes 2x above it.",
 "test":"No direct test of the leader. Check feasibility: if our idea-2 or idea-10 L-band kernel measures under 100 µs at 64,56,56, the story is physically plausible; if it floors near 130 µs, the story is unlikely and story 1 or 2 holds."}
]
```