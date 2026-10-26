### Audit

I ran six probes before the probe budget ran out: two failed to compile, one was cut off by the budget, and three gave numbers. All of them are about launch and barrier costs.

1. **Cooperative versus plain launch.** I10 and I15 rest on lineage F's r6 note that a plain launch is 1.8 µs cheaper than a cooperative one. I timed a kernel with 148 CTAs × 512 threads × 227 KB under the harness timer. Cooperative and plain launches take the same time. With no barriers, both run 1.9–2.1 µs. With two barriers, both run 5.6–5.7 µs. CUPTI shows no extra activity for the cooperative launch. The +0.005–0.012 these two ideas claim is not there, and a plain launch would add the risk of a hang for no gain.

2. **Barrier layout.** I14 and I15 assume that keeping the pollers off the arrival counter's cache line speeds up the release. With all CTAs arriving at once, it is the other way round:
   - The fastest form uses a single counter that only ever counts up. Each CTA does one atomic add, works out its target from the value returned, and polls that same counter. It costs 1.21 µs per barrier, needs no reset, and wraps only after about 29M barriers.
   - The counter plus a separate flag line costs 1.80 µs.
   - Per-CTA private flags cost 2.12 µs with one fence. They cost 4.0 µs with a release store per flag, because those stores run one after another.

   I could not test arrivals spread out in time, as on the sc rows, before the budget ran out.

3. **What actually limits the tensor core at small N.** The ledger (H31) reads the 1.57× slowdown at N=64 as an instruction-rate limit. Three separate measurements instead point to a cap on reading MMA operands from shared memory of about 123 B/clk:
   - r11 at N=64: 6 KB in 50 clk.
   - r11 at N=128: 8 KB in 66 clk.
   - Lineage L at N=112: 7.5 KB in 60 clk.

   This changes how small-N tile plans (I18, I20, sc_kernel, and G at N=64) should price the MMA. It also undercuts I7's mechanism:
   - At N≈112 the port caps a 64-k block at about 244 clk.
   - D measures about 550 clk per block (3.4–3.6 µs over 12 blocks).
   - That works out to 30 KB landed per block at about 55 B/clk, which is the TMA landing limit in H31, not the port.
   - Putting A in TMEM still lands A through shared memory (tcgen05.cp), so it does not touch that limit.

4. **Error claims for cheaper GELU forms.** By my arithmetic, the errors claimed in I27 and I1 are too low.
   - **I27:** its polynomial does reach about 3e-6, but only for |x| ≤ 3.9. At the clamp, x·(1−Φ(x)) leaves about 1.9e-4. Moving the clamp out to 5.5 raises the fit error of a degree-8 (in x²) polynomial to about 1e-4.
   - **I1:** a half2 erfc core picks up about 1–3e-3 relative error, which is about 3e-4 in h near |x|≈1, not 8e-5.

   Both are about the size of the fp16 rounding h already has. Neither blocks shipping, but each needs the 10-round error log.

```json premises
[{"claim":"A plain launch saves 1.8 us over cudaLaunchCooperativeKernel, so D's single-launch kernels can gain 1.5-2 us per row","where":"pool I10, I15; lineage F r6 K3 note","verdict":"wrong","why":"Same 148x512x227 KB kernel under harness_time: cooperative 1.92-2.06 us vs plain 1.90-2.08 (empty body); 5.60-5.66 vs 5.62-5.66 with two barriers; CUPTI shows only the kernel. F's 1.8 us must have come from its K3 configuration (several waves of 512-thread blocks), not from the launch mode","if_wrong":"I10 is worth 0 and I15 keeps only its barrier part; switching to a plain launch adds hang risk with no gain","test":"done (probe)"},
 {"claim":"Separating the arrival line from the wait line, or per-CTA private flags written by the last arriver, releases faster than a flat counter","where":"pool I14, I15; notebook r9 explanation of the flat barrier's loss on sc rows","verdict":"wrong","why":"With all CTAs arriving at once, marginal cost per barrier (8- vs 2-barrier kernels): one monotonic counter polled directly 1.21 us; relaxed atomic plus fences 1.29; counter plus separate flag line 1.80; counter and generation with reset 1.94-2.02; private flags with one fence 2.12; private flags with a release store each 4.0","if_wrong":"I14 is a regression and I15's barrier part has nothing to gain; the cheapest form is the monotonic counter with a target computed from the returned value (no reset or departure counter)","test":"a skewed-arrival version of the same microbench (one CTA late by 1.5 us, the rest 0-1 us) to settle the sc-row case"},
 {"claim":"Tensor-core issue at N=64 is 1.57x of ideal because of the instruction rate; pairs add rate only at N=64","where":"ledger H31 and lineage D notes (r11 K STEP 0a)","verdict":"overstated","why":"r11 N=64 (6 KB per 50 clk), r11 N=128 (8 KB per 66.4) and L's N=112 (7.5 KB per 60.4) all read about 123 B/clk of smem operands: a port cap. MMA time per k16 is about max(N/2, (4096+32N)/123) clk","if_wrong":"small-N tile plans (I18 N=24/72, I20 N=25-66, sc 64-column tiles, G at N=64) under-price the MMA by up to 3x; A in TMEM is the only lever, and it matters only where the MMA binds","test":"the r11 straight-line bench at N=32/64/96 with A in smem vs in TMEM"},
 {"claim":"D's 7x7 GEMM1 mainloop is bound by the smem port, so W1' in TMEM cuts it from 3.4 to 2 us","where":"pool I7","verdict":"wrong","why":"Port bound for N=112: 4 x 7.5 KB / 123 = 244 clk per 64-k block; measured about 550 clk = 30 KB landed at 55 B/clk, the H31 TMA landing cap. Loading A into TMEM goes through smem (tcgen05.cp), so the same bytes still land","if_wrong":"I7's +0.0035 drops to about the 9% port share at N=112","test":"stamp one D GEMM1 tile with A already resident (skip the A TMA): if the block time drops to <=300 clk, landing binds, as computed"},
 {"claim":"A polynomial erf (I27) is accurate to <3e-6; the half2 erfc core (I1) adds <=8e-5 to h","where":"pool I27, I1","verdict":"overstated","why":"Degree-8-in-x^2 fit up to erf argument 2.76 is about 3e-6, but the clamp at |x|=3.9 leaves x(1-Phi(x)) = 1.9e-4; extending to |x|=5.5 makes the fit error about 1e-4. fp16 Horner plus ex2.f16 gives 1-3e-3 relative on erfc, about 3e-4 in h near |x|=1. Both are about the size of the fp16 rounding of h","if_wrong":"each needs the 10-round fp64 log; margins shrink a little (about +5% rms), still passing on the tc rows","test":"numpy: max |GELU error| over [-9,9] for both forms, then a one-row 10-round log"}]
```
```json probes
[{"idea":"I10 / I15 (launch mode)","test":"harness_time, 148 CTAs x 512 thr x 227 KB, cooperative vs plain, empty body and 2-barrier body, 5 interleaved rounds","result":"empty 1.92-2.06 vs 1.90-2.08 us; 2 barriers 5.60-5.66 vs 5.62-5.66 us","verdict":"fails","note":"cooperative launch costs nothing on the GPU timeline"},
 {"idea":"I14 / I15 (barrier layout)","test":"marginal cost per barrier from 2- vs 8-barrier kernels, all CTAs arriving at once, 3-4 rounds","result":"monotonic counter polled directly 1.21 us; relaxed atomic plus fences 1.29; separate flag line 1.80; count and generation with reset 1.94-2.02; private flags with one fence 2.12; private flags with a release store each 4.0","verdict":"fails","note":"fan-out releases run one after another; skewed arrival not measured (probe budget used up)"},
 {"idea":"I2 / I17 / I25 (cache hints), I27, I24","test":"not run","result":"probe budget used up","verdict":"unclear","note":"I25 and I24 have the best expected value of the untested ideas; run their stated tests first"}]
```