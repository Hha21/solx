# Problem card: L1 #45 `045_fused_linear_gelu_grn_linear`

## 1. Semantics
Reference (ConvNeXtV2 MLP block, NHWC):
- `h = gelu_erf(x W1^T + b1)`. x is [B,H,W,128], W1 is [512,128] and h is [B,H,W,512].
- `G[b,c] = sqrt(sum_{hw} h²)`, `nf[b,c] = G / (mean_c G + eps)`, with eps = 1e-6.
- `x2 = gw*(h*nf) + gb + h`. gw and gb are [1,1,1,512].
- `out = x2 W2^T + b2`. W2 is [128,512] and out is [B,H,W,128].
- DPS: `run(hidden_states, pwconv1_weight, pwconv1_bias, grn_weight, grn_bias, pwconv2_weight, pwconv2_bias, eps, output)`; eps is a Python float.
- Write N = B·HW tokens. Per token: GEMM1 is 128→512 and GEMM2 is 512→128, so 2·2·128·512 = **262,144 FLOP**. Memory is 512 B in and 512 B out.
- **GRN is a global reduction over HW per batch.** GEMM2 for any token depends on every token of its batch, so a design must do one of: recompute GEMM1, store h (4× the I/O bytes), or keep h on chip across a grid-wide barrier.
- Algebra: `x2 = h ⊙ s_b + gb` with `s_b = 1 + gw·nf_b`. So out = h (W2 ⊙ s_b)^T + (W2 gb + b2), which means the per-batch scale can be folded into W2. The difference in numerics is about 1 ulp.
- HW ∈ {196, 784, 3136}. None of these is a multiple of 64, so token tiles need masking or a tile that divides HW (49 and 28 divide all three).
- All weights are randn: none of the names match the ones/zeros heuristics [harness io.py]. gw and gb are N(0,1), so s_b ranges roughly from -2 to 4.

## 2. Numerics and tolerance
- Tolerance per workload: atol 2.8e-3 to 3.4e-3 (three seen: 0.0029, 0.0028, 0.0034), rtol 1e-5, 99% matched. |out| has std ≈ 260 and max ≈ 1500, so the rtol term is ≤ 0.015 and atol dominates. The **relative budget is ≈ 1e-5 of the output std**: fp32 GEMM accuracy is required.
- fp32 reference against fp64: max error is already 1.3–1.5e-3, half the budget [probe].
- Emulated precision options on the harness distribution (fp32 accumulation) [probe_b200]:

| Option | max err | fail % |
|---|---|---|
| fp32 (any order) | 1.3–1.5e-3 | 0 |
| TF32 (1 pass) | 0.6–0.7 | 96% |
| TF32 with only W split (2 pass) | 0.4–0.5 | 95% |
| BF16x3 | 9.5e-3–1.1e-2 | 0.8–2.2% (fails) |
| **3xTF32** (hi·hi + hi·lo + lo·hi) | 1.8–2.3e-3 | **0** |
| **FP16x3** (same split, with or without 2^12 scaling of lo) | 1.8–2.3e-3 | **0** |
| cuBLAS BF16x9 emulation (`CUBLAS_EMULATE_SINGLE_PRECISION`) | better than fp32 | 0 |

- Both GEMMs need the split. TF32 error in GEMM1 alone (h error about 0.05) gives about 1 in out.
- The GRN statistics need relative error ≲ 1e-5 (Δout ≈ 22·|gw·h·nf|·ε), so pass 1 cannot use 1-pass TF32 either.
- GELU must use the erf form. `erff` with `--use_fast_math` passed. Keep `sqrt` and `/` in GRN exact (`__fsqrt_rn`, `__fdiv_rn`); they are cheap.

## 3. Workloads
Bytes = N·1024 + 0.53 MB weights. FLOP = N·262,144. The SIMT figure assumes the fp32 FMA peak of 56.8 TF/s at 1.5 GHz, without the recompute; 3xTF32 TC assumes 0.9 PF/s TF32 / 3; FP16x3 assumes 1.8 PF/s / 3. Floor and reference times are on the rented B200 (unlocked clocks).

| B,H,W | N tok | MB | GFLOP | band | copy floor µs | torch ref µs | SIMT µs | 3xTF32 TC µs | FP16x3 TC µs |
|---|---|---|---|---|---|---|---|---|---|
| 1,28,28 | 784 | 1.33 | 0.21 | S | 2.53 | 106 | 3.6 | 0.7 | 0.3 |
| 8,14,14 | 1568 | 2.14 | 0.41 | S | 2.69 | 108 | 7.2 | 1.4 | 0.7 |
| 4,28,28 | 3136 | 3.74 | 0.82 | S | 3.07 | 116 | 14.5 | 2.7 | 1.4 |
| 16,14,14 | 3136 | 3.74 | 0.82 | S | 3.04 | 113 | 14.5 | 2.7 | 1.4 |
| 2,56,56 | 6272 | 6.95 | 1.64 | M | 3.65 | 137 | 29 | 5.5 | 2.7 |
| 32,14,14 | 6272 | 6.95 | 1.64 | M | 3.78 | 128 | 29 | 5.5 | 2.7 |
| 64,14,14 | 12544 | 13.4 | 3.29 | M | 4.83 | 202 | 58 | 11 | 5.5 |
| 4,56,56 | 12544 | 13.4 | 3.29 | M | 4.82 | 211 | 58 | 11 | 5.5 |
| 16,28,28 | 12544 | 13.4 | 3.29 | M | 4.74 | 214 | 58 | 11 | 5.5 |
| 32,28,28 | 25088 | 26.2 | 6.58 | L | 6.91 | 404 | 116 | 22 | 11 |
| 64,28,28 | 50176 | 51.9 | 13.2 | L | 11.1 | 822 | 232 | 44 | 22 |
| 16,56,56 | 50176 | 51.9 | 13.2 | L | 11.0 | 816 | 232 | 44 | 22 |
| 32,56,56 | 100352 | 103 | 26.3 | L | 19.6 | 1559 | 463 | 88 | 44 |
| 64,56,56 | 200704 | 206 | 52.6 | L | 35.8 | 3028 | 926 | 175 | 88 |

Latency depends only on (B, HW): there are 11 distinct sizes. The table's 3xTF32 and FP16x3 columns count 2 GEMMs; recomputing GEMM1 adds 50% to the GEMM time.

## 4. Bounds per band
- Arithmetic intensity is 256 FLOP/B, against a B200 balance of about 225 FLOP/B at 1.8 PF and 8 TB/s for 16-bit MMA. SOLAR probably takes Tsol = max(FLOP / ~1.8 PF, bytes / 8 TB/s) ≈ 29 µs at 64,56,56, roughly balanced. This is INFERRED; the first portal page gives Tsol per workload.
- **Every fp32-accurate path is compute-bound at M and L**, even FP16x3 on tensor cores: 88 µs + recompute at 206 MB against a 36 µs copy. fp32 SIMT is 25–40× slower than the memory floor. At S, launch and latency plus the serial per-CTA work dominate: N/64 = 13–49 CTAs only.
- Splits of the stored h: x2 is ≤ ~300, fine for FP16 range. lo parts of small values become subnormal in fp16, but the probe shows no harm.

## 5. Where the reference loses (facts)
- About 7+ launches: GEMM1 (cuBLAS fp32 SIMT, 592 µs alone at 64,56,56, rented), GELU, the norm reduction, the mean, the affine (several passes over the 411 MB h tensor), then GEMM2 (500 µs). h is materialised (N·2 KB) and re-read 3–4 times.
- cuBLAS's BF16x9 emulation of fp32 is only about 2× faster on these skinny GEMMs (289 and 390 µs). K=128 and N=128 are too small to amortise the 9 products.

## 6. Decisions any design must make (options and their arithmetic, not chosen)
1. **GEMM precision engine.**
   - fp32 SIMT: exact; about 57 TF/s, so about 0.93 ms plus recompute at L.
   - 3xTF32 tcgen05 (kind::tf32): 3 MMAs per product; operand splits done in-kernel or by TMA plus conversion.
   - FP16x3 tcgen05 (kind::f16): 2× the TF32 rate; the split costs cvt per element.
   - BF16x9-style: too many products. Any 2-term split fails.
   - Note: the hi·lo terms can share one MMA by concatening along K: [a_hi | a_lo] · [w_lo | w_hi]^T, with K=256.
2. **How GRN's global dependency is crossed.**
   - (a) Recompute GEMM1: 3 GEMM-equivalents in 2–3 launches.
   - (b) Store h: +2 KB/token written and read, about +100 µs at L; it saves GEMM1 FLOPs.
   - (c) One persistent cooperative kernel with a grid barrier: keep h (or x) on chip. TMEM plus smem is about 37 MB of h in fp32, enough for N ≤ ~18k (all S and M). It needs a self-resetting barrier counter and partials without a memset.
   - (d) Store only x2's split operands in fp16 hi/lo (same bytes as fp32 h).
3. **Launch structure.** Options are 3 kernels (stats, scalars, main), 2 kernels (merge the scalars into main's prologue: each CTA reduces tiles × 512 partials from L2), or 1 cooperative kernel. Each extra launch costs about 2–5 µs at S.
4. **Folding.** s_b into W2 per batch (W2'_b = W2 ⊙ s_b, a per-batch weight, 256 KB × B) against applying it to h in registers. gb·W2 folded into the bias.
5. **Tiling.** Token tile of 49 or 28 (divides HW) or masked 64/128. Hidden chunking of GEMM1 → GEMM2 (on-chip x2 chunk as GEMM2's A operand, as in the first kernel). At S, smaller tiles or a split-hidden to occupy 148 SMs (784 tokens = 6 tiles of 128!).
6. **Weights.** W1 and W2 are 256 KB each in fp32, L2-resident. Per CTA they are streamed by chunks, or held in smem split-converted (hi and lo in fp16 = 256 KB per matrix: too big for one SM, so stream them).

## 7. Results so far (rented B200, run_tests, unlocked clocks)
`r0-simt-3k` (fp32 SIMT, 3 launches, TM=64, recompute) passes 14/14. Times in µs:

| Workload | µs |
|---|---|
| 1,28,28 | 139 |
| 8,14,14 | 137 |
| 4,28,28 | 138 |
| 16,14,14 | 137 |
| 2,56,56 | 143 |
| 32,14,14 | 138 |
| 64,14,14 | 221 |
| 4,56,56 | 221 |
| 16,28,28 | 220 |
| 32,28,28 | 325 |
| 64,28,28 | 613 |
| 16,56,56 | 613 |
| 32,56,56 | 1113 |
| 64,56,56 | 2191 |

It is slower than the torch reference at S and M, because of the serial per-CTA path with only 13–98 CTAs. It is 1.4× faster than the reference at the largest size.

## 8. Open questions
- What Tb and Tsol are per workload. Is Tb a TF32-free PyTorch, i.e. ≈ cuBLAS fp32 plus fused elementwise ops, so about 1.2 ms at the largest size?
- Real tcgen05 3xTF32 / FP16x3 error with the hardware's accumulation order (the emulation used fp32 GEMMs of split operands).
- Whether a cooperative single kernel's grid barrier is acceptable to the judge (it should be: the work is not hidden).
- The full list of per-workload atol values (only 3 seen).
