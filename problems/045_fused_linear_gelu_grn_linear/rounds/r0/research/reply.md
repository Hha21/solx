The first kernel passes all 14 workloads, and I've checked which precision options meet the tolerance. Next I'm writing the problem card, the ledger and the candidate.

### Rationale
The first kernel is plain fp32 SIMT with three launches: stats, GRN scalars, then the main pass. It passes 14/14 workloads on the rented B200. It is slow because the problem is **compute-bound, not memory-bound**: 262 kFLOP per token against 1 KB per token. The precision probes are the key finding for the design rounds. Plain TF32 fails about 96% of elements. BF16x3 fails 1–2%. 3xTF32 and FP16x3 pass with zero failures (max error 2.3e-3 against atol 2.8–3.4e-3). So a tensor-core path exists, but it costs 3 MMAs per product.

```markdown card
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
```

```yaml ledger
constraints:
- 'No deliberate delay before launch (CPU busy-wait, sleep, extra host work) and nothing else meant to change when our kernel reaches the GPU relative to the harness''s flush memset. It exploits the timing methodology, not kernel speed. Off-limits unless NVIDIA approves.'
- 'No discard/invalidate of cache lines (discard.global.L2 etc.), even on our own inputs: it targets the harness''s measurement, not kernel speed. Off-limits unless NVIDIA approves.'
- Never touch memory we do not own; no state carried between calls except a scratch counter that is correct for every call.
- 'No overlap with the harness''s own kernels (no PDL against the flush memset, no other streams): it hides work from the timed window.'
- 'Clusters only on small one-wave grids or where a GEMM tile needs them (2SM pairs).'
- 'Eviction-priority operations only on the current call''s own input/output buffers, inside our kernel.'
- 'No environment variables set from the solution (e.g. CUBLAS_EMULATE_SINGLE_PRECISION) without operator approval.'
hypotheses:
- id: H1
  statement: Every size is compute-bound for any fp32-accurate engine; the GEMM engine (SIMT vs 3xTF32 vs FP16x3 tcgen05) decides the score more than any memory-side tuning.
  status: open
  evidence:
  - 'arithmetic: 262 kFLOP/token vs 1 KB/token; SIMT 0.93 ms vs 36 us copy at 64,56,56'
  - 'r0-simt-3k: 2191 us at 64,56,56 (61x copy)'
- id: H2
  statement: Split-precision tensor-core GEMMs (3xTF32 or FP16x3, fp32 accumulate) meet the tolerance with zero failing elements; any 1- or 2-term scheme and BF16x3 fail.
  status: open
  evidence:
  - 'probe_b200 emulation: 3xTF32 and FP16x3 max err 1.8-2.3e-3, 0% fail; BF16x3 0.8-2.2% fail; TF32 96% fail; TF32 with W-only split 95%'
  - 'margin is thin: the fp32 reference itself is 1.3-1.5e-3 from fp64'
- id: H3
  statement: FP16x3 on kind::f16 beats 3xTF32 on kind::tf32 by ~2x MMA rate, outweighing the in-kernel split/convert cost.
  status: open
  evidence:
  - 'emulation shows identical error for both (probe_b200)'
- id: H4
  statement: For N <= ~18k tokens (all S and M) a single persistent kernel that keeps h on chip (TMEM+smem) across a grid barrier beats recompute and multi-launch designs.
  status: open
  evidence:
  - 'arithmetic: 148 x (256 KB TMEM) ~ 37.9 MB vs h = N x 2 KB'
- id: H5
  statement: At L, recomputing GEMM1 (+50% FLOPs) beats storing h (+2 KB/token of traffic) once GEMMs run on tensor cores.
  status: open
  evidence:
  - 'arithmetic at 64,56,56: recompute ~+44-88 us (FP16x3/3xTF32) vs store ~+100 us at 8 TB/s'
- id: H6
  statement: At S the first kernel is limited by too few CTAs (13-49) running long serial chunk loops; splitting work over hidden chunks or smaller token tiles to fill 148 SMs gives >5x at S.
  status: open
  evidence:
  - 'r0-simt-3k: 137-143 us flat from 784 to 6272 tokens'
- id: H7
  statement: Folding s_b = 1 + gw*nf into a per-batch W2 (and gb into the bias) is within tolerance and removes the per-element affine from the hot loop.
  status: open
  evidence:
  - 'algebra exact; rounding difference ~1 ulp of x2'
```

```json solution-spec
{"name": "r0-simt-3k", "definition": "045_fused_linear_gelu_grn_linear", "author": "solx-loop", "description": "fp32 SIMT, 3 launches: GEMM1+GELU stats partials, per-batch GRN scalars, recompute GEMM1 + GRN + chunked GEMM2", "spec": {"languages": ["cuda_cpp"], "target_hardware": ["B200", "LOCAL"], "entry_point": "binding.cpp::run", "dependencies": ["torch"], "destination_passing_style": true, "compile_options": {"cuda_cflags": ["-O3", "--use_fast_math", "-std=c++17"]}}}
```

```cuda file=kernel.cu
// First kernel for 045_fused_linear_gelu_grn_linear: fp32 SIMT, three launches.
//   k_stats: h = gelu(x W1^T + b1) per 64-token tile, per-tile sum of h^2 per channel -> part
//   k_grn:   per batch, G = sqrt(sum part), nf = G / (mean_c G + eps)
//   k_main:  recompute h, x2 = gw*(h*nf) + gb + h, out = x2 W2^T + b2
#include <cuda_runtime.h>
#include <stdint.h>

namespace {
constexpr int D = 128, HID = 512, TM = 64, NC = 64, NT = 256;
constexpr int NCHUNK = HID / NC;   // 8 hidden chunks of 64
constexpr int LDK = D + 4;         // 132: row stride of the x tile and the W1 chunk
constexpr int LDH = NC + 4;        // 68: row stride of the h chunk and the W2 chunk

__device__ __forceinline__ float gelu_erf(float v) {
  return v * 0.5f * (1.0f + erff(v * 0.7071067811865476f));
}

// rows x 128 floats, row-major source with stride 128, into smem with stride LDK; rows >= nvalid zero-filled
__device__ __forceinline__ void load_rows128(float* dst, const float* src, int nrows, int nvalid, int tid) {
  for (int i = tid; i < nrows * (D / 4); i += NT) {
    int r = i >> 5, k4 = (i & 31) << 2;
    float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
    if (r < nvalid) v = __ldg(reinterpret_cast<const float4*>(src + (size_t)r * D + k4));
    *reinterpret_cast<float4*>(dst + r * LDK + k4) = v;
  }
}

// acc[i][j] = sum_k As[tm*4+i][k] * Ws[tn+16j][k], K = 128
__device__ __forceinline__ void gemm1(float (&acc)[4][4], const float* As, const float* Ws, int tm, int tn) {
#pragma unroll
  for (int i = 0; i < 4; ++i)
#pragma unroll
    for (int j = 0; j < 4; ++j) acc[i][j] = 0.f;
#pragma unroll 4
  for (int k = 0; k < D; k += 4) {
    float4 a[4], w[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) a[i] = *reinterpret_cast<const float4*>(As + (tm * 4 + i) * LDK + k);
#pragma unroll
    for (int j = 0; j < 4; ++j) w[j] = *reinterpret_cast<const float4*>(Ws + (tn + 16 * j) * LDK + k);
#pragma unroll
    for (int i = 0; i < 4; ++i)
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        acc[i][j] = fmaf(a[i].x, w[j].x, acc[i][j]);
        acc[i][j] = fmaf(a[i].y, w[j].y, acc[i][j]);
        acc[i][j] = fmaf(a[i].z, w[j].z, acc[i][j]);
        acc[i][j] = fmaf(a[i].w, w[j].w, acc[i][j]);
      }
  }
}

__global__ void __launch_bounds__(NT) k_stats(const float* __restrict__ x, const float* __restrict__ w1,
                                              const float* __restrict__ b1, float* __restrict__ part,
                                              int HW, int tiles) {
  extern __shared__ float4 smem4[];
  float* As = reinterpret_cast<float*>(smem4);  // [TM][LDK]
  float* Ws = As + TM * LDK;                    // [NC][LDK]
  float* Red = Ws + NC * LDK;                   // [8 warps][NC]
  const int tile = blockIdx.x, b = blockIdx.y, tid = threadIdx.x;
  const int tm = tid >> 4, tn = tid & 15, warp = tid >> 5, lane = tid & 31;
  const int m0 = tile * TM, rows = min(TM, HW - m0);
  load_rows128(As, x + ((size_t)b * HW + m0) * D, TM, rows, tid);
  float* pout = part + ((size_t)b * tiles + tile) * HID;
  for (int c = 0; c < NCHUNK; ++c) {
    __syncthreads();
    load_rows128(Ws, w1 + (size_t)c * NC * D, NC, NC, tid);
    __syncthreads();
    float acc[4][4];
    gemm1(acc, As, Ws, tm, tn);
    float sq[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      float bias = __ldg(b1 + c * NC + tn + 16 * j);
      float s = 0.f;
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        float h = gelu_erf(acc[i][j] + bias);
        if (tm * 4 + i < rows) s = fmaf(h, h, s);
      }
      s += __shfl_xor_sync(0xffffffffu, s, 16);
      sq[j] = s;
    }
    if (lane < 16) {
#pragma unroll
      for (int j = 0; j < 4; ++j) Red[warp * NC + tn + 16 * j] = sq[j];
    }
    __syncthreads();
    if (tid < NC) {
      float s = 0.f;
#pragma unroll
      for (int w = 0; w < NT / 32; ++w) s += Red[w * NC + tid];
      pout[c * NC + tid] = s;
    }
  }
}

__global__ void __launch_bounds__(HID) k_grn(const float* __restrict__ part, float* __restrict__ nf, int tiles,
                                             float eps) {
  __shared__ float red[HID / 32];
  const int b = blockIdx.x, c = threadIdx.x;
  const float* p = part + (size_t)b * tiles * HID + c;
  float s = 0.f;
  for (int t = 0; t < tiles; ++t) s += p[(size_t)t * HID];
  float g = __fsqrt_rn(s);
  float r = g;
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) r += __shfl_xor_sync(0xffffffffu, r, o);
  if ((c & 31) == 0) red[c >> 5] = r;
  __syncthreads();
  float tot = 0.f;
#pragma unroll
  for (int w = 0; w < HID / 32; ++w) tot += red[w];
  float mean = __fdiv_rn(tot, (float)HID);
  nf[(size_t)b * HID + c] = __fdiv_rn(g, mean + eps);
}

__global__ void __launch_bounds__(NT) k_main(const float* __restrict__ x, const float* __restrict__ w1,
                                             const float* __restrict__ b1, const float* __restrict__ gw,
                                             const float* __restrict__ gb, const float* __restrict__ w2,
                                             const float* __restrict__ b2, const float* __restrict__ nf,
                                             float* __restrict__ out, int HW) {
  extern __shared__ float4 smem4[];
  float* As = reinterpret_cast<float*>(smem4);  // [TM][LDK]   x tile
  float* Ws = As + TM * LDK;                    // W1 chunk [NC][LDK] or W2 chunk [D][LDH]
  float* Hs = Ws + D * LDH;                     // [TM][LDH]   x2 chunk
  const int tile = blockIdx.x, b = blockIdx.y, tid = threadIdx.x;
  const int tm = tid >> 4, tn = tid & 15;
  const int m0 = tile * TM, rows = min(TM, HW - m0);
  load_rows128(As, x + ((size_t)b * HW + m0) * D, TM, rows, tid);
  const float* nfb = nf + (size_t)b * HID;
  float acc2[4][8];
#pragma unroll
  for (int i = 0; i < 4; ++i)
#pragma unroll
    for (int j = 0; j < 8; ++j) acc2[i][j] = 0.f;
  for (int c = 0; c < NCHUNK; ++c) {
    __syncthreads();
    load_rows128(Ws, w1 + (size_t)c * NC * D, NC, NC, tid);
    __syncthreads();
    float acc[4][4];
    gemm1(acc, As, Ws, tm, tn);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const int n = c * NC + tn + 16 * j;
      const float bias = __ldg(b1 + n), g = __ldg(gw + n), be = __ldg(gb + n), f = __ldg(nfb + n);
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        float h = gelu_erf(acc[i][j] + bias);
        Hs[(tm * 4 + i) * LDH + tn + 16 * j] = g * (h * f) + be + h;
      }
    }
    __syncthreads();
    // W2 chunk: Ws[o][k2] = w2[o][c*NC + k2], 128 rows x 16 float4
    for (int i = tid; i < D * (NC / 4); i += NT) {
      int o = i >> 4, k4 = (i & 15) << 2;
      *reinterpret_cast<float4*>(Ws + o * LDH + k4) =
          __ldg(reinterpret_cast<const float4*>(w2 + (size_t)o * HID + c * NC + k4));
    }
    __syncthreads();
#pragma unroll 2
    for (int k = 0; k < NC; k += 4) {
      float4 a[4];
#pragma unroll
      for (int i = 0; i < 4; ++i) a[i] = *reinterpret_cast<const float4*>(Hs + (tm * 4 + i) * LDH + k);
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        float4 w = *reinterpret_cast<const float4*>(Ws + (tn + 16 * j) * LDH + k);
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          acc2[i][j] = fmaf(a[i].x, w.x, acc2[i][j]);
          acc2[i][j] = fmaf(a[i].y, w.y, acc2[i][j]);
          acc2[i][j] = fmaf(a[i].z, w.z, acc2[i][j]);
          acc2[i][j] = fmaf(a[i].w, w.w, acc2[i][j]);
        }
      }
    }
  }
  float* ob = out + ((size_t)b * HW + m0) * D;
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    const int o = tn + 16 * j;
    const float bias = __ldg(b2 + o);
#pragma unroll
    for (int i = 0; i < 4; ++i)
      if (tm * 4 + i < rows) ob[(size_t)(tm * 4 + i) * D + o] = acc2[i][j] + bias;
  }
}

constexpr int SMEM_STATS = (TM * LDK + NC * LDK + (NT / 32) * NC) * 4;  // 69,632 B
constexpr int SMEM_MAIN = (TM * LDK + D * LDH + TM * LDH) * 4;         // 86,016 B
static_assert(D * LDH >= NC * LDK, "Ws must hold either chunk");
}  // namespace

extern "C" void launch_grn_mlp(const float* x, const float* w1, const float* b1, const float* gw, const float* gb,
                               const float* w2, const float* b2, float eps, float* out, float* part, float* nf,
                               int B, int HW, cudaStream_t stream) {
  static bool attrs_set = false;
  if (!attrs_set) {
    cudaFuncSetAttribute(k_stats, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_STATS);
    cudaFuncSetAttribute(k_main, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_MAIN);
    attrs_set = true;
  }
  const int tiles = (HW + TM - 1) / TM;
  dim3 grid(tiles, B);
  k_stats<<<grid, NT, SMEM_STATS, stream>>>(x, w1, b1, part, HW, tiles);
  k_grn<<<B, HID, 0, stream>>>(part, nf, tiles, eps);
  k_main<<<grid, NT, SMEM_MAIN, stream>>>(x, w1, b1, gw, gb, w2, b2, nf, out, HW);
}
```

```cpp file=binding.cpp
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

extern "C" void launch_grn_mlp(const float* x, const float* w1, const float* b1, const float* gw, const float* gb,
                               const float* w2, const float* b2, float eps, float* out, float* part, float* nf,
                               int B, int HW, cudaStream_t stream);

void run(torch::Tensor hidden_states, torch::Tensor pwconv1_weight, torch::Tensor pwconv1_bias,
         torch::Tensor grn_weight, torch::Tensor grn_bias, torch::Tensor pwconv2_weight, torch::Tensor pwconv2_bias,
         double eps, torch::Tensor output) {
  TORCH_CHECK(hidden_states.is_cuda() && hidden_states.scalar_type() == torch::kFloat32, "fp32 CUDA input");
  TORCH_CHECK(hidden_states.dim() == 4 && hidden_states.size(3) == 128, "[B,H,W,128]");
  TORCH_CHECK(pwconv1_weight.size(0) == 512 && pwconv1_weight.size(1) == 128, "W1 [512,128]");
  TORCH_CHECK(pwconv2_weight.size(0) == 128 && pwconv2_weight.size(1) == 512, "W2 [128,512]");
  for (auto* t : {&hidden_states, &pwconv1_weight, &pwconv1_bias, &grn_weight, &grn_bias, &pwconv2_weight,
                  &pwconv2_bias, &output})
    TORCH_CHECK(t->is_contiguous(), "contiguous tensors required");
  const c10::cuda::CUDAGuard guard(hidden_states.device());
  const int B = (int)hidden_states.size(0);
  const int HW = (int)(hidden_states.size(1) * hidden_states.size(2));
  const int tiles = (HW + 63) / 64;
  auto part = torch::empty({(int64_t)B * tiles * 512}, hidden_states.options());
  auto nf = torch::empty({(int64_t)B * 512}, hidden_states.options());
  launch_grn_mlp(hidden_states.data_ptr<float>(), pwconv1_weight.data_ptr<float>(), pwconv1_bias.data_ptr<float>(),
                 grn_weight.data_ptr<float>(), grn_bias.data_ptr<float>(), pwconv2_weight.data_ptr<float>(),
                 pwconv2_bias.data_ptr<float>(), (float)eps, output.data_ptr<float>(), part.data_ptr<float>(),
                 nf.data_ptr<float>(), B, HW, at::cuda::getCurrentCUDAStream());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("run", &run, "fused linear-gelu-grn-linear"); }
```

```yaml design-card
id: r0-simt-3k
parents: []
operation: new_design
language: cuda_cpp
niche:
  mem: ldg128
  st: direct
  grid: oneshot
  launch: split
  tile: rows64
  red: cta
  cache: default
  spec: all
hypothesis: >-
  Baseline anchor: exact fp32 SIMT with GEMM1 recomputed after a per-batch GRN statistics pass (three launches).
  Correctness first; its portal page yields Tb and Tsol per workload. It is compute- and latency-bound everywhere.
expected_effect:
  S:
    pct: 0
    confidence: medium
  M:
    pct: 0
    confidence: medium
  L:
    pct: 0
    confidence: medium
resources_sm100a:
  regs_per_thread: 128
  smem_per_cta_bytes: 86016
  threads_per_cta: 256
  ctas_per_sm: 2
  bytes_in_flight_per_sm: 67584
  launches_per_call: 3
knobs:
  TM: 64
  NC: 64
  NT: 256
tests: [H1, H6]
findings:
  - "run_tests: 14/14 pass; 137-143 us flat for 784-6272 tokens, 220 us at 12544, 2191 us at 200704 tokens (61x copy) [run_tests]"
  - "precision: TF32 fails 96% of elements, TF32 with only W split 95%, BF16x3 0.8-2.2% (fails), 3xTF32 and FP16x3 0% with max err 1.8-2.3e-3 vs atol ~3e-3 [probe_b200]"
  - "fp32 torch reference vs fp64: max err 1.3-1.5e-3, half the atol budget [probe_b200]"
  - "cuBLAS fp32 GEMM1 at M=200704: 592 us; GEMM2 500 us; TF32 GEMM1 85 us; cuBLAS BF16x9 emulation 289/390 us (more accurate than fp32) [probe_b200]"
  - "compile sm_100a: k_main 128 regs, k_stats 80 regs, k_grn 32 regs, 0 spills [compile_b200]"
paths:
  - {max_tokens: null, lang: cuda, width: 128, threads: 256, rows: 64, grid: oneshot, mem: ldg, launches: 3, st: none}
runs_on:
  H200:
    runs: true
    representative: false
    note: "SIMT fp32 compute-bound; clock ratio differs from B200 portal"
  A100:
    runs: true
    representative: false
  L40S:
    runs: true
    representative: false
risks:
  - "Very slow (compute-bound fp32 SIMT, only 13-98 CTAs at S); scores will be near 0 against any tensor-core baseline"
  - "Three launches; torch::empty workspace allocations each call (no kernel launches)"
measure_first: [B200]
compliance:
  single_stream: true
  no_cross_call_state: true
  fp32_math: true
  no_threads_or_fork: true
  no_precompiled_binaries: true
```