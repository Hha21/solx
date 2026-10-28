I'm handing in an untested kernel: the session's B200 budget ran out before any of this round's code compiled or ran.

The two probes (the instrumented stamps build and the precision log) each waited 15 minutes for Modal to schedule a B200 and never ran. The waits used up the 20-minute budget, so the compile_b200 call that followed was refused. None of E3's steps produced a measurement:
- **Step 0:** no precision log (err/tol).
- **Step 1:** no phase stamps for the 4,28,28 path.
- **Step 2:** the TT=196 instance is written but not compiled.
- **Step 3:** no run_tests.

The candidate below, `r2-b-e3-tt196`, adds the TT=196 instance for the two 6272-token rows. It needs compile_b200 and run_tests before it can ship. The TT ≤ 98 paths keep last round's shared-memory offsets and code, so they should behave as before. The stamps are compiled out unless built with `-DE2_STAMPS`.

### Rationale
The TT=196 instance covers 6272 tokens with 32 token tiles × 4 hidden slices = 128 CTAs in one wave. It fits by trading smem and TMEM space:
- **Shared memory:** the x hi/lo tile (104 KB) and the W1 hi/lo slice (64 KB) total 177 KB. W2 is loaded with plain loads while GEMM1 runs and converted over the W1 image once GEMM1 completes. The x2 split reuses the x buffer.
- **TMEM:** after GELU, h goes back into TMEM through the barrier. GEMM2's main accumulator reuses D1's 208 columns and the cross accumulator sits at columns 256–464, so GEMM2 runs in one pass, not the two halves the brief suggested.
- **Registers:** x and W1 loads need 84 registers per thread, and the output drain reads 4 columns at a time to stay under the 128 cap.

```json solution-spec
{"name": "r2-b-e3-tt196", "definition": "045_fused_linear_gelu_grn_linear", "author": "solx-loop", "description": "Hidden-slice fan-out extended to 6272 tokens (TT=196 x 4 slices, h parked in TMEM, W2 image over W1 after GEMM1, GEMM2 main accumulator over D1); TT<=98 paths unchanged; r0 SIMT elsewhere. UNTESTED.", "spec": {"languages": ["cuda_cpp"], "target_hardware": ["B200"], "entry_point": "binding.cpp::run", "dependencies": ["torch"], "destination_passing_style": true, "compile_options": {"cuda_cflags": ["-O3", "--use_fast_math", "-std=c++17"]}}}
```

```cuda file=kernel.cu
// 045_fused_linear_gelu_grn_linear, lineage B (r2 E3): hidden-slice fan-out for N <= 6272 tokens.
//   e2: grid = token tiles x 4 hidden slices of 128 (one wave, 1 CTA/SM, cooperative). Each CTA LDGs its fp32 slices
//       W1[slice,:], W2[:,slice] and its x tile, splits them into fp16 hi/lo SW128 smem images, runs GEMM1 swap-AB
//       (D1[hid][tok], FP16x3) into TMEM, + b1, A&S GELU, per-channel sumsq red.add into gsum[b][512] (zeroed by CTA 0
//       behind a flag), one grid counter (departure reset), GRN scale, x2 = gw*(h*nf) + gb + h split hi/lo, GEMM2
//       swap-AB (D2[out][tok], main + cross accumulators) and red.add into out (prefilled with b2 by the slice-0 CTA).
//       TT = 196 (BIG): h is parked in TMEM across the barrier, the W2 image is converted over the W1 image once GEMM1
//       has completed, and GEMM2's main accumulator reuses D1's columns.
//   other shapes: r0 fp32 SIMT, three launches.
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdint.h>

struct E2Params {
  const float* x; const float* w1; const float* b1; const float* gw; const float* gb; const float* w2; const float* b2;
  float* out; float* gsum; int B; int HW; float eps;
};

// [0] gsum-zeroed flag, [1] arrivals, [2] departures; the last departer resets all three for the next call
__device__ unsigned int g_e2ctr[4];

#ifdef E2_STAMPS
__device__ long long g_e2st[160 * 8];
#define E2_STAMP(i) if (threadIdx.x == 0) { long long t_; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t_)); g_e2st[blockIdx.x * 8 + (i)] = t_; }
extern "C" void e2_get_stamps(long long* h) { cudaMemcpyFromSymbol(h, g_e2st, sizeof(long long) * 160 * 8); }
#else
#define E2_STAMP(i)
#endif

namespace {
// ------------------------------------------------------------------ r0 fallback (fp32 SIMT, 3 launches)
constexpr int D = 128, HID = 512, TM = 64, NC = 64, NT = 256;
constexpr int NCHUNK = HID / NC;
constexpr int LDK = D + 4;
constexpr int LDH = NC + 4;

__device__ __forceinline__ float gelu_erf(float v) {
  return v * 0.5f * (1.0f + erff(v * 0.7071067811865476f));
}

__device__ __forceinline__ void load_rows128(float* dst, const float* src, int nrows, int nvalid, int tid) {
  for (int i = tid; i < nrows * (D / 4); i += NT) {
    int r = i >> 5, k4 = (i & 31) << 2;
    float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
    if (r < nvalid) v = __ldg(reinterpret_cast<const float4*>(src + (size_t)r * D + k4));
    *reinterpret_cast<float4*>(dst + r * LDK + k4) = v;
  }
}

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
  float* As = reinterpret_cast<float*>(smem4);
  float* Ws = As + TM * LDK;
  float* Red = Ws + NC * LDK;
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
  float* As = reinterpret_cast<float*>(smem4);
  float* Ws = As + TM * LDK;
  float* Hs = Ws + D * LDH;
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

constexpr int SMEM_STATS = (TM * LDK + NC * LDK + (NT / 32) * NC) * 4;
constexpr int SMEM_MAIN = (TM * LDK + D * LDH + TM * LDH) * 4;
static_assert(D * LDH >= NC * LDK, "Ws must hold either chunk");

// ------------------------------------------------------------------ e2: hidden-slice fan-out (tcgen05 FP16x3)
constexpr int E2_NTH = 512;
// smem layout per token tile: W1 slice [128 hid][128 k] K-major SW128 hi/lo at 0/32K; x tile [NP tok][128 k] (later x2)
// hi/lo; W2 slice [128 out][128 hid] hi/lo (BIG: over W1 once GEMM1 has completed); b1/gw/gb/nf; sumsq partials; bars
template <int TT>
struct E2L {
  static constexpr int NP = (TT + 15) / 16 * 16;  // MMA N (tokens, padded)
  static constexpr bool BIG = NP > 128;
  static constexpr int XATOM = NP * 128;          // bytes per 64-k atom of a token-row image
  static constexpr int W1H = 0, W1L = 32768;
  static constexpr int XH = 65536, XL = BIG ? 65536 + 2 * XATOM : 98304;
  static constexpr int W2H = BIG ? 0 : 131072, W2L = BIG ? 32768 : 163840;
  static constexpr int VEC = BIG ? XL + 2 * XATOM : 196608;
  static constexpr int SSP = VEC + 2048, RED = SSP + 2048, BAR = RED + 256;
  static constexpr int SMEM = BAR + 64 + 1024;
  static constexpr int D2M = BIG ? 0 : 128, D2C = 256;  // TMEM columns of the GEMM2 accumulators
  static_assert(NP <= 256 && D2C + NP <= 512, "TMEM");
  static_assert(SMEM <= 232448, "smem");
};

__device__ __forceinline__ uint32_t s_addr(const void* q) { return (uint32_t)__cvta_generic_to_shared(q); }
__device__ __forceinline__ void mbar_init(uint32_t b, uint32_t n) { asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(b), "r"(n)); }
__device__ __forceinline__ void mbar_wait(uint32_t b, uint32_t par) {
  asm volatile("{\n.reg .pred P;\nWAIT_%=:\nmbarrier.try_wait.parity.shared::cta.b64 P, [%0], %1;\n@!P bra WAIT_%=;\n}\n" ::"r"(b), "r"(par) : "memory");
}
__device__ __forceinline__ void fence_proxy_async() { asm volatile("fence.proxy.async.shared::cta;" ::: "memory"); }
__device__ __forceinline__ void tc_before() { asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory"); }
__device__ __forceinline__ void tc_after() { asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory"); }
__device__ __forceinline__ uint64_t sdesc(uint32_t addr) {  // K-major SWIZZLE_128B, SBO 1024, sm100 descriptor version 1
  return (uint64_t)((addr >> 4) & 0x3FFF) | ((uint64_t)1 << 16) | ((uint64_t)(1024 >> 4) << 32) | (1ull << 46) | (2ull << 61);
}
// kind::f16: fp32 D, fp16 A/B, both K-major, M = 128
__device__ __forceinline__ uint32_t idesc_kk(int n) { return (1u << 4) | ((uint32_t)(n >> 3) << 17) | ((128u >> 4) << 24); }
__device__ __forceinline__ void mma_issue(uint32_t d, uint64_t a, uint64_t b, uint32_t id, uint32_t acc) {
  asm volatile("{\n.reg .pred E, A;\nelect.sync _|E, 0xffffffff;\nsetp.ne.u32 A, %4, 0;\n"
               "@E tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, A;\n}\n" ::"r"(d), "l"(a), "l"(b), "r"(id), "r"(acc) : "memory");
}
__device__ __forceinline__ void mma_commit(uint32_t bar) {
  asm volatile("{\n.reg .pred E;\nelect.sync _|E, 0xffffffff;\n"
               "@E tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];\n}\n" ::"r"(bar) : "memory");
}
template <int N>
__device__ __forceinline__ void tmem_ld(uint32_t ta, float (&v)[N]) {
  uint32_t r[N];
#pragma unroll
  for (int i = 0; i < N / 4; i++)
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(r[4 * i]), "=r"(r[4 * i + 1]), "=r"(r[4 * i + 2]), "=r"(r[4 * i + 3]) : "r"(ta + 4 * i) : "memory");
  asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
#pragma unroll
  for (int i = 0; i < N; i++) {
    asm volatile("" : "+r"(r[i]));  // uses stay behind the wait
    v[i] = __uint_as_float(r[i]);
  }
}
template <int N>
__device__ __forceinline__ void tmem_st(uint32_t ta, const float (&v)[N]) {
#pragma unroll
  for (int i = 0; i < N / 4; i++)
    asm volatile("tcgen05.st.sync.aligned.32x32b.x4.b32 [%0], {%1,%2,%3,%4};" ::"r"(ta + 4 * i), "r"(__float_as_uint(v[4 * i])),
                 "r"(__float_as_uint(v[4 * i + 1])), "r"(__float_as_uint(v[4 * i + 2])), "r"(__float_as_uint(v[4 * i + 3]))
                 : "memory");
  asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
}
__device__ __forceinline__ uint32_t ld_acquire(const unsigned int* q) {
  uint32_t v;
  asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(q) : "memory");
  return v;
}
__device__ __forceinline__ void red_release(unsigned int* q, uint32_t v) {
  asm volatile("red.release.gpu.global.add.u32 [%0], %1;" ::"l"(q), "r"(v) : "memory");
}
__device__ __forceinline__ void st_release(unsigned int* q, uint32_t v) {
  asm volatile("st.release.gpu.global.u32 [%0], %1;" ::"l"(q), "r"(v) : "memory");
}
__device__ __forceinline__ uint32_t h2u(__half2 h) { return *reinterpret_cast<uint32_t*>(&h); }
__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}
__device__ __forceinline__ uint64_t f2pack(float a, float b) { uint64_t r; asm("mov.b64 %0, {%1, %2};" : "=l"(r) : "f"(a), "f"(b)); return r; }
__device__ __forceinline__ void f2unpack(uint64_t r, float& a, float& b) { asm("mov.b64 {%0, %1}, %2;" : "=f"(a), "=f"(b) : "l"(r)); }
__device__ __forceinline__ uint64_t fma2(uint64_t a, uint64_t b, uint64_t c) { uint64_t d; asm("fma.rn.f32x2 %0, %1, %2, %3;" : "=l"(d) : "l"(a), "l"(b), "l"(c)); return d; }
__device__ __forceinline__ uint64_t mul2(uint64_t a, uint64_t b) { uint64_t d; asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(d) : "l"(a), "l"(b)); return d; }
__device__ __forceinline__ uint64_t add2(uint64_t a, uint64_t b) { uint64_t d; asm("add.rn.f32x2 %0, %1, %2;" : "=l"(d) : "l"(a), "l"(b)); return d; }
// GELU (erf form) on a pair: Abramowitz-Stegun 7.1.26 erfc (|err| <= 1.5e-7), hc = 0.5 erfc(|x| / sqrt2),
// h = x hc + max(x - 2 x hc, 0)   (#129 gelu2 without the sumsq)
__device__ __forceinline__ void gelu2(float& x0, float& x1) {
  float t0, t1, e0, e1, q0, q1, d0, d1;
  asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(t0) : "f"(fmaf(fabsf(x0), 0.231641900f, 1.f)));
  asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(t1) : "f"(fmaf(fabsf(x1), 0.231641900f, 1.f)));
  const uint64_t T = f2pack(t0, t1), X = f2pack(x0, x1);
  uint64_t P = fma2(T, f2pack(0.5307027145f, 0.5307027145f), f2pack(-0.7265760135f, -0.7265760135f));
  P = fma2(T, P, f2pack(0.7107068705f, 0.7107068705f));
  P = fma2(T, P, f2pack(-0.142248368f, -0.142248368f));
  P = fma2(T, P, f2pack(0.127414796f, 0.127414796f));
  P = mul2(P, T);
  f2unpack(mul2(X, mul2(X, f2pack(-0.72134752044f, -0.72134752044f))), q0, q1);
  asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(e0) : "f"(q0));
  asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(e1) : "f"(q1));
  const uint64_t XH = mul2(X, mul2(P, f2pack(e0, e1)));
  f2unpack(fma2(XH, f2pack(-2.f, -2.f), X), d0, d1);
  f2unpack(add2(XH, f2pack(fmaxf(d0, 0.f), fmaxf(d1, 0.f))), x0, x1);
}
// 4 consecutive k of row r into the hi and lo K-major SW128 images (atom = 64 k = 128 B per row)
__device__ __forceinline__ void put_hilo(unsigned char* sm, int offh, int offl, int atom_bytes, int r, int c4, float4 v) {
  const int off = (c4 >> 4) * atom_bytes + r * 128 + ((((c4 >> 1) & 7) ^ (r & 7)) << 4) + (c4 & 1) * 8;
  const __half2 h01 = __floats2half2_rn(v.x, v.y), h23 = __floats2half2_rn(v.z, v.w);
  const float2 f01 = __half22float2(h01), f23 = __half22float2(h23);
  const __half2 l01 = __floats2half2_rn(v.x - f01.x, v.y - f01.y), l23 = __floats2half2_rn(v.z - f23.x, v.w - f23.y);
  *reinterpret_cast<uint2*>(sm + offh + off) = make_uint2(h2u(h01), h2u(h23));
  *reinterpret_cast<uint2*>(sm + offl + off) = make_uint2(h2u(l01), h2u(l23));
}
__device__ __forceinline__ void load_w2(float4 (&vw2)[8], const float* w2, int slice, int tid) {
#pragma unroll
  for (int i = 0; i < 8; i++) {
    const int f = tid + E2_NTH * i;
    vw2[i] = __ldg(reinterpret_cast<const float4*>(w2 + (size_t)(f >> 5) * 512 + slice * 128) + (f & 31));
  }
}

template <int TT>
__global__ void __launch_bounds__(E2_NTH, 1) e2_kernel(const E2Params p) {
  using L = E2L<TT>;
  constexpr int NP = L::NP;
  constexpr int CW = NP / 4;               // columns per thread in the epilogues
  constexpr int XATOM = L::XATOM;
  constexpr int NXL = NP / 16;             // x float4 per thread
  extern __shared__ unsigned char smraw[];
  unsigned char* sm = (unsigned char*)(((uintptr_t)smraw + 1023) & ~(uintptr_t)1023);
  const uint32_t sb = s_addr(sm);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  E2_STAMP(0)
  const int slice = blockIdx.x & 3, tile = blockIdx.x >> 2, tok0 = tile * TT, b = tok0 / p.HW;
  float* s_b1 = (float*)(sm + L::VEC);
  float* s_gw = s_b1 + 128;
  float* s_gb = s_b1 + 256;
  float* s_nf = s_b1 + 384;
  float* s_ssp = (float*)(sm + L::SSP);
  float* s_red = (float*)(sm + L::RED);
  const uint32_t bar1 = sb + L::BAR, bar2 = bar1 + 8;
  uint32_t* s_tmem = (uint32_t*)(sm + L::BAR + 16);
  unsigned int* ctr = g_e2ctr;

  // ---- global loads up front: W1 slice (contiguous 64 KB), x tile, W2 slice (not BIG), b1/gw/gb slices
  float4 vw1[8], vx[NXL], vw2[8], vv = make_float4(0.f, 0.f, 0.f, 0.f);
  const float4* w1s = reinterpret_cast<const float4*>(p.w1 + (size_t)slice * 128 * 128);
#pragma unroll
  for (int i = 0; i < 8; i++) vw1[i] = __ldg(w1s + tid + E2_NTH * i);
  const float4* xs = reinterpret_cast<const float4*>(p.x + (size_t)tok0 * 128);
#pragma unroll
  for (int i = 0; i < NXL; i++) {
    const int f = tid + E2_NTH * i;
    vx[i] = ((f >> 5) < TT) ? __ldg(xs + f) : make_float4(0.f, 0.f, 0.f, 0.f);
  }
  if constexpr (!L::BIG) load_w2(vw2, p.w2, slice, tid);
  if (tid < 96) {
    const float* src = tid < 32 ? p.b1 : (tid < 64 ? p.gw : p.gb);
    vv = __ldg(reinterpret_cast<const float4*>(src + slice * 128) + (tid & 31));
  }
  if (tid == 0) {
    mbar_init(bar1, 1);
    mbar_init(bar2, 1);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  if (warp == 0) {
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" ::"r"(s_addr(s_tmem)), "r"(512) : "memory");
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;" ::: "memory");
  }
  if (blockIdx.x == 0)
    for (int i = tid; i < p.B * 512; i += E2_NTH) p.gsum[i] = 0.f;
  if (tid < 96) reinterpret_cast<float4*>(s_b1)[tid] = vv;
#pragma unroll
  for (int i = 0; i < 8; i++) {
    const int f = tid + E2_NTH * i;
    put_hilo(sm, L::W1H, L::W1L, 16384, f >> 5, f & 31, vw1[i]);
  }
#pragma unroll
  for (int i = 0; i < NXL; i++) {
    const int f = tid + E2_NTH * i;
    put_hilo(sm, L::XH, L::XL, XATOM, f >> 5, f & 31, vx[i]);
  }
  fence_proxy_async();
  tc_before();
  __syncthreads();
  tc_after();
  E2_STAMP(1)
  if (blockIdx.x == 0 && tid == 0) { __threadfence(); st_release(ctr, 1u); }
  const uint32_t tmem = *s_tmem;
  // ---- GEMM1 swap-AB: D1[hid][tok] = W1h.Xh + W1h.Xl + W1l.Xh, K = 128 (cross terms first)
  if (warp == 0) {
    const uint32_t id = idesc_kk(NP);
#pragma unroll
    for (int pr = 0; pr < 3; pr++) {
      const uint32_t A = (pr == 1) ? L::W1L : L::W1H;
      const uint32_t Bo = (pr == 0) ? L::XL : L::XH;
#pragma unroll
      for (int kk = 0; kk < 8; kk++) {
        const uint32_t w = (kk & 3) * 32;
        mma_issue(tmem, sdesc(sb + A + (kk >> 2) * 16384 + w), sdesc(sb + Bo + (kk >> 2) * XATOM + w), id,
                  (pr | kk) ? 1u : 0u);
      }
    }
    mma_commit(bar1);
  }
  // ---- behind GEMM1: W2 slice image (BIG: loaded now, converted over W1 once GEMM1 has completed); b2 prefill
  if constexpr (L::BIG) {
    load_w2(vw2, p.w2, slice, tid);
  } else {
#pragma unroll
    for (int i = 0; i < 8; i++) {
      const int f = tid + E2_NTH * i;
      put_hilo(sm, L::W2H, L::W2L, 16384, f >> 5, f & 31, vw2[i]);
    }
    fence_proxy_async();
  }
  if (slice == 0) {
    const float4* b24 = reinterpret_cast<const float4*>(p.b2);
    float4* o4 = reinterpret_cast<float4*>(p.out + (size_t)tok0 * 128);
    for (int i = tid; i < TT * 32; i += E2_NTH) o4[i] = __ldg(b24 + (i & 31));
  }
  mbar_wait(bar1, 0);
  tc_after();
  E2_STAMP(2)
  if constexpr (L::BIG) {
#pragma unroll
    for (int i = 0; i < 8; i++) {
      const int f = tid + E2_NTH * i;
      put_hilo(sm, L::W2H, L::W2L, 16384, f >> 5, f & 31, vw2[i]);
    }
    fence_proxy_async();
  }
  // ---- GEMM1 epilogue: + b1, GELU, masked per-channel sumsq; h stays in registers (BIG: back into TMEM)
  const int q = warp & 3, cq = warp >> 2, hid = 32 * q + lane;
  const uint32_t tq = tmem + ((uint32_t)(32 * q) << 16) + cq * CW;
  float h[CW];
  tmem_ld<CW>(tq, h);
  {
    const float bk = s_b1[hid];
    float ss = 0.f;
#pragma unroll
    for (int e = 0; e < CW; e += 2) {
      float x0 = h[e] + bk, x1 = h[e + 1] + bk;
      gelu2(x0, x1);
      const int t = cq * CW + e;
      if (t >= TT) x0 = 0.f;
      if (t + 1 >= TT) x1 = 0.f;
      h[e] = x0;
      h[e + 1] = x1;
      ss = fmaf(x0, x0, ss);
      ss = fmaf(x1, x1, ss);
    }
    s_ssp[cq * 128 + hid] = ss;
  }
  if constexpr (L::BIG) tmem_st<CW>(tq, h);
  if (tid == 0) { while (ld_acquire(ctr) == 0u) { } }
  __syncthreads();
  if (tid < 128) atomicAdd(p.gsum + b * 512 + slice * 128 + tid, s_ssp[tid] + s_ssp[128 + tid] + s_ssp[256 + tid] + s_ssp[384 + tid]);
  __syncthreads();
  E2_STAMP(3)
  // ---- grid barrier: one arrival counter; the last departer resets the counters
  if (tid == 0) {
    __threadfence();
    red_release(ctr + 1, 1u);
    while (ld_acquire(ctr + 1) < gridDim.x) { }
    const unsigned old = atomicAdd(ctr + 2, 1u);
    if (old == gridDim.x - 1) {
      atomicExch(ctr, 0u);
      atomicExch(ctr + 1, 0u);
      atomicExch(ctr + 2, 0u);
    }
  }
  __syncthreads();
  E2_STAMP(4)
  // ---- GRN: G = sqrt(gsum), nf = G / (mean G + eps) for this slice
  {
    const float g = __fsqrt_rn(__ldcg(p.gsum + b * 512 + tid));
    const float ws = warp_sum(g);
    if (lane == 0) s_red[warp] = ws;
    __syncthreads();
    float tot = 0.f;
#pragma unroll
    for (int i = 0; i < 16; i++) tot += s_red[i];
    const float mean = __fdiv_rn(tot, 512.f);
    if ((tid >> 7) == slice) s_nf[tid & 127] = __fdiv_rn(g, mean + p.eps);
    __syncthreads();
  }
  // ---- x2 = gw*(h*nf) + gb + h, split hi/lo into the K-major token-row images (x space)
  if constexpr (L::BIG) tmem_ld<CW>(tq, h);
  {
    const float nfk = s_nf[hid], gwk = s_gw[hid], gbk = s_gb[hid];
    const int at = (hid >> 6) * XATOM, ch = (hid >> 3) & 7, sub = (hid & 7) * 2;
#pragma unroll
    for (int e = 0; e < CW; e++) {
      const int t = cq * CW + e;
      const float v = h[e];
      const float x2 = gwk * (v * nfk) + gbk + v;
      const __half hi = __float2half_rn(x2);
      const __half lo = __float2half_rn(x2 - __half2float(hi));
      const int off = at + t * 128 + ((ch ^ (t & 7)) << 4) + sub;
      *reinterpret_cast<__half*>(sm + L::XH + off) = hi;
      *reinterpret_cast<__half*>(sm + L::XL + off) = lo;
    }
  }
  fence_proxy_async();
  tc_before();
  __syncthreads();
  tc_after();
  E2_STAMP(5)
  // ---- GEMM2 swap-AB: D2[out][tok] = W2h.X2h (main) and W2h.X2l + W2l.X2h (cross), K = 128
  if (warp == 0) {
    const uint32_t id = idesc_kk(NP);
#pragma unroll
    for (int kk = 0; kk < 8; kk++) {
      const uint32_t w = (kk & 3) * 32;
      mma_issue(tmem + L::D2M, sdesc(sb + L::W2H + (kk >> 2) * 16384 + w), sdesc(sb + L::XH + (kk >> 2) * XATOM + w), id,
                kk ? 1u : 0u);
    }
#pragma unroll
    for (int pr = 0; pr < 2; pr++) {
      const uint32_t A = pr ? L::W2L : L::W2H;
      const uint32_t Bo = pr ? L::XH : L::XL;
#pragma unroll
      for (int kk = 0; kk < 8; kk++) {
        const uint32_t w = (kk & 3) * 32;
        mma_issue(tmem + L::D2C, sdesc(sb + A + (kk >> 2) * 16384 + w), sdesc(sb + Bo + (kk >> 2) * XATOM + w), id,
                  (pr | kk) ? 1u : 0u);
      }
    }
    mma_commit(bar2);
  }
  mbar_wait(bar2, 0);
  tc_after();
  E2_STAMP(6)
  {
    float* ob = p.out + (size_t)tok0 * 128 + hid;  // hid = output channel here
    if constexpr (L::BIG) {
#pragma unroll 1
      for (int e0 = 0; e0 < CW; e0 += 4) {
        float m[4], c[4];
        tmem_ld<4>(tq + L::D2M + e0, m);
        tmem_ld<4>(tq + L::D2C + e0, c);
#pragma unroll
        for (int j = 0; j < 4; j++) {
          const int t = cq * CW + e0 + j;
          if (t < TT) atomicAdd(ob + (size_t)t * 128, m[j] + c[j]);
        }
      }
    } else {
      float m[CW], c[CW];
      tmem_ld<CW>(tq + L::D2M, m);
      tmem_ld<CW>(tq + L::D2C, c);
#pragma unroll
      for (int e = 0; e < CW; e++) {
        const int t = cq * CW + e;
        if (t < TT) atomicAdd(ob + (size_t)t * 128, m[e] + c[e]);
      }
    }
  }
  tc_before();
  __syncthreads();
  E2_STAMP(7)
  if (warp == 0) {
    tc_after();
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" ::"r"(tmem), "r"(512) : "memory");
  }
}

template <int TT>
const char* e2_launch(const E2Params& p, int N, cudaStream_t stream) {
  static bool attr_set = false;
  cudaError_t e;
  if (!attr_set) {
    e = cudaFuncSetAttribute(e2_kernel<TT>, cudaFuncAttributeMaxDynamicSharedMemorySize, E2L<TT>::SMEM);
    if (e != cudaSuccess) return cudaGetErrorString(e);
    attr_set = true;
  }
  E2Params q = p;
  void* args[] = {(void*)&q};
  e = cudaLaunchCooperativeKernel((const void*)e2_kernel<TT>, dim3(4 * (N / TT)), dim3(E2_NTH), args, (size_t)E2L<TT>::SMEM,
                                  stream);
  return e == cudaSuccess ? nullptr : cudaGetErrorString(e);
}
}  // namespace

extern "C" const char* launch_grn_mlp(const float* x, const float* w1, const float* b1, const float* gw, const float* gb,
                                      const float* w2, const float* b2, float eps, float* out, float* part, float* nf,
                                      int B, int HW, cudaStream_t stream) {
  const int N = B * HW;
  // e2 path: smallest token tile that divides HW and keeps 4 slices x tiles within one wave of 148 CTAs
  int tt = 0;
  const int cands[4] = {28, 49, 98, 196};
  for (int i = 0; i < 4 && !tt; i++)
    if (HW % cands[i] == 0 && 4 * (N / cands[i]) <= 148) tt = cands[i];
  if (tt) {
    E2Params p{x, w1, b1, gw, gb, w2, b2, out, nf, B, HW, eps};
    if (tt == 28) return e2_launch<28>(p, N, stream);
    if (tt == 49) return e2_launch<49>(p, N, stream);
    if (tt == 98) return e2_launch<98>(p, N, stream);
    return e2_launch<196>(p, N, stream);
  }
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
  return nullptr;
}
```

```cpp file=binding.cpp
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

extern "C" const char* launch_grn_mlp(const float* x, const float* w1, const float* b1, const float* gw, const float* gb,
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
  const char* err = launch_grn_mlp(hidden_states.data_ptr<float>(), pwconv1_weight.data_ptr<float>(),
                                   pwconv1_bias.data_ptr<float>(), grn_weight.data_ptr<float>(), grn_bias.data_ptr<float>(),
                                   pwconv2_weight.data_ptr<float>(), pwconv2_bias.data_ptr<float>(), (float)eps,
                                   output.data_ptr<float>(), part.data_ptr<float>(), nf.data_ptr<float>(), B, HW,
                                   at::cuda::getCurrentCUDAStream());
  TORCH_CHECK(err == nullptr, err ? err : "");
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("run", &run, "fused linear-gelu-grn-linear"); }
```

```yaml design-card
id: r2-b-e3-tt196
parents:
  - r1-b-e2-fanout-v3
operation: structural_mutation
language: cuda_cpp
niche:
  mem: ldg128
  st: direct
  grid: oneshot
  launch: fused
  tile: rows196
  red: cta
  cache: default
  spec: dispatch:size
hypothesis: >-
  Extends the hidden-slice fan-out to the 6272-token rows with TT=196 x 4 slices = 128 CTAs in one wave. It fits by
  parking h in TMEM through the barrier, converting the W2 slice over the W1 image once GEMM1 completes (177 KB smem),
  and placing GEMM2's main accumulator over D1's 208 columns and the cross accumulator at columns 256-464, so GEMM2 runs
  in one pass. The prediction from scaling r1 (16.5 us at 98 tokens per CTA, about 5 us per extra 70 tokens) is 22-25 us
  rented at 2,56,56 and 32,14,14, against e1's 25.3 / 28.3 us. UNTESTED: not compiled or run this session.
expected_effect:
  S:
    pct: 0
    confidence: medium
  M:
    pct: -85
    confidence: low
  L:
    pct: 0
    confidence: high
resources_sm100a:
  regs_per_thread: 126
  smem_per_cta_bytes: 177472
  threads_per_cta: 512
  ctas_per_sm: 1
  bytes_in_flight_per_sm: 184320
  launches_per_call: 1
knobs:
  slices: 4
  slice_width: 128
  TT_candidates: [28, 49, 98, 196]
  threads: 512
  tmem_cols: 512
dispatch:
  - {max_tokens: 784, kernel: "e2_kernel<28>", meta: {grid: 112}}
  - {max_tokens: 1568, kernel: "e2_kernel<49>", meta: {grid: 128}}
  - {max_tokens: 3136, kernel: "e2_kernel<98>", meta: {grid: 128}}
  - {max_tokens: 6272, kernel: "e2_kernel<196>", meta: {grid: 128, h_in_tmem: true, w2_over_w1: true}}
  - {max_tokens: null, kernel: "r0 k_stats/k_grn/k_main", meta: {launches: 3}}
tests: [H21]
findings:
  - "infrastructure: this session's B200 budget (20 min) was used up by two probes that each waited 15 min for Modal to schedule a B200 (a TimeoutError before any code ran); the compile_b200 call that followed was refused. Nothing in this round was compiled, run or timed [probe_b200, compile_b200]"
  - "not measured: step 0 (10-seed fp64 err/tol), step 1 (globaltimer stamps of 4,28,28; the probe build with -DE2_STAMPS was written but never ran), step 3 (run_tests). H21 is neither supported nor refuted"
  - "design (arithmetic): TT=196 fits without the brief's two token halves. With h parked in TMEM through the barrier, D1 is dead after the x2 split, so GEMM2 main can go in D1's columns 0-207 and cross in 256-463. Shared memory is 177 KB (W1/W2 images share 64 KB, x/x2 images 104 KB) [INFERRED]"
  - "design (registers): the BIG path holds about 84 payload registers at entry (13 x float4 + 8 W1 float4); W2's 32 load registers are live only while GEMM1 runs; the output drain reads 4 columns at a time, so the 128-register cap should hold without spills. Unverified [INFERRED]"
  - "carried over from r1 (measured): 14/14 pass; 1,28,28 11.6 us, 8,14,14 13.7, 4,28,28 16.5, 16,14,14 16.7 rented; the TT<=98 paths keep the same smem offsets and code here [run_tests r1]"
  - "next session: compile_b200 (check regs and STACK for e2_kernel<196>), run_tests all 14, then the stamp probe (build kernel.cu with -DE2_STAMPS plus a binding exposing e2_get_stamps) and the 10-seed fp64 err/tol log for the 6 served rows"
paths:
  - {max_tokens: 3136, lang: cuda, width: 128, threads: 512, rows: 98, grid: oneshot, mem: ldg, launches: 1, x: none, w: nc, st: none}
  - {max_tokens: 6272, lang: cuda, width: 128, threads: 512, rows: 196, grid: oneshot, mem: ldg, launches: 1, x: none, w: nc, st: none}
  - {max_tokens: null, lang: cuda, width: 128, threads: 256, rows: 64, grid: oneshot, mem: ldg, launches: 3, st: none}
runs_on:
  H200:
    runs: false
    representative: false
    note: "tcgen05/TMEM are sm_100a only"
  A100:
    runs: false
    representative: false
  L40S:
    runs: false
    representative: false
risks:
  - "UNTESTED: not compiled; the BIG path (if constexpr branches, tmem_st, D2 over D1) may fail to build or produce wrong results; it must pass compile_b200 and run_tests before shipping"
  - "Register pressure at TT=196 could spill under the 128-reg cap (__launch_bounds__(512,1))"
  - "Precision err/tol per row still unlogged (gate <= 0.8)"
  - "Grid barrier relies on cooperative co-residency at 1 CTA/SM; an aborted call leaves the counters dirty"
measure_first: [B200]
compliance:
  single_stream: true
  no_cross_call_state: true
  fp32_math: true
  no_threads_or_fork: true
  no_precompiled_binaries: true
```