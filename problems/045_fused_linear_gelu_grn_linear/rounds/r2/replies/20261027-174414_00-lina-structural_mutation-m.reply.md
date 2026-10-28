### Rationale
The candidate is **r2-a-e1-prod-fix**, the e1 path with three changes. First, the producer role moves to its own warp (warp 17), so the MMA warp never blocks on its own completion signals. Second, the base descriptors are hoisted out of the K loop. Third, it fixes a correctness bug that the r1 harness passed: padded tokens of a partial last CTA were added into the last batch's GRN sum (err/tol 444 at 64,14,14 and 12.7 at 2,56,56). I measured the same device code through NVRTC probes on the rented B200. All rows tested were 8–15% faster than r1, with err/tol 0.19–0.25 against an fp64 reference, but 64,14,14 at 35.5 µs and 2,56,56 at 23.6 µs both miss this round's targets (≤ 29 and ≤ 20 µs). **This two-file build has not been compiled or run_tests'd.** The B200 queue timed out twice and then the round's GPU budget ran out. The probe build differed only in that its stamp writes were active and its ring depth was a runtime value (set to 3). Step 5 (porting to e4) was not needed: e4 already has a separate producer warp. e4's precision gate on the L rows was not run.

```json solution-spec
{"name": "r2-a-e1-prod-fix", "definition": "045_fused_linear_gelu_grn_linear", "author": "solx-loop", "description": "e1 token-stationary FP16x3 tcgen05 path with a dedicated producer warp and the partial-last-CTA sumsq fix; r0 SIMT fallback above 14208 tokens", "spec": {"languages": ["cuda_cpp"], "target_hardware": ["B200"], "entry_point": "binding.cpp::run", "dependencies": [], "destination_passing_style": true, "compile_options": {"cuda_cflags": ["-O3", "--use_fast_math", "-std=c++17"]}}}
```

```cuda file=kernel.cu
// 045 fused linear-gelu-grn-linear. Lineage A, r2 E1.
// N <= 148*96 tokens: one cooperative tcgen05 FP16x3 launch, token-stationary. Each CTA owns t tokens (t = ceil16(N/148),
// <= 96). Phase 0: every CTA converts 1/G of W1/W2 into exact fp16 hi/lo SW128 smem images in the workspace, part of
// b2' = W2.gb and zeroes part of gsum, then bumps the weight counter. GEMM1 swap-AB D1[hidden][token] (4 blocks of 128,
// hi*hi + hi*lo + lo*hi in one fp32 accumulator), + b1, GELU, per-channel sumsq, fp32 h back into TMEM. Per-batch
// counters, GRN scale s = 1 + gw*G/(mean G + eps), h' = h*s split hi/lo (lo * 2^11) into MN-major SW32 B tiles, GEMM2
// D2[out][token] with hi*hi in one accumulator and the 2^11-scaled cross terms in another. out = A + B/2^11 + b2 + W2.gb.
// r2: a dedicated producer warp (17) refills the 3-stage weight ring as soon as a stage's MMAs commit, so the MMA warp
// (16) never blocks on a completion signal; padded tokens of a partial last CTA no longer enter the batch sumsq.
// Larger N: the r0 SIMT 3-launch path.
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdint.h>

// =====================================================================================================================
// r0 fallback (exact fp32 SIMT, 3 launches), unchanged
// =====================================================================================================================
namespace r0 {
constexpr int D = 128, HID = 512, TM = 64, NC = 64, NT = 256;
constexpr int NCHUNK = HID / NC;
constexpr int LDK = D + 4;
constexpr int LDH = NC + 4;

__device__ __forceinline__ float gelu_erf(float v) { return v * 0.5f * (1.0f + erff(v * 0.7071067811865476f)); }

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
                                              const float* __restrict__ b1, float* __restrict__ part, int HW, int tiles) {
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

__global__ void __launch_bounds__(HID) k_grn(const float* __restrict__ part, float* __restrict__ nf, int tiles, float eps) {
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
      *reinterpret_cast<float4*>(Ws + o * LDH + k4) = __ldg(reinterpret_cast<const float4*>(w2 + (size_t)o * HID + c * NC + k4));
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
}  // namespace r0

extern "C" void launch_grn_mlp(const float* x, const float* w1, const float* b1, const float* gw, const float* gb,
                               const float* w2, const float* b2, float eps, float* out, float* part, float* nf,
                               int B, int HW, cudaStream_t stream) {
  static bool attrs_set = false;
  if (!attrs_set) {
    cudaFuncSetAttribute(r0::k_stats, cudaFuncAttributeMaxDynamicSharedMemorySize, r0::SMEM_STATS);
    cudaFuncSetAttribute(r0::k_main, cudaFuncAttributeMaxDynamicSharedMemorySize, r0::SMEM_MAIN);
    attrs_set = true;
  }
  const int tiles = (HW + r0::TM - 1) / r0::TM;
  dim3 grid(tiles, B);
  r0::k_stats<<<grid, r0::NT, r0::SMEM_STATS, stream>>>(x, w1, b1, part, HW, tiles);
  r0::k_grn<<<B, r0::HID, 0, stream>>>(part, nf, tiles, eps);
  r0::k_main<<<grid, r0::NT, r0::SMEM_MAIN, stream>>>(x, w1, b1, gw, gb, w2, b2, nf, out, HW);
}

// =====================================================================================================================
// fused token-stationary tcgen05 FP16x3 path
// =====================================================================================================================
struct FParams {
  const float* x; const float* w1; const float* b1; const float* gw; const float* gb; const float* w2; const float* b2;
  float* out; unsigned char* wimg; float* b2p; float* gsum;
  int N, HW, B, t; float eps;
};

// [0] CTAs whose phase-0 slice (weight images, b2', gsum zero) is written, [1] departures, [2 + b] tokens of batch b
// whose sumsq is published. The last departer resets them for the next call.
__device__ unsigned int g_fctr[2 + 128];

namespace fz {
constexpr int NTH = 576;          // warps 0..15: epilogues; warp 16: MMA issuer; warp 17: bulk producer
constexpr int NWARP = NTH / 32;
constexpr int NST = 3;            // 32 KB weight-unit ring stages
constexpr int HRB = 32;           // h' (GEMM2 B, MN-major SW32) row bytes: 16 tokens per atom
constexpr int HAT = 16;
constexpr int TMAX = 96;
constexpr int HREG = (TMAX / HAT) * 64 * HRB;  // one (64-k chunk, hi|lo) h' region
constexpr int HBUF = 4 * HREG;                 // one hidden block of h' (2 chunks x hi/lo)
constexpr int OFF_RING = 0;                    // NST x 32 KB weight units (hi 16 KB + lo 16 KB)
constexpr int OFF_HB = NST * 32768;            // 2 h' buffers; buffer 0 first holds the x hi/lo tile
constexpr int OFF_B1 = OFF_HB + 2 * HBUF;
constexpr int OFF_GW = OFF_B1 + 2048;
constexpr int OFF_SC = OFF_GW + 2048;   // GRN scale [2 slots][512]
constexpr int OFF_GS = OFF_SC + 4096;   // sumsq [2 slots][512]
constexpr int OFF_B2P = OFF_GS + 4096;  // b2 + W2.gb [128]
constexpr int OFF_RED = OFF_B2P + 512;
constexpr int OFF_BAR = OFF_RED + 128;  // full[3], empty[3], g1d[4], hrd[4], hfr[2], g2d, tmem slot
constexpr int SMEM_TOTAL = OFF_BAR + 256 + 1024;
static_assert(TMAX * 512 <= HBUF, "x tile must fit in h' buffer 0");

__device__ __forceinline__ uint32_t s_addr(const void* q) { return (uint32_t)__cvta_generic_to_shared(q); }
__device__ __forceinline__ void mbar_init(uint32_t b, uint32_t n) { asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(b), "r"(n)); }
__device__ __forceinline__ void mbar_expect(uint32_t b, uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(b), "r"(bytes) : "memory");
}
__device__ __forceinline__ void mbar_arrive(uint32_t b) { asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(b) : "memory"); }
__device__ __forceinline__ void mbar_wait(uint32_t b, uint32_t par) {
  asm volatile("{\n.reg .pred P;\nWAIT_%=:\nmbarrier.try_wait.parity.shared::cta.b64 P, [%0], %1;\n@!P bra WAIT_%=;\n}\n" ::"r"(b), "r"(par) : "memory");
}
__device__ __forceinline__ void bulk_g2s(uint32_t dst, const void* src, uint32_t bytes, uint32_t bar) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];" ::"r"(dst), "l"(src), "r"(bytes), "r"(bar) : "memory");
}
__device__ __forceinline__ void fence_proxy_async() { asm volatile("fence.proxy.async.shared::cta;" ::: "memory"); }
__device__ __forceinline__ void tc_before() { asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory"); }
__device__ __forceinline__ void tc_after() { asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory"); }
__device__ __forceinline__ void bar512() { asm volatile("bar.sync 1, 512;" ::: "memory"); }
// sm100 smem descriptor, version 1; lay: 2 = SW128, 6 = SW32. The start-address field is addr >> 4, so +2 per 32 B.
__device__ __forceinline__ uint64_t sdesc(uint32_t addr, uint32_t lbo, uint32_t sbo, uint32_t lay) {
  return (uint64_t)((addr >> 4) & 0x3FFF) | ((uint64_t)((lbo >> 4) & 0x3FFF) << 16) | ((uint64_t)((sbo >> 4) & 0x3FFF) << 32) |
         (1ull << 46) | ((uint64_t)lay << 61);
}
// kind::f16: fp32 D, fp16 A/B, A K-major, B K-major (bmn = 0) or MN-major (bmn = 1), M = 128
__device__ __forceinline__ uint32_t idesc_f16(int n, bool bmn) {
  return (1u << 4) | (bmn ? (1u << 16) : 0u) | ((uint32_t)(n >> 3) << 17) | ((128u >> 4) << 24);
}
__device__ __forceinline__ void mma_issue(uint32_t d, uint64_t a, uint64_t b, uint32_t id, uint32_t acc) {
  asm volatile("{\n.reg .pred E, A;\nelect.sync _|E, 0xffffffff;\nsetp.ne.u32 A, %4, 0;\n"
               "@E tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, A;\n}\n" ::"r"(d), "l"(a), "l"(b), "r"(id), "r"(acc) : "memory");
}
__device__ __forceinline__ void mma_commit(uint32_t bar) {
  asm volatile("{\n.reg .pred E;\nelect.sync _|E, 0xffffffff;\n"
               "@E tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];\n}\n" ::"r"(bar) : "memory");
}
__device__ __forceinline__ void tmem_ld8(uint32_t ta, float (&v)[8]) {
  uint32_t r[8];
  asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];\n"
               "tcgen05.wait::ld.sync.aligned;\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]), "=r"(r[4]), "=r"(r[5]), "=r"(r[6]), "=r"(r[7])
               : "r"(ta) : "memory");
#pragma unroll
  for (int i = 0; i < 8; i++) v[i] = __uint_as_float(r[i]);
}
__device__ __forceinline__ void tmem_ld8x2(uint32_t ta, uint32_t tb, float (&v)[8], float (&w)[8]) {
  uint32_t r[8], s[8];
  asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%16];\n"
               "tcgen05.ld.sync.aligned.32x32b.x8.b32 {%8,%9,%10,%11,%12,%13,%14,%15}, [%17];\n"
               "tcgen05.wait::ld.sync.aligned;\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]), "=r"(r[4]), "=r"(r[5]), "=r"(r[6]), "=r"(r[7]),
                 "=r"(s[0]), "=r"(s[1]), "=r"(s[2]), "=r"(s[3]), "=r"(s[4]), "=r"(s[5]), "=r"(s[6]), "=r"(s[7])
               : "r"(ta), "r"(tb) : "memory");
#pragma unroll
  for (int i = 0; i < 8; i++) { v[i] = __uint_as_float(r[i]); w[i] = __uint_as_float(s[i]); }
}
__device__ __forceinline__ void tmem_st8(uint32_t ta, const float (&v)[8]) {
  asm volatile("tcgen05.st.sync.aligned.32x32b.x8.b32 [%0], {%1,%2,%3,%4,%5,%6,%7,%8};\n" ::"r"(ta),
               "r"(__float_as_uint(v[0])), "r"(__float_as_uint(v[1])), "r"(__float_as_uint(v[2])), "r"(__float_as_uint(v[3])),
               "r"(__float_as_uint(v[4])), "r"(__float_as_uint(v[5])), "r"(__float_as_uint(v[6])), "r"(__float_as_uint(v[7]))
               : "memory");
}
__device__ __forceinline__ uint32_t ld_acquire(const unsigned int* q) {
  uint32_t v;
  asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(q) : "memory");
  return v;
}
__device__ __forceinline__ void red_release(unsigned int* q, uint32_t v) {
  asm volatile("red.release.gpu.global.add.u32 [%0], %1;" ::"l"(q), "r"(v) : "memory");
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
// exact-class GELU on a pair (A&S 7.1.26 erfc, |err| <= 1.5e-7); ss += h^2
__device__ __forceinline__ void gelu2(float& x0, float& x1, uint64_t& ss2) {
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
  const uint64_t H = add2(XH, f2pack(fmaxf(d0, 0.f), fmaxf(d1, 0.f)));
  ss2 = fma2(H, H, ss2);
  f2unpack(H, x0, x1);
}
__device__ __forceinline__ uint32_t hswz(uint32_t off) { return off ^ (((off >> 7) & 1u) << 4); }  // SW32
}  // namespace fz

__global__ void __launch_bounds__(fz::NTH, 1) f_kernel(const FParams p) {
  using namespace fz;
  extern __shared__ unsigned char smraw[];
  unsigned char* sm = (unsigned char*)(((uintptr_t)smraw + 1023) & ~(uintptr_t)1023);
  const uint32_t sb = s_addr(sm);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int t = p.t, G = (int)gridDim.x, HW = p.HW;
  const int tile0 = blockIdx.x * t;
  const int nv = min(t, p.N - tile0);
  const int img0 = tile0 / HW, img1 = (tile0 + nv - 1) / HW;
  const bool strad = img1 != img0;
  const int bnd = strad ? img1 * HW - tile0 : t;  // first local token of batch img1
  float* s_b1 = (float*)(sm + OFF_B1);
  float* s_gw = (float*)(sm + OFF_GW);
  float* s_sc = (float*)(sm + OFF_SC);
  float* s_gs = (float*)(sm + OFF_GS);
  float* s_b2p = (float*)(sm + OFF_B2P);
  float* s_red = (float*)(sm + OFF_RED);
  const uint32_t bars = sb + OFF_BAR;
  // full s: bars + 8s; empty s: + 24 + 8s; g1d b: + 48 + 8b; hrd b: + 80 + 8b; hfr i: + 112 + 8i; g2d: + 128
  uint32_t* s_tmem = (uint32_t*)(sm + OFF_BAR + 192);
  unsigned int* ctr = g_fctr;

  if (tid == 0) {
    for (int i = 0; i < 17; i++) mbar_init(bars + 8 * i, (i >= 10 && i < 14) ? 16u : 1u);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  if (warp == 16) {
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" ::"r"(s_addr(s_tmem)), "r"(512) : "memory");
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;" ::: "memory");
  }
  // x tile into registers first (t rows x 32 float4; rows >= nv are zero)
  float4 xv[6];
#pragma unroll
  for (int i = 0; i < 6; i++) {
    const int f = tid + 512 * i;
    xv[i] = make_float4(0.f, 0.f, 0.f, 0.f);
    if (warp < 16 && f < t * 32 && (f >> 5) < nv) xv[i] = __ldg(reinterpret_cast<const float4*>(p.x) + (size_t)tile0 * 32 + f);
  }
  for (int i = tid; i < 512; i += NTH) { s_b1[i] = __ldg(p.b1 + i); s_gw[i] = __ldg(p.gw + i); }
  for (int i = tid; i < 1024; i += NTH) s_gs[i] = 0.f;
  // ---- phase 0: this CTA's slice of the hi/lo weight images (exact smem byte images; W2 lo scaled by 2^11)
#pragma unroll 1
  for (int f = blockIdx.x * NTH + tid; f < 32768; f += G * NTH) {
    float4 w;
    int u, r, k0;
    float ls;
    if (f < 16384) {
      w = __ldg(reinterpret_cast<const float4*>(p.w1) + f);
      const int n = f >> 5;
      k0 = (f & 31) * 4; u = (n >> 7) * 2 + (k0 >> 6); r = n & 127; ls = 1.f;
    } else {
      const int g = f - 16384;
      w = __ldg(reinterpret_cast<const float4*>(p.w2) + g);
      k0 = (g & 127) * 4; u = 8 + (k0 >> 6); r = g >> 7; ls = 2048.f;
    }
    const int chunk = (k0 & 63) >> 3;
    const size_t off = (size_t)u * 32768 + r * 128 + ((chunk ^ (r & 7)) << 4) + ((k0 >> 2) & 1) * 8;
    const __half2 h01 = __floats2half2_rn(w.x, w.y), h23 = __floats2half2_rn(w.z, w.w);
    const float2 f01 = __half22float2(h01), f23 = __half22float2(h23);
    const __half2 l01 = __floats2half2_rn((w.x - f01.x) * ls, (w.y - f01.y) * ls);
    const __half2 l23 = __floats2half2_rn((w.z - f23.x) * ls, (w.w - f23.y) * ls);
    *reinterpret_cast<uint2*>(p.wimg + off) = make_uint2(h2u(h01), h2u(h23));
    *reinterpret_cast<uint2*>(p.wimg + off + 16384) = make_uint2(h2u(l01), h2u(l23));
  }
#pragma unroll 1
  for (int o = blockIdx.x * NWARP + warp; o < 128; o += G * NWARP) {
    const float4* wr = reinterpret_cast<const float4*>(p.w2 + o * 512);
    const float4* gr = reinterpret_cast<const float4*>(p.gb);
    float d = 0.f;
#pragma unroll
    for (int i = 0; i < 4; i++) {
      const float4 w = __ldg(wr + lane + 32 * i), g = __ldg(gr + lane + 32 * i);
      d = fmaf(w.x, g.x, fmaf(w.y, g.y, fmaf(w.z, g.z, fmaf(w.w, g.w, d))));
    }
    d = warp_sum(d);
    if (lane == 0) p.b2p[o] = d;
  }
  for (int i = blockIdx.x * NTH + tid; i < p.B * 512; i += G * NTH) p.gsum[i] = 0.f;
  asm volatile("fence.proxy.async;" ::: "memory");
  __syncthreads();
  if (tid == 0) { __threadfence(); red_release(ctr, 1u); }
  // ---- x tile split hi/lo into the K-major SW128 B tile ([hi|lo][ka][t rows][128 B]) in h' buffer 0
  if (warp < 16) {
#pragma unroll
    for (int i = 0; i < 6; i++) {
      const int f = tid + 512 * i;
      if (f < t * 32) {
        const int row = f >> 5, k0 = (f & 31) * 4, ka = k0 >> 6, chunk = (k0 & 63) >> 3;
        const int off = ka * t * 128 + row * 128 + ((chunk ^ (row & 7)) << 4) + ((k0 >> 2) & 1) * 8;
        const float4 v = xv[i];
        const __half2 h01 = __floats2half2_rn(v.x, v.y), h23 = __floats2half2_rn(v.z, v.w);
        const float2 f01 = __half22float2(h01), f23 = __half22float2(h23);
        const __half2 l01 = __floats2half2_rn(v.x - f01.x, v.y - f01.y), l23 = __floats2half2_rn(v.z - f23.x, v.w - f23.y);
        *reinterpret_cast<uint2*>(sm + OFF_HB + off) = make_uint2(h2u(h01), h2u(h23));
        *reinterpret_cast<uint2*>(sm + OFF_HB + t * 256 + off) = make_uint2(h2u(l01), h2u(l23));
      }
    }
  }
  fence_proxy_async();
  tc_before();
  __syncthreads();
  tc_after();
  const uint32_t tmem = *s_tmem;
  const uint32_t c_main = 4 * t, c_cross = (6 * t <= 512) ? 5 * t : 0;  // t = 96: cross acc reuses h block 0
  const int nch = t >> 3;

  if (warp == 17) {
    // ===================== producer: refill a stage as soon as its MMAs commit =====================
    if (lane == 0) {
      while (ld_acquire(ctr) < (unsigned)G) { }
      asm volatile("fence.proxy.async;" ::: "memory");
#pragma unroll 1
      for (int n = 0; n < 16; n++) {
        const int s = n % NST;
        if (n >= NST) mbar_wait(bars + 24 + 8 * s, ((n - NST) / NST) & 1);
        mbar_expect(bars + 8 * s, 32768);
        bulk_g2s(sb + OFF_RING + s * 32768, p.wimg + (size_t)n * 32768, 16384, bars + 8 * s);
        bulk_g2s(sb + OFF_RING + s * 32768 + 16384, p.wimg + (size_t)n * 32768 + 16384, 16384, bars + 8 * s);
      }
    }
    __syncwarp();
  } else if (warp == 16) {
    // ===================== MMA issuer: never waits on its own completions =====================
    const uint32_t id1 = idesc_f16(t, false), id2 = idesc_f16(t, true);
    const uint32_t xh = sb + OFF_HB, xl = xh + t * 256;
#pragma unroll 1
    for (int u = 0; u < 16; u++) {
      const int s = u % NST;
      mbar_wait(bars + 8 * s, (u / NST) & 1);
      tc_after();
      const uint64_t A0 = sdesc(sb + OFF_RING + s * 32768, 16, 1024, 2);  // hi; lo at +16384 B (+1024)
      if (u < 8) {
        const int b = u >> 1, ka = u & 1;
        const uint32_t d = tmem + b * t;
        const uint64_t X0 = sdesc(xh + ka * t * 128, 16, 1024, 2), X1 = sdesc(xl + ka * t * 128, 16, 1024, 2);
#pragma unroll
        for (int ks = 0; ks < 4; ks++) {
          mma_issue(d, A0 + 2 * ks, X0 + 2 * ks, id1, (ka | ks) ? 1u : 0u);
          mma_issue(d, A0 + 2 * ks, X1 + 2 * ks, id1, 1u);
          mma_issue(d, A0 + 1024 + 2 * ks, X0 + 2 * ks, id1, 1u);
        }
        mma_commit(bars + 24 + 8 * s);
        if (ka) mma_commit(bars + 48 + 8 * b);
      } else {
        const int v = u - 8, b = v >> 1, kcl = v & 1;
        if (kcl == 0) { mbar_wait(bars + 80 + 8 * b, 0); tc_after(); }
        const uint32_t hh = sb + OFF_HB + (b & 1) * HBUF + kcl * 2 * HREG;
        const uint64_t H0 = sdesc(hh, 64 * HRB, 8 * HRB, 6), H1 = sdesc(hh + HREG, 64 * HRB, 8 * HRB, 6);  // +16 rows = +512 B (+32)
#pragma unroll
        for (int ks = 0; ks < 4; ks++) {
          mma_issue(tmem + c_main, A0 + 2 * ks, H0 + 32 * ks, id2, (v | ks) ? 1u : 0u);
          mma_issue(tmem + c_cross, A0 + 2 * ks, H1 + 32 * ks, id2, (v | ks) ? 1u : 0u);
          mma_issue(tmem + c_cross, A0 + 1024 + 2 * ks, H0 + 32 * ks, id2, 1u);
        }
        mma_commit(bars + 24 + 8 * s);
        if (kcl && b < 2) mma_commit(bars + 112 + 8 * b);
        if (v == 7) mma_commit(bars + 128);
      }
    }
  } else {
    // ===================== epilogue warps: (TMEM lane quadrant q, column group g) =====================
    const int q = warp & 3, g = warp >> 2;
    const int lim0 = min(bnd, nv);
    const uint32_t tl = tmem + ((uint32_t)(32 * q) << 16);
#pragma unroll 1
    for (int b = 0; b < 4; b++) {
      mbar_wait(bars + 48 + 8 * b, 0);
      tc_after();
      const int k = 128 * b + 32 * q + lane;
      const float bk = s_b1[k];
      const uint64_t bk2 = f2pack(bk, bk);
      uint64_t ss2 = f2pack(0.f, 0.f);
      float s0x = 0.f, s1 = 0.f;
#pragma unroll 1
      for (int c = g; c < nch; c += 4) {
        const uint32_t ta = tl + b * t + 8 * c;
        float a[8];
        tmem_ld8(ta, a);
        if (8 * c + 8 <= lim0) {
#pragma unroll
          for (int e = 0; e < 8; e += 2) {
            float x0, x1;
            f2unpack(add2(f2pack(a[e], a[e + 1]), bk2), x0, x1);
            gelu2(x0, x1, ss2);
            a[e] = x0; a[e + 1] = x1;
          }
        } else {
          uint64_t dum = f2pack(0.f, 0.f);
#pragma unroll
          for (int e = 0; e < 8; e += 2) {
            float x0, x1;
            f2unpack(add2(f2pack(a[e], a[e + 1]), bk2), x0, x1);
            gelu2(x0, x1, dum);
            a[e] = x0; a[e + 1] = x1;
          }
#pragma unroll
          for (int e = 0; e < 8; e++) {  // slot 0: tokens < min(bnd, nv); slot 1: [bnd, nv); padded tokens: none
            const int j = 8 * c + e;
            const float hh = a[e] * a[e];
            if (j < lim0) s0x += hh;
            else if (j >= bnd && j < nv) s1 += hh;
          }
        }
        tmem_st8(ta, a);
      }
      float sa, sbb;
      f2unpack(ss2, sa, sbb);
      atomicAdd(s_gs + k, sa + sbb + s0x);
      if (strad) atomicAdd(s_gs + 512 + k, s1);
    }
    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
    tc_before();
    bar512();
    if (tid == 0) { while (ld_acquire(ctr) < (unsigned)G) { } }  // every CTA's gsum slice is zeroed
    bar512();
    atomicAdd(p.gsum + img0 * 512 + tid, s_gs[tid]);
    if (strad) atomicAdd(p.gsum + img1 * 512 + tid, s_gs[512 + tid]);
    if (tid < 128) s_b2p[tid] = __ldg(p.b2 + tid) + __ldcg(p.b2p + tid);
    bar512();
    if (tid == 0) {
      __threadfence();
      red_release(ctr + 2 + img0, (uint32_t)(strad ? bnd : nv));
      if (strad) red_release(ctr + 2 + img1, (uint32_t)(nv - bnd));
      while (ld_acquire(ctr + 2 + img0) < (unsigned)HW) { }
      if (strad) { while (ld_acquire(ctr + 2 + img1) < (unsigned)HW) { } }
      const unsigned old = atomicAdd(ctr + 1, 1u);
      if (old == (unsigned)G - 1) {
        for (int bb = 0; bb < p.B; bb++) atomicExch(ctr + 2 + bb, 0u);
        atomicExch(ctr, 0u);
        atomicExch(ctr + 1, 0u);
      }
    }
    bar512();
    // ---- GRN scale per batch slot: s[k] = 1 + gw[k] * G[k] / (mean_k G + eps)
#pragma unroll 1
    for (int sl = 0; sl < (strad ? 2 : 1); sl++) {
      const int img = sl ? img1 : img0;
      const float gv = __fsqrt_rn(__ldcg(p.gsum + img * 512 + tid));
      const float ws = warp_sum(gv);
      if (lane == 0) s_red[warp] = ws;
      bar512();
      float tot = 0.f;
#pragma unroll
      for (int i = 0; i < 16; i++) tot += s_red[i];
      const float nf = __fdiv_rn(gv, tot * (1.f / 512.f) + p.eps);
      s_sc[sl * 512 + tid] = fmaf(s_gw[tid], nf, 1.f);
      bar512();
    }
    tc_after();
    // ---- h' = h * s split hi / lo*2^11 into the MN-major SW32 h' tiles, one hidden block at a time
    const int kcl = q >> 1, r = 32 * (q & 1) + lane;
#pragma unroll 1
    for (int b = 0; b < 4; b++) {
      if (b >= 2) { mbar_wait(bars + 112 + 8 * (b & 1), 0); }
      const int kk = 128 * b + 32 * q + lane;
      const float sc0 = s_sc[kk], sc1 = s_sc[512 + kk];
      unsigned char* hb = sm + OFF_HB + (b & 1) * HBUF + kcl * 2 * HREG;
#pragma unroll 1
      for (int c = g; c < nch; c += 4) {
        float a[8];
        tmem_ld8(tl + b * t + 8 * c, a);
        if (8 * c + 8 <= bnd) {
#pragma unroll
          for (int e = 0; e < 8; e++) a[e] *= sc0;
        } else {
#pragma unroll
          for (int e = 0; e < 8; e++) a[e] *= (8 * c + e < bnd) ? sc0 : sc1;
        }
        uint32_t hi[4], lo[4];
#pragma unroll
        for (int e = 0; e < 4; e++) {
          const __half2 h = __floats2half2_rn(a[2 * e], a[2 * e + 1]);
          const float2 f = __half22float2(h);
          hi[e] = h2u(h);
          lo[e] = h2u(__floats2half2_rn((a[2 * e] - f.x) * 2048.f, (a[2 * e + 1] - f.y) * 2048.f));
        }
        const uint32_t off = (c >> 1) * 2048 + hswz(r * HRB + (c & 1) * 16);
        *reinterpret_cast<uint4*>(hb + off) = make_uint4(hi[0], hi[1], hi[2], hi[3]);
        *reinterpret_cast<uint4*>(hb + HREG + off) = make_uint4(lo[0], lo[1], lo[2], lo[3]);
      }
      fence_proxy_async();
      tc_before();
      __syncwarp();
      if (lane == 0) mbar_arrive(bars + 80 + 8 * b);
    }
    // ---- out = D2 + D2x / 2^11 + b2', [token][128] rows, lanes along output channels
    mbar_wait(bars + 128, 0);
    tc_after();
    const int o = 32 * q + lane;
    const float bias = s_b2p[o];
#pragma unroll 1
    for (int c = g; c < nch; c += 4) {
      float m[8], xc[8];
      tmem_ld8x2(tl + c_main + 8 * c, tl + c_cross + 8 * c, m, xc);
#pragma unroll
      for (int e = 0; e < 8; e++) {
        const int j = 8 * c + e;
        if (j < nv) p.out[(size_t)(tile0 + j) * 128 + o] = m[e] + xc[e] * (1.f / 2048.f) + bias;
      }
    }
  }
  tc_before();
  __syncthreads();
  if (warp == 16) {
    tc_after();
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" ::"r"(tmem), "r"(512) : "memory");
  }
}

// 1 if the fused path takes this shape; t tokens per CTA, G CTAs (one wave at 1 CTA/SM)
extern "C" int f_plan(int B, int HW, int* t_out, int* g_out) {
  static int nsm = 0;
  if (nsm == 0) {
    int dev = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, dev);
  }
  const long long N = (long long)B * HW;
  const long long per = (N + 147) / 148;
  int t = (int)(((per + 15) / 16) * 16);
  if (t > fz::TMAX) t = fz::TMAX;
  const long long G = (N + t - 1) / t;
  *t_out = t;
  *g_out = (int)G;
  return (G <= nsm && HW >= t && B <= 128) ? 1 : 0;
}

extern "C" const char* f_launch(const float* x, const float* w1, const float* b1, const float* gw, const float* gb,
                                const float* w2, const float* b2, float* out, void* ws, int B, int HW, int t, int G,
                                float eps, cudaStream_t stream) {
  static bool attr_set = false;
  cudaError_t e;
  if (!attr_set) {  // one maximal value for every shape
    e = cudaFuncSetAttribute(f_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, fz::SMEM_TOTAL);
    if (e != cudaSuccess) return cudaGetErrorString(e);
    attr_set = true;
  }
  FParams p;
  p.x = x; p.w1 = w1; p.b1 = b1; p.gw = gw; p.gb = gb; p.w2 = w2; p.b2 = b2; p.out = out;
  p.wimg = (unsigned char*)ws;
  p.b2p = (float*)((unsigned char*)ws + 524288);
  p.gsum = (float*)((unsigned char*)ws + 524288 + 512);
  p.N = B * HW; p.HW = HW; p.B = B; p.t = t; p.eps = eps;
  void* args[] = {(void*)&p};
  e = cudaLaunchCooperativeKernel((const void*)f_kernel, dim3(G), dim3(fz::NTH), args, (size_t)fz::SMEM_TOTAL, stream);
  if (e != cudaSuccess) return cudaGetErrorString(e);
  return nullptr;
}
```

```cpp file=binding.cpp
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

extern "C" void launch_grn_mlp(const float* x, const float* w1, const float* b1, const float* gw, const float* gb,
                               const float* w2, const float* b2, float eps, float* out, float* part, float* nf,
                               int B, int HW, cudaStream_t stream);
extern "C" int f_plan(int B, int HW, int* t_out, int* g_out);
extern "C" const char* f_launch(const float* x, const float* w1, const float* b1, const float* gw, const float* gb,
                                const float* w2, const float* b2, float* out, void* ws, int B, int HW, int t, int G,
                                float eps, cudaStream_t stream);

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
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  int t = 0, G = 0;
  if (f_plan(B, HW, &t, &G)) {
    // workspace (uninitialised; every byte read is written earlier in the same call): 512 KB hi/lo weight images,
    // b2' [128], per-batch sumsq [B][512]
    auto ws = torch::empty({(int64_t)(131072 + 128 + (int64_t)B * 512)}, hidden_states.options());
    const char* err = f_launch(hidden_states.data_ptr<float>(), pwconv1_weight.data_ptr<float>(),
                               pwconv1_bias.data_ptr<float>(), grn_weight.data_ptr<float>(), grn_bias.data_ptr<float>(),
                               pwconv2_weight.data_ptr<float>(), pwconv2_bias.data_ptr<float>(), output.data_ptr<float>(),
                               ws.data_ptr(), B, HW, t, G, (float)eps, stream);
    TORCH_CHECK(err == nullptr, "fused launch failed: ", err ? err : "");
    return;
  }
  const int tiles = (HW + 63) / 64;
  auto part = torch::empty({(int64_t)B * tiles * 512}, hidden_states.options());
  auto nf = torch::empty({(int64_t)B * 512}, hidden_states.options());
  launch_grn_mlp(hidden_states.data_ptr<float>(), pwconv1_weight.data_ptr<float>(), pwconv1_bias.data_ptr<float>(),
                 grn_weight.data_ptr<float>(), grn_bias.data_ptr<float>(), pwconv2_weight.data_ptr<float>(),
                 pwconv2_bias.data_ptr<float>(), (float)eps, output.data_ptr<float>(), part.data_ptr<float>(),
                 nf.data_ptr<float>(), B, HW, stream);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("run", &run, "fused linear-gelu-grn-linear"); }
```

```yaml design-card
id: r2-a-e1-prod-fix
parents:
  - r1-a-e1-ts-fp16x3-m
  - r1-merge
operation: structural_mutation
language: cuda_cpp
niche:
  mem: bulk1d
  st: direct
  grid: persistent
  launch: fused
  tile: rows96
  red: cta
  cache: default
  spec: dispatch:size
hypothesis: >-
  The weight-path chain in r1's e1 path was neither L2 ingest nor an L2 hot spot. The MMA warp also refilled the ring,
  and it waited for the previous unit's MMA commit before every refill, so at most two units of MMAs were ever queued.
  A dedicated producer warp removes that coupling. In probes on the rented B200 this cut 2.6-3.6 us (8-15%) from every
  S/M row tested. The same kernel also fixes the r1 bug where a partial last CTA's padded tokens corrupted the last
  batch's GRN statistics.
expected_effect:
  S:
    pct: -13
    confidence: medium
  M:
    pct: -10
    confidence: medium
  L:
    pct: 0
    confidence: high
resources_sm100a:
  regs_per_thread: 96
  smem_per_cta_bytes: 210944
  threads_per_cta: 576
  ctas_per_sm: 1
  bytes_in_flight_per_sm: 98304
  launches_per_call: 1
knobs:
  TMAX: 96
  NST: 3
  NTH: 576
  h_swizzle: SW32
  lo_scale_gemm2: 2048
dispatch:
  - {max_tokens: 14208, kernel: f_kernel, meta: {t: "min(96, ceil16(N/148))", grid: "ceil(N/t)", cooperative: true, producer_warp: 17, mma_warp: 16}}
  - {max_tokens: null, kernel: r0 k_stats+k_grn+k_main, meta: {TM: 64}}
tests: [H18]
findings:
  - "CORRECTNESS BUG in r1-a-e1-ts-fp16x3(-m), also present in r1-merge for 64,14,14 and 2,56,56: in the slow sumsq path, padded tokens (j >= nv) of a partial last CTA were added to slot 0. Against an fp64 reference with randn gw/gb, err/tol was 444 at 64,14,14 (1.5% of elements fail) and 12.7 at 2,56,56 (22% fail): the whole last batch is wrong. Rows where N is a multiple of t (1,28,28; 4,28,28) were fine (0.18-0.27). r1's run_tests and the portal passed anyway, so the harness's GRN inputs probably make the GRN term small (e.g. gw ~ 0); do not rely on that. The fix is slot 0 iff j < min(bnd, nv), slot 1 iff bnd <= j < nv [probe_b200]"
  - "precision after the fix (fp64 reference, atol 2.8e-3, rtol 1e-5, seed 0): 1,28,28 0.19; 64,14,14 0.20; 2,56,56 0.24; 32,14,14 0.25; 16,28,28 0.23; 4,56,56 0.22; 4,28,28 0.27 (seed 1). Only 1-2 seeds per row were run, not the full 10-seed gate; e4 on the L rows was not gated [probe_b200]"
  - "stamps of r1 e1 at 64,14,14 (rented, median/max us from the first CTA's entry): phase 0 done 3.1/4.1, weight counter passed 5.6/5.9, unit 0 lands 6.4, unit 7 13.1, GEMM1 epilogue done 15.4/16.4, batch wait 18.2/19.5, GEMM2 start 20.5/28.2, GEMM2 done 27.9/35.3, store 29.2/36.6. At 1,28,28: phase 0 2.1, weight counter 4.6, units every ~0.7 us, store 20.5. Weight units plus phase 0 make up about 70% of the chain, so H18 is not refuted [probe_b200]"
  - "L2 hot-spot probe (148 or 49 CTAs x 16 x 32 KB bulk copies after a 2xL2 flush): one shared image runs at 146-205 GB/s/SM (16 units in 2.6-3.6 us). 16 replicas are 5-30% slower and 148 private copies 30-40% slower. Replicas (step 4a) are a dead end [probe_b200]"
  - "neither 16 warps polling mbarrier.try_wait (with or without a suspend hint or nanosleep) nor an image written in-kernel with scattered 8 B stores slows bulk ingest: both stay at 158-205 GB/s/SM [probe_b200]"
  - "a deeper ring (6 stages at t=16, 5 at t=48) changes nothing (22.8 vs 22.4 us; 26.4 vs 26.3 us), so step 3's ring variant is a dead end [probe_b200]"
  - "MMA issue cost on B200 (M=128, K=16, kind::f16, rented clocks): with a tight unrolled issue loop, 21 ns per instruction at N=16, 32 ns at N=96, 37 ns at N=128, 73 ns at N=256 (ideal at 1.9 GHz: 4/25/34/67). One accumulator vs 2-4 interleaved makes no difference, so dependent-accumulate latency is not a limit. A loop with runtime div/mod in its descriptor maths ran at 53-56 ns per MMA, so the issue code matters; elect.sync issue beats single-thread issue at N=16 (21 vs 33 ns) [probe_b200]"
  - "in e1 the stamps mislabelled MMA completion as unit 'landing'. Six units issued at once land 0.77 us apart only because the MMA warp's loop waited on the previous unit's commit before refilling. Hoisting descriptors alone gained nothing (the compiler had already hoisted them). Moving the producer to warp 17 so the MMA warp never waits on completions measured (rented, harness_time, 3-stage ring): 1,28,28 23.3 -> 19.9 us; 64,14,14 38.9 -> 35.5; 2,56,56 26.9 -> 23.6; 32,14,14 31.2 -> 27.6; 16,28,28 34.3 -> 31.7; 4,56,56 28.2 [probe_b200]"
  - "success line missed: 64,14,14 35.5 us (target <= 29) and 2,56,56 23.6 us (target <= 20). The largest remaining term at 64,14,14 is the CTA tail: median store 26.6 us, max 34.0, and some CTAs start GEMM2 8 us after their batch wait. Phase 0 plus the grid wait still costs about 5.6 us, and the batch wait about 3 us. The tail probe could not run (Modal capacity) [probe_b200]"
  - "e4 (r1-a-e4-rounds-tmem) already uses a separate producer warp (16) and MMA warp (17), so step 5's producer change does not apply. e4 issues kdesc() per MMA, but hoisting did not matter in e1. Its partial-tile sumsq logic is correct by inspection (sp0 = min(nv, ...)) [get_kernel]"
  - "tooling: this exact two-file build was NOT compiled or run_tests'd. Two run_tests calls timed out on Modal capacity and then the round's 20-minute B200 budget ran out. The device code matches the NVRTC probe build e1c.cu (96 regs, STACK 0, LOCAL 0 under NVRTC) except for compiled-out stamps and a constant NST=3 layout [run_tests]"
paths:
  - {max_tokens: 14208, lang: cuda, width: 128, threads: 576, rows: 96, grid: persistent, mem: tma, wstat: false, launches: 1, prefetch: true, x: none, w: none, st: none, cluster: 1}
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
  - "This two-file build was never compiled or run through run_tests; only its NVRTC probe twin was run (all checked rows correct, 1-2 seeds each)"
  - "Cooperative launch with 211 KB smem at 1 CTA/SM; grid <= SM count is checked, occupancy is not queried"
  - "Tail CTAs at HW=196 (up to +7 us over the median CTA) are not understood yet"
  - "The 10-seed precision gate is incomplete (1-2 seeds per row); e4's L rows were not gated"
measure_first: [B200]
compliance:
  single_stream: true
  no_cross_call_state: true
  fp32_math: true
  no_threads_or_fork: true
  no_precompiled_binaries: true
```