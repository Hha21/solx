// From this project's own work on L2 #129 (035_convnextv2_block_with_grn, ConvNeXtV2 block), round 13, lineage M
// (r13-m-e4-token-stationary; our code). Token-stationary tcgen05 kernel for C=96: weights resident in shared
// memory, the 4C hidden kept in TMEM, GRN via per-image counters. Reference only: use, adapt or ignore.

// Lineage M: token-stationary one-launch tcgen05 kernel for the C=96, 56x56 rows (B = 1, 2, 4).
// Each CTA owns T 8x8-token patches (64 tokens each). Warps 0..11: x halo by cp.async (zero-filled outside the image),
// depthwise 7x7 conv in registers, LN statistics, LN affine in registers into the MN-major SW128 a16 tile. Warps 12..15
// convert this CTA's 1/G slice of W1/W2 into exact fp16 smem byte images in a workspace and bump one weight counter; the
// W1 image then lands by bulk copy behind the conv and the W2 image behind GEMM1. GEMM1 swap-AB (3 x 128 hidden rows x
// 64 tokens, K = 96) into TMEM, + b1, A&S erfc GELU, fp32 h back into TMEM, per-(image,k) sumsq red.add, per-image
// counters, GRN scale, h' = fp16(h * s) into an smem B tile, GEMM2 (96 out channels x 64T tokens, K = 384) and
// out = x + acc + b2 + W2.gb stored NCHW. The hidden is rounded once.
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdint.h>

struct MParams {
  const float* x; const float* dww; const float* dwb; const float* lnw; const float* lnb;
  const float* w1; const float* b1; const float* gw; const float* gb; const float* w2; const float* b2;
  float* out; float* gsum; unsigned char* w1img; unsigned char* w2img; float* b2pg;
  int B; float eps, ln_eps;
};

// [0] gsum-zeroed flag, [1] departure count, [2 + b] GEMM1 sub-tiles published for image b, [6] CTAs whose weight-image
// slice is written. Every CTA departs after its last poll; the last departer resets all of them for the next call.
__device__ unsigned int g_mctr[8];

namespace {
constexpr int NTH = 512;
constexpr int XPS = 912;          // x plane stride: 4 consecutive channels hit 4 different 16-B bank slots
constexpr int OFF_XS = 0;         // x halo fp32 [96][14][16] (87552 B); later the W2 fp16 K-major SW128 image (6 x 12288 B)
constexpr int OFF_A1 = 88064;     // W1 fp16 K-major SW128 image: k 0..63 of blocks 0..2, k 64..95 of blocks 0+1, of block 2
                                  // (80 KB); later the h' tiles [T][384][128 B] (T = 2 runs over B1, unused by then)
constexpr int OFF_SSP = 169984;   // GRN sumsq partials [4 column groups][384]
constexpr int OFF_LNP = 176128;   // LN partials [12 warps][64 tokens][2]
constexpr int OFF_B1 = 184320;    // a16 MN-major SW128 [96][128 B]
constexpr int OFF_DW = 196608;    // dw taps [96][49] fp32
constexpr int OFF_VEC = 215424;   // lnw, lnb, dwb, b2 [96]; b1, gw, gb [384]
constexpr int OFF_SS = 221568;    // GRN scale [2][384]
constexpr int OFF_ST = 224640;    // mu[64], rs[64]
constexpr int OFF_RED = 225152;   // reduction scratch [32]
constexpr int OFF_BAR = 225280;   // 5 mbarriers + TMEM slot
constexpr int SMEM_TOTAL = 225408 + 1024;
constexpr uint32_t VEC_BYTES = 18816 + 6144;
constexpr uint32_t W1IMG = 81920, W2IMG = 73728;

__device__ __forceinline__ uint32_t s_addr(const void* q) { return (uint32_t)__cvta_generic_to_shared(q); }
__device__ __forceinline__ void mbar_init(uint32_t b, uint32_t n) { asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(b), "r"(n)); }
__device__ __forceinline__ void mbar_expect(uint32_t b, uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(b), "r"(bytes) : "memory");
}
__device__ __forceinline__ void mbar_wait(uint32_t b, uint32_t par) {
  asm volatile("{\n.reg .pred P;\nWAIT_%=:\nmbarrier.try_wait.parity.shared::cta.b64 P, [%0], %1;\n@!P bra WAIT_%=;\n}\n" ::"r"(b), "r"(par) : "memory");
}
__device__ __forceinline__ void bulk_g2s(uint32_t dst, const void* src, uint32_t bytes, uint32_t bar) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];" ::"r"(dst), "l"(src), "r"(bytes), "r"(bar) : "memory");
}
__device__ __forceinline__ void fence_proxy_async() { asm volatile("fence.proxy.async.shared::cta;" ::: "memory"); }
__device__ __forceinline__ void tc_before() { asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory"); }
__device__ __forceinline__ void tc_after() { asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory"); }
__device__ __forceinline__ uint64_t sdesc(uint32_t addr, uint32_t lbo, uint32_t sbo) {  // SWIZZLE_128B, sm100 descriptor version 1
  return (uint64_t)((addr >> 4) & 0x3FFF) | ((uint64_t)((lbo >> 4) & 0x3FFF) << 16) | ((uint64_t)((sbo >> 4) & 0x3FFF) << 32) |
         (1ull << 46) | (2ull << 61);
}
// kind::f16: fp32 D, fp16 A/B, A K-major, B MN-major, M = 128
__device__ __forceinline__ uint32_t idesc_f16(int n) { return (1u << 4) | (1u << 16) | ((uint32_t)(n >> 3) << 17) | ((128u >> 4) << 24); }
__device__ __forceinline__ void mma_issue(uint32_t d, uint64_t a, uint64_t b, uint32_t id, uint32_t acc) {  // whole warp, one lane issues
  asm volatile("{\n.reg .pred E, A;\nelect.sync _|E, 0xffffffff;\nsetp.ne.u32 A, %4, 0;\n"
               "@E tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, A;\n}\n" ::"r"(d), "l"(a), "l"(b), "r"(id), "r"(acc) : "memory");
}
__device__ __forceinline__ void mma_commit(uint32_t bar) {  // whole warp, the same elected lane commits
  asm volatile("{\n.reg .pred E;\nelect.sync _|E, 0xffffffff;\n"
               "@E tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];\n}\n" ::"r"(bar) : "memory");
}
__device__ __forceinline__ void tmem_ld16(uint32_t ta, float (&v)[16]) {
  uint32_t r[16];
  asm volatile("tcgen05.ld.sync.aligned.32x32b.x16.b32 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, [%16];\n"
               "tcgen05.wait::ld.sync.aligned;\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]), "=r"(r[4]), "=r"(r[5]), "=r"(r[6]), "=r"(r[7]), "=r"(r[8]),
                 "=r"(r[9]), "=r"(r[10]), "=r"(r[11]), "=r"(r[12]), "=r"(r[13]), "=r"(r[14]), "=r"(r[15])
               : "r"(ta) : "memory");
#pragma unroll
  for (int i = 0; i < 16; i++) v[i] = __uint_as_float(r[i]);
}
__device__ __forceinline__ void tmem_st16(uint32_t ta, const float (&v)[16]) {
  asm volatile("tcgen05.st.sync.aligned.32x32b.x16.b32 [%0], {%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16};\n" ::"r"(ta),
               "r"(__float_as_uint(v[0])), "r"(__float_as_uint(v[1])), "r"(__float_as_uint(v[2])), "r"(__float_as_uint(v[3])),
               "r"(__float_as_uint(v[4])), "r"(__float_as_uint(v[5])), "r"(__float_as_uint(v[6])), "r"(__float_as_uint(v[7])),
               "r"(__float_as_uint(v[8])), "r"(__float_as_uint(v[9])), "r"(__float_as_uint(v[10])), "r"(__float_as_uint(v[11])),
               "r"(__float_as_uint(v[12])), "r"(__float_as_uint(v[13])), "r"(__float_as_uint(v[14])), "r"(__float_as_uint(v[15]))
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
__device__ __forceinline__ void st_release(unsigned int* q, uint32_t v) {
  asm volatile("st.release.gpu.global.u32 [%0], %1;" ::"l"(q), "r"(v) : "memory");
}
__device__ __forceinline__ uint32_t h2u(__half2 h) { return *reinterpret_cast<uint32_t*>(&h); }
__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}
// packed f32x2 helpers (sm_100): one instruction for two lanes' worth of FMA-pipe work
__device__ __forceinline__ uint64_t f2pack(float a, float b) { uint64_t r; asm("mov.b64 %0, {%1, %2};" : "=l"(r) : "f"(a), "f"(b)); return r; }
__device__ __forceinline__ void f2unpack(uint64_t r, float& a, float& b) { asm("mov.b64 {%0, %1}, %2;" : "=f"(a), "=f"(b) : "l"(r)); }
__device__ __forceinline__ uint64_t fma2(uint64_t a, uint64_t b, uint64_t c) { uint64_t d; asm("fma.rn.f32x2 %0, %1, %2, %3;" : "=l"(d) : "l"(a), "l"(b), "l"(c)); return d; }
__device__ __forceinline__ uint64_t mul2(uint64_t a, uint64_t b) { uint64_t d; asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(d) : "l"(a), "l"(b)); return d; }
__device__ __forceinline__ uint64_t add2(uint64_t a, uint64_t b) { uint64_t d; asm("add.rn.f32x2 %0, %1, %2;" : "=l"(d) : "l"(a), "l"(b)); return d; }
// exact-class GELU on a pair: Abramowitz-Stegun 7.1.26 erfc (|err| <= 1.5e-7), hc = 0.5 erfc(|x| / sqrt2),
// h = x hc + max(x - 2 x hc, 0) (= x (1 - hc) for x >= 0, x hc otherwise); ss += h^2
__device__ __forceinline__ void gelu2(float& x0, float& x1, uint64_t& ss2) {
  float t0, t1, e0, e1, q0, q1, d0, d1;
  asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(t0) : "f"(fmaf(fabsf(x0), 0.231641900f, 1.f)));
  asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(t1) : "f"(fmaf(fabsf(x1), 0.231641900f, 1.f)));
  const uint64_t T = f2pack(t0, t1), X = f2pack(x0, x1);
  uint64_t P = fma2(T, f2pack(0.5307027145f, 0.5307027145f), f2pack(-0.7265760135f, -0.7265760135f));  // 0.5 a5, 0.5 a4, ...
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
// x halo of one 8x8 patch for this warp's 8 channels: [8][14 rows][4 x 16-B granules] by cp.async, OOB granules zero-filled
__device__ __forceinline__ void load_x_warp(const float* __restrict__ x, uint32_t sb, int tile, int warp, int lane) {
  const int img = tile / 49, pi = tile - img * 49, h0 = (pi / 7) * 8, w0 = (pi % 7) * 8;
#pragma unroll 2
  for (int e = lane; e < 448; e += 32) {
    const int cl = e / 56, rem = e - cl * 56, row = rem >> 2, g = rem & 3;
    const int c = warp * 8 + cl, gh = h0 - 3 + row, gw = w0 - 4 + 4 * g;
    const bool ok = (gh >= 0) && (gh < 56) && (gw >= 0) && (gw < 56);
    const float* src = ok ? x + ((size_t)(img * 96 + c) * 56 + gh) * 56 + gw : x;
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::"r"(sb + OFF_XS + c * XPS + row * 64 + g * 16), "l"(src),
                 "r"(ok ? 16 : 0) : "memory");
  }
  asm volatile("cp.async.commit_group;" ::: "memory");
}
}  // namespace

template <int T>
__global__ void __launch_bounds__(NTH, 1) m_kernel(const MParams p) {
  extern __shared__ unsigned char smraw[];
  unsigned char* sm = (unsigned char*)(((uintptr_t)smraw + 1023) & ~(uintptr_t)1023);
  const uint32_t sb = s_addr(sm);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const float* vec = (const float*)(sm + OFF_VEC);
  const float* s_lnw = vec;
  const float* s_lnb = vec + 96;
  const float* s_dwb = vec + 192;
  const float* s_b2 = vec + 288;
  const float* s_b1 = vec + 384;
  const float* s_gw = vec + 768;
  const float* s_dw = (const float*)(sm + OFF_DW);
  float* s_ssp = (float*)(sm + OFF_SSP);
  float* s_lnp = (float*)(sm + OFF_LNP);
  float* s_ss = (float*)(sm + OFF_SS);
  float* s_mu = (float*)(sm + OFF_ST);
  float* s_rs = s_mu + 64;
  float* s_red = (float*)(sm + OFF_RED);
  const uint32_t bar_v = sb + OFF_BAR, bar_m1 = bar_v + 8, bar_m2 = bar_v + 16, bar_w1 = bar_v + 24, bar_w2 = bar_v + 32;
  uint32_t* s_tmem = (uint32_t*)(sm + OFF_BAR + 64);
  unsigned int* ctr = g_mctr;
  constexpr uint32_t NCOL = (T == 1) ? 256 : 512;
  const int tile0 = blockIdx.x * T;
  const int img0 = tile0 / 49;
  const int img1 = (tile0 + T - 1) / 49;

  if (tid == 0) {
    mbar_init(bar_v, 1); mbar_init(bar_m1, 1); mbar_init(bar_m2, 1); mbar_init(bar_w1, 1); mbar_init(bar_w2, 1);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  if (warp == 0) {
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" ::"r"(s_addr(s_tmem)), "r"(NCOL) : "memory");
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;" ::: "memory");
  }
  for (int i = tid; i < 1536; i += NTH) s_ssp[i] = 0.f;
  if (blockIdx.x == 0)
    for (int i = tid; i < p.B * 384; i += NTH) p.gsum[i] = 0.f;
  tc_before();
  __syncthreads();
  tc_after();
  const uint32_t tmem = *s_tmem;
  if (warp < 12) load_x_warp(p.x, sb, tile0, warp, lane);
  if (tid == 0) {
    if (blockIdx.x == 0) { __threadfence(); st_release(ctr, 1u); }
    mbar_expect(bar_v, VEC_BYTES);
    bulk_g2s(sb + OFF_DW, p.dww, 18816, bar_v);
    bulk_g2s(sb + OFF_VEC, p.lnw, 384, bar_v);
    bulk_g2s(sb + OFF_VEC + 384, p.lnb, 384, bar_v);
    bulk_g2s(sb + OFF_VEC + 768, p.dwb, 384, bar_v);
    bulk_g2s(sb + OFF_VEC + 1152, p.b2, 384, bar_v);
    bulk_g2s(sb + OFF_VEC + 1536, p.b1, 1536, bar_v);
    bulk_g2s(sb + OFF_VEC + 3072, p.gw, 1536, bar_v);
    bulk_g2s(sb + OFF_VEC + 4608, p.gb, 1536, bar_v);
  }
  // ---- warps 12..15 (no conv work): this CTA's slice of the fp16 weight images (exact smem byte images of the W1 and W2
  // A operands; 18432 float4 of fp32 weights in total) and of b2' = W2.gb, one release on the weight counter, then the
  // W1 image lands behind the conv
  if (warp >= 12) {
    const int t4 = tid - 384;
    const int per = (18432 + (int)gridDim.x - 1) / (int)gridDim.x;
#pragma unroll 1
    for (int j = t4; j < per; j += 128) {
      const int f = blockIdx.x * per + j;
      if (f >= 18432) break;
      if (f < 9216) {
        const float4 w = __ldg(reinterpret_cast<const float4*>(p.w1) + f);
        const int k = f / 24, c4 = f - 24 * k, b = k >> 7, r = k & 127, cs = c4 >> 1;
        const int atom = (cs < 8) ? b : 3 + (b >> 1);
        const int chunk = (cs < 8) ? cs : (cs - 8) + 4 * (b & 1);
        *reinterpret_cast<uint2*>(p.w1img + atom * 16384 + r * 128 + ((chunk ^ (r & 7)) << 4) + (c4 & 1) * 8) =
            make_uint2(h2u(__floats2half2_rn(w.x, w.y)), h2u(__floats2half2_rn(w.z, w.w)));
      } else {
        const int g = f - 9216;
        const float4 w = __ldg(reinterpret_cast<const float4*>(p.w2) + g);
        const int c = g / 96, k0 = (g - 96 * c) * 4;
        *reinterpret_cast<uint2*>(p.w2img + (k0 >> 6) * 12288 + c * 128 + ((((k0 >> 3) & 7) ^ (c & 7)) << 4) + ((k0 >> 2) & 1) * 8) =
            make_uint2(h2u(__floats2half2_rn(w.x, w.y)), h2u(__floats2half2_rn(w.z, w.w)));
      }
    }
    for (int c = blockIdx.x * 4 + (warp - 12); c < 96; c += gridDim.x * 4) {
      const float4* wr = reinterpret_cast<const float4*>(p.w2 + c * 384);
      const float4* gr = reinterpret_cast<const float4*>(p.gb);
      float d = 0.f;
#pragma unroll
      for (int i = 0; i < 3; i++) {
        const float4 w = __ldg(wr + lane + 32 * i), g = __ldg(gr + lane + 32 * i);
        d = fmaf(w.x, g.x, fmaf(w.y, g.y, fmaf(w.z, g.z, fmaf(w.w, g.w, d))));
      }
      d = warp_sum(d);
      if (lane == 0) p.b2pg[c] = d;
    }
    asm volatile("fence.proxy.async;" ::: "memory");
    asm volatile("bar.sync 1, 128;" ::: "memory");
    if (t4 == 0) {
      __threadfence();
      red_release(ctr + 6, 1u);
      while (ld_acquire(ctr + 6) < gridDim.x) { }
      asm volatile("fence.proxy.async;" ::: "memory");
      mbar_expect(bar_w1, W1IMG);
#pragma unroll
      for (int i = 0; i < 5; i++) bulk_g2s(sb + OFF_A1 + i * 16384, p.w1img + i * 16384, 16384, bar_w1);
    }
  }
  mbar_wait(bar_v, 0);

  const int q4 = warp & 3, g4 = warp >> 2;
#pragma unroll 1
  for (int s = 0; s < T; s++) {
    const int tile = tile0 + s;
    // ---- depthwise 7x7 conv: lane = (4 channels) x (row parity) x (4-col half) x (channel group); 4 rows x 4 cols each
    float acc[4][4];
    if (warp < 12) {
      asm volatile("cp.async.wait_group 0;" ::: "memory");
      __syncwarp();
      const int c = warp * 8 + (lane >> 4) * 4 + (lane & 3);
      const int pp = (lane >> 2) & 1, ch = (lane >> 3) & 1;
      float tap[49];
#pragma unroll
      for (int i = 0; i < 49; i++) tap[i] = s_dw[c * 49 + i];
#pragma unroll
      for (int m = 0; m < 4; m++)
#pragma unroll
        for (int cc = 0; cc < 4; cc++) acc[m][cc] = 0.f;
#pragma unroll
      for (int i = 0; i < 13; i++) {
        const int br = pp + i;
        float xr[12];
#pragma unroll
        for (int qq = 0; qq < 3; qq++) {
          const float4 v = *reinterpret_cast<const float4*>(sm + OFF_XS + c * XPS + br * 64 + (ch + qq) * 16);
          xr[4 * qq] = v.x; xr[4 * qq + 1] = v.y; xr[4 * qq + 2] = v.z; xr[4 * qq + 3] = v.w;
        }
#pragma unroll
        for (int m = 0; m < 4; m++) {
          const int ti = i - 2 * m;
          if (ti >= 0 && ti < 7) {
#pragma unroll
            for (int cc = 0; cc < 4; cc++)
#pragma unroll
              for (int j = 0; j < 7; j++) acc[m][cc] = fmaf(xr[cc + 1 + j], tap[ti * 7 + j], acc[m][cc]);
          }
        }
      }
      const float bias = s_dwb[c];
#pragma unroll
      for (int m = 0; m < 4; m++)
#pragma unroll
        for (int cc = 0; cc < 4; cc++) acc[m][cc] += bias;
      float ps[16], pq[16];
#pragma unroll
      for (int m = 0; m < 4; m++)
#pragma unroll
        for (int cc = 0; cc < 4; cc++) { ps[4 * m + cc] = acc[m][cc]; pq[4 * m + cc] = acc[m][cc] * acc[m][cc]; }
#pragma unroll
      for (int e = 0; e < 16; e++) {
        float a = ps[e], b = pq[e];
        a += __shfl_xor_sync(0xffffffffu, a, 1); b += __shfl_xor_sync(0xffffffffu, b, 1);
        a += __shfl_xor_sync(0xffffffffu, a, 2); b += __shfl_xor_sync(0xffffffffu, b, 2);
        a += __shfl_xor_sync(0xffffffffu, a, 16); b += __shfl_xor_sync(0xffffffffu, b, 16);
        ps[e] = a; pq[e] = b;
      }
      if ((lane & 0x13) == 0) {
#pragma unroll
        for (int m = 0; m < 4; m++)
#pragma unroll
          for (int cc = 0; cc < 4; cc++) {
            const int t = (pp + 2 * m) * 8 + 4 * ch + cc;
            *reinterpret_cast<float2*>(s_lnp + (warp * 64 + t) * 2) = make_float2(ps[4 * m + cc], pq[4 * m + cc]);
          }
      }
    }
    __syncthreads();
    if (s == T - 1 && tid == 0) {  // the x buffer is free after the last conv: land the W2 image behind GEMM1
      asm volatile("fence.proxy.async;" ::: "memory");
      mbar_expect(bar_w2, W2IMG);
#pragma unroll
      for (int i = 0; i < 6; i++) bulk_g2s(sb + OFF_XS + i * 12288, p.w2img + i * 12288, 12288, bar_w2);
    }
    if (s + 1 < T && warp < 12) load_x_warp(p.x, sb, tile + 1, warp, lane);  // next patch lands behind GEMM1 + epilogue
    if (tid < 64) {
      float S1 = 0.f, S2 = 0.f;
#pragma unroll
      for (int w = 0; w < 12; w++) { const float2 v = *reinterpret_cast<const float2*>(s_lnp + (w * 64 + tid) * 2); S1 += v.x; S2 += v.y; }
      const float mu = S1 * (1.f / 96.f);
      const float var = fmaxf(S2 * (1.f / 96.f) - mu * mu, 0.f);
      s_mu[tid] = mu;
      s_rs[tid] = rsqrtf(var + p.ln_eps);
    }
    __syncthreads();
    // ---- LN affine in registers -> a16 = fp16((d - mu) * rs * lnw + lnb) into the MN-major SW128 B tile
    if (warp < 12) {
      const int c = warp * 8 + (lane >> 4) * 4 + (lane & 3);
      const int pp = (lane >> 2) & 1, ch = (lane >> 3) & 1;
      const float lw = s_lnw[c], lb = s_lnb[c];
#pragma unroll
      for (int m = 0; m < 4; m++) {
        const int R = pp + 2 * m, t0 = R * 8 + 4 * ch;
        const float4 mu4 = *reinterpret_cast<const float4*>(s_mu + t0);
        const float4 rs4 = *reinterpret_cast<const float4*>(s_rs + t0);
        const float a0 = fmaf((acc[m][0] - mu4.x) * rs4.x, lw, lb), a1 = fmaf((acc[m][1] - mu4.y) * rs4.y, lw, lb);
        const float a2 = fmaf((acc[m][2] - mu4.z) * rs4.z, lw, lb), a3 = fmaf((acc[m][3] - mu4.w) * rs4.w, lw, lb);
        *reinterpret_cast<uint2*>(sm + OFF_B1 + c * 128 + ((R ^ (c & 7)) << 4) + ch * 8) =
            make_uint2(h2u(__floats2half2_rn(a0, a1)), h2u(__floats2half2_rn(a2, a3)));
      }
    }
    fence_proxy_async();
    tc_before();
    __syncthreads();
    tc_after();
    // ---- GEMM1 (swap-AB): D1[hidden][token] = W1[hidden][c] * a16[c][token], 3 hidden blocks of 128, K = 96
    if (warp == 0) {
      mbar_wait(bar_w1, 0);
      const uint32_t id = idesc_f16(64);
#pragma unroll
      for (int b = 0; b < 3; b++)
#pragma unroll
        for (int j = 0; j < 6; j++) {
          const uint32_t aa = (j < 4) ? (uint32_t)(OFF_A1 + b * 16384 + j * 32)
                                      : (uint32_t)(OFF_A1 + (3 + (b >> 1)) * 16384 + (b & 1) * 64 + (j - 4) * 32);
          mma_issue(tmem + 192 * s + 64 * b, sdesc(sb + aa, 16, 1024), sdesc(sb + OFF_B1 + j * 2048, 12288, 1024), id, j > 0 ? 1u : 0u);
        }
      mma_commit(bar_m1);
    }
    mbar_wait(bar_m1, s & 1);
    tc_after();
    // ---- GEMM1 epilogue: + b1, GELU, per-k sumsq, fp32 h back into TMEM. Warp = (lane quadrant, 3 of 12 16-column units)
#pragma unroll 1
    for (int i = 0; i < 3; i++) {
      const int u = 3 * g4 + i, b = u >> 2, c16 = (u & 3) * 16;
      const int k = 128 * b + 32 * q4 + lane;
      const uint32_t ta = tmem + ((uint32_t)(32 * q4) << 16) + 192 * s + 64 * b + c16;
      float a[16];
      tmem_ld16(ta, a);
      const float bk = s_b1[k];
      uint64_t ss2 = f2pack(0.f, 0.f);
      const uint64_t bk2 = f2pack(bk, bk);
#pragma unroll
      for (int e = 0; e < 16; e += 2) {
        float x0, x1;
        f2unpack(add2(f2pack(a[e], a[e + 1]), bk2), x0, x1);
        gelu2(x0, x1, ss2);
        a[e] = x0; a[e + 1] = x1;
      }
      float s0, s1;
      f2unpack(ss2, s0, s1);
      const float ss = s0 + s1;
      tmem_st16(ta, a);
      s_ssp[g4 * 384 + k] += ss;  // owned by this thread for every unit of block b
    }
    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
    if (T == 2 && s == 0 && img1 != img0) {  // straddling CTA: flush image img0's sums before sub-tile 1
      tc_before();
      if (tid == 0) { while (ld_acquire(ctr) == 0u) { } }
      __syncthreads();
      if (tid < 384) {
        const float v = s_ssp[tid] + s_ssp[384 + tid] + s_ssp[768 + tid] + s_ssp[1152 + tid];
        s_ssp[tid] = 0.f; s_ssp[384 + tid] = 0.f; s_ssp[768 + tid] = 0.f; s_ssp[1152 + tid] = 0.f;
        atomicAdd(p.gsum + img0 * 384 + tid, v);
      }
      __syncthreads();
      tc_after();
    }
  }
  // ---- publish this CTA's GRN sums (gsum was zeroed by CTA 0 in this call before it released the flag)
  if (tid == 0) { while (ld_acquire(ctr) == 0u) { } }
  tc_before();
  __syncthreads();
  if (tid < 384) atomicAdd(p.gsum + img1 * 384 + tid, s_ssp[tid] + s_ssp[384 + tid] + s_ssp[768 + tid] + s_ssp[1152 + tid]);
  __syncthreads();
  if (tid == 0) {
    __threadfence();
    if (T == 1) red_release(ctr + 2 + img0, 1u);
    else if (img1 == img0) red_release(ctr + 2 + img0, 2u);
    else { red_release(ctr + 2 + img0, 1u); red_release(ctr + 2 + img1, 1u); }
    // ---- wait for every sub-tile of our image(s); the last departer resets the counters for the next call
    while (ld_acquire(ctr + 2 + img0) < 49u) { }
    if (img1 != img0) { while (ld_acquire(ctr + 2 + img1) < 49u) { } }
    const unsigned old = atomicAdd(ctr + 1, 1u);
    if (old == gridDim.x - 1) {
      for (int b = 0; b < p.B; b++) atomicExch(ctr + 2 + b, 0u);
      atomicExch(ctr, 0u);
      atomicExch(ctr + 6, 0u);
      atomicExch(ctr + 1, 0u);
    }
  }
  __syncthreads();
  // ---- GRN scale per image: s[k] = 1 + gw[k] * gf[k] / (mean_k gf + eps)
  {
    const int nslot = (img1 != img0) ? 2 : 1;
#pragma unroll 1
    for (int sl = 0; sl < nslot; sl++) {
      const int img = sl ? img1 : img0;
      float g = 0.f;
      if (tid < 384) g = sqrtf(__ldcg(p.gsum + img * 384 + tid));
      const float ws = warp_sum(g);
      if (lane == 0) s_red[warp] = ws;
      __syncthreads();
      float tot = 0.f;
#pragma unroll
      for (int i = 0; i < 12; i++) tot += s_red[i];
      const float inv = 1.f / (tot * (1.f / 384.f) + p.eps);
      if (tid < 384) s_ss[sl * 384 + tid] = fmaf(s_gw[tid] * g, inv, 1.f);
      __syncthreads();
    }
  }
  // ---- prefetch the x residual and b2' of this warp's output rows (they land while epilogue 2 and GEMM2 run)
  constexpr int NR = 2 * T;
  const int fs = (T == 1) ? 0 : (g4 >> 1);
  const int fr0 = (T == 1) ? 2 * g4 : (g4 & 1) * 4;
  size_t foff = 0;
  float4 xres[2 * NR];
  float bb = 0.f;
  if (q4 < 3) {
    const int c = 32 * q4 + lane;
    const int tile = tile0 + fs, img = tile / 49, pi = tile - img * 49;
    foff = ((size_t)(img * 96 + c) * 56 + (pi / 7) * 8 + fr0) * 56 + (pi % 7) * 8;
#pragma unroll
    for (int r = 0; r < NR; r++) {
      xres[2 * r] = __ldg(reinterpret_cast<const float4*>(p.x + foff + r * 56));
      xres[2 * r + 1] = __ldg(reinterpret_cast<const float4*>(p.x + foff + r * 56 + 4));
    }
    bb = s_b2[c] + __ldcg(p.b2pg + c);
  }
  // ---- h' = fp16(h * s) into the MN-major SW128 B tiles (A1 space)
  tc_after();
#pragma unroll 1
  for (int s = 0; s < T; s++) {
    const int slot = ((tile0 + s) / 49 != img0) ? 1 : 0;
#pragma unroll 1
    for (int i = 0; i < 3; i++) {
      const int u = 3 * g4 + i, b = u >> 2, c16 = (u & 3) * 16;
      const int k = 128 * b + 32 * q4 + lane;
      float a[16];
      tmem_ld16(tmem + ((uint32_t)(32 * q4) << 16) + 192 * s + 64 * b + c16, a);
      const float sc = s_ss[slot * 384 + k];
      uint32_t pk[8];
#pragma unroll
      for (int e = 0; e < 8; e++) pk[e] = h2u(__floats2half2_rn(a[2 * e] * sc, a[2 * e + 1] * sc));
      unsigned char* row = sm + OFF_A1 + s * 49152 + k * 128;
      const int c8 = c16 >> 3;
      *reinterpret_cast<uint4*>(row + ((c8 ^ (k & 7)) << 4)) = make_uint4(pk[0], pk[1], pk[2], pk[3]);
      *reinterpret_cast<uint4*>(row + (((c8 + 1) ^ (k & 7)) << 4)) = make_uint4(pk[4], pk[5], pk[6], pk[7]);
    }
  }
  fence_proxy_async();
  tc_before();
  __syncthreads();
  tc_after();
  // ---- GEMM2: D2[out c][token] = W2[c][k] * h'[k][token], K = 384, N = 64T (rows 96..127 of D2 are unused)
  if (warp == 0) {
    mbar_wait(bar_w2, 0);
    const uint32_t id = idesc_f16(64 * T);
#pragma unroll
    for (int j = 0; j < 24; j++)
      mma_issue(tmem + 192 * T, sdesc(sb + OFF_XS + (j >> 2) * 12288 + (j & 3) * 32, 16, 1024),
                sdesc(sb + OFF_A1 + j * 2048, 49152, 1024), id, j > 0 ? 1u : 0u);
    mma_commit(bar_m2);
  }
  // ---- out = x + D2 + b2', NCHW, lanes along channels, one 8-token patch row per float4 pair
  if (q4 < 3) {
    mbar_wait(bar_m2, 0);
    tc_after();
#pragma unroll
    for (int hf = 0; hf < T; hf++) {
      float a[16];
      tmem_ld16(tmem + ((uint32_t)(32 * q4) << 16) + 192 * T + 16 * T * g4 + 16 * hf, a);
#pragma unroll
      for (int rr = 0; rr < 2; rr++) {
        const int r = 2 * hf + rr;
        const float4 x0 = xres[2 * r], x1 = xres[2 * r + 1];
        float4 o0, o1;
        o0.x = x0.x + a[8 * rr] + bb; o0.y = x0.y + a[8 * rr + 1] + bb; o0.z = x0.z + a[8 * rr + 2] + bb; o0.w = x0.w + a[8 * rr + 3] + bb;
        o1.x = x1.x + a[8 * rr + 4] + bb; o1.y = x1.y + a[8 * rr + 5] + bb; o1.z = x1.z + a[8 * rr + 6] + bb; o1.w = x1.w + a[8 * rr + 7] + bb;
        *reinterpret_cast<float4*>(p.out + foff + r * 56) = o0;
        *reinterpret_cast<float4*>(p.out + foff + r * 56 + 4) = o1;
      }
    }
  }
  tc_before();
  __syncthreads();
  if (warp == 0) {
    tc_after();
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" ::"r"(tmem), "r"(NCOL) : "memory");
  }
}

extern "C" const char* m_launch(MParams p, void* stream) {
  if (!(p.B == 1 || p.B == 2 || p.B == 4)) return "m_launch: unsupported batch";
  static bool attr_set = false;
  cudaError_t e;
  if (!attr_set) {  // one maximal value for every shape
    e = cudaFuncSetAttribute(m_kernel<1>, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_TOTAL);
    if (e != cudaSuccess) return cudaGetErrorString(e);
    e = cudaFuncSetAttribute(m_kernel<2>, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_TOTAL);
    if (e != cudaSuccess) return cudaGetErrorString(e);
    attr_set = true;
  }
  const int T = (p.B == 4) ? 2 : 1;
  const int grid = p.B * 49 / T;
  void* args[] = {(void*)&p};
  e = cudaLaunchCooperativeKernel(T == 1 ? (const void*)m_kernel<1> : (const void*)m_kernel<2>, dim3(grid), dim3(NTH), args,
                                  (size_t)SMEM_TOTAL, (cudaStream_t)stream);
  if (e != cudaSuccess) return cudaGetErrorString(e);
  return nullptr;
}
