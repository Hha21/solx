The B200 backend stayed down for the rest of the session: probe and compile both returned "function is stopped" four more times. I stopped calling it.

### Rationale
**Nothing in g_g2 has been compiled or run.** The one result this session is the numerical check, and it changes the design. Scaling the h tile in place by s_b (P1) rounds the hidden twice and fails at 8,320: 1 element past atol, max 101 % of atol. The fp16(fp16(W2)·s) variant reaches 97 %. Only the current roundings, fp16(W2·s_b) times h16 (74-80 %), or a hi/lo B split are safe. A hi/lo split doubles the MMA count.

So g_g2 builds A = fp16(W2·s_b) per image itself, with the same arithmetic as K3, in a short phase before one flat grid barrier. It is still one cooperative launch, so the row runs 3 launches instead of 4. Before the barrier it also prefetches the h B-stages by TMA. After it, a 4-stage TMA ring (A 16 KB + B 32 KB per stage) feeds 1SM tcgen05 at N=256. The epilogue writes out = x + acc + b2' (b2' in fp32) in coalesced 128-B NCHW rows through smem. A desk check found and fixed one PTX error in the barrier atomic. Before shipping, it needs compile_b200, a torch check, timing and run_tests.

```json solution-spec
{"name": "r10-g-e2-g2-coop", "definition": "035_convnextv2_block_with_grn", "author": "solx-loop", "description": "Lineage G r10 E2 (H42): K3 + cuBLASLt GEMM2 replaced at 8,320 / 4,96 by one cooperative tcgen05 launch g_g2 (per-image A = fp16(W2*s_b) phase, flat grid barrier, TMA-fed swap-AB GEMM, coalesced NCHW epilogue). UNTESTED: backend down for the whole session.", "spec": {"languages": ["cuda_cpp"], "target_hardware": ["B200"], "entry_point": "binding.cpp::run", "dependencies": ["torch"], "destination_passing_style": true, "compile_options": {"cuda_cflags": ["-O3", "--use_fast_math", "-std=c++17"], "ld_flags": ["-lcuda", "-lcublasLt"]}}}
```

```cuda file=kernel.cu
// Lineage G r10 (H42): fused pass 1 for the HW >= 1024 rows plus a custom tcgen05 GEMM2 (g_g2) at 8,320 / 4,96.
//  64,96,112,112 and 32,128,64,64: r7 warp-specialised g_pass1 with mma.sync, K3 g_grn + cuBLASLt GEMM2 (unchanged).
//  4,96,56,56 and 8,320,32,32: K0 + g_tc (r9, unchanged) + g_g2, one cooperative launch: phase P builds the per-image
//  A operand fp16(W2*s_b) (s_b from the pass-1 red.add sums, the same roundings as K3) and b2' = W2.gb + b2 in fp32,
//  one flat grid barrier, then a 1SM tcgen05 kind::f16 swap-AB GEMM per (c-tile, image, token chunk): A = W2_b 128 x 64
//  K-major SW128 TMA boxes, B = the pass-1 h [k][token] fp16 tile by 64x64 MN-major SW128 TMA boxes, D in TMEM
//  (lanes = channels, columns = tokens); epilogue out = x + acc + b2' staged through smem for coalesced NCHW rows.
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublasLt.h>
#include <stdint.h>
#include <map>
#include <tuple>

#ifndef TC_STAMPS
#define TC_STAMPS 0
#endif
#ifndef G2_STAMPS
#define G2_STAMPS 0
#endif

struct GParams {
  const float* x; const float* dww; const float* dwb; const float* lnw; const float* lnb;
  const float* w1; const float* b1; const float* gw; const float* gb; const float* w2; const float* b2;
  float* out;
  void* w1f; float* uv; void* h; float* gsum; void* w2b; float* dwp;
  int B, C, H, W, HW, C4, K2, TPI, ntiles;
  float eps, ln_eps;
  unsigned long long* stamp;  // only written in stamp probe builds
  float* b2p;                 // g_g2: b2' = W2.gb + b2 (fp32, written in phase P)
};

__device__ __forceinline__ float g_wsum(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}
__device__ __forceinline__ uint32_t h2u(__half2 h) { return *reinterpret_cast<uint32_t*>(&h); }
// exact-erf GELU z*Phi(z) with Phi from the A&S 7.1.26 erfc (|err| <= 1.5e-7), rcp and ex2 on the MUFU pipe
__device__ __forceinline__ float gelu_as(float z) {
  const float u = fabsf(z) * 0.70710678118654752f;
  const float t = __fdividef(1.f, fmaf(0.3275911f, u, 1.f));
  float y = fmaf(1.061405429f, t, -1.453152027f);
  y = fmaf(y, t, 1.421413741f);
  y = fmaf(y, t, -0.284496736f);
  y = fmaf(y, t, 0.254829592f);
  const float e = exp2f(z * z * -0.72134752044448170f);
  const float hq = 0.5f * (y * t) * e;
  return z * (z >= 0.f ? 1.f - hq : hq);
}
__device__ __forceinline__ void ldsm_x4_t(uint32_t (&r)[4], const void* ptr) {
  const uint32_t a = (uint32_t)__cvta_generic_to_shared(ptr);
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
}
__device__ __forceinline__ void mma16816(float (&d)[4], const uint4& a, uint32_t b0, uint32_t b1) {
  asm("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a.x), "r"(a.y), "r"(a.z), "r"(a.w), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void nbar_sync(int id, int n) { asm volatile("bar.sync %0, %1;" ::"r"(id), "r"(n) : "memory"); }
__device__ __forceinline__ void nbar_arrive(int id, int n) { asm volatile("bar.arrive %0, %1;" ::"r"(id), "r"(n) : "memory"); }
__device__ __forceinline__ bool mbar_try(uint32_t a, uint32_t par) {
  uint32_t ok;
  asm volatile("{.reg .pred P; mbarrier.try_wait.parity.shared::cta.b64 P, [%1], %2; selp.b32 %0, 1, 0, P;}"
               : "=r"(ok) : "r"(a), "r"(par) : "memory");
  return ok != 0;
}
__device__ __forceinline__ void mbar_wait(uint32_t a, uint32_t par) { while (!mbar_try(a, par)) {} }
__device__ __forceinline__ void bulk_g2s(uint32_t dst, const void* src, uint32_t bytes, uint32_t mb) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(mb), "r"(bytes) : "memory");
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
               ::"r"(dst), "l"(src), "r"(bytes), "r"(mb) : "memory");
}
__device__ __forceinline__ void mbar_expect(uint32_t mb, uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(mb), "r"(bytes) : "memory");
}
__device__ __forceinline__ void tma2d(uint32_t dst, const CUtensorMap* tm, int x, int y, uint32_t mb) {
  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];"
               ::"r"(dst), "l"(reinterpret_cast<uint64_t>(tm)), "r"(x), "r"(y), "r"(mb) : "memory");
}
__device__ __forceinline__ uint32_t elect_one() {
  uint32_t p;
  asm volatile("{.reg .pred P; elect.sync _|P, 0xffffffff; selp.b32 %0, 1, 0, P;}" : "=r"(p));
  return p;
}
__device__ __forceinline__ void tc_before() { asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory"); }
__device__ __forceinline__ void tc_after() { asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory"); }
// SW128 shared-memory descriptor (sm_100: version bit 46, swizzle mode 2 = 128 B)
__device__ __forceinline__ uint64_t sdesc(uint32_t addr, uint32_t lbo, uint32_t sbo) {
  return (uint64_t)((addr >> 4) & 0x3FFF) | ((uint64_t)((lbo >> 4) & 0x3FFF) << 16) |
         ((uint64_t)((sbo >> 4) & 0x3FFF) << 32) | (1ull << 46) | (2ull << 61);
}
// byte offset of (channel c, token t) in the MN-major SW128 B tile: 64-token chunks LBO apart, 128 B per channel,
// 16-B unit (t&63)>>3 XORed with c&7 (validated against torch in the r9 engine gate)
__device__ __forceinline__ uint32_t swzB(int c, int t, int LBO) {
  return (uint32_t)((t >> 6) * LBO + c * 128 + ((((t & 63) >> 3) ^ (c & 7)) << 4) + (t & 7) * 2);
}
#define TC_LD32(v, ta)                                                                                               \
  asm volatile("tcgen05.ld.sync.aligned.32x32b.x32.b32 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17," \
               "%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, [%32];\n\ttcgen05.wait::ld.sync.aligned;"      \
               : "=r"(v[0]), "=r"(v[1]), "=r"(v[2]), "=r"(v[3]), "=r"(v[4]), "=r"(v[5]), "=r"(v[6]), "=r"(v[7]),       \
                 "=r"(v[8]), "=r"(v[9]), "=r"(v[10]), "=r"(v[11]), "=r"(v[12]), "=r"(v[13]), "=r"(v[14]),             \
                 "=r"(v[15]), "=r"(v[16]), "=r"(v[17]), "=r"(v[18]), "=r"(v[19]), "=r"(v[20]), "=r"(v[21]),            \
                 "=r"(v[22]), "=r"(v[23]), "=r"(v[24]), "=r"(v[25]), "=r"(v[26]), "=r"(v[27]), "=r"(v[28]),            \
                 "=r"(v[29]), "=r"(v[30]), "=r"(v[31])                                                                \
               : "r"(ta) : "memory")
#if TC_STAMPS || G2_STAMPS
__device__ __forceinline__ unsigned long long g_gtime() {
  unsigned long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)::"memory"); return t;
}
#endif

// K0 (swz = 1: W1' as pre-swizzled K-major SW128 images [hidden block][k-block] of 128 x 64 fp16 for g_tc)
__global__ void __launch_bounds__(256) g_prep(GParams p, int nfb, int nub, int ndb, int swz) {
  const int C = p.C, C4 = p.C4, KS = C / 16;
  int blk = blockIdx.x;
  if (blk < nfb) {
    const int i = blk * 256 + threadIdx.x;
    if (swz) {
      const int KB = (C + 63) / 64;
      if (i >= (C4 / 128) * KB * 1024) return;
      const int st = i >> 10, rem = i & 1023, r = rem >> 3, ch = rem & 7, j = st / KB, kb = st - j * KB;
      const int k = j * 128 + r, c0 = kb * 64 + ch * 8;
      uint4 v = make_uint4(0u, 0u, 0u, 0u);
      if (c0 < C) {
        const float4 wa = __ldg(reinterpret_cast<const float4*>(p.w1 + (size_t)k * C + c0));
        const float4 wb = __ldg(reinterpret_cast<const float4*>(p.w1 + (size_t)k * C + c0 + 4));
        const float4 la = __ldg(reinterpret_cast<const float4*>(p.lnw + c0));
        const float4 lb = __ldg(reinterpret_cast<const float4*>(p.lnw + c0 + 4));
        v.x = h2u(__floats2half2_rn(wa.x * la.x, wa.y * la.y));
        v.y = h2u(__floats2half2_rn(wa.z * la.z, wa.w * la.w));
        v.z = h2u(__floats2half2_rn(wb.x * lb.x, wb.y * lb.y));
        v.w = h2u(__floats2half2_rn(wb.z * lb.z, wb.w * lb.w));
      }
      *reinterpret_cast<uint4*>(reinterpret_cast<unsigned char*>(p.w1f) + (size_t)st * 16384 + r * 128 +
                                ((ch ^ (r & 7)) << 4)) = v;
      return;
    }
    if (i >= (C4 / 16) * KS * 32) return;
    const int lane = i & 31, f = i >> 5, mt = f / KS, ks = f - mt * KS;
    const int g = lane >> 2, t = lane & 3;
    const int r0 = mt * 16 + g, r1 = r0 + 8, c0 = ks * 16 + 2 * t, c2 = c0 + 8;
    auto pk = [&](int r, int c) {
      const float a = __ldg(p.w1 + (size_t)r * C + c) * __ldg(p.lnw + c);
      const float b = __ldg(p.w1 + (size_t)r * C + c + 1) * __ldg(p.lnw + c + 1);
      return h2u(__floats2half2_rn(a, b));
    };
    uint4 v; v.x = pk(r0, c0); v.y = pk(r1, c0); v.z = pk(r0, c2); v.w = pk(r1, c2);
    reinterpret_cast<uint4*>(p.w1f)[i] = v;
    return;
  }
  blk -= nfb;
  if (blk < nub) {
    const int k = blk * 8 + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    if (k >= C4) return;
    const float* wr = p.w1 + (size_t)k * C;
    float u = 0.f, v = 0.f;
    for (int c = lane; c < C; c += 32) {
      const float w = __ldg(wr + c);
      u += __half2float(__float2half_rn(w * __ldg(p.lnw + c)));
      v = fmaf(w, __ldg(p.lnb + c), v);
    }
    u = g_wsum(u); v = g_wsum(v);
    if (lane == 0) { p.uv[2 * k] = u; p.uv[2 * k + 1] = v + __ldg(p.b1 + k); }
    return;
  }
  blk -= nub;
  if (blk < ndb) {
    const int i = blk * 256 + threadIdx.x;
    if (i >= C * 56) return;
    const int c = i / 56, rem = i - c * 56, ki = rem >> 3, j = rem & 7;
    p.dwp[i] = j < 7 ? __ldg(p.dww + c * 49 + ki * 7 + j) : 0.f;
    return;
  }
  blk -= ndb;
  const int i = blk * 256 + threadIdx.x;
  if (i < p.B * C4) p.gsum[i] = 0.f;
}

template <int C, int R, int W, int NCW, int NGW, int MG, bool DWS>
struct P1Cfg {
  static constexpr int C4 = 4 * C, T = R * W, LDT = T + 8, QO = W / 8, NT = (NCW + NGW) * 32, NCT = NCW * 32,
                       NGT = NGW * 32, KS = C / 16, MT = C4 / 16;
  static constexpr int SM_W1F = MT * KS * 32 * 16, SM_D = C * LDT * 2, SM_ST = T * 8, SM_UV = C4 * 8, SM_PS = C4 * 4,
                       SM_DW = DWS ? C * 56 * 4 : 0;
  static constexpr int OFF_D = SM_W1F, OFF_ST = OFF_D + 2 * SM_D, OFF_UV = OFF_ST + 2 * SM_ST, OFF_PS = OFF_UV + SM_UV,
                       OFF_DW = OFF_PS + 2 * SM_PS, SMEM = OFF_DW + SM_DW;
};

// token t of a 32-token chunk sits at position n*8 + 2q + e (t = 8q + 2n + e), so that after ldmatrix.trans + mma each
// lane's accumulators cover 8 consecutive tokens of a hidden row (one 16-B store per row)
__device__ __forceinline__ int tok_pos(int t) { return (t & ~31) | (((t >> 1) & 3) << 3) | (((t >> 3) & 3) << 1) | (t & 1); }

template <int C, int R, int W, int NCW, int NGW, int MG, bool DWS>
__global__ void __launch_bounds__((NCW + NGW) * 32, 1) g_pass1(GParams p) {
  using Cf = P1Cfg<C, R, W, NCW, NGW, MG, DWS>;
  constexpr int C4 = Cf::C4, T = Cf::T, LDT = Cf::LDT, QO = Cf::QO, NT = Cf::NT, NCT = Cf::NCT, NGT = Cf::NGT, KS = Cf::KS;
  constexpr int P = T / 2, S = NCT / P;
  static_assert(NCT % P == 0 && (S & (S - 1)) == 0 && S <= 32, "stats split");
  constexpr int NMG = C4 / (16 * MG), NCH = T / 32;
  static_assert(C4 % (16 * MG) == 0 && T % 32 == 0 && W % 8 == 0 && R % 2 == 0 && C % 16 == 0, "tiles");
  constexpr int PF = 3;  // conv register prefetch distance (input rows)
  extern __shared__ __align__(16) unsigned char sm[];
  uint4* w1s = reinterpret_cast<uint4*>(sm);
  __half* dsb = reinterpret_cast<__half*>(sm + Cf::OFF_D);
  float2* stb = reinterpret_cast<float2*>(sm + Cf::OFF_ST);
  float2* uvs = reinterpret_cast<float2*>(sm + Cf::OFF_UV);
  float* psb = reinterpret_cast<float*>(sm + Cf::OFF_PS);
  float* dws = reinterpret_cast<float*>(sm + Cf::OFF_DW);
  const int tid = threadIdx.x;
  const int H = p.H, HW = p.HW, K2 = p.K2, TPI = p.TPI;
  __half* hg = reinterpret_cast<__half*>(p.h);

  for (int i = tid; i < Cf::SM_W1F / 16; i += NT) w1s[i] = reinterpret_cast<const uint4*>(p.w1f)[i];
  for (int i = tid; i < C4; i += NT) { uvs[i] = reinterpret_cast<const float2*>(p.uv)[i]; psb[i] = 0.f; psb[C4 + i] = 0.f; }
  if (DWS)
    for (int i = tid; i < C * 56; i += NT) dws[i] = p.dwp[i];
  __syncthreads();
  const int warp = __shfl_sync(0xffffffffu, tid >> 5, 0), lane = tid & 31;

  if (warp < NCW) {
    // ================= conv warps
    int i = 0;
#pragma unroll 1
    for (int tile = blockIdx.x; tile < p.ntiles; tile += gridDim.x, i++) {
      const int buf = i & 1;
      const int b = tile / TPI, rb = tile - b * TPI, h0 = rb * R, hw0 = h0 * W;
      if (i >= 2) nbar_sync(3 + buf, NT);  // GEMM warps are done with this buffer
      __half* ds = dsb + buf * (C * LDT);
      constexpr int NIT = C * (R / 2) * QO;
#pragma unroll 1
      for (int it = tid; it < NIT; it += NCT) {
        const int c = it / ((R / 2) * QO), rem = it - c * ((R / 2) * QO), rs = rem / QO, o = rem - rs * QO;
        const int w0 = 8 * o, r0 = 2 * rs;
        const float* xc = p.x + (size_t)(b * C + c) * HW;
        const float* wbase = DWS ? dws + c * 56 : p.dwp + c * 56;
        const float bias = __ldg(p.dwb + c);
        float acc[2][8];
#pragma unroll
        for (int r = 0; r < 2; r++)
#pragma unroll
          for (int cc = 0; cc < 8; cc++) acc[r][cc] = 0.f;
        float4 rows[8][4];
        auto ldrow = [&](int ir) {
          const int hh = h0 + r0 - 3 + ir;
          rows[ir][0] = rows[ir][1] = rows[ir][2] = rows[ir][3] = make_float4(0.f, 0.f, 0.f, 0.f);
          if (hh >= 0 && hh < H) {
            const float* rp = xc + hh * W + w0;
            if (o > 0) rows[ir][0] = __ldg(reinterpret_cast<const float4*>(rp - 4));
            rows[ir][1] = __ldg(reinterpret_cast<const float4*>(rp));
            rows[ir][2] = __ldg(reinterpret_cast<const float4*>(rp + 4));
            if (o < QO - 1) rows[ir][3] = __ldg(reinterpret_cast<const float4*>(rp + 8));
          }
        };
#pragma unroll
        for (int ir = 0; ir < PF; ir++) ldrow(ir);
#pragma unroll
        for (int ir = 0; ir < 8; ir++) {
          if (ir + PF < 8) ldrow(ir + PF);
          const float v[14] = {rows[ir][0].y, rows[ir][0].z, rows[ir][0].w, rows[ir][1].x, rows[ir][1].y,
                               rows[ir][1].z, rows[ir][1].w, rows[ir][2].x, rows[ir][2].y, rows[ir][2].z,
                               rows[ir][2].w, rows[ir][3].x, rows[ir][3].y, rows[ir][3].z};
#pragma unroll
          for (int r = 0; r < 2; r++) {
            const int ki = ir - r;
            if (ki >= 0 && ki < 7) {
              const float4 wa = DWS ? *reinterpret_cast<const float4*>(wbase + ki * 8)
                                    : __ldg(reinterpret_cast<const float4*>(wbase + ki * 8));
              const float4 wb = DWS ? *reinterpret_cast<const float4*>(wbase + ki * 8 + 4)
                                    : __ldg(reinterpret_cast<const float4*>(wbase + ki * 8 + 4));
              const float w[7] = {wa.x, wa.y, wa.z, wa.w, wb.x, wb.y, wb.z};
#pragma unroll
              for (int j = 0; j < 7; j++)
#pragma unroll
                for (int cc = 0; cc < 8; cc++) acc[r][cc] = fmaf(v[cc + j], w[j], acc[r][cc]);
            }
          }
        }
#pragma unroll
        for (int r = 0; r < 2; r++) {
          const int t0 = (r0 + r) * W + w0;  // multiple of 8: q fixed, n = 0..3
          __half* dp = ds + c * LDT + (t0 & ~31) + (((t0 >> 3) & 3) << 1);
#pragma unroll
          for (int n = 0; n < 4; n++)
            *reinterpret_cast<uint32_t*>(dp + 8 * n) = h2u(__floats2half2_rn(acc[r][2 * n] + bias, acc[r][2 * n + 1] + bias));
        }
      }
      nbar_sync(5, NCT);
      {  // per-token LayerNorm statistics of the fp16 d tile (2 tokens per thread, S channel slices)
        const int pp = tid / S, s = tid - pp * S;
        const int pos = tok_pos(2 * pp);
        float s1a = 0.f, s2a = 0.f, s1b = 0.f, s2b = 0.f;
#pragma unroll 8
        for (int c = s; c < C; c += S) {
          const float2 f = __half22float2(*reinterpret_cast<const __half2*>(ds + c * LDT + pos));
          s1a += f.x; s2a = fmaf(f.x, f.x, s2a); s1b += f.y; s2b = fmaf(f.y, f.y, s2b);
        }
#pragma unroll
        for (int o = 1; o < S; o <<= 1) {
          s1a += __shfl_xor_sync(0xffffffffu, s1a, o); s2a += __shfl_xor_sync(0xffffffffu, s2a, o);
          s1b += __shfl_xor_sync(0xffffffffu, s1b, o); s2b += __shfl_xor_sync(0xffffffffu, s2b, o);
        }
        if (s == 0) {
          constexpr float inv = 1.f / (float)C;
          const float m0 = s1a * inv, m1 = s1b * inv;
          const float q0 = rsqrtf(fmaxf(s2a * inv - m0 * m0, 0.f) + p.ln_eps);
          const float q1 = rsqrtf(fmaxf(s2b * inv - m1 * m1, 0.f) + p.ln_eps);
          *reinterpret_cast<float4*>(stb + buf * T + 2 * pp) = make_float4(q0, q0 * m0, q1, q1 * m1);
          __half* ha = hg + ((size_t)b * K2 + C4) * HW + hw0 + 2 * pp;
          const uint32_t one = h2u(__floats2half2_rn(1.f, 1.f));
#pragma unroll
          for (int j = 0; j < 8; j++) *reinterpret_cast<uint32_t*>(ha + (size_t)j * HW) = j < 2 ? one : 0u;
        }
      }
      nbar_arrive(1 + buf, NT);  // d tile + stats ready
    }
    for (int j = (i >= 2 ? i - 2 : 0); j < i; j++) nbar_sync(3 + (j & 1), NT);
  } else {
    // ================= GEMM + epilogue warps
    const int gw = warp - NCW, gt = tid - NCT, g = lane >> 2, t4 = lane & 3;
    int i = 0;
#pragma unroll 1
    for (int tile = blockIdx.x; tile < p.ntiles; tile += gridDim.x, i++) {
      const int buf = i & 1;
      const int b = tile / TPI, rb = tile - b * TPI, hw0 = rb * R * W;
      nbar_sync(1 + buf, NT);
      const __half* ds = dsb + buf * (C * LDT);
      const float2* sts = stb + buf * T;
      float* pss = psb + buf * C4;
#pragma unroll 1
      for (int wt = gw; wt < NMG * NCH; wt += NGW) {
        const int mg = wt / NCH, nc = wt - mg * NCH;
        float acc[MG][4][4];
#pragma unroll
        for (int m = 0; m < MG; m++)
#pragma unroll
          for (int n = 0; n < 4; n++)
#pragma unroll
            for (int e = 0; e < 4; e++) acc[m][n][e] = 0.f;
#pragma unroll
        for (int ks = 0; ks < KS; ks++) {
          uint32_t bf[2][4];
#pragma unroll
          for (int pr = 0; pr < 2; pr++) {
            const int mi = lane >> 3, rr = lane & 7;
            ldsm_x4_t(bf[pr], ds + (ks * 16 + (mi & 1) * 8 + rr) * LDT + nc * 32 + pr * 16 + (mi >> 1) * 8);
          }
#pragma unroll
          for (int m = 0; m < MG; m++) {
            const uint4 a = w1s[((mg * MG + m) * KS + ks) * 32 + lane];
#pragma unroll
            for (int n = 0; n < 4; n++) mma16816(acc[m][n], a, bf[n >> 1][(n & 1) * 2], bf[n >> 1][(n & 1) * 2 + 1]);
          }
        }
        // lane holds tokens nc*32 + 8*t4 + 2n + e for hidden rows k0 (e = 0,1) and k1 = k0 + 8 (e = 2,3)
        const int tb = nc * 32 + 8 * t4;
        float4 st[4];
#pragma unroll
        for (int n = 0; n < 4; n++) st[n] = *reinterpret_cast<const float4*>(sts + tb + 2 * n);
#pragma unroll
        for (int m = 0; m < MG; m++) {
          const int k0 = (mg * MG + m) * 16 + g, k1 = k0 + 8;
          const float2 uv0 = uvs[k0], uv1 = uvs[k1];
          float ss0 = 0.f, ss1 = 0.f;
          uint32_t o0[4], o1[4];
#pragma unroll
          for (int n = 0; n < 4; n++) {
            const float g00 = gelu_as(fmaf(st[n].x, acc[m][n][0], fmaf(-st[n].y, uv0.x, uv0.y)));
            const float g01 = gelu_as(fmaf(st[n].z, acc[m][n][1], fmaf(-st[n].w, uv0.x, uv0.y)));
            const float g10 = gelu_as(fmaf(st[n].x, acc[m][n][2], fmaf(-st[n].y, uv1.x, uv1.y)));
            const float g11 = gelu_as(fmaf(st[n].z, acc[m][n][3], fmaf(-st[n].w, uv1.x, uv1.y)));
            ss0 = fmaf(g00, g00, fmaf(g01, g01, ss0));
            ss1 = fmaf(g10, g10, fmaf(g11, g11, ss1));
            o0[n] = h2u(__floats2half2_rn(g00, g01));
            o1[n] = h2u(__floats2half2_rn(g10, g11));
          }
          __half* hp0 = hg + ((size_t)b * K2 + k0) * HW + hw0 + tb;
          *reinterpret_cast<uint4*>(hp0) = make_uint4(o0[0], o0[1], o0[2], o0[3]);
          *reinterpret_cast<uint4*>(hp0 + (size_t)8 * HW) = make_uint4(o1[0], o1[1], o1[2], o1[3]);
          ss0 += __shfl_xor_sync(0xffffffffu, ss0, 1); ss0 += __shfl_xor_sync(0xffffffffu, ss0, 2);
          ss1 += __shfl_xor_sync(0xffffffffu, ss1, 1); ss1 += __shfl_xor_sync(0xffffffffu, ss1, 2);
          if (t4 == 0) { atomicAdd(pss + k0, ss0); atomicAdd(pss + k1, ss1); }
        }
      }
      nbar_arrive(3 + buf, NT);  // done reading d tile + stats of this buffer
      nbar_sync(6, NGT);
      float* gs = p.gsum + (size_t)b * C4;
      for (int k = gt; k < C4; k += NGT) { atomicAdd(gs + k, pss[k]); pss[k] = 0.f; }
    }
  }
}

// ---------------------------------------------------------------- r9: single-tile fused pass 1 on tcgen05 (g_tc)
// warps 2-15 run the conv in channel rounds (one item per thread per round) and signal each round on an mbarrier;
// warp 0 issues the MMAs k-block-outer over the resident accumulators as the rounds land (the MMA overlaps the conv),
// then the remaining hidden blocks into TMEM slots freed by the epilogue; warp 1 feeds the W1' ring; warps 2-15 then
// compute the LN statistics; all 16 warps drain TMEM items (32 hidden rows x 32 tokens of their lane quadrant).
template <int C, int R, int W, int TP, int NS>
struct TCfg {
  static constexpr int NT = 512, C4 = 4 * C, T = R * W, KB = (C + 63) / 64, NK16 = C / 16, NJ = C4 / 128,
                       NSTG = NJ * KB, LBO = C * 128, NQ = TP / 32, NSLOT = (NJ < 512 / TP) ? NJ : 512 / TP,
                       NA = NJ < NSLOT ? NJ : NSLOT, NITEM = NJ * NQ, NPRE = NS < NSTG ? NS : NSTG;
  static constexpr int IPC = (R / 2) * (W / 8), CPR0 = 448 / IPC, CPR = CPR0 < C ? CPR0 : C, NR = (C + CPR - 1) / CPR;
  static constexpr int OFF_D = NS * 16384, OFF_ST = OFF_D + (TP / 64) * LBO, OFF_BAR = OFF_ST + TP * 8,
                       NBAR = 2 * NS + 2 * NSLOT + NR, OFF_STG = (OFF_BAR + NBAR * 8 + 32 + 127) / 128 * 128,
                       SMEM = OFF_STG + 16 * 2048 + 1024;
  static constexpr int P = T / 2, Q = 448 / P;
  static constexpr int S = Q >= 32 ? 32 : Q >= 16 ? 16 : Q >= 8 ? 8 : Q >= 4 ? 4 : Q >= 2 ? 2 : 1;
  static constexpr int NST = P * S, PADU = (TP - T) / 8 > 0 ? (TP - T) / 8 : 1;
  static constexpr uint32_t IDESC = (1u << 4) | (1u << 16) | ((uint32_t)(TP >> 3) << 17) | ((128u >> 4) << 24);
  // ring sequence -> W1' image (hidden block j, k-block kb): k-block-outer over the NA resident blocks, then block-major
  __host__ __device__ static constexpr int img_of(int i) {
    return i < NA * KB ? (i % NA) * KB + i / NA : (NA + (i - NA * KB) / KB) * KB + (i - NA * KB) % KB;
  }
};

template <int C, int R, int W, int TP, int NS>
__global__ void __launch_bounds__(512, 1) g_tc(GParams p) {
  using Cf = TCfg<C, R, W, TP, NS>;
  constexpr int NT = Cf::NT, C4 = Cf::C4, T = Cf::T, KB = Cf::KB, NK16 = Cf::NK16, NJ = Cf::NJ, NSTG = Cf::NSTG,
                LBO = Cf::LBO, NQ = Cf::NQ, NSLOT = Cf::NSLOT, NA = Cf::NA, NITEM = Cf::NITEM, S = Cf::S,
                NST = Cf::NST, PADU = Cf::PADU, IPC = Cf::IPC, CPR = Cf::CPR, NR = Cf::NR;
  constexpr int QO = W / 8, PF = 3;
  static_assert(T <= TP && TP % 64 == 0 && TP <= 256 && T % 8 == 0 && W % 8 == 0 && R % 2 == 0 && C % 16 == 0, "tile");
  static_assert(C4 % 128 == 0 && NST % 32 == 0 && NST <= 448 && S <= 32 && CPR * IPC <= 448, "split");
  // TMEM slot reuse: an item of use u waits accf parity u&1, so a quadrant must never run two uses ahead
  static_assert(NJ <= NSLOT || NSLOT * NQ >= 8, "slot parity");
  static_assert(Cf::SMEM <= 232448, "smem");
  extern __shared__ __align__(1024) unsigned char smraw[];
  const uint32_t sraw = (uint32_t)__cvta_generic_to_shared(smraw), sA = (sraw + 1023u) & ~1023u;
  unsigned char* sm = smraw + (sA - sraw);
  const uint32_t sD = sA + Cf::OFF_D, bar = sA + Cf::OFF_BAR;
  const uint32_t b_full = bar, b_empty = bar + 8 * NS, b_accf = bar + 16 * NS, b_acce = b_accf + 8 * NSLOT,
                 b_drdy = b_acce + 8 * NSLOT;
  unsigned char* dsm = sm + Cf::OFF_D;
  float2* stb = reinterpret_cast<float2*>(sm + Cf::OFF_ST);
  volatile uint32_t* tslot = reinterpret_cast<volatile uint32_t*>(sm + Cf::OFF_BAR + Cf::NBAR * 8);
  int* ctr = reinterpret_cast<int*>(sm + Cf::OFF_BAR + Cf::NBAR * 8 + 16);
  const int tid = threadIdx.x, lane = tid & 31;
  const int warp = __shfl_sync(0xffffffffu, tid >> 5, 0);
  const int H = p.H, HW = p.HW, K2 = p.K2, TPI = p.TPI;
  const int tile = blockIdx.x, b = tile / TPI, h0 = (tile - b * TPI) * R, hw0 = h0 * W;
  const unsigned char* img = reinterpret_cast<const unsigned char*>(p.w1f);
  __half* hg = reinterpret_cast<__half*>(p.h);
#if TC_STAMPS
  unsigned long long* stp = p.stamp ? p.stamp + blockIdx.x * 16 : nullptr;
  if (stp && tid == 0) stp[0] = g_gtime();
#endif
  if (tid == 0) {
    for (int s = 0; s < NS; s++) {
      asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::"r"(b_full + 8 * s) : "memory");
      asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::"r"(b_empty + 8 * s) : "memory");
    }
    for (int s = 0; s < NSLOT; s++) {
      asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::"r"(b_accf + 8 * s) : "memory");
      asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(b_acce + 8 * s), "r"(4 * NQ) : "memory");
    }
    for (int r = 0; r < NR; r++) asm volatile("mbarrier.init.shared::cta.b64 [%0], 14;" ::"r"(b_drdy + 8 * r) : "memory");
    for (int q = 0; q < 4; q++) ctr[q] = 0;
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  if (T < TP)  // padding tokens of the B tile are zero (their accumulator columns are never stored)
    for (int i = tid; i < C * PADU; i += NT) {
      const int c = i / PADU, t = T + 8 * (i - c * PADU);
      *reinterpret_cast<uint4*>(dsm + swzB(c, t, LBO)) = make_uint4(0u, 0u, 0u, 0u);
    }
  for (int i = tid; i < TP; i += NT) stb[i] = make_float2(0.f, 0.f);
  if (warp == 0) {
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], 512;" ::"r"(bar + Cf::NBAR * 8) : "memory");
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;" ::: "memory");
  }
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  tc_before();
  __syncthreads();
  tc_after();
  const uint32_t tbase = *tslot;

  if (warp == 0) {
    // ---- MMA issuer. Phase 1: k-block-outer over the NA resident accumulators, gated by the conv rounds.
    //      Phase 2: hidden blocks NA.. into TMEM slots freed by the epilogue.
    int i = 0;
    auto stage = [&](uint32_t dt, int kb, uint32_t accf) {
      const int s = i % NS;
      mbar_wait(b_full + 8 * s, (i / NS) & 1);
      tc_after();
      if (elect_one()) {
        const int nk = (NK16 - 4 * kb) < 4 ? (NK16 - 4 * kb) : 4;
        for (int kk = 0; kk < nk; kk++) {
          const uint64_t ad = sdesc(sA + s * 16384 + kk * 32, 16, 1024);
          const uint64_t bd = sdesc(sD + (4 * kb + kk) * 2048, LBO, 1024);
          const uint32_t accum = (kb | kk) ? 1u : 0u;
          asm volatile("{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %4, 0;\n\t"
                       "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n\t}"
                       ::"r"(dt), "l"(ad), "l"(bd), "r"(Cf::IDESC), "r"(accum) : "memory");
        }
        asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];"
                     ::"r"(b_empty + 8 * s) : "memory");
        if (accf)
          asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];" ::"r"(accf) : "memory");
      }
      __syncwarp();
      i++;
    };
#pragma unroll 1
    for (int kb = 0; kb < KB; kb++) {
      const int lastc = (64 * kb + 63 < C ? 64 * kb + 63 : C - 1);
      mbar_wait(b_drdy + 8 * (lastc / CPR), 0);  // conv round holding this k-block's last channel has landed
      tc_after();
#pragma unroll 1
      for (int j = 0; j < NA; j++) stage(tbase + j * TP, kb, kb == KB - 1 ? b_accf + 8 * j : 0u);
    }
#pragma unroll 1
    for (int j = NA; j < NJ; j++) {
      const int slot = j % NSLOT;
      mbar_wait(b_acce + 8 * slot, ((j / NSLOT) - 1) & 1);
      tc_after();
#pragma unroll 1
      for (int kb = 0; kb < KB; kb++) stage(tbase + slot * TP, kb, kb == KB - 1 ? b_accf + 8 * slot : 0u);
    }
#if TC_STAMPS
    if (stp && lane == 0) stp[3] = g_gtime();
#endif
  } else if (warp == 1) {
    // ---- W1' ring producer: prefill lands while the conv runs; refill once the MMAs that read a stage completed
    if (elect_one())
      for (int i = 0; i < Cf::NPRE; i++) bulk_g2s(sA + i * 16384, img + (size_t)Cf::img_of(i) * 16384, 16384, b_full + 8 * i);
    __syncwarp();
#pragma unroll 1
    for (int i = Cf::NPRE; i < NSTG; i++) {
      const int s = i % NS;
      mbar_wait(b_empty + 8 * s, ((i / NS) - 1) & 1);
      if (elect_one()) bulk_g2s(sA + s * 16384, img + (size_t)Cf::img_of(i) * 16384, 16384, b_full + 8 * s);
      __syncwarp();
    }
  } else {
    // ---- 7x7 depthwise conv, warps 2-15: round r covers channels [r*CPR, (r+1)*CPR), one item (8 columns x 2 rows of
    //      one channel) per thread, written as one 16-B unit per row of the MN-major SW128 B tile
    const int ctid = tid - 64;
#pragma unroll 1
    for (int r = 0; r < NR; r++) {
      const int it = r * CPR * IPC + ctid;
      if (ctid < CPR * IPC && it < C * IPC) {
        const int c = it / IPC, rem = it - c * IPC, rs = rem / QO, o = rem - rs * QO;
        const int w0 = 8 * o, r0 = 2 * rs;
        const float* xc = p.x + (size_t)(b * C + c) * HW;
        const float* wbase = p.dwp + c * 56;
        const float bias = __ldg(p.dwb + c);
        float acc[2][8];
#pragma unroll
        for (int rr = 0; rr < 2; rr++)
#pragma unroll
          for (int cc = 0; cc < 8; cc++) acc[rr][cc] = 0.f;
        float4 rows[8][4];
        auto ldrow = [&](int ir) {
          const int hh = h0 + r0 - 3 + ir;
          rows[ir][0] = rows[ir][1] = rows[ir][2] = rows[ir][3] = make_float4(0.f, 0.f, 0.f, 0.f);
          if (hh >= 0 && hh < H) {
            const float* rp = xc + hh * W + w0;
            if (o > 0) rows[ir][0] = __ldg(reinterpret_cast<const float4*>(rp - 4));
            rows[ir][1] = __ldg(reinterpret_cast<const float4*>(rp));
            rows[ir][2] = __ldg(reinterpret_cast<const float4*>(rp + 4));
            if (o < QO - 1) rows[ir][3] = __ldg(reinterpret_cast<const float4*>(rp + 8));
          }
        };
#pragma unroll
        for (int ir = 0; ir < PF; ir++) ldrow(ir);
#pragma unroll
        for (int ir = 0; ir < 8; ir++) {
          if (ir + PF < 8) ldrow(ir + PF);
          const float v[14] = {rows[ir][0].y, rows[ir][0].z, rows[ir][0].w, rows[ir][1].x, rows[ir][1].y,
                               rows[ir][1].z, rows[ir][1].w, rows[ir][2].x, rows[ir][2].y, rows[ir][2].z,
                               rows[ir][2].w, rows[ir][3].x, rows[ir][3].y, rows[ir][3].z};
#pragma unroll
          for (int rr = 0; rr < 2; rr++) {
            const int ki = ir - rr;
            if (ki >= 0 && ki < 7) {
              const float4 wa = __ldg(reinterpret_cast<const float4*>(wbase + ki * 8));
              const float4 wb = __ldg(reinterpret_cast<const float4*>(wbase + ki * 8 + 4));
              const float w[7] = {wa.x, wa.y, wa.z, wa.w, wb.x, wb.y, wb.z};
#pragma unroll
              for (int j = 0; j < 7; j++)
#pragma unroll
                for (int cc = 0; cc < 8; cc++) acc[rr][cc] = fmaf(v[cc + j], w[j], acc[rr][cc]);
            }
          }
        }
#pragma unroll
        for (int rr = 0; rr < 2; rr++) {
          const int t0 = (r0 + rr) * W + w0;
          uint4 u;
          u.x = h2u(__floats2half2_rn(acc[rr][0] + bias, acc[rr][1] + bias));
          u.y = h2u(__floats2half2_rn(acc[rr][2] + bias, acc[rr][3] + bias));
          u.z = h2u(__floats2half2_rn(acc[rr][4] + bias, acc[rr][5] + bias));
          u.w = h2u(__floats2half2_rn(acc[rr][6] + bias, acc[rr][7] + bias));
          *reinterpret_cast<uint4*>(dsm + swzB(c, t0, LBO)) = u;
        }
      }
      asm volatile("fence.proxy.async.shared::cta;" ::: "memory");  // this round's d units visible to the tensor core
      __syncwarp();
      if (lane == 0) asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(b_drdy + 8 * r) : "memory");
    }
#if TC_STAMPS
    if (stp && tid == 64) stp[1] = g_gtime();
#endif
    nbar_sync(1, 448);
    // ---- per-token LayerNorm statistics of the fp16 d tile (2 tokens per thread, S channel slices)
    if (ctid < NST) {
      const int pp = ctid / S, s = ctid - pp * S, t = 2 * pp;
      float s1a = 0.f, s2a = 0.f, s1b = 0.f, s2b = 0.f;
#pragma unroll 4
      for (int c = s; c < C; c += S) {
        const float2 f = __half22float2(*reinterpret_cast<const __half2*>(dsm + swzB(c, t, LBO)));
        s1a += f.x; s2a = fmaf(f.x, f.x, s2a); s1b += f.y; s2b = fmaf(f.y, f.y, s2b);
      }
#pragma unroll
      for (int o = 1; o < S; o <<= 1) {
        s1a += __shfl_xor_sync(0xffffffffu, s1a, o); s2a += __shfl_xor_sync(0xffffffffu, s2a, o);
        s1b += __shfl_xor_sync(0xffffffffu, s1b, o); s2b += __shfl_xor_sync(0xffffffffu, s2b, o);
      }
      if (s == 0) {
        constexpr float inv = 1.f / (float)C;
        const float m0 = s1a * inv, m1 = s1b * inv;
        const float q0 = rsqrtf(fmaxf(s2a * inv - m0 * m0, 0.f) + p.ln_eps);
        const float q1 = rsqrtf(fmaxf(s2b * inv - m1 * m1, 0.f) + p.ln_eps);
        *reinterpret_cast<float4*>(stb + t) = make_float4(q0, q0 * m0, q1, q1 * m1);
        __half* ha = hg + ((size_t)b * K2 + C4) * HW + hw0 + t;
        const uint32_t one = h2u(__floats2half2_rn(1.f, 1.f));
#pragma unroll
        for (int j = 0; j < 8; j++) *reinterpret_cast<uint32_t*>(ha + (size_t)j * HW) = j < 2 ? one : 0u;
      }
    }
    nbar_sync(1, 448);
    nbar_arrive(2, 512);
#if TC_STAMPS
    if (stp && tid == 64) stp[4] = g_gtime();
#endif
  }
  if (warp < 2) nbar_sync(2, 512);  // stats ready (the stats warps only arrive, so the MMA warp never blocks them)

  // ---- epilogue: items of 32 hidden rows (this warp's TMEM lane quadrant) x 32 tokens, claimed per quadrant in order;
  //      the fp16 h item goes through a per-warp smem transpose so each store instruction writes 8 rows x 64 B
  {
    const int qd = warp & 3;
    unsigned char* stg = sm + Cf::OFF_STG + warp * 2048;
    const float2* uvg = reinterpret_cast<const float2*>(p.uv);
    __half* hb = hg + (size_t)b * K2 * HW + hw0;
    float* gs = p.gsum + (size_t)b * C4;
#pragma unroll 1
    for (;;) {
      int n = 0;
      if (lane == 0) n = atomicAdd(ctr + qd, 1);
      n = __shfl_sync(0xffffffffu, n, 0);
      if (n >= NITEM) break;
      const int j = n / NQ, q = n - j * NQ, slot = j % NSLOT;
      mbar_wait(b_accf + 8 * slot, (j / NSLOT) & 1);
      tc_after();
      uint32_t v[32];
      TC_LD32(v, tbase + ((uint32_t)(qd * 32) << 16) + (uint32_t)(slot * TP + q * 32));
      tc_before();
      __syncwarp();
      if (lane == 0) asm volatile("mbarrier.arrive.relaxed.cta.shared::cta.b64 _, [%0];" ::"r"(b_acce + 8 * slot) : "memory");
      const int k = j * 128 + qd * 32 + lane;
      const float2 uv = __ldg(uvg + k);
      const float cz = uv.y;
      float ss = 0.f;
#pragma unroll
      for (int g = 0; g < 4; g++) {
        const int t = q * 32 + 8 * g;
        uint32_t o[4];
        float sg = 0.f;
#pragma unroll
        for (int e = 0; e < 4; e++) {
          const float4 st = *reinterpret_cast<const float4*>(stb + t + 2 * e);
          const float g0 = gelu_as(fmaf(st.x, __uint_as_float(v[8 * g + 2 * e]), fmaf(-st.y, uv.x, cz)));
          const float g1 = gelu_as(fmaf(st.z, __uint_as_float(v[8 * g + 2 * e + 1]), fmaf(-st.w, uv.x, cz)));
          sg = fmaf(g0, g0, fmaf(g1, g1, sg));
          o[e] = h2u(__floats2half2_rn(g0, g1));
        }
        if (T % 32 == 0 || t < T) ss += sg;
        *reinterpret_cast<uint4*>(stg + lane * 64 + ((g ^ ((lane >> 1) & 3)) << 4)) = make_uint4(o[0], o[1], o[2], o[3]);
      }
      __syncwarp();
#pragma unroll
      for (int r = 0; r < 4; r++) {  // 8 rows x 64 B per store instruction
        const int row = r * 8 + (lane >> 2), c = lane & 3, t = q * 32 + 8 * c;
        if (T % 32 == 0 || t < T)
          *reinterpret_cast<uint4*>(hb + (size_t)(j * 128 + qd * 32 + row) * HW + t) =
              *reinterpret_cast<const uint4*>(stg + row * 64 + ((c ^ ((row >> 1) & 3)) << 4));
      }
      __syncwarp();
      atomicAdd(gs + k, ss);
    }
  }
#if TC_STAMPS
  if (stp && lane == 0 && (warp == 0 || warp == 4)) stp[5 + (warp >> 2)] = g_gtime();
#endif
  tc_before();
  __syncthreads();
  if (warp == 0) {
    tc_after();
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, 512;" ::"r"(tbase) : "memory");
  }
#if TC_STAMPS
  if (stp && tid == 0) stp[7] = g_gtime();
#endif
}

// K3: grid (B, C/16): per image s_b[k] = 1 + gw*gf/(mean gf + eps); one W2_b row per warp, the row loaded up front
__global__ void __launch_bounds__(512, 2) g_grn(GParams p) {
  extern __shared__ __align__(16) float gsm[];
  constexpr int MAXI = 12;  // C4 <= 1536
  const int C = p.C, C4 = p.C4, K2 = p.K2;
  float* sc = gsm; float* red = gsm + C4;
  const int b = blockIdx.x, tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int c = blockIdx.y * 16 + warp;
  const float* wr = p.w2 + (size_t)(c < C ? c : 0) * C4;
  float4 wv[MAXI];
#pragma unroll
  for (int i = 0; i < MAXI; i++) {
    const int k = lane * 4 + i * 128;
    wv[i] = k < C4 ? __ldg(reinterpret_cast<const float4*>(wr + k)) : make_float4(0.f, 0.f, 0.f, 0.f);
  }
  float loc = 0.f;
  for (int k = tid; k < C4; k += 512) {
    const float gf = sqrtf(__ldcg(p.gsum + (size_t)b * C4 + k));
    sc[k] = gf; loc += gf;
  }
  loc = g_wsum(loc);
  if (lane == 0) red[warp] = loc;
  __syncthreads();
  if (warp == 0) {
    float v = lane < 16 ? red[lane] : 0.f;
    v = g_wsum(v);
    if (lane == 0) red[32] = 1.f / (v / (float)C4 + p.eps);
  }
  __syncthreads();
  const float inv = red[32];
  for (int k = tid; k < C4; k += 512) sc[k] = fmaf(__ldg(p.gw + k) * sc[k], inv, 1.f);
  __syncthreads();
  if (c >= C) return;
  __half* w2b = reinterpret_cast<__half*>(p.w2b) + ((size_t)b * C + c) * K2;
  float dot = 0.f;
#pragma unroll
  for (int i = 0; i < MAXI; i++) {
    const int k = lane * 4 + i * 128;
    if (k < C4) {
      const float4 w = wv[i];
      const float4 gv = __ldg(reinterpret_cast<const float4*>(p.gb + k));
      const float4 s = *reinterpret_cast<const float4*>(sc + k);
      dot = fmaf(w.x, gv.x, dot); dot = fmaf(w.y, gv.y, dot); dot = fmaf(w.z, gv.z, dot); dot = fmaf(w.w, gv.w, dot);
      uint2 u;
      u.x = h2u(__floats2half2_rn(w.x * s.x, w.y * s.y));
      u.y = h2u(__floats2half2_rn(w.z * s.z, w.w * s.w));
      *reinterpret_cast<uint2*>(w2b + k) = u;
    }
  }
  dot = g_wsum(dot);
  if (lane == 0) {
    dot += __ldg(p.b2 + c);
    const __half hh = __float2half_rn(dot);
    const __half hl = __float2half_rn(dot - __half2float(hh));
    *reinterpret_cast<uint4*>(w2b + C4) = make_uint4(h2u(__halves2half2(hh, hl)), 0u, 0u, 0u);
  }
}

// ---------------------------------------------------------------- r10: g_g2, GRN scale + GEMM2 in one cooperative launch
// flat grid barrier: a module-level arrival counter reset by the last arriver, which then bumps the generation word
__device__ unsigned int g2_bar_cnt = 0;
__device__ unsigned int g2_bar_gen = 0;
__device__ __forceinline__ void g2_grid_bar(unsigned int nb) {
  __syncthreads();
  if (threadIdx.x == 0) {
    unsigned int gen, old;
    asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(gen) : "l"(&g2_bar_gen) : "memory");
    asm volatile("atom.acq_rel.gpu.global.add.u32 %0, [%1], 1;" : "=r"(old) : "l"(&g2_bar_cnt) : "memory");
    if (old == nb - 1) {
      asm volatile("st.relaxed.gpu.global.u32 [%0], 0;" ::"l"(&g2_bar_cnt) : "memory");
      asm volatile("st.release.gpu.global.u32 [%0], %1;" ::"l"(&g2_bar_gen), "r"(gen + 1u) : "memory");
    } else {
      unsigned int g;
      do { asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(g) : "l"(&g2_bar_gen) : "memory"); } while (g == gen);
    }
  }
  __syncthreads();
}

template <int C, int N, int NS>
struct G2Cfg {
  static constexpr int C4 = 4 * C, KB = C4 / 64, NB = (N + 63) / 64, NCT = (C + 127) / 128, NCH = (N + 31) / 32,
                       IPW = (NCH + 3) / 4, MAXI = C4 / 128, NPRE = NS < KB ? NS : KB;
  static constexpr int SA = 16384, SB = NB * 8192, STG = SA + SB;
  static constexpr int OFF_S = NS * STG, OFF_BAR = OFF_S + C4 * 4 + 128, NBAR = 3 * NS + 1;
  static constexpr int SMEM = OFF_BAR + NBAR * 8 + 16 + 1024;
  static constexpr int TCOLS = N <= 32 ? 32 : N <= 64 ? 64 : N <= 128 ? 128 : 256;
  static constexpr uint32_t IDESC = (1u << 4) | (1u << 16) | ((uint32_t)(N >> 3) << 17) | ((128u >> 4) << 24);
};

// unit u = (c-tile ct, image ub, token chunk uq): out[ub][ct*128 + 0..127][uq*N + 0..N-1]
template <int C, int N, int NS>
__global__ void __launch_bounds__(512, 1) g_g2(GParams p, const __grid_constant__ CUtensorMap tmA,
                                                const __grid_constant__ CUtensorMap tmB) {
  using Cf = G2Cfg<C, N, NS>;
  constexpr int C4 = Cf::C4, KB = Cf::KB, NB = Cf::NB, NCT = Cf::NCT, NCH = Cf::NCH, IPW = Cf::IPW, MAXI = Cf::MAXI,
                NPRE = Cf::NPRE, SA = Cf::SA, SB = Cf::SB, STG = Cf::STG;
  static_assert(N % 16 == 0 && N <= 256 && C4 % 128 == 0 && C % 4 == 0, "shape");
  static_assert(NS * STG >= 16 * 4096 && Cf::SMEM <= 232448, "smem");
  extern __shared__ __align__(1024) unsigned char smraw[];
  const uint32_t sraw = (uint32_t)__cvta_generic_to_shared(smraw), sbase = (sraw + 1023u) & ~1023u;
  unsigned char* sm = smraw + (sbase - sraw);
  float* ssm = reinterpret_cast<float*>(sm + Cf::OFF_S);
  float* red = ssm + C4;
  const uint32_t bar = sbase + Cf::OFF_BAR;
  const uint32_t b_fa = bar, b_fb = bar + 8 * NS, b_emp = bar + 16 * NS, b_acc = bar + 24 * NS;
  volatile uint32_t* tslot = reinterpret_cast<volatile uint32_t*>(sm + Cf::OFF_BAR + Cf::NBAR * 8);
  const int tid = threadIdx.x, lane = tid & 31;
  const int warp = __shfl_sync(0xffffffffu, tid >> 5, 0);
  const int B = p.B, HW = p.HW, K2 = p.K2, NQ = HW / N;
  const int nunits = NCT * B * NQ, u = blockIdx.x;
  const bool has_unit = u < nunits;
  const int ct = u % NCT, rest = u / NCT, ub = rest / NQ, hw0 = (rest - ub * NQ) * N;
#if G2_STAMPS
  unsigned long long* stp = p.stamp ? p.stamp + blockIdx.x * 8 : nullptr;
  if (stp && tid == 0) stp[0] = g_gtime();
#endif
  if (tid == 0) {
    for (int s = 0; s < NS; s++) {
      asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::"r"(b_fa + 8 * s) : "memory");
      asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::"r"(b_fb + 8 * s) : "memory");
      asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::"r"(b_emp + 8 * s) : "memory");
    }
    asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::"r"(b_acc) : "memory");
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  if (has_unit && warp == 1) {
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" ::"r"(bar + Cf::NBAR * 8),
                 "n"(Cf::TCOLS) : "memory");
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;" ::: "memory");
  }
  tc_before();
  __syncthreads();
  tc_after();
  const uint32_t tbase = has_unit ? *tslot : 0u;

  // h B stages do not depend on phase P: issue the first NPRE before it
  if (has_unit && warp == 0) {
    if (elect_one()) {
      for (int kb = 0; kb < NPRE; kb++) {
        mbar_expect(b_fb + 8 * kb, SB);
        for (int j = 0; j < NB; j++)
          tma2d(sbase + kb * STG + SA + j * 8192, &tmB, hw0 + 64 * j, ub * K2 + 64 * kb, b_fb + 8 * kb);
      }
    }
    __syncwarp();
  }

  // ---- phase P: CTA j < G*B builds rows c = j/B + G*i of image j%B: A_b = fp16(W2 * s_b) [B][C][C4]; image-0 CTAs
  //      also write b2' = W2.gb + b2. s_b and the conversion use K3's arithmetic (same roundings).
  {
    const int G = gridDim.x / B;
    if ((int)blockIdx.x < G * B) {
      const int pb = blockIdx.x % B, pg = blockIdx.x / B;
      int c = pg + G * warp;
      float4 wv[MAXI];
      auto ldrow = [&](int cc) {
        const float* wr = p.w2 + (size_t)cc * C4;
#pragma unroll
        for (int i = 0; i < MAXI; i++) wv[i] = __ldg(reinterpret_cast<const float4*>(wr + lane * 4 + i * 128));
      };
      if (c < C) ldrow(c);
      float loc = 0.f;
      for (int k = tid; k < C4; k += 512) {
        const float gf = sqrtf(__ldcg(p.gsum + (size_t)pb * C4 + k));
        ssm[k] = gf; loc += gf;
      }
      loc = g_wsum(loc);
      if (lane == 0) red[warp] = loc;
      __syncthreads();
      if (warp == 0) {
        float v = lane < 16 ? red[lane] : 0.f;
        v = g_wsum(v);
        if (lane == 0) red[16] = 1.f / (v / (float)C4 + p.eps);
      }
      __syncthreads();
      const float inv = red[16];
      for (int k = tid; k < C4; k += 512) ssm[k] = fmaf(__ldg(p.gw + k) * ssm[k], inv, 1.f);
      __syncthreads();
      __half* w2b = reinterpret_cast<__half*>(p.w2b);
#pragma unroll 1
      while (c < C) {
        __half* dst = w2b + ((size_t)pb * C + c) * C4;
        float dot = 0.f;
#pragma unroll
        for (int i = 0; i < MAXI; i++) {
          const int k = lane * 4 + i * 128;
          const float4 w = wv[i];
          const float4 s = *reinterpret_cast<const float4*>(ssm + k);
          uint2 o;
          o.x = h2u(__floats2half2_rn(w.x * s.x, w.y * s.y));
          o.y = h2u(__floats2half2_rn(w.z * s.z, w.w * s.w));
          *reinterpret_cast<uint2*>(dst + k) = o;
          if (pb == 0) {
            const float4 gv = __ldg(reinterpret_cast<const float4*>(p.gb + k));
            dot = fmaf(w.x, gv.x, dot); dot = fmaf(w.y, gv.y, dot); dot = fmaf(w.z, gv.z, dot); dot = fmaf(w.w, gv.w, dot);
          }
        }
        if (pb == 0) {
          dot = g_wsum(dot);
          if (lane == 0) p.b2p[c] = dot + __ldg(p.b2 + c);
        }
        c += 16 * G;
        if (c < C) ldrow(c);
      }
    }
  }
  asm volatile("fence.proxy.async;" ::: "memory");  // A_b is read by TMA in other CTAs
#if G2_STAMPS
  if (stp && tid == 0) stp[1] = g_gtime();
#endif
  g2_grid_bar(gridDim.x);
#if G2_STAMPS
  if (stp && tid == 0) stp[2] = g_gtime();
#endif
  if (!has_unit) return;

  if (warp == 0) {
    // ---- TMA producer: A every stage, B for the stages not prefetched
    asm volatile("fence.proxy.async;" ::: "memory");
#pragma unroll 1
    for (int kb = 0; kb < KB; kb++) {
      const int s = kb % NS;
      if (kb >= NS) mbar_wait(b_emp + 8 * s, ((kb / NS) - 1) & 1);
      if (elect_one()) {
        mbar_expect(b_fa + 8 * s, SA);
        tma2d(sbase + s * STG, &tmA, 64 * kb, ub * C + ct * 128, b_fa + 8 * s);
        if (kb >= NPRE) {
          mbar_expect(b_fb + 8 * s, SB);
          for (int j = 0; j < NB; j++)
            tma2d(sbase + s * STG + SA + j * 8192, &tmB, hw0 + 64 * j, ub * K2 + 64 * kb, b_fb + 8 * s);
        }
      }
      __syncwarp();
    }
  } else if (warp == 1) {
    // ---- MMA issuer: 4 x (128 x N x 16) per 64-k stage into one accumulator
#if G2_STAMPS
    long long wclk = 0, c0 = 0;
#endif
#pragma unroll 1
    for (int kb = 0; kb < KB; kb++) {
      const int s = kb % NS;
      const uint32_t par = (kb / NS) & 1;
#if G2_STAMPS
      const long long w0 = clock64();
#endif
      mbar_wait(b_fa + 8 * s, par);
      mbar_wait(b_fb + 8 * s, par);
#if G2_STAMPS
      if (kb == 0) { c0 = clock64(); if (stp && lane == 0) stp[3] = g_gtime(); }
      else wclk += clock64() - w0;
#endif
      tc_after();
      if (elect_one()) {
        const uint32_t aS = sbase + s * STG, bS = aS + SA;
#pragma unroll
        for (int kk = 0; kk < 4; kk++) {
          const uint64_t ad = sdesc(aS + kk * 32, 16, 1024);
          const uint64_t bd = sdesc(bS + kk * 2048, 8192, 1024);
          const uint32_t accum = (kb | kk) ? 1u : 0u;
          asm volatile("{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %4, 0;\n\t"
                       "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n\t}"
                       ::"r"(tbase), "l"(ad), "l"(bd), "r"(Cf::IDESC), "r"(accum) : "memory");
        }
        asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];"
                     ::"r"(b_emp + 8 * s) : "memory");
        if (kb == KB - 1)
          asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];" ::"r"(b_acc) : "memory");
      }
      __syncwarp();
    }
#if G2_STAMPS
    mbar_wait(b_acc, 0);
    if (stp && lane == 0) { stp[4] = (unsigned long long)(clock64() - c0); stp[6] = (unsigned long long)wclk; }
#endif
  }

  // ---- epilogue (all 16 warps): item = 32 channels (this warp's TMEM lane quadrant) x 32 tokens; x is prefetched into
  //      registers before the accumulator is ready; the item goes through a per-warp smem transpose (the ring is free
  //      once the accumulator barrier fired) so that each load/store instruction covers 4 rows x 128 B of NCHW
  {
    const int q = warp & 3, jw = warp >> 2;
    const size_t ibase = (size_t)ub * C * HW + hw0;
    float4 xv[IPW][8];
    float bb[8];
#pragma unroll
    for (int r = 0; r < 8; r++) {
      const int c = ct * 128 + q * 32 + 4 * r + (lane >> 3);
      bb[r] = c < C ? __ldcg(p.b2p + c) : 0.f;
#pragma unroll
      for (int it = 0; it < IPW; it++) {
        const int j = jw + 4 * it, t = j * 32 + (lane & 7) * 4;
        xv[it][r] = (j < NCH && c < C && t < N) ? __ldg(reinterpret_cast<const float4*>(p.x + ibase + (size_t)c * HW + t))
                                                : make_float4(0.f, 0.f, 0.f, 0.f);
      }
    }
    mbar_wait(b_acc, 0);
    tc_after();
#if G2_STAMPS
    if (stp && tid == 64) stp[5] = g_gtime();
#endif
    unsigned char* stg = sm + warp * 4096;
#pragma unroll
    for (int it = 0; it < IPW; it++) {
      const int j = jw + 4 * it;
      if (j < NCH) {
        uint32_t v[32];
        TC_LD32(v, tbase + ((uint32_t)(q * 32) << 16) + (uint32_t)(j * 32));
#pragma unroll
        for (int e = 0; e < 8; e++)
          *reinterpret_cast<uint4*>(stg + lane * 128 + ((e ^ (lane & 7)) << 4)) =
              make_uint4(v[4 * e], v[4 * e + 1], v[4 * e + 2], v[4 * e + 3]);
        __syncwarp();
#pragma unroll
        for (int r = 0; r < 8; r++) {
          const int row = 4 * r + (lane >> 3), un = lane & 7, c = ct * 128 + q * 32 + row, t = j * 32 + un * 4;
          const float4 a = *reinterpret_cast<const float4*>(stg + row * 128 + ((un ^ (row & 7)) << 4));
          if (c < C && t < N) {
            const float4 xx = xv[it][r];
            *reinterpret_cast<float4*>(p.out + ibase + (size_t)c * HW + t) =
                make_float4(xx.x + a.x + bb[r], xx.y + a.y + bb[r], xx.z + a.z + bb[r], xx.w + a.w + bb[r]);
          }
        }
        __syncwarp();
      }
    }
  }
  tc_before();
  __syncthreads();
  if (warp == 1) {
    tc_after();
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" ::"r"(tbase), "n"(Cf::TCOLS) : "memory");
  }
#if G2_STAMPS
  if (stp && tid == 0) stp[7] = g_gtime();
#endif
}

// ---------------------------------------------------------------- host
extern "C" int g_rows_per_tile(int C, int H, int W) {
  if (C == 96 && H == 112 && W == 112) return 2;
  if (C == 128 && H == 64 && W == 64) return 2;
  if (C == 96 && H == 56 && W == 56) return 2;
  if (C == 320 && H == 32 && W == 32) return 2;
  return 0;
}
static int g_num_sms() {
  static int n = 0;
  if (n == 0) { int dev = 0; cudaGetDevice(&dev); cudaDeviceGetAttribute(&n, cudaDevAttrMultiProcessorCount, dev); }
  return n;
}
static bool g_is_tc(int C, int W) { return (C == 96 && W == 56) || (C == 320 && W == 32); }
template <int C, int R, int W, int NCW, int NGW, int MG, bool DWS>
static const char* launch_p1(const GParams& p, cudaStream_t s) {
  using Cf = P1Cfg<C, R, W, NCW, NGW, MG, DWS>;
  static bool init = false;  // one attribute value per instantiation, never changed
  if (!init) {
    cudaError_t e = cudaFuncSetAttribute(g_pass1<C, R, W, NCW, NGW, MG, DWS>, cudaFuncAttributeMaxDynamicSharedMemorySize, Cf::SMEM);
    if (e != cudaSuccess) return cudaGetErrorString(e);
    init = true;
  }
  const int grid = p.ntiles < g_num_sms() ? p.ntiles : g_num_sms();
  g_pass1<C, R, W, NCW, NGW, MG, DWS><<<grid, Cf::NT, Cf::SMEM, s>>>(p);
  return nullptr;
}
// one CTA per tile (112 / 128 tiles <= 148 SMs; 1 CTA per SM by registers, so the 512-column TMEM alloc never waits)
template <int C, int R, int W, int TP, int NS>
static const char* launch_tc(const GParams& p, cudaStream_t s) {
  using Cf = TCfg<C, R, W, TP, NS>;
  static bool init = false;  // one attribute value per instantiation, never changed
  if (!init) {
    cudaError_t e = cudaFuncSetAttribute(g_tc<C, R, W, TP, NS>, cudaFuncAttributeMaxDynamicSharedMemorySize, Cf::SMEM);
    if (e != cudaSuccess) return cudaGetErrorString(e);
    init = true;
  }
  g_tc<C, R, W, TP, NS><<<p.ntiles, Cf::NT, Cf::SMEM, s>>>(p);
  return nullptr;
}
// tensor maps are encoded on every call (pointers shift); the driver entry point is resolved once
typedef CUresult (*g_encode_fn)(CUtensorMap*, CUtensorMapDataType, cuuint32_t, void*, const cuuint64_t*, const cuuint64_t*,
                                const cuuint32_t*, const cuuint32_t*, CUtensorMapInterleave, CUtensorMapSwizzle,
                                CUtensorMapL2promotion, CUtensorMapFloatOOBfill);
static g_encode_fn g_encoder() {
  static g_encode_fn fn = nullptr;
  if (!fn) {
    void* f = nullptr;
    cudaDriverEntryPointQueryResult q;
    if (cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled", &f, 12000, cudaEnableDefault, &q) == cudaSuccess &&
        q == cudaDriverEntryPointSuccess)
      fn = (g_encode_fn)f;
  }
  return fn;
}
static bool g_map2d(CUtensorMap* m, void* base, uint64_t inner, uint64_t outer, uint64_t row_bytes, uint32_t bi, uint32_t bo) {
  g_encode_fn enc = g_encoder();
  if (!enc) return false;
  const cuuint64_t dims[2] = {inner, outer}, strides[1] = {row_bytes};
  const cuuint32_t box[2] = {bi, bo}, es[2] = {1, 1};
  return enc(m, CU_TENSOR_MAP_DATA_TYPE_FLOAT16, 2, base, dims, strides, box, es, CU_TENSOR_MAP_INTERLEAVE_NONE,
             CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) == CUDA_SUCCESS;
}
// cooperative launch on every SM (1 CTA per SM by smem); units beyond NCT*B*HW/N only take part in phase P
template <int C, int N, int NS>
static const char* launch_g2(const GParams& p, cudaStream_t s) {
  using Cf = G2Cfg<C, N, NS>;
  static bool init = false;  // one attribute value per instantiation, never changed
  if (!init) {
    cudaError_t e = cudaFuncSetAttribute(g_g2<C, N, NS>, cudaFuncAttributeMaxDynamicSharedMemorySize, Cf::SMEM);
    if (e != cudaSuccess) return cudaGetErrorString(e);
    init = true;
  }
  if (p.HW % N != 0 || Cf::NCT * p.B * (p.HW / N) > g_num_sms() || g_num_sms() < p.B) return "g_g2: bad unit split";
  CUtensorMap ta, tb;
  if (!g_map2d(&ta, p.w2b, (uint64_t)Cf::C4, (uint64_t)p.B * C, (uint64_t)Cf::C4 * 2, 64, 128)) return "g_g2: A map";
  if (!g_map2d(&tb, p.h, (uint64_t)p.HW, (uint64_t)p.B * p.K2, (uint64_t)p.HW * 2, 64, 64)) return "g_g2: B map";
  GParams pp = p;
  void* args[] = {&pp, &ta, &tb};
  cudaError_t e = cudaLaunchCooperativeKernel((const void*)g_g2<C, N, NS>, dim3(g_num_sms()), dim3(512), args,
                                              (size_t)Cf::SMEM, s);
  return e == cudaSuccess ? nullptr : cudaGetErrorString(e);
}
static cublasLtHandle_t g_lt() {
  static cublasLtHandle_t h = nullptr;
  if (!h) cublasLtCreate(&h);
  return h;
}
#define LTCHK(x) do { if ((x) != CUBLAS_STATUS_SUCCESS) { err = "cublasLt: " #x; goto done; } } while (0)
// descriptors, batch strides and pointers are set on every call; only the heuristic result is cached per shape
static const char* g_gemm2(const GParams& p, void* ws, size_t wsb, cudaStream_t s) {
  static std::map<std::tuple<int, int, int>, cublasLtMatmulHeuristicResult_t> cache;
  const char* err = nullptr;
  cublasLtHandle_t lt = g_lt();
  cublasLtMatmulDesc_t op = nullptr;
  cublasLtMatrixLayout_t la = nullptr, lb = nullptr, lc = nullptr;
  const cublasOperation_t tn = CUBLAS_OP_N;
  const int32_t bc = p.B;
  const int64_t sa = (int64_t)p.K2 * p.HW, sb = (int64_t)p.C * p.K2, scd = (int64_t)p.C * p.HW;
  const float alpha = 1.f, beta = 1.f;
  auto key = std::make_tuple(p.B, p.C, p.HW);
  LTCHK(cublasLtMatmulDescCreate(&op, CUBLAS_COMPUTE_32F, CUDA_R_32F));
  LTCHK(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSA, &tn, sizeof(tn)));
  LTCHK(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSB, &tn, sizeof(tn)));
  LTCHK(cublasLtMatrixLayoutCreate(&la, CUDA_R_16F, p.HW, p.K2, p.HW));
  LTCHK(cublasLtMatrixLayoutCreate(&lb, CUDA_R_16F, p.K2, p.C, p.K2));
  LTCHK(cublasLtMatrixLayoutCreate(&lc, CUDA_R_32F, p.HW, p.C, p.HW));
  LTCHK(cublasLtMatrixLayoutSetAttribute(la, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &bc, sizeof(bc)));
  LTCHK(cublasLtMatrixLayoutSetAttribute(lb, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &bc, sizeof(bc)));
  LTCHK(cublasLtMatrixLayoutSetAttribute(lc, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &bc, sizeof(bc)));
  LTCHK(cublasLtMatrixLayoutSetAttribute(la, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &sa, sizeof(sa)));
  LTCHK(cublasLtMatrixLayoutSetAttribute(lb, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &sb, sizeof(sb)));
  LTCHK(cublasLtMatrixLayoutSetAttribute(lc, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &scd, sizeof(scd)));
  if (cache.find(key) == cache.end()) {
    cublasLtMatmulPreference_t pref = nullptr;
    const uint64_t wsl = wsb;
    cublasLtMatmulHeuristicResult_t res;
    int n = 0;
    LTCHK(cublasLtMatmulPreferenceCreate(&pref));
    cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &wsl, sizeof(wsl));
    const cublasStatus_t st = cublasLtMatmulAlgoGetHeuristic(lt, op, la, lb, lc, lc, pref, 1, &res, &n);
    cublasLtMatmulPreferenceDestroy(pref);
    if (st != CUBLAS_STATUS_SUCCESS || n == 0) { err = "cublasLt: no heuristic for GEMM2"; goto done; }
    cache[key] = res;
  }
  LTCHK(cublasLtMatmul(lt, op, &alpha, p.h, la, p.w2b, lb, &beta, p.x, lc, p.out, lc, &cache[key].algo, ws, wsb, s));
done:
  if (la) cublasLtMatrixLayoutDestroy(la);
  if (lb) cublasLtMatrixLayoutDestroy(lb);
  if (lc) cublasLtMatrixLayoutDestroy(lc);
  if (op) cublasLtMatmulDescDestroy(op);
  return err;
}
// mask selects launches (15 = all; lower values only for per-launch timing). g2mode (tc rows only): 0 = K3 + cuBLASLt
// GEMM2 (r9), 1 = g_g2 (4,96 with 224-token units), 2 = g_g2 (4,96 with 112-token units)
extern "C" const char* g_launch(GParams p, void* ltws, size_t ltws_bytes, int mask, int g2mode, void* stream_v) {
  cudaStream_t s = (cudaStream_t)stream_v;
  const int R = g_rows_per_tile(p.C, p.H, p.W);
  if (R == 0) return "unsupported shape";
  p.HW = p.H * p.W; p.C4 = 4 * p.C; p.K2 = p.C4 + 8; p.TPI = p.H / R; p.ntiles = p.B * p.TPI;
  const char* err = nullptr;
  const int swz = g_is_tc(p.C, p.W) ? 1 : 0;
  const bool use_g2 = swz && g2mode > 0;
  if (mask & 1) {
    const int nfb = swz ? ((p.C4 / 128) * ((p.C + 63) / 64) * 1024 + 255) / 256
                        : ((p.C4 / 16) * (p.C / 16) * 32 + 255) / 256;
    const int nub = (p.C4 + 7) / 8, ndb = (p.C * 56 + 255) / 256, nzb = (p.B * p.C4 + 255) / 256;
    g_prep<<<nfb + nub + ndb + nzb, 256, 0, s>>>(p, nfb, nub, ndb, swz);
  }
  if (mask & 2) {
    if (p.C == 96 && p.W == 112) err = launch_p1<96, 2, 112, 7, 9, 3, true>(p, s);
    else if (p.C == 128 && p.W == 64) err = launch_p1<128, 2, 64, 8, 8, 2, false>(p, s);
    else if (p.C == 96 && p.W == 56) err = launch_tc<96, 2, 56, 128, 6>(p, s);
    else if (p.C == 320 && p.W == 32) err = launch_tc<320, 2, 32, 64, 9>(p, s);
    if (err) return err;
  }
  if (mask & 4) {
    if (use_g2) {
      if (p.C == 320) err = launch_g2<320, 256, 4>(p, s);
      else if (g2mode == 2) err = launch_g2<96, 112, 6>(p, s);
      else err = launch_g2<96, 224, 4>(p, s);
      if (err) return err;
    } else {
      g_grn<<<dim3(p.B, (p.C + 15) / 16), 512, (p.C4 + 64) * 4, s>>>(p);
    }
  }
  if ((mask & 8) && !use_g2) { err = g_gemm2(p, ltws, ltws_bytes, s); if (err) return err; }
  cudaError_t e = cudaGetLastError();
  return e == cudaSuccess ? nullptr : cudaGetErrorString(e);
}
```

```cpp file=binding.cpp
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <stdexcept>
#include <string>

// r0 anchor (cooperative fp32 kernel): every shape outside the lineage-G set
struct Params {
  const float* x; const float* dww; const float* dwb; const float* lnw; const float* lnb;
  const float* w1; const float* b1; const float* gw; const float* gb; const float* w2; const float* b2;
  float* out;
  float* xn; float* part; float* hidden; float* sumsq; float* w1t; float* w2t;
  int B, C, H, W, HW, M, Mp, C4, nch;
  float eps, ln_eps;
  int p1_mode, p1_twr, p1_tiles, p1_lpr, p1_rg, p1_hb, p1_wb;
  int cfg2, cfg3, split3;
};
extern "C" const char* convnext_launch(Params p, void* stream);

// lineage G (kernel.cu): K0 prep, fused pass 1, then K3 + cuBLASLt GEMM2 (64,96 / 32,128) or g_g2 (8,320 / 4,96)
struct GParams {
  const float* x; const float* dww; const float* dwb; const float* lnw; const float* lnb;
  const float* w1; const float* b1; const float* gw; const float* gb; const float* w2; const float* b2;
  float* out;
  void* w1f; float* uv; void* h; float* gsum; void* w2b; float* dwp;
  int B, C, H, W, HW, C4, K2, TPI, ntiles;
  float eps, ln_eps;
  unsigned long long* stamp;
  float* b2p;
};
extern "C" int g_rows_per_tile(int C, int H, int W);
extern "C" const char* g_launch(GParams p, void* ltws, size_t ltws_bytes, int mask, int g2mode, void* stream);

static inline size_t align256(size_t v) { return (v + 255) & ~(size_t)255; }

static void run_g(torch::Tensor& x, torch::Tensor& dww, torch::Tensor& dwb, torch::Tensor& lnw, torch::Tensor& lnb,
                  torch::Tensor& w1, torch::Tensor& b1, torch::Tensor& gw, torch::Tensor& gb, torch::Tensor& w2,
                  torch::Tensor& b2, double eps, double leps, torch::Tensor& out, cudaStream_t stream) {
  const int B = (int)x.size(0), C = (int)x.size(1), H = (int)x.size(2), W = (int)x.size(3);
  const int HW = H * W, C4 = 4 * C, K2 = C4 + 8;
  // W1' scratch: m16n8k16 fragments (C4*C fp16) or pre-swizzled 128x64 images ((C4/128) * ceil(C/64) * 16 KB)
  const size_t w1f_frag = (size_t)C4 * C * 2, w1f_img = (size_t)(C4 / 128) * ((C + 63) / 64) * 16384;
  const size_t o_w1f = 0, o_uv = align256(o_w1f + (w1f_frag > w1f_img ? w1f_frag : w1f_img)),
               o_dwp = align256(o_uv + (size_t)C4 * 8),
               o_gs = align256(o_dwp + (size_t)C * 56 * 4), o_w2b = align256(o_gs + (size_t)B * C4 * 4),
               o_h = align256(o_w2b + (size_t)B * C * K2 * 2), o_b2p = align256(o_h + (size_t)B * K2 * HW * 2),
               o_lt = align256(o_b2p + (size_t)C * 4), ltb = (size_t)32 << 20, tot = o_lt + ltb;
  torch::Tensor ws = torch::empty({(long)tot}, x.options().dtype(torch::kUInt8));
  char* wp = (char*)ws.data_ptr();
  GParams p;
  p.x = x.data_ptr<float>(); p.dww = dww.data_ptr<float>(); p.dwb = dwb.data_ptr<float>(); p.lnw = lnw.data_ptr<float>();
  p.lnb = lnb.data_ptr<float>(); p.w1 = w1.data_ptr<float>(); p.b1 = b1.data_ptr<float>(); p.gw = gw.data_ptr<float>();
  p.gb = gb.data_ptr<float>(); p.w2 = w2.data_ptr<float>(); p.b2 = b2.data_ptr<float>(); p.out = out.data_ptr<float>();
  p.w1f = wp + o_w1f; p.uv = (float*)(wp + o_uv); p.dwp = (float*)(wp + o_dwp); p.gsum = (float*)(wp + o_gs);
  p.w2b = wp + o_w2b; p.h = wp + o_h; p.b2p = (float*)(wp + o_b2p);
  p.B = B; p.C = C; p.H = H; p.W = W; p.HW = 0; p.C4 = 0; p.K2 = 0; p.TPI = 0; p.ntiles = 0;
  p.eps = (float)eps; p.ln_eps = (float)leps;
  p.stamp = nullptr;
  const char* err = g_launch(p, wp + o_lt, ltb, 15, 1, (void*)stream);
  if (err != nullptr) throw std::runtime_error(std::string("g_launch failed: ") + err);
}

static void run_anchor(torch::Tensor& x, torch::Tensor& dwconv_weight, torch::Tensor& dwconv_bias,
                       torch::Tensor& layernorm_weight, torch::Tensor& layernorm_bias, torch::Tensor& pwconv1_weight,
                       torch::Tensor& pwconv1_bias, torch::Tensor& grn_weight, torch::Tensor& grn_bias,
                       torch::Tensor& pwconv2_weight, torch::Tensor& pwconv2_bias, double eps, double layer_norm_eps,
                       torch::Tensor& output, cudaStream_t stream) {
  const int B = (int)x.size(0), C = (int)x.size(1), H = (int)x.size(2), W = (int)x.size(3);
  TORCH_CHECK(C % 32 == 0, "dim must be a multiple of 32");
  TORCH_CHECK(H * W >= 32, "H*W must be >= 32");
  const int HW = H * W, M = B * HW, C4 = 4 * C, nch = C / 32;
  const int Mp = ((M + 127) / 128) * 128;
  const size_t off_xn = 0;
  const size_t off_part = align256(off_xn + (size_t)C * Mp * 4);
  const size_t off_hidden = align256(off_part + (size_t)nch * Mp * 2 * 4);
  const size_t off_sumsq = align256(off_hidden + (size_t)C4 * Mp * 4);
  const size_t off_w1t = align256(off_sumsq + (size_t)B * C4 * 4);
  const size_t off_w2t = align256(off_w1t + (size_t)C * C4 * 4);
  const size_t total = align256(off_w2t + (size_t)C4 * C * 4);
  torch::Tensor ws = torch::empty({(long)total}, x.options().dtype(torch::kUInt8));
  char* wp = (char*)ws.data_ptr();
  Params p;
  p.x = x.data_ptr<float>(); p.dww = dwconv_weight.data_ptr<float>(); p.dwb = dwconv_bias.data_ptr<float>();
  p.lnw = layernorm_weight.data_ptr<float>(); p.lnb = layernorm_bias.data_ptr<float>();
  p.w1 = pwconv1_weight.data_ptr<float>(); p.b1 = pwconv1_bias.data_ptr<float>();
  p.gw = grn_weight.data_ptr<float>(); p.gb = grn_bias.data_ptr<float>();
  p.w2 = pwconv2_weight.data_ptr<float>(); p.b2 = pwconv2_bias.data_ptr<float>();
  p.out = output.data_ptr<float>();
  p.xn = (float*)(wp + off_xn); p.part = (float*)(wp + off_part); p.hidden = (float*)(wp + off_hidden);
  p.sumsq = (float*)(wp + off_sumsq); p.w1t = (float*)(wp + off_w1t); p.w2t = (float*)(wp + off_w2t);
  p.B = B; p.C = C; p.H = H; p.W = W; p.HW = HW; p.M = M; p.Mp = Mp; p.C4 = C4; p.nch = nch;
  p.eps = (float)eps; p.ln_eps = (float)layer_norm_eps;
  p.p1_mode = 0; p.p1_twr = 1; p.p1_tiles = 0; p.p1_lpr = 0; p.p1_rg = 0; p.p1_hb = 0; p.p1_wb = 0;
  p.cfg2 = 0; p.cfg3 = 0; p.split3 = 1;
  const char* err = convnext_launch(p, (void*)stream);
  if (err != nullptr) throw std::runtime_error(std::string("convnext_launch failed: ") + err);
}

void run(torch::Tensor x, torch::Tensor dwconv_weight, torch::Tensor dwconv_bias, torch::Tensor layernorm_weight,
         torch::Tensor layernorm_bias, torch::Tensor pwconv1_weight, torch::Tensor pwconv1_bias, torch::Tensor grn_weight,
         torch::Tensor grn_bias, torch::Tensor pwconv2_weight, torch::Tensor pwconv2_bias, double eps, double layer_norm_eps,
         torch::Tensor output) {
  TORCH_CHECK(x.is_cuda() && x.dtype() == torch::kFloat32 && x.is_contiguous(), "x must be contiguous fp32 cuda");
  TORCH_CHECK(output.is_contiguous() && output.dtype() == torch::kFloat32, "output must be contiguous fp32");
  TORCH_CHECK(x.dim() == 4, "x must be NCHW");
  TORCH_CHECK(pwconv1_weight.is_contiguous() && pwconv2_weight.is_contiguous() && dwconv_weight.is_contiguous(), "weights must be contiguous");
  TORCH_CHECK(layernorm_weight.is_contiguous() && layernorm_bias.is_contiguous() && grn_weight.is_contiguous() && grn_bias.is_contiguous(), "params contiguous");
  TORCH_CHECK(dwconv_bias.is_contiguous() && pwconv1_bias.is_contiguous() && pwconv2_bias.is_contiguous(), "biases contiguous");
  const c10::cuda::OptionalCUDAGuard guard(x.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream(x.device().index()).stream();
  if (g_rows_per_tile((int)x.size(1), (int)x.size(2), (int)x.size(3)) > 0)
    run_g(x, dwconv_weight, dwconv_bias, layernorm_weight, layernorm_bias, pwconv1_weight, pwconv1_bias, grn_weight,
          grn_bias, pwconv2_weight, pwconv2_bias, eps, layer_norm_eps, output, stream);
  else
    run_anchor(x, dwconv_weight, dwconv_bias, layernorm_weight, layernorm_bias, pwconv1_weight, pwconv1_bias, grn_weight,
               grn_bias, pwconv2_weight, pwconv2_bias, eps, layer_norm_eps, output, stream);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("run", &run, "ConvNeXtV2 block with GRN (lineage G fused pass 1 + g_g2 + r0 fallback)"); }
```

```cuda file=anchor.cu
// ConvNeXtV2 block with GRN: one cooperative launch, three phases, fp32 SIMT maths (r0 anchor, fallback only).
#include <cuda_runtime.h>
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

#define NT 256
#define MAXIMG 5
#define GEMM_SMEM_FLOATS (128 * 132)
#define SMEM_BYTES (GEMM_SMEM_FLOATS * 4 + MAXIMG * 128 * 4 + 64)
#define P1_SLOTS 512

struct Params {
  const float* x; const float* dww; const float* dwb; const float* lnw; const float* lnb;
  const float* w1; const float* b1; const float* gw; const float* gb; const float* w2; const float* b2;
  float* out;
  float* xn; float* part; float* hidden; float* sumsq; float* w1t; float* w2t;
  int B, C, H, W, HW, M, Mp, C4, nch;
  float eps, ln_eps;
  int p1_mode, p1_twr, p1_tiles, p1_lpr, p1_rg, p1_hb, p1_wb;
  int cfg2, cfg3, split3;
};

__device__ __forceinline__ float gelu_f(float v) { return 0.5f * v * (1.f + erff(v * 0.70710678118654752f)); }

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}

__device__ void transpose_mat(const float* __restrict__ src, int R, int Cc, float* __restrict__ dst, float* tile) {
  const int rt = (R + 31) / 32, ct = (Cc + 31) / 32;
  const int tid = threadIdx.x, j = tid & 31, i0 = tid >> 5;
  for (int t = blockIdx.x; t < rt * ct; t += gridDim.x) {
    const int tr = t / ct, tc = t % ct;
#pragma unroll
    for (int s = 0; s < 4; s++) {
      const int i = i0 + 8 * s, r = tr * 32 + i, c = tc * 32 + j;
      tile[i * 33 + j] = (r < R && c < Cc) ? src[(size_t)r * Cc + c] : 0.f;
    }
    __syncthreads();
#pragma unroll
    for (int s = 0; s < 4; s++) {
      const int i = i0 + 8 * s, c = tc * 32 + i, r = tr * 32 + j;
      if (r < R && c < Cc) dst[(size_t)c * R + r] = tile[j * 33 + i];
    }
    __syncthreads();
  }
}

template <int TW>
__device__ __forceinline__ void conv_patch(const float* __restrict__ xc, const float* wt, int H, int W, int h0, int w0,
                                           float (&acc)[4][TW]) {
#pragma unroll
  for (int r = 0; r < 4; r++)
#pragma unroll
    for (int cc = 0; cc < TW; cc++) acc[r][cc] = 0.f;
#pragma unroll
  for (int i = 0; i < 10; i++) {
    const int hh = h0 - 3 + i;
    const bool rowok = (hh >= 0) && (hh < H);
    float v[TW + 6];
    if (TW == 4) {
      const float* rp = xc + hh * W;
      float4 L = make_float4(0.f, 0.f, 0.f, 0.f), Cq = L, R = L;
      if (rowok) {
        if (w0 >= 4) L = __ldg(reinterpret_cast<const float4*>(rp + w0 - 4));
        Cq = __ldg(reinterpret_cast<const float4*>(rp + w0));
        if (w0 + 4 < W) R = __ldg(reinterpret_cast<const float4*>(rp + w0 + 4));
      }
      v[0] = L.y; v[1] = L.z; v[2] = L.w; v[3] = Cq.x; v[4] = Cq.y; v[5] = Cq.z; v[6] = Cq.w;
      v[7] = R.x; v[8] = R.y; v[9] = R.z;
    } else {
#pragma unroll
      for (int jj = 0; jj < TW + 6; jj++) {
        const int ww = w0 - 3 + jj;
        v[jj] = (rowok && ww >= 0 && ww < W) ? __ldg(xc + hh * W + ww) : 0.f;
      }
    }
#pragma unroll
    for (int r = 0; r < 4; r++) {
      const int ki = i - r;
      if (ki >= 0 && ki < 7) {
#pragma unroll
        for (int cc = 0; cc < TW; cc++)
#pragma unroll
          for (int j = 0; j < 7; j++) acc[r][cc] = fmaf(v[cc + j], wt[ki * 7 + j], acc[r][cc]);
      }
    }
  }
}

template <int MODE, int TWR>
__device__ void phase1_conv(const Params& p, float* smem) {
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int C = p.C, H = p.H, W = p.W, HW = p.HW, M = p.M, Mp = p.Mp, nch = p.nch;
  float* st_s = smem;
  float* wsm = smem + 8 * P1_SLOTS * 2;
  const int lpr = p.p1_lpr, rg = p.p1_rg, hb = p.p1_hb, wb = p.p1_wb;
  const int rows_per_tile = 4 * rg;
  const int rpt = lpr * TWR;

  for (int tile = blockIdx.x; tile < p.p1_tiles; tile += gridDim.x) {
    int b = 0, cch = 0, hbi = 0, wbi = 0, tok0 = 0;
    if (MODE == 1) {
      const int per = nch * hb * wb;
      b = tile / per; int rem = tile - b * per;
      cch = rem / (hb * wb); rem -= cch * (hb * wb);
      hbi = rem / wb; wbi = rem - hbi * wb;
    } else {
      cch = tile % nch; tok0 = (tile / nch) * (32 * TWR);
    }
    for (int i = tid; i < 8 * P1_SLOTS * 2; i += NT) st_s[i] = 0.f;
    for (int i = tid; i < 32 * 49; i += NT) wsm[i] = __ldg(p.dww + (size_t)cch * 32 * 49 + i);
    __syncthreads();

    int rgi = 0, cgi = 0, h0 = 0, w0 = 0; bool lane_ok = true;
    if (MODE == 1) {
      rgi = lane / lpr; cgi = lane - rgi * lpr;
      lane_ok = rgi < rg;
      h0 = hbi * rows_per_tile + rgi * 4;
      w0 = (wbi * 32 + cgi) * TWR;
      lane_ok = lane_ok && (w0 < W);
    }
    int tk[MODE == 0 ? TWR : 1], tb[MODE == 0 ? TWR : 1], th[MODE == 0 ? TWR : 1], tw_[MODE == 0 ? TWR : 1];
    bool tv[MODE == 0 ? TWR : 1];
    if (MODE == 0) {
#pragma unroll
      for (int r = 0; r < TWR; r++) {
        const int t = tok0 + r * 32 + lane;
        tv[r] = t < M;
        const int tt = tv[r] ? t : 0;
        const int bb = tt / HW; const int rem = tt - bb * HW; const int hh = rem / W;
        tk[r] = t; tb[r] = bb; th[r] = hh; tw_[r] = rem - hh * W;
      }
    }

#pragma unroll 1
    for (int cl = 0; cl < 4; cl++) {
      const int clc = warp * 4 + cl;
      const int c = cch * 32 + clc;
      float wt[49];
#pragma unroll
      for (int i = 0; i < 49; i++) wt[i] = wsm[clc * 49 + i];
      const float bias = __ldg(p.dwb + c);
      float* xrow = p.xn + (size_t)c * Mp;
      float* sts = st_s + warp * P1_SLOTS * 2;
      if (MODE == 1) {
        if (lane_ok) {
          float acc[4][TWR];
          conv_patch<TWR>(p.x + ((size_t)b * C + c) * HW, wt, H, W, h0, w0, acc);
#pragma unroll
          for (int r = 0; r < 4; r++) {
            const int h = h0 + r;
            if (h < H) {
              const int t = b * HW + h * W + w0;
              const int slot = (rgi * 4 + r) * rpt + cgi * TWR;
              if (TWR == 4) {
                float4 o = make_float4(acc[r][0] + bias, acc[r][1] + bias, acc[r][2] + bias, acc[r][3] + bias);
                *reinterpret_cast<float4*>(xrow + t) = o;
                sts[(slot + 0) * 2] += o.x; sts[(slot + 0) * 2 + 1] += o.x * o.x;
                sts[(slot + 1) * 2] += o.y; sts[(slot + 1) * 2 + 1] += o.y * o.y;
                sts[(slot + 2) * 2] += o.z; sts[(slot + 2) * 2 + 1] += o.z * o.z;
                sts[(slot + 3) * 2] += o.w; sts[(slot + 3) * 2 + 1] += o.w * o.w;
              } else {
#pragma unroll
                for (int cc = 0; cc < TWR; cc++) {
                  const float o = acc[r][cc] + bias;
                  xrow[t + cc] = o;
                  sts[(slot + cc) * 2] += o; sts[(slot + cc) * 2 + 1] += o * o;
                }
              }
            }
          }
        }
      } else {
        float acc[TWR];
#pragma unroll
        for (int r = 0; r < TWR; r++) acc[r] = 0.f;
#pragma unroll
        for (int i = 0; i < 7; i++)
#pragma unroll
          for (int j = 0; j < 7; j++) {
            const float wv = wt[i * 7 + j];
#pragma unroll
            for (int r = 0; r < TWR; r++) {
              const int hh = th[r] - 3 + i, ww = tw_[r] - 3 + j;
              if (tv[r] && hh >= 0 && hh < H && ww >= 0 && ww < W)
                acc[r] = fmaf(__ldg(p.x + ((size_t)tb[r] * C + c) * HW + hh * W + ww), wv, acc[r]);
            }
          }
#pragma unroll
        for (int r = 0; r < TWR; r++) {
          if (tv[r]) {
            const float o = acc[r] + bias;
            xrow[tk[r]] = o;
            const int slot = r * 32 + lane;
            sts[slot * 2] += o; sts[slot * 2 + 1] += o * o;
          }
        }
      }
    }
    __syncthreads();
    const int nslots = (MODE == 1) ? rows_per_tile * rpt : 32 * TWR;
    for (int slot = tid; slot < nslots; slot += NT) {
      int t; bool valid;
      if (MODE == 1) {
        const int row = slot / rpt, col = slot - row * rpt;
        const int h = hbi * rows_per_tile + row, w = wbi * 32 * TWR + col;
        valid = (h < H) && (w < W);
        t = b * HW + h * W + w;
      } else {
        t = tok0 + slot; valid = t < M;
      }
      if (valid) {
        float S1 = 0.f, S2 = 0.f;
#pragma unroll
        for (int ws = 0; ws < 8; ws++) { S1 += st_s[(ws * P1_SLOTS + slot) * 2]; S2 += st_s[(ws * P1_SLOTS + slot) * 2 + 1]; }
        float* pp = p.part + ((size_t)cch * Mp + t) * 2;
        pp[0] = S1; pp[1] = S2;
      }
    }
    __syncthreads();
  }
}

__device__ void phase1(const Params& p, float* smem) {
  const int tid = threadIdx.x;
  for (int i = blockIdx.x * NT + tid; i < p.B * p.C4; i += gridDim.x * NT) p.sumsq[i] = 0.f;
  transpose_mat(p.w1, p.C4, p.C, p.w1t, smem);
  transpose_mat(p.w2, p.C, p.C4, p.w2t, smem);
  if (p.p1_mode == 1) {
    if (p.p1_twr == 4) phase1_conv<1, 4>(p, smem); else phase1_conv<1, 1>(p, smem);
  } else {
    if (p.p1_twr == 4) phase1_conv<0, 4>(p, smem);
    else if (p.p1_twr == 2) phase1_conv<0, 2>(p, smem);
    else phase1_conv<0, 1>(p, smem);
  }
}

template <int BM, int BN, int WM, int WN, int KS, int BKc, int PHASE>
__device__ void gemm_phase(const Params& p, float* smem, float* sq_s, float* inv_s, float* red_s, int split) {
  constexpr int WS = WM * WN;
  constexpr int GT = WS * 32;
  constexpr int TM = BM / (WM * 8), TN = BN / (WN * 4);
  constexpr int AM4 = BM / 4, BN4 = BN / 4;
  constexpr int AQ = (BKc * AM4) / GT, BQ = (BKc * BN4) / GT;
  constexpr int STAGE = BKc * (BM + BN);
  constexpr int CS_LD = BM + 4;
  constexpr int NSTEP = NT / BM, NPT = BN / NSTEP;
  static_assert(WS * KS == 8, "8 warps");
  static_assert((BKc * AM4) % GT == 0 && (BKc * BN4) % GT == 0, "quad mapping");
  static_assert(GT % AM4 == 0, "token quad constant per thread");
  static_assert(KS * 2 * STAGE <= GEMM_SMEM_FLOATS && KS * BN * CS_LD <= GEMM_SMEM_FLOATS, "smem");
  static_assert(TM == 8 || TM == 4, "TM");
  static_assert(TN == 8 || TN == 6, "TN");

  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int kg = warp / WS, ws = warp - kg * WS, wm = ws / WN, wn = ws - wm * WN;
  const int lm = lane >> 2, ln = lane & 3;
  const int gt = ws * 32 + lane;
  const int M = p.M, Mp = p.Mp, HW = p.HW;
  const int N = (PHASE == 2) ? p.C4 : p.C;
  const int K = (PHASE == 2) ? p.C : p.C4;
  const float* __restrict__ Ag = (PHASE == 2) ? p.xn : p.hidden;
  const float* __restrict__ Bg = (PHASE == 2) ? p.w1t : p.w2t;
  const int mt = (M + BM - 1) / BM, nt = (N + BN - 1) / BN;
  const int ntiles = mt * nt * split, nk = K / BKc;
  float* As = smem + kg * 2 * STAGE;
  float* Bs = As + 2 * BKc * BM;
  const int row_m = wm * (BM / WM) + lm * TM;
  const int col_n = wn * (BN / WN) + ln * TN;
  const int a_m4 = (gt % AM4) * 4;
  const int a_kq = gt / AM4;
  constexpr int KSTEP_A = GT / AM4;

  for (int tile = blockIdx.x; tile < ntiles; tile += gridDim.x) {
    int rem = tile;
    const int ni = rem % nt; rem /= nt;
    const int mi = rem % mt; const int s = rem / mt;
    const int m0 = mi * BM, n0 = ni * BN;
    const int c0 = (int)(((long)nk * s) / split), c1 = (int)(((long)nk * (s + 1)) / split);
    const int b_first = m0 / HW;
    const int nimg = min(MAXIMG, (min(m0 + BM, M) - 1) / HW - b_first + 1);

    float tc0[4], tc1[4]; int timg[4]; bool tval[4];
#pragma unroll
    for (int e = 0; e < 4; e++) {
      const int t = m0 + a_m4 + e;
      tval[e] = t < M;
      const int tt = tval[e] ? t : 0;
      if (PHASE == 2) {
        float S1 = 0.f, S2 = 0.f;
        for (int ch = 0; ch < p.nch; ch++) {
          const float2 pp = __ldg(reinterpret_cast<const float2*>(p.part + ((size_t)ch * Mp + tt) * 2));
          S1 += pp.x; S2 += pp.y;
        }
        const float mean = S1 / (float)p.C;
        float var = S2 / (float)p.C - mean * mean; var = fmaxf(var, 0.f);
        tc0[e] = mean; tc1[e] = rsqrtf(var + p.ln_eps); timg[e] = 0;
      } else {
        timg[e] = tt / HW - b_first; tc0[e] = 0.f; tc1[e] = 0.f;
      }
    }
    if (PHASE == 3) {
      for (int img = 0; img < nimg; img++) {
        const float* sq = p.sumsq + (size_t)(b_first + img) * K;
        float sacc = 0.f;
        for (int k = tid; k < K; k += NT) sacc += sqrtf(sq[k]);
        sacc = warp_sum(sacc);
        if (lane == 0) red_s[warp] = sacc;
        __syncthreads();
        if (tid == 0) {
          float tot = 0.f;
#pragma unroll
          for (int w2 = 0; w2 < 8; w2++) tot += red_s[w2];
          inv_s[img] = 1.f / (tot / (float)K + p.eps);
        }
        __syncthreads();
      }
#pragma unroll
      for (int e = 0; e < 4; e++) tc1[e] = inv_s[timg[e] < MAXIMG ? timg[e] : 0];
    }

    float acc[TM][TN];
#pragma unroll
    for (int i = 0; i < TM; i++)
#pragma unroll
      for (int j = 0; j < TN; j++) acc[i][j] = 0.f;

    float4 ra[AQ], rb[BQ];
    auto load_chunk = [&](int kc) {
      const int k0 = kc * BKc;
#pragma unroll
      for (int q = 0; q < AQ; q++) {
        const int ka = k0 + q * KSTEP_A + a_kq;
        float4 v = __ldg(reinterpret_cast<const float4*>(Ag + (size_t)ka * Mp + m0 + a_m4));
        float vv[4] = {v.x, v.y, v.z, v.w};
        if (PHASE == 2) {
          const float lw = __ldg(p.lnw + ka), lb = __ldg(p.lnb + ka);
#pragma unroll
          for (int e = 0; e < 4; e++) vv[e] = tval[e] ? fmaf((vv[e] - tc0[e]) * tc1[e], lw, lb) : 0.f;
        } else {
          const float gwk = __ldg(p.gw + ka), gbk = __ldg(p.gb + ka);
#pragma unroll
          for (int e = 0; e < 4; e++) {
            const float sq = __ldg(p.sumsq + (size_t)(b_first + timg[e]) * K + ka);
            vv[e] = tval[e] ? fmaf(vv[e], fmaf(gwk * tc1[e], sqrtf(sq), 1.f), gbk) : 0.f;
          }
        }
        ra[q] = make_float4(vv[0], vv[1], vv[2], vv[3]);
      }
#pragma unroll
      for (int q = 0; q < BQ; q++) {
        const int id = q * GT + gt, kb = id / BN4, n4 = (id - kb * BN4) * 4;
        const int col = n0 + n4;
        rb[q] = (col < N) ? __ldg(reinterpret_cast<const float4*>(Bg + (size_t)(k0 + kb) * N + col))
                          : make_float4(0.f, 0.f, 0.f, 0.f);
      }
    };
    auto store_regs = [&](int buf) {
#pragma unroll
      for (int q = 0; q < AQ; q++)
        *reinterpret_cast<float4*>(As + (buf * BKc + q * KSTEP_A + a_kq) * BM + a_m4) = ra[q];
#pragma unroll
      for (int q = 0; q < BQ; q++) {
        const int id = q * GT + gt, kb = id / BN4, n4 = (id - kb * BN4) * 4;
        *reinterpret_cast<float4*>(Bs + (buf * BKc + kb) * BN + n4) = rb[q];
      }
    };
    auto gsync = [&]() {
      if (KS == 1) __syncthreads();
      else if (GT == 32) __syncwarp();
      else asm volatile("bar.sync %0, %1;" ::"r"(kg + 1), "r"(GT));
    };

    int kc = c0 + kg;
    if (kc < c1) { load_chunk(kc); store_regs(0); }
    gsync();
    for (int it = 0; kc < c1; it++, kc += KS) {
      const int buf = it & 1;
      const bool more = (kc + KS) < c1;
      if (more) load_chunk(kc + KS);
      const float* Ab = As + buf * BKc * BM + row_m;
      const float* Bb = Bs + buf * BKc * BN + col_n;
#pragma unroll 4
      for (int k = 0; k < BKc; k++) {
        float a[TM], b[TN];
        {
          const float4 a0 = *reinterpret_cast<const float4*>(Ab + k * BM);
          a[0] = a0.x; a[1] = a0.y; a[2] = a0.z; a[3] = a0.w;
          if (TM == 8) {
            const float4 a1 = *reinterpret_cast<const float4*>(Ab + k * BM + 4);
            a[4] = a1.x; a[5] = a1.y; a[6] = a1.z; a[7] = a1.w;
          }
        }
        if (TN == 8) {
          const float4 b0 = *reinterpret_cast<const float4*>(Bb + k * BN);
          const float4 b1 = *reinterpret_cast<const float4*>(Bb + k * BN + 4);
          b[0] = b0.x; b[1] = b0.y; b[2] = b0.z; b[3] = b0.w; b[4] = b1.x; b[5] = b1.y; b[6] = b1.z; b[7] = b1.w;
        } else {
          const float2 b0 = *reinterpret_cast<const float2*>(Bb + k * BN);
          const float2 b1 = *reinterpret_cast<const float2*>(Bb + k * BN + 2);
          const float2 b2 = *reinterpret_cast<const float2*>(Bb + k * BN + 4);
          b[0] = b0.x; b[1] = b0.y; b[2] = b1.x; b[3] = b1.y; b[4] = b2.x; b[5] = b2.y;
        }
#pragma unroll
        for (int i = 0; i < TM; i++)
#pragma unroll
          for (int j = 0; j < TN; j++) acc[i][j] = fmaf(a[i], b[j], acc[i][j]);
      }
      if (more) store_regs(buf ^ 1);
      gsync();
    }

    __syncthreads();
    float* Cs = smem;
#pragma unroll
    for (int j = 0; j < TN; j++) {
      float* dst = Cs + (kg * BN + col_n + j) * CS_LD + row_m;
      *reinterpret_cast<float4*>(dst) = make_float4(acc[0][j], acc[1][j], acc[2][j], acc[3][j]);
      if (TM == 8) *reinterpret_cast<float4*>(dst + 4) = make_float4(acc[4][j], acc[5][j], acc[6][j], acc[7][j]);
    }
    __syncthreads();
    const int m_l = tid % BM, n_l0 = tid / BM;
    const int t = m0 + m_l;
    const bool tv = t < M;
    const int img = tv ? t / HW : b_first;
    const int hw = t - img * HW;
    if (PHASE == 2) {
      const int img0 = __shfl_sync(0xffffffffu, img, 0);
      const bool uniform = __all_sync(0xffffffffu, (img == img0) && tv);
      const int il0 = img0 - b_first;
#pragma unroll 4
      for (int i = 0; i < NPT; i++) {
        const int n_l = n_l0 + NSTEP * i, col = n0 + n_l;
        float v = 0.f;
#pragma unroll
        for (int g = 0; g < KS; g++) v += Cs[(g * BN + n_l) * CS_LD + m_l];
        const float gl = gelu_f(v + __ldg(p.b1 + col));
        if (tv) p.hidden[(size_t)col * Mp + t] = gl;
        float g2 = tv ? gl * gl : 0.f;
        if (uniform) {
          g2 = warp_sum(g2);
          if (lane == 0) {
            if (il0 < MAXIMG) atomicAdd(&sq_s[il0 * BN + n_l], g2);
            else atomicAdd(p.sumsq + (size_t)img0 * N + col, g2);
          }
        } else if (tv) {
          const int il = img - b_first;
          if (il < MAXIMG) atomicAdd(&sq_s[il * BN + n_l], g2);
          else atomicAdd(p.sumsq + (size_t)img * N + col, g2);
        }
      }
      __syncthreads();
      for (int idx = tid; idx < nimg * BN; idx += NT) {
        const int im = idx / BN, n = idx - im * BN;
        const float v = sq_s[idx]; sq_s[idx] = 0.f;
        atomicAdd(p.sumsq + (size_t)(b_first + im) * N + n0 + n, v);
      }
      __syncthreads();
    } else {
#pragma unroll 4
      for (int i = 0; i < NPT; i++) {
        const int n_l = n_l0 + NSTEP * i, col = n0 + n_l;
        if (col < N && tv) {
          float v = 0.f;
#pragma unroll
          for (int g = 0; g < KS; g++) v += Cs[(g * BN + n_l) * CS_LD + m_l];
          const size_t idx = ((size_t)img * N + col) * HW + hw;
          if (split == 1) p.out[idx] = v + __ldg(p.b2 + col) + __ldg(p.x + idx);
          else atomicAdd(p.out + idx, v);
        }
      }
      __syncthreads();
    }
  }
}

template <int PHASE>
__device__ __forceinline__ void run_gemm(const Params& p, int cfg, int split, float* smem, float* sq_s, float* inv_s, float* red_s) {
  switch (cfg) {
    case 0: gemm_phase<128, 128, 2, 4, 1, 32, PHASE>(p, smem, sq_s, inv_s, red_s, split); break;
    case 1: gemm_phase<128, 96, 2, 4, 1, 32, PHASE>(p, smem, sq_s, inv_s, red_s, split); break;
    case 2: gemm_phase<64, 64, 2, 2, 2, 32, PHASE>(p, smem, sq_s, inv_s, red_s, split); break;
    default: gemm_phase<32, 32, 1, 1, 8, 16, PHASE>(p, smem, sq_s, inv_s, red_s, split); break;
  }
}

__global__ void __launch_bounds__(NT, 2) convnext_fused(Params p) {
  extern __shared__ float4 smem4[];
  float* smem = reinterpret_cast<float*>(smem4);
  float* sq_s = smem + GEMM_SMEM_FLOATS;
  float* inv_s = sq_s + MAXIMG * 128;
  float* red_s = inv_s + 8;
  for (int i = threadIdx.x; i < MAXIMG * 128; i += NT) sq_s[i] = 0.f;
  __syncthreads();
  cg::grid_group grid = cg::this_grid();

  phase1(p, smem);
  grid.sync();
  if (p.split3 > 1) {
    const long total = (long)p.M * p.C;
    for (long i = (long)blockIdx.x * NT + threadIdx.x; i < total; i += (long)gridDim.x * NT) {
      const int c = (int)((i / p.HW) % p.C);
      p.out[i] = p.x[i] + p.b2[c];
    }
  }
  run_gemm<2>(p, p.cfg2, 1, smem, sq_s, inv_s, red_s);
  grid.sync();
  run_gemm<3>(p, p.cfg3, p.split3, smem, sq_s, inv_s, red_s);
}

struct Pick { int cfg; int split; };
static Pick pick_cfg(long M, long N, long K, int grid, bool allow_split) {
  struct CfgD { int id, BM, BN, BK; double eff; };
  const CfgD cfgs[4] = {{0, 128, 128, 32, 0.80}, {1, 128, 96, 32, 0.78}, {2, 64, 64, 32, 0.60}, {3, 32, 32, 16, 0.40}};
  double best = 1e300; Pick r{0, 1};
  for (int ci = 0; ci < 4; ci++) {
    const CfgD& c = cfgs[ci];
    if (c.id == 1 && (N % 96) != 0) continue;
    const long tiles = ((M + c.BM - 1) / c.BM) * ((N + c.BN - 1) / c.BN);
    const long nk = K / c.BK;
    for (int S = 1; S <= (allow_split ? 16 : 1); S *= 2) {
      if (S > 1 && nk / S < 2) break;
      const long total = tiles * S;
      const long waves = (total + grid - 1) / grid;
      double cost = (double)c.BM * c.BN * ((double)K / S) / c.eff * (double)waves;
      if (S > 1) cost *= 1.08 + 0.03 * S;
      if (cost < best) { best = cost; r.cfg = c.id; r.split = S; }
    }
  }
  return r;
}

extern "C" const char* convnext_launch(Params p, void* stream_v) {
  static int grid_size = 0;
  cudaError_t e;
  if (grid_size == 0) {
    e = cudaFuncSetAttribute(convnext_fused, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_BYTES);
    if (e != cudaSuccess) return cudaGetErrorString(e);
    int dev = 0; cudaGetDevice(&dev);
    int sms = 0; cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
    int bps = 0;
    e = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&bps, convnext_fused, NT, SMEM_BYTES);
    if (e != cudaSuccess) return cudaGetErrorString(e);
    if (bps < 1) bps = 1;
    grid_size = sms * bps;
  }
  const Pick k2 = pick_cfg(p.M, p.C4, p.C, grid_size, false);
  const Pick k3 = pick_cfg(p.M, p.C, p.C4, grid_size, true);
  p.cfg2 = k2.cfg; p.cfg3 = k3.cfg; p.split3 = k3.split;
  {
    const int tw = (p.W % 4 == 0) ? 4 : 1;
    const int cgt = p.W / tw;
    int lpr, rg, wb;
    if (cgt >= 32) { lpr = 32; rg = 1; wb = (cgt + 31) / 32; }
    else { lpr = cgt; rg = 32 / cgt; wb = 1; }
    if (rg > 8) rg = 8;
    const int rows = 4 * rg;
    if (rows <= p.H) {
      p.p1_mode = 1; p.p1_twr = tw; p.p1_lpr = lpr; p.p1_rg = rg; p.p1_wb = wb;
      p.p1_hb = (p.H + rows - 1) / rows;
      p.p1_tiles = p.B * p.nch * p.p1_hb * p.p1_wb;
    } else {
      p.p1_mode = 0; p.p1_lpr = 0; p.p1_rg = 0; p.p1_wb = 0; p.p1_hb = 0;
      int r0 = 4;
      while (r0 > 1 && ((p.M + 32 * r0 - 1) / (32 * r0)) * p.nch < grid_size) r0 >>= 1;
      p.p1_twr = r0;
      p.p1_tiles = ((p.M + 32 * r0 - 1) / (32 * r0)) * p.nch;
    }
  }
  void* args[] = {&p};
  e = cudaLaunchCooperativeKernel((const void*)convnext_fused, dim3(grid_size), dim3(NT), args, (size_t)SMEM_BYTES,
                                  (cudaStream_t)stream_v);
  if (e != cudaSuccess) return cudaGetErrorString(e);
  return nullptr;
}
```

```yaml design-card
id: r10-g-e2-g2-coop
parents:
  - r9-g-e3-tc05-p1s
operation: structural_mutation
language: cuda_cpp
niche:
  lin: "G:E2:H42"
  mem: tma-tensor
  st: direct
  grid: oneshot
  launch: split
  tile: token
  red: cta
  cache: default
  spec: dispatch:size
hypothesis: >-
  At 8,320 and 4,96, K3 plus the cuBLASLt GEMM2 (20.2 / 12.2 us rented) are replaced by one cooperative tcgen05 launch,
  g_g2, so these rows run 3 launches instead of 4. Phase P computes s_b and writes A_b = fp16(W2*s_b) with K3's
  arithmetic, and the image-0 CTAs write b2' in fp32. The h B-stages are TMA-prefetched before one flat
  self-resetting grid barrier. After it, a 4-stage TMA ring (A 16 KB + B 32 KB) feeds 1SM 128x256x16 MMAs into one
  TMEM accumulator, 96 units at 8,320 and 56 at 4,96. The epilogue writes out = x + acc + b2' in coalesced NCHW rows,
  with x prefetched into registers. The task's in-place B scale was dropped because it fails precision (findings).
  Estimate: g_g2 ~10-11 us at 8,320 and ~6 us at 4,96, so rows ~49-50 / ~31 us rented. UNTESTED: the B200 backend
  was down for the whole session; the code has never been compiled.
expected_effect:
  S:
    pct: 0
    confidence: high
  M:
    pct: -3
    confidence: low
  L:
    pct: -3
    confidence: low
resources_sm100a:
  regs_per_thread: 128
  smem_per_cta_bytes: 203000
  threads_per_cta: 512
  ctas_per_sm: 1
  bytes_in_flight_per_sm: 196608
  launches_per_call: 3
knobs:
  g2_8_320: "g_g2<320,256,4>: 96 units (3 c-tiles x 8 images x 4 chunks of 256 tokens; c-tile 3 is half zero), KB=20 stages of 64 k, TMEM 256 cols, 148 CTAs cooperative (52 do phase P only)"
  g2_4_96: "g_g2<96,224,4> (56 units, default g2mode 1) or g_g2<96,112,6> (112 units, g2mode 2): time both, pick in binding"
  phaseP: "CTA j < (148/B)*B handles image j%B, rows c = j/B + (148/B)*m, one row per warp, W2 row prefetched before the s_b reduce"
  barrier: "module-level g2_bar_cnt/g2_bar_gen, atom.acq_rel arrive, last arriver resets the count and st.release bumps gen; waiters spin on ld.acquire (no nanosleep)"
  roles: "warp 0 TMA producer, warp 1 MMA issuer (elect.sync issue only, whole-warp loops), all 16 warps epilogue (2 items of 32x32 each at N=256)"
dispatch:
  - {shape: "64,96,112,112", kernel: "K0 + g_pass1<96,...> + g_grn + cublasLt (unchanged)"}
  - {shape: "32,128,64,64", kernel: "K0 + g_pass1<128,...> + g_grn + cublasLt (unchanged)"}
  - {shape: "8,320,32,32", kernel: "K0 + g_tc<320,2,32,64,9> + g_g2<320,256,4>"}
  - {shape: "4,96,56,56", kernel: "K0 + g_tc<96,2,56,128,6> + g_g2<96,224,4>"}
  - {shape: "all others", kernel: "r0 anchor convnext_fused (fallback only; do not merge)"}
tests:
  - H42
findings:
  - "PRECISION SIM (torch fp64 emulation of the shipped roundings: d16, LN stats of d16, fp16(lnw*W1), u = rowsum of fp16 W1', exact GELU, h16, sumsq from unrounded g; harness generator rules; seeds 0-3 via torch.Generator, not the harness seed 200). 8,320 (atol 2.4e-3): P0 current fp16(W2*s_b).h16 max 1.77-1.91e-3 (74-80 %), 0 beyond; P1 task design fp16(W2).fp16(h16*s) max 2.09-2.44e-3 (87-101 %), 1 element beyond atol at seed 0; P2 fp16(fp16(W2)*s).h16 2.02-2.33e-3 (84-97 %), 0 beyond; P3 hi/lo B split 1.81-1.99e-3 (75-83 %). 4,96 (atol 2.8e-3): P0 61-70 %, P1 73-78 %, P2 65-76 %, P3 62-68 %, all 0 beyond. P1 is refuted at 8,320 and P2 has no margin; only P0 or a 2-MMA hi/lo split meet the 0-beyond bar [probe_b200]"
  - "Design consequence: the B-tile in-place scale (and HFMA2 hi/lo-s variants of it, which give the same fp16(h16*s)) is dead at 8,320. A cheaper fix than hi/lo-B (2x MMA = ~20k clk per unit at N=256): build A_b = fp16(W2*s_b) once per image inside g_g2 (3.3M elements over 148 CTAs, ~22K per CTA) before one flat grid barrier. This keeps P0 roundings exactly (b2' now fp32 instead of hi/lo fp16) and makes the mainloop a pure TMA->MMA ring with no in-ring work [INFERRED]"
  - "Rejected alternatives (arithmetic only): per-stage A build from fp32 W2 by producer warps needs 64 KB of L2->SM per 64-k stage (125 B/clk at the N=256 MMA ideal, above the 68-92 B/clk measured in H31) and 256 regs/lane per warp-owned stage; per-CTA private A build is 655 KB fp32 per CTA (~4 us); tokens-as-M orientation needs N=160x2 MMAs at the 105-clk floor (16.8k clk per 128-token unit) [INFERRED]"
  - "INFRA: the B200 backend (Modal function) was stopped for the whole session after the first probe: compile_b200 and 5 probe_b200 calls all returned 'function is stopped'. Nothing in g_g2 was compiled, validated or timed. Desk check found and fixed one PTX error (atom.add.acq_rel.gpu.global.u32 -> atom.acq_rel.gpu.global.add.u32) [compile_b200, probe_b200]"
  - "OWED before any merge: compile_b200 (regs <= 128, STACK 0, UTMALDG/UTCHMMA present); standalone g_g2 vs torch (x + W2_b.h + b2') at both rows; harness_time of the 3-launch vs 4-launch pipeline interleaved x3 at 8,320 / 4,96; g_g2 cold alone (target <= 9 / <= 4.5 us); G2_STAMPS=1 build to report clk per 128x256x16 instruction (stp[4]-stp[6] over 4*KB) for the audit; N=224 vs N=112 at 4,96; run_tests 16/16 with portal-order re-runs; 10-round fp64 precision log (P0 class, expected 74-82 % at 8,320) [INFERRED]"
  - "Risks to check first: TMA box rows (128) larger than the A map's per-image rows at C=96 (the 2-D map spans B*C rows, so only the last image hits OOB fill; the junk rows are masked in the epilogue); mbarrier tx counts for partially OOB B boxes at the image end of 4,96 (TMA counts full box bytes); cudaGetDriverEntryPointByVersion availability on the portal's CUDA 13.1 [INFERRED]"
paths:
  - {max_tokens: 8192, lang: cuda, width: 128, threads: 512, rows: 2, grid: oneshot, mem: tma, launches: 3, x: none, w: none, st: none, cluster: 1}
  - {max_tokens: 12544, lang: cuda, width: 128, threads: 512, rows: 2, grid: oneshot, mem: tma, launches: 3, x: none, w: none, st: none, cluster: 1}
  - {max_tokens: 131072, lang: cuda, width: 128, threads: 512, rows: 2, grid: persistent, mem: ldg, launches: 4, x: none, w: none, st: none, cluster: 1}
  - {max_tokens: null, lang: cuda, width: 128, threads: 512, rows: 2, grid: persistent, mem: ldg, launches: 4, x: none, w: none, st: none, cluster: 1}
runs_on:
  H200:
    runs: false
    representative: false
    note: "tcgen05 / TMEM: sm_100a only (target_hardware B200)"
  A100:
    runs: false
    representative: false
  L40S:
    runs: false
    representative: false
risks:
  - "UNTESTED: never compiled or run (backend down all session); must pass compile_b200, a torch check and run_tests 16/16 before any merge"
  - "Merge-only candidate: 12 rows run the slow r0 anchor; only 8,320 and 4,96 change vs the parent (64,96 / 32,128 code paths identical)"
  - "Cooperative launch of 148 CTAs needs 1 CTA/SM residency (203 KB smem); the grid barrier counter is module-level state, self-reset by the last arriver"
  - "4,96 default uses 56 units of N=224 (56 SMs busy); the 112-unit N=112 variant may be faster and is one g2mode switch away"
  - "fp32 GEMM accumulation order differs from cuBLASLt; roundings equal the logged P0 class, but the 10-round log must be redone on the built kernel"
measure_first:
  - B200
compliance:
  single_stream: true
  no_cross_call_state: true
  fp32_math: true
  no_threads_or_fork: true
  no_precompiled_binaries: true
```