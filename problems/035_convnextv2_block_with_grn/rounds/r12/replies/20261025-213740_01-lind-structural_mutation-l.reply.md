### Rationale
I built H49 in `tc_kernel`. The GEMM1 epilogue computes fp32 h and keeps it in the tile's TMEM columns with `tcgen05.st`, so the fp32 h costs no registers (STACK stays 0). It then publishes the tile's gf sums on a per-image-group counter and waits for the rest of the group. Finally it stores h' = fp16(h·(1+gw·gf·inv_b)) once. GEMM2 now runs one accumulator over h' alone, the g tensor is gone, and 16,768 moves to KS 6 / MP 2.

All 16 rows pass. Against r11-merge-b on the same GPU, 32,768 is -8.8 % (-9.0 % in an earlier run) and 16,768 is -3.1 %. The other wide-C rows tie or are up to 3 % slower.

The GEMM2 phase drops sharply: 11.3 → 6.7 µs at 32,768. But the image wait costs ~2 µs per tile and the extra h' pass costs 1–2.5 µs, which together cancel the removed g pass at B ≤ 16. The ≥10 % success bar is missed. The refutation clause does not fire, because 32,768 is −9 % and the image-wait median is under 2.5 µs.

This candidate carries only the wide-C rows. Every other row runs the r0 anchor and must come from r11-merge-b in the merge.

```json solution-spec
{"name": "r12-d-e2-grn-epi-counters", "definition": "035_convnextv2_block_with_grn", "author": "solx-loop", "spec": {"languages": ["cuda_cpp"], "target_hardware": ["B200"], "entry_point": "binding.cpp::run", "dependencies": ["torch"], "destination_passing_style": true, "compile_options": {"cuda_cflags": ["-O3", "--use_fast_math", "-std=c++17"], "ld_flags": ["-lcuda"]}}}
```

```cuda file=kernel.cu
// ConvNeXtV2 block with GRN (r12 lineage D, E2 / H49).
// Wide-C rows (2/8/16/32,768,7,7 and 8/16,384,14,14): cooperative kernel tc_kernel with a tcgen05 kind::f16 swap-AB
//   engine (A = fp16 weight tile, 128 rows, K-major; B = fp16 activations [channel][token], MN-major; TMA 128B-swizzled
//   ring; whole-warp TMA producer and MMA warp; double-buffered TMEM accumulators; 512 threads).
//   r12 (H49): the GRN scale is applied inside the GEMM1 epilogue behind per-image-group counters, so no g tensor and
//   no second GEMM2 accumulator exist; GEMM2 runs one accumulator over h' = fp16(h * (1 + gw*gf*inv_b)).
// Other rows: the r0 fp32 SIMT cooperative anchor (convnext_fused). This candidate carries only the wide-C rows; the
//   sc_kernel / tc2_kernel / G paths of r11-merge-b are unchanged there and are left to the merge.
#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cooperative_groups.h>
#include <stdint.h>
#include <string.h>
namespace cg = cooperative_groups;

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}

#ifndef TC_ONLY
// =====================================================================================================
// r0 anchor path (unchanged)
// =====================================================================================================
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
#endif  // TC_ONLY

// =====================================================================================================
// tcgen05 swap-AB cooperative path for the wide-C rows (512 threads, 16 warps)
// Phase A (unchanged from r11): plane loads (+ LN vectors, dwb/b2 slices) by bulk copy at entry, W1 loads in flight
//   during the depthwise conv, LN partials, fp16 d, out = x + b2, then the W1 conversion. Sync 1 (flat barrier).
// Phase B (r12, H49): GEMM1 tiles = 128 hidden rows x NI whole images, image-major (tile t = hidden tile t % HT of
//   image group t / HT; a CTA's tiles t, t + P, ... are in increasing groups). Epilogue on all 16 warps (warps 4-15 on
//   every tile, warps 0-3 join the CTA's last tile). Pass 1: LN fold + A&S erfc GELU in fp32, per-(image,k) sumsq of
//   the fp32 h, and h written back into the tile's own TMEM columns (tcgen05.st). gf = sqrt(sumsq); sum_k gf is
//   red.add-ed into gsum[b] (zeroed in phase A). Then bar.sync and one thread release-adds the group's counter
//   (red.release.gpu), spins with ld.acquire until all HT tiles of the group have published, computes
//   inv_b = 1 / (mean_k gf + eps) and counts its departure (the last departing tile of the group resets both cells).
//   A tile publishes before it waits, and its CTA's earlier tiles belong to earlier groups, so with every CTA
//   co-resident (cooperative launch) the wait graph has no cycle. Pass 2: h re-read from TMEM, h' = fp16(h * (1 +
//   gw*gf*inv_b)) stored once through the per-warp staged coalesced store. The TMEM buffer is released after pass 2;
//   the MMA warp issues the CTA's next tile into the other buffer meanwhile. No g tensor. Sync 2.
// Phase C: GEMM2 split-K, out += W2h x h' (+ W2.gb on the first k-split): one accumulator per 128-channel tile, two
//   TMEM buffers, MP tiles per unit share each landed B stage, up to 8 stages; drain 7x7 by whole-image bulk reduce,
//   14x14 by float4 red.add, into the x + b2 prefill.
// TC_STAMPS: globaltimer diagnostics (off in the shipped build; the timed path writes no stamp buffer).
// =====================================================================================================
#ifndef TC_STAMPS
#define TC_STAMPS 0
#endif
#define TC_NT 512
#define TC_RING (192 * 1024)
#define TC_STGB 2560
#define TC_OFF_STG TC_RING
#define TC_OFF_MS (TC_OFF_STG + 8 * TC_STGB)
#define TC_OFF_SQ (TC_OFF_MS + 4096)
#define TC_OFF_INV (TC_OFF_SQ + 4096)
#define TC_OFF_CB (TC_OFF_INV + 512)
#define TC_OFF_BAR (TC_OFF_CB + 512)
#define TC_NBARS 34
#define TC_SMEM (TC_OFF_BAR + (TC_NBARS + 1) * 8 + 1024)
#define TC_NS2MAX 8
// phase-C bulk-reduce staging tile [128 ch][49] fp32 (7x7 units that are not the CTA's last): after the ring, over
// buffers used only in phase B; the last unit stages its tiles in the (then free) ring
#define TC_OFF_RS TC_RING
static_assert(TC_OFF_RS + 128 * 49 * 4 <= TC_OFF_INV, "bulk staging overlaps inv/cb");
// phase-B buffers of the all-warp epilogue inside the ring (GEMM1 stages use <= 160 KB): per-warp staging, sumsq parts
#define TC_B_STG (160 * 1024)
#define TC_B_STGW 1536
#define TC_B_SQ (TC_B_STG + 16 * TC_B_STGW)
// phase-A small-vector area inside the ring: LN weight/bias (2*C floats), per-unit dwb and b2 slices
#define TC_PA_LN (116 * 1024)
#define TC_PA_PAR (126 * 1024)
#define TC_PA_MAX (116 * 1024)
struct TCP {
  CUtensorMap mW1, mW2, mD, mH;
  const float *x, *dww, *dwb, *lnw, *lnb, *w1, *b1, *gw, *gb, *w2, *b2;
  float* out;
  __half *d16, *w1h, *w2h, *hid;
  float *part, *uk, *vk, *wgb, *gsum;
  unsigned long long* st;
  int B, Mcols, units, NI, T1, STG1, NST1, KS, KB2, U2, NI2, NAT2, MP, NS2;
  float eps, ln_eps;
};
__device__ __forceinline__ unsigned long long gtime() { unsigned long long g; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(g)); return g; }
#if TC_STAMPS
#define STAMP(i) do { p.st[cta * 24 + (i)] = gtime(); } while (0)
#else
#define STAMP(i) do {} while (0)
#endif
__device__ __forceinline__ uint32_t sa(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
__device__ __forceinline__ void mb_init(uint64_t* b, uint32_t c) { asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(sa(b)), "r"(c)); }
__device__ __forceinline__ void mb_expect(uint64_t* b, uint32_t tx) { asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(sa(b)), "r"(tx) : "memory"); }
__device__ __forceinline__ void mb_arrive1(uint64_t* b) { asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(sa(b)) : "memory"); }
__device__ __forceinline__ void mb_wait(uint64_t* b, uint32_t ph) {
  asm volatile("{\n .reg .pred p;\n WAIT_%=: mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n @!p bra WAIT_%=;\n}" ::"r"(sa(b)), "r"(ph) : "memory"); }
__device__ __forceinline__ void bulk_g2s(void* dst, const void* src, uint32_t bytes, uint64_t* b) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];" ::"r"(sa(dst)), "l"(src), "r"(bytes), "r"(sa(b)) : "memory"); }
__device__ __forceinline__ void bulk_red_add(float* dst, const void* src, uint32_t bytes) {
  asm volatile("cp.reduce.async.bulk.global.shared::cta.bulk_group.add.f32 [%0], [%1], %2;" ::"l"(dst), "r"(sa(src)), "r"(bytes) : "memory");
  asm volatile("cp.async.bulk.commit_group;" ::: "memory"); }
__device__ __forceinline__ float4 ldnc4(const float4* ptr) {
  float4 r;
  asm volatile("ld.global.nc.v4.f32 {%0,%1,%2,%3}, [%4];" : "=f"(r.x), "=f"(r.y), "=f"(r.z), "=f"(r.w) : "l"(ptr));
  return r;
}
__device__ __forceinline__ void tma2d(void* dst, const CUtensorMap* m, int x, int y, uint64_t* b) {
  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];" ::"r"(sa(dst)), "l"((uint64_t)m), "r"(x), "r"(y), "r"(sa(b)) : "memory"); }
__device__ __forceinline__ bool elect1() { uint32_t r; asm volatile("{\n.reg .pred p;\n.reg .b32 t;\nelect.sync t|p, 0xffffffff;\nselp.u32 %0, 1, 0, p;\n}" : "=r"(r)); return r != 0; }
__device__ __forceinline__ uint64_t sdesc(uint32_t addr, uint32_t lbo, uint32_t sbo) {
  return (uint64_t)((addr & 0x3FFFFu) >> 4) | ((uint64_t)((lbo >> 4) & 0x3FFFu) << 16) | ((uint64_t)((sbo >> 4) & 0x3FFFu) << 32) | (1ull << 46) | (2ull << 61); }
__device__ __forceinline__ void umma(uint32_t d, uint64_t a, uint64_t b, uint32_t idesc, uint32_t acc) {
  asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %4, 0;\ntcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n}" ::"r"(d), "l"(a), "l"(b), "r"(idesc), "r"(acc)); }
__device__ __forceinline__ void umma_commit(uint64_t* b) { asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];" ::"r"(sa(b)) : "memory"); }
__device__ __forceinline__ void tfb() { asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory"); }
__device__ __forceinline__ void tfa() { asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory"); }
__device__ __forceinline__ void tld32(uint32_t a, uint32_t (&v)[32]) {
  asm volatile("tcgen05.ld.sync.aligned.32x32b.x32.b32 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, [%32];"
      : "=r"(v[0]), "=r"(v[1]), "=r"(v[2]), "=r"(v[3]), "=r"(v[4]), "=r"(v[5]), "=r"(v[6]), "=r"(v[7]), "=r"(v[8]), "=r"(v[9]),
        "=r"(v[10]), "=r"(v[11]), "=r"(v[12]), "=r"(v[13]), "=r"(v[14]), "=r"(v[15]), "=r"(v[16]), "=r"(v[17]), "=r"(v[18]),
        "=r"(v[19]), "=r"(v[20]), "=r"(v[21]), "=r"(v[22]), "=r"(v[23]), "=r"(v[24]), "=r"(v[25]), "=r"(v[26]), "=r"(v[27]),
        "=r"(v[28]), "=r"(v[29]), "=r"(v[30]), "=r"(v[31]) : "r"(a));
  asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory"); }
__device__ __forceinline__ void tld16(uint32_t a, uint32_t (&v)[16]) {
  asm volatile("tcgen05.ld.sync.aligned.32x32b.x16.b32 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, [%16];"
      : "=r"(v[0]), "=r"(v[1]), "=r"(v[2]), "=r"(v[3]), "=r"(v[4]), "=r"(v[5]), "=r"(v[6]), "=r"(v[7]), "=r"(v[8]), "=r"(v[9]),
        "=r"(v[10]), "=r"(v[11]), "=r"(v[12]), "=r"(v[13]), "=r"(v[14]), "=r"(v[15]) : "r"(a));
  asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory"); }
// 16 fp32 columns of this lane's TMEM row back into the accumulator (h kept on chip across the image wait)
__device__ __forceinline__ void tst16(uint32_t a, const uint32_t (&v)[16]) {
  asm volatile("tcgen05.st.sync.aligned.32x32b.x16.b32 [%0], {%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16};"
      ::"r"(a), "r"(v[0]), "r"(v[1]), "r"(v[2]), "r"(v[3]), "r"(v[4]), "r"(v[5]), "r"(v[6]), "r"(v[7]), "r"(v[8]), "r"(v[9]),
        "r"(v[10]), "r"(v[11]), "r"(v[12]), "r"(v[13]), "r"(v[14]), "r"(v[15]) : "memory"); }
__device__ __forceinline__ void tst_wait() { asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory"); }
__device__ __forceinline__ void ebar() { asm volatile("bar.sync 1, 256;" ::: "memory"); }
__device__ __forceinline__ void nbar(int id, int n) { asm volatile("bar.sync %0, %1;" ::"r"(id), "r"(n) : "memory"); }

// Flat self-resetting grid barrier (r4 B microbench form): one arrival counter and one generation word in module-level
// device memory (zero at module load, never in the torch workspace); all CTAs are co-resident (cooperative launch).
// Callers run fence.proxy.async before it and the TMA warps run fence.proxy.async after it.
__device__ unsigned int g_bar_cnt[32];
__device__ unsigned int g_bar_gen[32];
__device__ __forceinline__ void gbar(unsigned int nctas) {
  __syncthreads();
  if (threadIdx.x == 0) {
    unsigned int gen, old;
    asm volatile("ld.acquire.gpu.u32 %0, [%1];" : "=r"(gen) : "l"(g_bar_gen) : "memory");
    asm volatile("atom.acq_rel.gpu.add.u32 %0, [%1], 1;" : "=r"(old) : "l"(g_bar_cnt) : "memory");
    if (old == nctas - 1u) {
      asm volatile("st.relaxed.gpu.u32 [%0], 0;" ::"l"(g_bar_cnt) : "memory");
      asm volatile("st.release.gpu.u32 [%0], %1;" ::"l"(g_bar_gen), "r"(gen + 1u) : "memory");
    } else {
      unsigned int cur;
      do {
        asm volatile("ld.acquire.gpu.u32 %0, [%1];" : "=r"(cur) : "l"(g_bar_gen) : "memory");
      } while (cur == gen);
    }
  }
  __syncthreads();
}
// r12 per-image-group counters (GEMM1 tiles of one image group publish here), one 128-B line per group, module-level
// device memory (zero at module load), reset by the last departing tile of the group within the same call
#define TC_MAXGRP 64
__device__ unsigned int g_img_cnt[TC_MAXGRP * 32];
__device__ unsigned int g_img_dep[TC_MAXGRP * 32];

__device__ __forceinline__ uint32_t h2u(__half2 h) { return *reinterpret_cast<uint32_t*>(&h); }
// exact-erf GELU v*Phi(v), Phi from Abramowitz-Stegun 7.1.26 (|erf error| <= 1.5e-7): one MUFU.RCP + one MUFU.EX2
__device__ __forceinline__ float gelu_as(float v) {
  const float a = fabsf(v) * 0.70710678118654752f;
  const float t = __fdividef(1.f, fmaf(0.3275911f, a, 1.f));
  float y = fmaf(1.061405429f, t, -1.453152027f);
  y = fmaf(y, t, 1.421413741f);
  y = fmaf(y, t, -0.284496736f);
  y = fmaf(y, t, 0.254829592f);
  const float hc = 0.5f * t * y * __expf(-a * a);
  return (v >= 0.f) ? fmaf(-v, hc, v) : v * hc;
}
// 16 fp16 columns of 32 rows (one per lane) -> per-warp smem transpose -> 2 lanes per 32-B row segment
__device__ __forceinline__ void tc_stage16(uint32_t* stg, const uint32_t (&w)[8], __half* base, int pitch, int ch, int vcols, int lane) {
  uint4* srow = reinterpret_cast<uint4*>(stg + lane * 12);
  srow[0] = make_uint4(w[0], w[1], w[2], w[3]);
  srow[1] = make_uint4(w[4], w[5], w[6], w[7]);
  __syncwarp();
#pragma unroll
  for (int i2 = 0; i2 < 2; i2++) {
    const int row = i2 * 16 + (lane >> 1), colb = ch * 16 + (lane & 1) * 8;
    const uint4 val = *reinterpret_cast<const uint4*>(stg + row * 12 + (lane & 1) * 4);
    if (colb < vcols) *reinterpret_cast<uint4*>(base + (size_t)row * pitch + colb) = val;
  }
  __syncwarp();
}

// W2 -> fp16 conversion with the per-quarter W2.gb partial (deferred out of phase A); unchanged from r10/r11
template <int C_, class PT>
__device__ __forceinline__ void tc_w2_work(const PT& p, int wid, int nwk, int lane) {
  constexpr int NQ = C_ / 128, C4 = 4 * C_, MW = (C_ >= 768) ? 2 : 1;
  const int qtr = wid & 3;
  float4 v[MW][NQ];
#pragma unroll
  for (int k = 0; k < MW; k++) {
    const int u = wid + k * nwk;
    if (u < C4) {
      const float4* src = reinterpret_cast<const float4*>(p.w2 + (size_t)(u >> 2) * C4 + qtr * C_);
#pragma unroll
      for (int j = 0; j < NQ; j++) v[k][j] = __ldg(src + lane + 32 * j);
    }
  }
  const float4* gb4 = reinterpret_cast<const float4*>(p.gb + qtr * C_);
#pragma unroll
  for (int k = 0; k < MW; k++) {
    const int u = wid + k * nwk;
    if (u < C4) {
      const int row = u >> 2;
      uint2* dst = reinterpret_cast<uint2*>(p.w2h + (size_t)row * C4 + qtr * C_);
      float sg = 0.f;
#pragma unroll
      for (int j = 0; j < NQ; j++) {
        const int c4 = lane + 32 * j;
        const float4 g4 = __ldg(gb4 + c4);
        sg += v[k][j].x * g4.x + v[k][j].y * g4.y + v[k][j].z * g4.z + v[k][j].w * g4.w;
        uint2 o; o.x = h2u(__floats2half2_rn(v[k][j].x, v[k][j].y)); o.y = h2u(__floats2half2_rn(v[k][j].z, v[k][j].w));
        dst[c4] = o;
      }
      sg = warp_sum(sg);
      if (lane == 0) p.wgb[row * 4 + qtr] = sg;
    }
  }
}

template <int C_, int HW_>
__global__ void __launch_bounds__(TC_NT, 1) tc_kernel(const __grid_constant__ TCP p) {
  constexpr int W_ = (HW_ == 49) ? 7 : 14, H_ = W_;
  constexpr int C4 = 4 * C_, G = C_ / 32, HT = C4 / 128, CT = C_ / 128, KB1 = C_ / 64;
  constexpr int HWS = (HW_ + 7) / 8 * 8;
  extern __shared__ __align__(1024) unsigned char tc_raw[];
  unsigned char* sm = tc_raw + ((1024u - (sa(tc_raw) & 1023u)) & 1023u);
  unsigned char* ring = sm;
  float* inv_s = reinterpret_cast<float*>(sm + TC_OFF_INV);
  int* cb_s = reinterpret_cast<int*>(sm + TC_OFF_CB);
  uint64_t* bars = reinterpret_cast<uint64_t*>(sm + TC_OFF_BAR);
  uint64_t *full1 = bars, *empty1 = bars + 4, *tf1 = bars + 8, *te1 = bars + 10;
  uint64_t *full2 = bars + 12, *empty2 = bars + 20, *tf2 = bars + 28, *te2 = bars + 30, *cbar = bars + 32, *sbar = bars + 33;
  uint32_t* tslot = reinterpret_cast<uint32_t*>(bars + TC_NBARS);
  const int tid = threadIdx.x, warp = __shfl_sync(0xffffffffu, tid >> 5, 0), lane = tid & 31;
  const int cta = blockIdx.x, P = gridDim.x;
  if (tid == 0) {
    for (int i = 0; i < TC_NBARS; i++) mb_init(bars + i, (i == 30 || i == 31) ? 8u : 1u);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  if (warp == 2) {
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], 512;" ::"r"(sa(tslot)));
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
  }
  tfb(); __syncthreads(); tfa();
  const uint32_t tbase = *tslot;
  if (tid == 0) STAMP(0);

  // ---------------- phase A: plane loads, W1 loads in flight, depthwise conv, LN partials, fp16 d, out = x + b2, W1 conversion
  const int nu = (cta < p.units) ? (p.units - cta + P - 1) / P : 0;
  float* xs = reinterpret_cast<float*>(ring);
  float* wsm = xs + nu * 32 * HW_;
  float* dbuf = wsm + nu * 32 * 49;
  float* lnw_s = reinterpret_cast<float*>(ring + TC_PA_LN);
  float* lnb_s = lnw_s + C_;
  float* dwb_s = reinterpret_cast<float*>(ring + TC_PA_PAR);
  float* b2_s = dwb_s + 256;
  if (tid == 0) {
    mb_expect(sbar, 2 * C_ * 4);
    bulk_g2s(lnw_s, p.lnw, C_ * 4, sbar);
    bulk_g2s(lnb_s, p.lnb, C_ * 4, sbar);
    if (nu > 0) {
      mb_expect(cbar, nu * (32 * HW_ * 4 + 32 * 49 * 4 + 2 * 32 * 4));
      for (int i = 0; i < nu; i++) {
        const int u = cta + i * P, b = u / G, g = u - b * G;
        bulk_g2s(xs + i * 32 * HW_, p.x + ((size_t)b * C_ + g * 32) * HW_, 32 * HW_ * 4, cbar);
        bulk_g2s(wsm + i * 32 * 49, p.dww + (size_t)g * 32 * 49, 32 * 49 * 4, cbar);
        bulk_g2s(dwb_s + i * 32, p.dwb + g * 32, 128, cbar);
        bulk_g2s(b2_s + i * 32, p.b2 + g * 32, 128, cbar);
      }
    }
  }
  if (cta == 0 && tid < p.B) p.gsum[tid] = 0.f;
  constexpr int NQ1 = C_ / 128, MW1 = (C_ >= 768) ? 2 : 1;
  const int nw1 = P * (TC_NT / 32), wb1 = (P - 1 - cta) * (TC_NT / 32) + warp;
  float4 w1v[MW1][NQ1];
#pragma unroll
  for (int k = 0; k < MW1; k++) {
    const int wu = wb1 + k * nw1;
    if (wu < C4) {
      const float4* src = reinterpret_cast<const float4*>(p.w1 + (size_t)wu * C_);
#pragma unroll
      for (int j = 0; j < NQ1; j++) w1v[k][j] = ldnc4(src + lane + 32 * j);
    }
  }
  if (nu > 0) {
    mb_wait(cbar, 0);
    constexpr int SEG = W_ / 7, per_plane = H_ * SEG;
    const int items = nu * 32 * per_plane;
    for (int itm = tid; itm < items; itm += TC_NT) {
      const int pl = itm / per_plane;
      const int rem = itm - pl * per_plane;
      const int h = rem / SEG, w0 = (rem - h * SEG) * 7;
      const int ui = pl >> 5, pp = pl & 31;
      const float* xp = xs + pl * HW_;
      const float* wp = wsm + pl * 49;
      float acc[7];
#pragma unroll
      for (int o = 0; o < 7; o++) acc[o] = 0.f;
#pragma unroll
      for (int i = 0; i < 7; i++) {
        const int hh = h + i - 3;
        if (hh >= 0 && hh < H_) {
          float v[13];
#pragma unroll
          for (int jj = 0; jj < 13; jj++) { const int col = w0 - 3 + jj; v[jj] = (col >= 0 && col < W_) ? xp[hh * W_ + col] : 0.f; }
          float wt[7];
#pragma unroll
          for (int jj = 0; jj < 7; jj++) wt[jj] = wp[i * 7 + jj];
#pragma unroll
          for (int o = 0; o < 7; o++)
#pragma unroll
            for (int jj = 0; jj < 7; jj++) acc[o] = fmaf(v[o + jj], wt[jj], acc[o]);
        }
      }
      const float bias = dwb_s[ui * 32 + pp];
      float* dp = dbuf + pl * HW_ + h * W_ + w0;
#pragma unroll
      for (int o = 0; o < 7; o++) dp[o] = acc[o] + bias;
    }
    __syncthreads();
    for (int e = tid; e < nu * HW_; e += TC_NT) {
      const int ui = e / HW_, hw = e - ui * HW_;
      const int u = cta + ui * P, b = u / G, g = u - b * G;
      float s1 = 0.f, s2 = 0.f;
#pragma unroll 8
      for (int pp = 0; pp < 32; pp++) { const float v = dbuf[(ui * 32 + pp) * HW_ + hw]; s1 += v; s2 = fmaf(v, v, s2); }
      reinterpret_cast<float2*>(p.part)[((size_t)b * G + g) * HW_ + hw] = make_float2(s1, s2);
    }
    constexpr int q8n = HWS / 8;
    for (int e = tid; e < nu * 32 * q8n; e += TC_NT) {
      const int pl = e / q8n, q8 = e - pl * q8n;
      const int ui = pl >> 5, pp = pl & 31;
      const int u = cta + ui * P, b = u / G, g = u - b * G;
      float v[8];
#pragma unroll
      for (int t = 0; t < 8; t++) { const int hw = q8 * 8 + t; v[t] = hw < HW_ ? dbuf[pl * HW_ + hw] : 0.f; }
      uint4 o;
      o.x = h2u(__floats2half2_rn(v[0], v[1])); o.y = h2u(__floats2half2_rn(v[2], v[3]));
      o.z = h2u(__floats2half2_rn(v[4], v[5])); o.w = h2u(__floats2half2_rn(v[6], v[7]));
      *reinterpret_cast<uint4*>(p.d16 + (size_t)(g * 32 + pp) * p.Mcols + b * HWS + q8 * 8) = o;
    }
    constexpr int n4u = 8 * HW_;
    for (int e = tid; e < nu * n4u; e += TC_NT) {
      const int ui = e / n4u, r = e - ui * n4u;
      const int u = cta + ui * P, b = u / G, g = u - b * G;
      float4 v = reinterpret_cast<const float4*>(xs)[e];
      const int i0 = r * 4;
      const float* b2g = b2_s + ui * 32;
      v.x += b2g[(i0 + 0) / HW_]; v.y += b2g[(i0 + 1) / HW_];
      v.z += b2g[(i0 + 2) / HW_]; v.w += b2g[(i0 + 3) / HW_];
      reinterpret_cast<float4*>(p.out + ((size_t)b * C_ + g * 32) * HW_)[r] = v;
    }
  }
  {
    // W1 conversion: fp16(W1*lnw), u = row sum of the fp16 values, v = W1.lnb + b1 (arithmetic and order unchanged)
    mb_wait(sbar, 0);
    const float4* lw4 = reinterpret_cast<const float4*>(lnw_s);
    const float4* lb4 = reinterpret_cast<const float4*>(lnb_s);
#pragma unroll
    for (int k = 0; k < MW1; k++) {
      const int wu = wb1 + k * nw1;
      if (wu < C4) {
        uint2* dst = reinterpret_cast<uint2*>(p.w1h + (size_t)wu * C_);
        float su = 0.f, sv = 0.f;
#pragma unroll
        for (int j = 0; j < NQ1; j++) {
          const int c4 = lane + 32 * j;
          const float4 lw = lw4[c4];
          const float4 lb = lb4[c4];
          const __half2 h0 = __floats2half2_rn(w1v[k][j].x * lw.x, w1v[k][j].y * lw.y);
          const __half2 h1 = __floats2half2_rn(w1v[k][j].z * lw.z, w1v[k][j].w * lw.w);
          const float2 f0 = __half22float2(h0), f1 = __half22float2(h1);
          su += (f0.x + f0.y) + (f1.x + f1.y);
          sv += w1v[k][j].x * lb.x + w1v[k][j].y * lb.y + w1v[k][j].z * lb.z + w1v[k][j].w * lb.w;
          uint2 o; o.x = h2u(h0); o.y = h2u(h1);
          dst[c4] = o;
        }
        su = warp_sum(su); sv = warp_sum(sv);
        if (lane == 0) { p.uk[wu] = su; p.vk[wu] = sv + __ldg(p.b1 + wu); }
      }
    }
  }
#if TC_STAMPS
  __syncthreads(); if (tid == 0) STAMP(1);
#endif
  asm volatile("fence.proxy.async.global;" ::: "memory");
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  tfb(); gbar(P); tfa();
  if (tid == 0) STAMP(2);

  // ---------------- phase B: GEMM1 (hidden = W1h x d16), LN fold, GELU, GRN scale behind the image-group counter
  const int Bz = min(P, p.T1), Ez = P - Bz, Wt = Ez * 16 + Bz * 12;
  if (cta >= p.T1) tc_w2_work<C_>(p, (cta - Bz) * 16 + warp, Wt, lane);
  if (warp == 0) {
    asm volatile("fence.proxy.async.global;" ::: "memory");
    const uint32_t nst = (uint32_t)p.NST1;
    uint32_t it = 0;
    for (int t = cta; t < p.T1; t += P) {
      const int ht = t % HT, tt = t / HT, m0 = ht * 128, img0 = tt * p.NI;
      const int nimg = min(p.NI, p.B - img0), tok0 = img0 * HWS, N = (nimg * HWS + 15) & ~15, nat = (N + 63) >> 6;
      for (int kb = 0; kb < KB1; kb++, it++) {
        const int s = (int)(it % nst); const uint32_t n = it / nst;
        if (it >= nst) mb_wait(empty1 + s, (n & 1) ^ 1);
        if (elect1()) {
          unsigned char* st = ring + s * p.STG1;
          mb_expect(full1 + s, 16384 + nat * 8192);
          tma2d(st, &p.mW1, kb * 64, m0, full1 + s);
          for (int a = 0; a < nat; a++) tma2d(st + 16384 + a * 8192, &p.mD, tok0 + a * 64, kb * 64, full1 + s);
        }
        __syncwarp();
      }
    }
  } else if (warp == 1) {
    const uint32_t nst = (uint32_t)p.NST1;
    uint32_t it = 0, ti = 0;
    for (int t = cta; t < p.T1; t += P, ti++) {
      const int tt = t / HT, img0 = tt * p.NI;
      const int nimg = min(p.NI, p.B - img0), N = (nimg * HWS + 15) & ~15;
      const uint32_t buf = ti & 1;
      if (ti >= 2) { mb_wait(te1 + buf, ((ti >> 1) & 1) ^ 1); tfa(); }
      const uint32_t d = tbase + buf * 256;
      const uint32_t idesc = (1u << 4) | (1u << 16) | ((uint32_t)(N >> 3) << 17) | (8u << 24);
      for (int kb = 0; kb < KB1; kb++, it++) {
        const int s = (int)(it % nst); const uint32_t n = it / nst;
        mb_wait(full1 + s, n & 1);
        tfa();
        if (elect1()) {
          const uint32_t a0 = sa(ring + s * p.STG1), b0 = a0 + 16384;
#pragma unroll
          for (int k = 0; k < 4; k++) umma(d, sdesc(a0 + k * 32, 16, 1024), sdesc(b0 + k * 2048, 8192, 1024), idesc, (uint32_t)(kb | k));
          umma_commit(empty1 + s);
          if (kb == KB1 - 1) umma_commit(tf1 + buf);
        }
        __syncwarp();
      }
    }
  }
  {
    // GEMM1 epilogue on all 16 warps. Warps 4-15 take every tile (3 per TMEM quadrant); warps 0-3 join the CTA's last
    // tile (warps 0/1 after issuing), which then runs 4 warps per quadrant. Warp j of quadrant q takes the 16-column
    // chunks j, j+nwq, ...
    const int q = warp & 3, r = q * 32 + lane;
    const int ntc = (cta < p.T1) ? (p.T1 - cta + P - 1) / P : 0;
    uint32_t* stg = reinterpret_cast<uint32_t*>(ring + TC_B_STG) + warp * (TC_B_STGW / 4);
    float* sqb = reinterpret_cast<float*>(ring + TC_B_SQ);
#if TC_STAMPS
    unsigned long long a_pre = 0, a_wait = 0, a_p1 = 0, a_img = 0, a_p2 = 0, a_imax = 0;
#endif
    if constexpr (C_ >= 768) {
      if (ntc > 0 && warp >= 4) tc_w2_work<C_>(p, Ez * 16 + cta * 12 + (warp - 4), Wt, lane);
    }
    int ti = 0;
    for (int t = cta; t < p.T1; t += P, ti++) {
      const bool last = (ti == ntc - 1);
      if (!last && warp < 4) continue;
#if TC_STAMPS
      const unsigned long long q0 = gtime();
#endif
      const int nwq = last ? 4 : 3, j = last ? (warp >> 2) : (warp >> 2) - 1;
      const int bid = last ? 1 : 2, bn = nwq * 128;
      const int ht = t % HT, tt = t / HT, m0 = ht * 128, img0 = tt * p.NI;
      const int nimg = min(p.NI, p.B - img0), tok0 = img0 * HWS, N = (nimg * HWS + 15) & ~15, vcols = nimg * HWS;
      const int k = m0 + r;
      const float uk = __ldcg(p.uk + k), vk = __ldcg(p.vk + k), gwk = __ldg(p.gw + k);
      float4* ms = reinterpret_cast<float4*>(sm + ((ti & 1) ? TC_OFF_SQ : TC_OFF_MS));
      const bool early = last && ntc == 2;
      const int pw0 = early ? 2 : 4, pnw = early ? 2 : 8;
      if (warp >= pw0 && warp < pw0 + pnw) {
        for (int et = (warp - pw0) * 32 + lane; et < 256; et += pnw * 32) {
          const int i = et / HWS, hw = et - i * HWS;
          float4 m = make_float4(0.f, 0.f, 0.f, 0.f);
          if (et < vcols && hw < HW_) {
            const float2* pp = reinterpret_cast<const float2*>(p.part) + (size_t)(img0 + i) * G * HW_ + hw;
            float2 v[G];
#pragma unroll
            for (int g = 0; g < G; g++) v[g] = __ldcg(pp + g * HW_);
            float s1 = 0.f, s2 = 0.f;
#pragma unroll
            for (int g = 0; g < G; g++) { s1 += v[g].x; s2 += v[g].y; }
            const float mu = s1 / (float)C_;
            const float var = fmaxf(s2 / (float)C_ - mu * mu, 0.f);
            m = make_float4(mu, rsqrtf(var + p.ln_eps), 0.f, 1.f);
          }
          ms[et] = m;
        }
      }
      if (C_ < 768 && ti == 0 && warp >= 4) tc_w2_work<C_>(p, Ez * 16 + cta * 12 + (warp - 4), Wt, lane);
      nbar(bid, bn);
#if TC_STAMPS
      const unsigned long long q1 = gtime();
#endif
      const uint32_t buf = ti & 1;
      mb_wait(tf1 + buf, (ti >> 1) & 1);
      tfa();
#if TC_STAMPS
      const unsigned long long q2 = gtime();
#endif
      const uint32_t taddr = tbase + ((uint32_t)(q * 32) << 16) + buf * 256;
      const int nch = (N + 15) >> 4;
      __half* hbase = p.hid + (size_t)(m0 + q * 32) * p.Mcols + tok0;
      // pass 1: LN fold + GELU in fp32, sumsq per (image, k) per 8-column group, fp32 h back into TMEM
      float sq0 = 0.f, sq1 = 0.f, sq2 = 0.f;
#pragma unroll 1
      for (int ch = j; ch < nch; ch += nwq) {
        uint32_t v[16];
        tld16(taddr + ch * 16, v);
        float s8a = 0.f, s8b = 0.f;
#pragma unroll
        for (int e = 0; e < 16; e++) {
          const float4 m = ms[ch * 16 + e];
          const float h = gelu_as(fmaf(m.y, fmaf(-m.x, uk, __uint_as_float(v[e])), vk)) * m.w;
          v[e] = __float_as_uint(h);
          if (e < 8) s8a = fmaf(h, h, s8a); else s8b = fmaf(h, h, s8b);
        }
        tst16(taddr + ch * 16, v);
        const int ia = (ch * 16) / HWS, ib = (ch * 16 + 8) / HWS;
        sq0 += (ia == 0 ? s8a : 0.f) + (ib == 0 ? s8b : 0.f);
        sq1 += (ia == 1 ? s8a : 0.f) + (ib == 1 ? s8b : 0.f);
        sq2 += (ia == 2 ? s8a : 0.f) + (ib == 2 ? s8b : 0.f);
      }
      tst_wait();
      { float* w_ = sqb + (j * 128 + r) * 4; w_[0] = sq0; w_[1] = sq1; w_[2] = sq2; }
      nbar(bid, bn);
      float g0, g1, g2;
      {
        float s0 = 0.f, s1 = 0.f, s2 = 0.f;
#pragma unroll
        for (int jj = 0; jj < 4; jj++)
          if (jj < nwq) { const float* w_ = sqb + (jj * 128 + r) * 4; s0 += w_[0]; s1 += w_[1]; s2 += w_[2]; }
        g0 = sqrtf(s0); g1 = sqrtf(s1); g2 = sqrtf(s2);
        if (j == 0) {
          const float w0 = warp_sum(g0); if (lane == 0) atomicAdd(p.gsum + img0, w0);
          if (nimg > 1) { const float w1_ = warp_sum(g1); if (lane == 0) atomicAdd(p.gsum + img0 + 1, w1_); }
          if (nimg > 2) { const float w2_ = warp_sum(g2); if (lane == 0) atomicAdd(p.gsum + img0 + 2, w2_); }
        }
      }
      nbar(bid, bn);
#if TC_STAMPS
      const unsigned long long q3 = gtime();
#endif
      if (warp == 4 && lane == 0) {
        // publish this tile's gf sums, wait for the whole image group, inv_b per image, departure
        unsigned int* cnt = g_img_cnt + tt * 32;
        unsigned int* dep = g_img_dep + tt * 32;
        asm volatile("red.release.gpu.global.add.u32 [%0], 1;" ::"l"(cnt) : "memory");
        unsigned int cur;
        do {
          asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(cur) : "l"(cnt) : "memory");
        } while (cur < (unsigned int)HT);
#pragma unroll
        for (int i = 0; i < 3; i++) inv_s[i] = (i < nimg) ? 1.f / (__ldcg(p.gsum + img0 + i) / (float)C4 + p.eps) : 0.f;
        unsigned int old;
        asm volatile("atom.relaxed.gpu.global.add.u32 %0, [%1], 1;" : "=r"(old) : "l"(dep) : "memory");
        if (old == (unsigned int)HT - 1u) {
          asm volatile("st.relaxed.gpu.global.u32 [%0], 0;" ::"l"(cnt) : "memory");
          asm volatile("st.relaxed.gpu.global.u32 [%0], 0;" ::"l"(dep) : "memory");
        }
      }
      nbar(bid, bn);
#if TC_STAMPS
      const unsigned long long q4 = gtime();
#endif
      const float sc0 = fmaf(gwk * g0, inv_s[0], 1.f), sc1 = fmaf(gwk * g1, inv_s[1], 1.f), sc2 = fmaf(gwk * g2, inv_s[2], 1.f);
      // pass 2: h from TMEM, h' = fp16(h * s_img), staged coalesced stores
#pragma unroll 1
      for (int ch = j; ch < nch; ch += nwq) {
        uint32_t v[16];
        tld16(taddr + ch * 16, v);
        const int ia = (ch * 16) / HWS, ib = (ch * 16 + 8) / HWS;
        const float fa = (ia == 0) ? sc0 : (ia == 1) ? sc1 : sc2;
        const float fb = (ib == 0) ? sc0 : (ib == 1) ? sc1 : sc2;
        uint32_t w[8];
#pragma unroll
        for (int x = 0; x < 4; x++) {
          w[x] = h2u(__floats2half2_rn(__uint_as_float(v[2 * x]) * fa, __uint_as_float(v[2 * x + 1]) * fa));
          w[4 + x] = h2u(__floats2half2_rn(__uint_as_float(v[8 + 2 * x]) * fb, __uint_as_float(v[9 + 2 * x]) * fb));
        }
        tc_stage16(stg, w, hbase, p.Mcols, ch, vcols, lane);
      }
      tfb();
      nbar(bid, bn);
      if (warp == 4 && lane == 0) mb_arrive1(te1 + buf);
#if TC_STAMPS
      if (warp == 4) {
        const unsigned long long q5 = gtime();
        a_pre += q1 - q0; a_wait += q2 - q1; a_p1 += q3 - q2; a_img += q4 - q3; a_p2 += q5 - q4;
        if (q4 - q3 > a_imax) a_imax = q4 - q3;
      }
#endif
    }
#if TC_STAMPS
    if (warp == 4 && lane == 0) {
      STAMP(4); p.st[cta * 24 + 11] = a_pre; p.st[cta * 24 + 12] = a_wait; p.st[cta * 24 + 19] = a_p1;
      p.st[cta * 24 + 18] = a_img; p.st[cta * 24 + 20] = a_p2; p.st[cta * 24 + 21] = a_imax; p.st[cta * 24 + 22] = ntc;
    }
#endif
  }
  asm volatile("fence.proxy.async.global;" ::: "memory");
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  tfb(); gbar(P); tfa();
  if (tid == 0) STAMP(7);

  // ---------------- phase C: GEMM2 split-K, out += W2h x h' (+ W2.gb on the first k-split)
  // Unit = (MP x 128 output channels, column group, K slice). Column group = NI2 whole images (7x7) or 128 padded
  // columns (14x14, NI2 = 0). B stage = NAT2 64-token h' chunks after the MP W2 boxes; accumulator of channel tile m
  // at TMEM column buf*256 + m*128 (two buffers per CTA).
  const int NI2 = p.NI2, NAT2 = p.NAT2, MP = p.MP, NS2 = p.NS2;
  const uint32_t NB2 = (uint32_t)NAT2 * 64u;
  const uint32_t SS2 = (uint32_t)MP * 16384u + (uint32_t)NAT2 * 8192u;
  const int NCP = CT / MP;
  if (warp == 0) {
    asm volatile("fence.proxy.async.global;" ::: "memory");
    const uint32_t bo = (uint32_t)MP * 16384u;
    uint32_t s = 0, n = 0, it = 0;
    for (int u = cta; u < p.U2; u += P) {
      const int pair = u / p.KS, ks = u - pair * p.KS, cp = pair % NCP, tt2 = pair / NCP;
      const int c0 = cp * MP * 128, col0 = NI2 ? tt2 * NI2 * HWS : tt2 * 128, kb0 = ks * p.KB2;
      for (int kk = 0; kk < p.KB2; kk++, it++) {
        const int kb = kb0 + kk;
        if (it >= (uint32_t)NS2) mb_wait(empty2 + s, (n & 1) ^ 1);
        if (elect1()) {
          unsigned char* st = ring + s * SS2;
          mb_expect(full2 + s, SS2);
          for (int m = 0; m < MP; m++) tma2d(st + m * 16384, &p.mW2, kb * 64, c0 + m * 128, full2 + s);
          for (int a = 0; a < NAT2; a++) tma2d(st + bo + a * 8192, &p.mH, col0 + a * 64, kb * 64, full2 + s);
        }
        __syncwarp();
        if (++s == (uint32_t)NS2) { s = 0; n++; }
      }
    }
  } else if (warp == 1) {
    uint32_t s = 0, n = 0, ui = 0;
    const uint32_t idesc = (1u << 4) | (1u << 16) | ((NB2 >> 3) << 17) | (8u << 24);
    for (int u = cta; u < p.U2; u += P, ui++) {
      const uint32_t buf = ui & 1, nb = ui >> 1;
      if (nb >= 1) { mb_wait(te2 + buf, (nb - 1) & 1); tfa(); }
      const uint32_t d1 = tbase + buf * 256;
      for (int kk = 0; kk < p.KB2; kk++) {
        mb_wait(full2 + s, n & 1);
        tfa();
#if TC_STAMPS
        if (lane == 0 && ui == 0 && kk == 0) STAMP(14);
#endif
        if (elect1()) {
          const uint32_t a0 = sa(ring + s * SS2), bh = a0 + (uint32_t)MP * 16384u;
#pragma unroll
          for (int k = 0; k < 4; k++) {
            const uint64_t bd = sdesc(bh + k * 2048, 8192, 1024);
            umma(d1, sdesc(a0 + k * 32, 16, 1024), bd, idesc, (uint32_t)(kk | k));
            if (MP == 2) umma(d1 + 128, sdesc(a0 + 16384 + k * 32, 16, 1024), bd, idesc, (uint32_t)(kk | k));
          }
          umma_commit(empty2 + s);
          if (kk == p.KB2 - 1) umma_commit(tf2 + buf);
        }
        __syncwarp();
        if (++s == (uint32_t)NS2) { s = 0; n++; }
      }
    }
#if TC_STAMPS
    if (lane == 0) STAMP(15);
#endif
  } else if (warp >= 4 && warp < 12) {
    const int ew = warp - 4, grp = ew >> 2, q = warp & 3, r = q * 32 + lane, et = ew * 32 + lane;
    if constexpr (HW_ == 49) {
      // whole-image units: per (channel tile, image), TMEM -> dense [128 ch][49] fp32 smem tile -> one bulk add into
      // out[b][c..c+127][0..48]. Thread et 0 owns the bulk group. On the CTA's last unit every tile gets its own ring
      // slot; otherwise one staging tile is reused after cp.async.bulk.wait_group.read.
      float* sst = reinterpret_cast<float*>(sm + TC_OFF_RS);
      if (et == 0) asm volatile("fence.proxy.async.global;" ::: "memory");
      uint32_t ui = 0;
      for (int u = cta; u < p.U2; u += P, ui++) {
        const int pair = u / p.KS, ks = u - pair * p.KS, cp = pair % NCP, gi = pair / NCP;
        const int c0 = cp * MP * 128, bi0 = gi * NI2, nimg = min(NI2, p.B - bi0);
        const bool lastu = (u + P >= p.U2);
        float wg0 = 0.f, wg1 = 0.f;
        if (ks == 0) {
          const float4 w4 = __ldcg(reinterpret_cast<const float4*>(p.wgb) + (c0 + r)); wg0 = (w4.x + w4.y) + (w4.z + w4.w);
          if (MP == 2) { const float4 w5 = __ldcg(reinterpret_cast<const float4*>(p.wgb) + (c0 + 128 + r)); wg1 = (w5.x + w5.y) + (w5.z + w5.w); }
        }
        const uint32_t buf = ui & 1, nb = ui >> 1;
        mb_wait(tf2 + buf, nb & 1);
        tfa();
#if TC_STAMPS
        if (warp == 4 && lane == 0) { if (ui == 0) STAMP(17); STAMP(16); }
#endif
        int jt = 0;
        for (int m = 0; m < MP; m++) {
          const uint32_t tq = tbase + ((uint32_t)(q * 32) << 16) + buf * 256 + m * 128;
          const float wg = m ? wg1 : wg0;
          for (int i = 0; i < nimg; i++, jt++) {
            float* dsm = lastu ? reinterpret_cast<float*>(ring + jt * (128 * HW_ * 4)) : sst;
            if (!lastu) {
              if (et == 0) asm volatile("cp.async.bulk.wait_group.read 0;" ::: "memory");
              ebar();
            }
            const int ch0 = (i * HWS) >> 4, ch1 = (i * HWS + HW_ - 1) >> 4;
            for (int ch = ch0 + grp; ch <= ch1; ch += 2) {
              uint32_t v1[16];
              tld16(tq + ch * 16, v1);
#pragma unroll
              for (int jx = 0; jx < 16; jx++) {
                const int hw = ch * 16 + jx - i * HWS;
                if (hw >= 0 && hw < HW_) dsm[r * HW_ + hw] = __uint_as_float(v1[jx]) + wg;
              }
            }
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            ebar();
            if (et == 0) bulk_red_add(p.out + ((size_t)(bi0 + i) * C_ + c0 + m * 128) * HW_, dsm, 128u * HW_ * 4u);
          }
        }
        tfb(); __syncwarp();
        if (lane == 0) mb_arrive1(te2 + buf);
      }
      if (et == 0) asm volatile("cp.async.bulk.wait_group.read 0;" ::: "memory");
    } else {
      // 14x14: 4 consecutive tokens of one image are 16-B aligned in NCHW, so each lane (output channel) reduces
      // straight from registers with a float4 red.add
      uint32_t ui = 0;
      for (int u = cta; u < p.U2; u += P, ui++) {
        const int pair = u / p.KS, ks = u - pair * p.KS, ct = pair % CT, tt2 = pair / CT;
        const int c0 = ct * 128, col0 = tt2 * 128;
        float wg = 0.f;
        if (ks == 0) { const float4 w4 = __ldcg(reinterpret_cast<const float4*>(p.wgb) + (c0 + r)); wg = (w4.x + w4.y) + (w4.z + w4.w); }
        ebar();
        if (et < 128) {
          const int col = col0 + et, b = col / HWS, hw = col - b * HWS;
          const bool valid = (b < p.B) && (hw < HW_);
          cb_s[et] = valid ? b * C_ * HW_ + hw : -1;
        }
        ebar();
        const uint32_t buf = ui & 1;
        mb_wait(tf2 + buf, (ui >> 1) & 1);
        tfa();
#if TC_STAMPS
        if (warp == 4 && lane == 0) { if (ui == 0) STAMP(17); STAMP(16); }
#endif
        float* orow = p.out + (size_t)(c0 + r) * HW_;
#pragma unroll
        for (int cs = 0; cs < 2; cs++) {
          const int ch = grp + 2 * cs;
          uint32_t v1[32];
          tld32(tbase + ((uint32_t)(q * 32) << 16) + buf * 256 + ch * 32, v1);
#pragma unroll
          for (int g4 = 0; g4 < 8; g4++) {
            const int jc = g4 * 4;
            const int cb = cb_s[ch * 32 + jc];
            if (cb >= 0)
              atomicAdd(reinterpret_cast<float4*>(orow + cb),
                        make_float4(__uint_as_float(v1[jc]) + wg, __uint_as_float(v1[jc + 1]) + wg,
                                    __uint_as_float(v1[jc + 2]) + wg, __uint_as_float(v1[jc + 3]) + wg));
          }
        }
        tfb(); __syncwarp();
        if (lane == 0) mb_arrive1(te2 + buf);
      }
    }
#if TC_STAMPS
    if (warp == 4 && lane == 0) STAMP(9);
#endif
  }
  tfb(); __syncthreads();
  if (tid == 0) STAMP(10);
  if (warp == 2) asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, 512;" ::"r"(tbase));
}

// ---------------------------------------------------------------- host side of the tcgen05 path
// NI = images per GEMM1 tile, KS = GEMM2 K split, NI2 = images per GEMM2 unit (7x7 whole-image units with the
// bulk-reduce drain; 0 = 128-column units with red.add at 14x14), MP = 128-channel tiles per GEMM2 unit.
// r12: 16,768 uses KS 6 / MP 2 (144 units; -1.5 us rented vs KS 3 / MP 1 once g is gone); probed and rejected:
// 8,768 KS 12 MP 2 or NI2 1 MP 2, 32,768 NI2 1 or KS 6, 8,384 KS 6, 16,384 KS 2.
struct TCRow { int B, C, H, W, NI, KS, NI2, MP; };
static const TCRow kTCRows[6] = {
    {2, 768, 7, 7, 1, 12, 1, 1}, {8, 768, 7, 7, 2, 6, 2, 1}, {16, 768, 7, 7, 3, 6, 2, 2},
    {32, 768, 7, 7, 3, 3, 2, 2}, {8, 384, 14, 14, 1, 3, 0, 1}, {16, 384, 14, 14, 1, 3, 0, 1},
};

struct WSArgs {
  const float *x, *dww, *dwb, *lnw, *lnb, *w1, *b1, *gw, *gb, *w2, *b2;
  float* out; void* ws; unsigned long long* stamps;
  int B, C, H, W; float eps, ln_eps;
};

static inline size_t tc_al(size_t v) { return (v + 255) / 256 * 256; }

struct TCPlan { TCRow r; int HW, HWS, Mcols, C4, G, T1, NAT1, STG1, U2, KB2, NI2, NAT2, MP, SS2, NS2;
                size_t o_d16, o_w1h, o_w2h, o_hid, o_part, o_uk, o_vk, o_wgb, o_gsum, bytes; };

static bool tc_plan(int B, int C, int H, int W, TCPlan& pl) {
  int idx = -1;
  for (int i = 0; i < 6; i++)
    if (kTCRows[i].B == B && kTCRows[i].C == C && kTCRows[i].H == H && kTCRows[i].W == W) idx = i;
  if (idx < 0) return false;
  const TCRow r = kTCRows[idx];
  pl.r = r;
  pl.HW = H * W; pl.HWS = (pl.HW + 7) / 8 * 8; pl.Mcols = B * pl.HWS; pl.C4 = 4 * C; pl.G = C / 32;
  if ((B + r.NI - 1) / r.NI > TC_MAXGRP) return false;
  pl.T1 = (pl.C4 / 128) * ((B + r.NI - 1) / r.NI);
  pl.NAT1 = (((r.NI * pl.HWS + 15) & ~15) + 63) / 64;
  if (pl.NAT1 > 4 || r.NI > 3) return false;
  pl.STG1 = 16384 + pl.NAT1 * 8192;
  pl.KB2 = (pl.C4 / 64) / r.KS;
  if (pl.KB2 * r.KS * 64 != pl.C4) return false;
  pl.NI2 = (pl.HW == 49) ? r.NI2 : 0;
  if (pl.HW == 49 && (pl.NI2 < 1 || pl.NI2 > 2)) return false;
  pl.NAT2 = pl.NI2 ? (pl.NI2 * pl.HWS + 63) / 64 : 2;
  if (pl.NAT2 > 2) return false;
  pl.MP = (pl.NI2 > 0 && (C / 128) % 2 == 0 && r.MP == 2) ? 2 : 1;
  pl.SS2 = pl.MP * 16384 + pl.NAT2 * 8192;
  pl.NS2 = TC_RING / pl.SS2; if (pl.NS2 > TC_NS2MAX) pl.NS2 = TC_NS2MAX;
  const int ng2 = pl.NI2 ? (B + pl.NI2 - 1) / pl.NI2 : (pl.Mcols + 127) / 128;
  pl.U2 = (C / 128 / pl.MP) * ng2 * r.KS;
  size_t o = 0;
  pl.o_d16 = o; o = tc_al(o + (size_t)C * pl.Mcols * 2);
  pl.o_w1h = o; o = tc_al(o + (size_t)pl.C4 * C * 2);
  pl.o_w2h = o; o = tc_al(o + (size_t)pl.C4 * C * 2);
  pl.o_hid = o; o = tc_al(o + (size_t)pl.C4 * pl.Mcols * 2);
  pl.o_part = o; o = tc_al(o + (size_t)B * pl.G * pl.HW * 8);
  pl.o_uk = o; o = tc_al(o + (size_t)pl.C4 * 4);
  pl.o_vk = o; o = tc_al(o + (size_t)pl.C4 * 4);
  pl.o_wgb = o; o = tc_al(o + (size_t)C * 16);
  pl.o_gsum = o; o = tc_al(o + (size_t)B * 4);
  pl.bytes = o;
  return true;
}

// fp16 2-D map over a row-major [outer][inner] tensor with row pitch inner*2 bytes, 128B swizzle
static bool enc16(CUtensorMap* m, const void* base, uint64_t inner, uint64_t outer, uint32_t box_in, uint32_t box_out) {
  const cuuint64_t dims[2] = {(cuuint64_t)inner, (cuuint64_t)outer};
  const cuuint64_t strides[1] = {(cuuint64_t)inner * 2};
  const cuuint32_t box[2] = {box_in, box_out};
  const cuuint32_t estr[2] = {1u, 1u};
  CUresult r = cuTensorMapEncodeTiled(m, CU_TENSOR_MAP_DATA_TYPE_FLOAT16, 2, (void*)base, dims, strides, box, estr,
                                      CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
                                      CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  return r == CUDA_SUCCESS;
}

static int sm_count() {
  static int nsm = 0;
  if (nsm == 0) { int dev = 0; cudaGetDevice(&dev); cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, dev); }
  return nsm;
}

static const char* tc_launch(const WSArgs* a, int nsm, void* stream_v) {
  TCPlan pl;
  if (!tc_plan(a->B, a->C, a->H, a->W, pl)) return "ws_launch: unsupported shape";
  {
    const int HWv = pl.HW, units = a->B * pl.G, numax = (units + nsm - 1) / nsm;
    if ((size_t)numax * (2 * 32 * HWv + 32 * 49) * 4 > (size_t)TC_PA_MAX) return "ws_launch: phase A smem budget";
    const int MW1 = (a->C >= 768) ? 2 : 1;
    if (MW1 * nsm * (TC_NT / 32) < pl.C4) return "ws_launch: W1 worker count";
    const int Bz = nsm < pl.T1 ? nsm : pl.T1, Wt = (nsm - Bz) * 16 + Bz * 12;
    if (MW1 * Wt < pl.C4) return "ws_launch: W2 worker count";
  }
  TCP p;
  memset(&p, 0, sizeof(p));
  char* wsb = (char*)a->ws;
  p.d16 = (__half*)(wsb + pl.o_d16); p.w1h = (__half*)(wsb + pl.o_w1h); p.w2h = (__half*)(wsb + pl.o_w2h);
  p.hid = (__half*)(wsb + pl.o_hid);
  p.part = (float*)(wsb + pl.o_part); p.uk = (float*)(wsb + pl.o_uk); p.vk = (float*)(wsb + pl.o_vk);
  p.wgb = (float*)(wsb + pl.o_wgb); p.gsum = (float*)(wsb + pl.o_gsum);
  p.st = a->stamps;  // diagnostics only: read by the kernel only when built with -DTC_STAMPS=1
  const int C = a->C, C4 = pl.C4;
  if (!enc16(&p.mW1, p.w1h, C, C4, 64, 128)) return "ws_launch: map W1";
  if (!enc16(&p.mW2, p.w2h, C4, C, 64, 128)) return "ws_launch: map W2";
  if (!enc16(&p.mD, p.d16, pl.Mcols, C, 64, 64)) return "ws_launch: map D";
  if (!enc16(&p.mH, p.hid, pl.Mcols, C4, 64, 64)) return "ws_launch: map H";
  p.x = a->x; p.dww = a->dww; p.dwb = a->dwb; p.lnw = a->lnw; p.lnb = a->lnb; p.w1 = a->w1; p.b1 = a->b1;
  p.gw = a->gw; p.gb = a->gb; p.w2 = a->w2; p.b2 = a->b2; p.out = a->out;
  p.B = a->B; p.Mcols = pl.Mcols; p.units = a->B * pl.G; p.NI = pl.r.NI; p.T1 = pl.T1; p.STG1 = pl.STG1;
  p.NST1 = (TC_B_STG / pl.STG1 < 4) ? TC_B_STG / pl.STG1 : 4;
  p.KS = pl.r.KS; p.KB2 = pl.KB2; p.U2 = pl.U2; p.NI2 = pl.NI2; p.NAT2 = pl.NAT2; p.MP = pl.MP; p.NS2 = pl.NS2;
  p.eps = a->eps; p.ln_eps = a->ln_eps;
  void* args[] = {&p};
  const void* fn = (C == 768) ? (const void*)tc_kernel<768, 49> : (const void*)tc_kernel<384, 196>;
  cudaError_t e = cudaLaunchCooperativeKernel(fn, dim3(nsm), dim3(TC_NT), args, (size_t)TC_SMEM, (cudaStream_t)stream_v);
  if (e != cudaSuccess) return cudaGetErrorString(e);
  return nullptr;
}

extern "C" long long ws_workspace_bytes(int B, int C, int H, int W) {
  TCPlan pl;
  if (tc_plan(B, C, H, W, pl)) return (long long)pl.bytes;
  return 0;
}

extern "C" const char* ws_launch(const WSArgs* a, void* stream_v) {
  static bool attr_done = false;
  const int nsm = sm_count();
  if (!attr_done) {
    // one maximal value for every instantiation (never shrunk later; the portal plans all shapes first)
    cudaError_t e;
    e = cudaFuncSetAttribute(tc_kernel<768, 49>, cudaFuncAttributeMaxDynamicSharedMemorySize, 227 * 1024);
    if (e != cudaSuccess) return cudaGetErrorString(e);
    e = cudaFuncSetAttribute(tc_kernel<384, 196>, cudaFuncAttributeMaxDynamicSharedMemorySize, 227 * 1024);
    if (e != cudaSuccess) return cudaGetErrorString(e);
    attr_done = true;
  }
  return tc_launch(a, nsm, stream_v);
}
```

```cpp file=binding.cpp
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <stdexcept>
#include <string>

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

struct WSArgs {
  const float *x, *dww, *dwb, *lnw, *lnb, *w1, *b1, *gw, *gb, *w2, *b2;
  float* out; void* ws; unsigned long long* stamps;
  int B, C, H, W; float eps, ln_eps;
};
extern "C" long long ws_workspace_bytes(int B, int C, int H, int W);
extern "C" const char* ws_launch(const WSArgs* a, void* stream);

static inline size_t align256(size_t v) { return (v + 255) & ~(size_t)255; }

void run(torch::Tensor x, torch::Tensor dwconv_weight, torch::Tensor dwconv_bias, torch::Tensor layernorm_weight,
         torch::Tensor layernorm_bias, torch::Tensor pwconv1_weight, torch::Tensor pwconv1_bias, torch::Tensor grn_weight,
         torch::Tensor grn_bias, torch::Tensor pwconv2_weight, torch::Tensor pwconv2_bias, double eps, double layer_norm_eps,
         torch::Tensor output) {
  TORCH_CHECK(x.is_cuda() && x.dtype() == torch::kFloat32 && x.is_contiguous(), "x must be contiguous fp32 cuda");
  TORCH_CHECK(output.is_contiguous() && output.dtype() == torch::kFloat32, "output must be contiguous fp32");
  TORCH_CHECK(x.dim() == 4, "x must be NCHW");
  const c10::cuda::OptionalCUDAGuard guard(x.device());
  const int B = (int)x.size(0), C = (int)x.size(1), H = (int)x.size(2), W = (int)x.size(3);
  TORCH_CHECK(C % 32 == 0, "dim must be a multiple of 32");
  TORCH_CHECK(H * W >= 32, "H*W must be >= 32");
  TORCH_CHECK(pwconv1_weight.is_contiguous() && pwconv2_weight.is_contiguous() && dwconv_weight.is_contiguous(), "weights must be contiguous");
  TORCH_CHECK(layernorm_weight.is_contiguous() && layernorm_bias.is_contiguous() && grn_weight.is_contiguous() && grn_bias.is_contiguous(), "params contiguous");
  TORCH_CHECK(dwconv_bias.is_contiguous() && pwconv1_bias.is_contiguous() && pwconv2_bias.is_contiguous(), "biases contiguous");
  cudaStream_t stream = at::cuda::getCurrentCUDAStream(x.device().index()).stream();

  const long long wsb = ws_workspace_bytes(B, C, H, W);
  if (wsb > 0) {
    torch::Tensor ws = torch::empty({(long)wsb}, x.options().dtype(torch::kUInt8));
    WSArgs a;
    a.x = x.data_ptr<float>(); a.dww = dwconv_weight.data_ptr<float>(); a.dwb = dwconv_bias.data_ptr<float>();
    a.lnw = layernorm_weight.data_ptr<float>(); a.lnb = layernorm_bias.data_ptr<float>();
    a.w1 = pwconv1_weight.data_ptr<float>(); a.b1 = pwconv1_bias.data_ptr<float>();
    a.gw = grn_weight.data_ptr<float>(); a.gb = grn_bias.data_ptr<float>();
    a.w2 = pwconv2_weight.data_ptr<float>(); a.b2 = pwconv2_bias.data_ptr<float>();
    a.out = output.data_ptr<float>(); a.ws = ws.data_ptr(); a.stamps = nullptr;
    a.B = B; a.C = C; a.H = H; a.W = W; a.eps = (float)eps; a.ln_eps = (float)layer_norm_eps;
    const char* err = ws_launch(&a, (void*)stream);
    if (err != nullptr) throw std::runtime_error(std::string("ws_launch failed: ") + err);
    return;
  }

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

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("run", &run, "ConvNeXtV2 block with GRN (tcgen05 swap-AB wide-C path, H49 + fp32 anchor)"); }
```

```yaml design-card
id: r12-d-e2-grn-epi-counters
parents:
  - r11-d-e1-g2-mp2-bulk
operation: structural_mutation
language: cuda_cpp
niche:
  mem: tma-tensor
  st: bulk
  grid: persistent
  launch: fused
  tile: token
  red: cta
  cache: default
  spec: dispatch:size
hypothesis: >-
  H49, partly supported. The GEMM1 epilogue keeps fp32 h in its own TMEM columns (tcgen05.st, so no register cost)
  while the tile publishes its gf sums on a per-image-group counter and waits for the group. It then stores
  h' = fp16(h*(1+gw*gf*inv_b)) once, so g and the second GEMM2 accumulator disappear. The GEMM2 phase drops 30-55 %
  (32,768: 11.3 -> 6.7 us). But the image wait (~2 us per tile) and the extra h' pass (1-2.5 us) cancel the removed
  g pass, so the wide-C rows move -9 % at 32,768, -3 % at 16,768 and 0..+3 % elsewhere (rented, same GPU as
  r11-merge-b).
expected_effect:
  S:
    pct: 0
    confidence: medium
  M:
    pct: 0
    confidence: medium
  L:
    pct: -3
    confidence: medium
resources_sm100a:
  regs_per_thread: 128
  smem_per_cta_bytes: 227608
  threads_per_cta: 512
  ctas_per_sm: 1
  bytes_in_flight_per_sm: 196608
  launches_per_call: 1
knobs:
  tc_rows: "NI/KS/NI2/MP: 2,768 1/12/1/1; 8,768 2/6/2/1; 16,768 3/6/2/2; 32,768 3/3/2/2; 8,384 1/3/0/1; 16,384 1/3/0/1 (tc_kernel, not tc2)"
  gemm2_stage: "MP x 16 KB W2 + NAT2 x 8 KB h' (24-48 KB), NS2 = min(8, 192 KB / stage), two TMEM buffers of MP x 128 cols"
  image_counter: "g_img_cnt/g_img_dep module-level, one 128-B line per image group, target HT tiles, reset by the last departer"
  TC_STAMPS: 0
dispatch:
  - shapes: "2,768,7,7 8,768,7,7 16,768,7,7 32,768,7,7"
    kernel: "tc_kernel<768,49> (H49 epilogue, one-accumulator GEMM2 over h', whole-image bulk-reduce drain)"
    meta: {grid: "num SMs, cooperative"}
  - shapes: "8,384,14,14 16,384,14,14"
    kernel: "tc_kernel<384,196> (H49 epilogue, float4 red.add drain)"
    meta: {grid: "num SMs, cooperative"}
  - shapes: "all other rows"
    kernel: "convnext_fused (r0 anchor; merge-only candidate, take those rows from r11-merge-b)"
    meta: {}
tests:
  - H49
findings:
  - "Parent paths read from the card: r11-d-e1 dispatches the four C=768 rows to tc_kernel<768,49> (MP=2 only at 32,768), 8,384 to tc_kernel<384,196> and 16,384 to tc2_kernel. No pair path exists in this source. The parent card records the flat-barrier review and the 1500-call stress test as done in r10, so they were not repeated. The new counters got their own 400-call stress (below) [get_kernel parent text]"
  - "NOT DONE from STEP 0: clk-per-MMA-instruction stamps of D's mainloops against r11 K's table (50/66/114 clk), and the precomputed-descriptor unrolled k16 loop. The budget went to H49 itself [INFERRED]"
  - "Design: fp32 h is written back into its own TMEM accumulator columns with tcgen05.st.32x32b.x16 and re-read after the wait. This avoids holding 48-80 fp32 per thread across the wait (that would spill at the 128-reg cap). No hreg array remains, so the shipped build has REG 128 / STACK 0 (768,49) and REG 114 / STACK 0 (384,196), checked with cuobjdump -res-usage on an nvcc sm_100a build of the identical device code [probe_b200, compile_b200]"
  - "run_tests 16/16 PASS twice (portal-order re-runs included). Rented us, this vs r11-merge-b on the same GPU (final build / first build with 16,768 at KS3 MP1): 32,768 54.4 / 54.2 vs 59.6 (-8.8 / -9.0 %); 16,768 39.4 / 39.5 vs 40.6 (-3.1 / -2.8 %); 8,384 33.4 vs 33.6 (-0.6 %); 8,768 30.3 vs 30.1 (+0.7 %); 16,384 50.5 / 51.0 vs 50.1 (tc_kernel H49 vs r10 tc2, +0.7..+1.7 %); 2,768 24.1 / 23.8 vs 23.4 (+1.8..+3.0 %). The other 10 rows run the r0 anchor (merge-only) [run_tests]"
  - "Stamps (stamped build, flushed, CTA median/max, us). GEMM2 phase sync2->end: 2,768 1.28; 8,768 2.30; 16,768 5.12; 32,768 6.66; 8,384 2.30; 16,384 4.86. The parent's were 1.5 / 3.3 / 7.3 / 11.3 (MP=2) / 3.0, so the GEMM2 stage time does fall without g (that refutation clause does not fire) [probe_b200]"
  - "Per-tile GEMM1 epilogue (median us: pass 1 / image wait / pass 2). 2,768 1.28 / 1.95 / 0.96; 8,768 2.08 / 2.14 / 1.57; 16,768 2.98 / 2.30 / 2.34; 32,768 3.47 / 2.18 / 2.51 (max wait 10.7 on the 4 first-tile CTAs whose group needs second tiles); 8,384 3.46 / 1.79 / 2.43; 16,384 3.55 / 1.95 / 2.32. The sum equals or exceeds r9's post-accumulator time without g (2.05 / 3.36 / 4.77 / 5.17), so the epilogue gains nothing net; the row gains come from GEMM2 [probe_b200]"
  - "Image-wait median 1.8-2.3 us: above the 1.5 us success line, below the 2.5 us refutation line. A no-wait timing stub (wrong results, timing only, never shipped) shows the whole wait costs only 0.9-1.4 us end-to-end (2,768 24.0->22.7, 8,768 30.2->29.3, 16,768 40.2->38.8, 32,768 54.6->53.2, 8,384 33.8->33.4, 16,384 51.1->49.9, harness_time x3). A perfect counter cannot reach the 10 % bar at 16,768 [probe_b200]"
  - "Dead end: running the deferred W2 conversion inside the image-wait window (publish, then W2 work, then spin) was slower on every row (8,768 +2.4 us, 32,768 +2.0, 16,768 +1.6) and put tc_kernel<768,49> at STACK 88. The W2 pass is already hidden under the GEMM1 mainloop [probe_b200]"
  - "GEMM2 tables after g removal (harness_time x3, same container): 16,768 KS6 MP2 38.5 vs KS3 MP1 40.0 (adopted; the two TMEM buffers now hold MP=2 units). Rejected: 8,768 KS12 MP2 30.6 / NI2=1 MP2 30.2 vs 30.2; 32,768 NI2=1 MP2 57.8 / KS6 MP2 56.7 vs 54.4; 8,384 KS6 35.7 vs 33.6; 16,384 KS2 52.2 vs 51.3 [probe_b200]"
  - "PRECISION LOG (fp64 reference, harness generator rules, torch.manual_seed(200), 10 rounds per row, KS3 build): 2,768 1.674e-3 / 2.6e-3 (64 %), 0 beyond; 8,768 1.883e-3 / 2.9e-3 (65 %), 0; 16,768 1.891e-3 / 2.9e-3 (65 %), 0; 32,768 1.876e-3 / 2.9e-3 (65 %), 0; 8,384 1.863e-3 / 2.8e-3 (67 %), 0; 16,384 1.870e-3 / 3.3e-3 (57 %), 0. h' is rounded once from fp32 (h * s); gf uses fp32 h. The shipped 16,768 KS6 MP2 differs only in the fp32 split-K summation order (max diff vs KS3 < 1e-3, checked); 16/16 x 10 rounds pass in run_tests [probe_b200, run_tests]"
  - "Counter stress: 400 consecutive calls each at 8,768, 32,768 and 16,384 (output NaN-filled before each call) gave worst |err| vs fp64 1.61e-3 / 1.81e-3 / 1.81e-3 and no NaN. The self-reset through the departure counter holds across calls and across the two instantiations sharing the counter cells [probe_b200]"
  - "Verdict H49: success missed (needed >= 10 % at 32,768 and 16,768 plus >= 5 % at 8,768; got -9 / -3 / +1 %). Refutation not fired (32,768 is -9 %, so not < 4 % on both; image wait < 2.5 us; GEMM2 phase falls). Mechanism: removing g pays in GEMM2 (L2-bound stages carry 25-33 % fewer bytes) but not in GEMM1, where the per-image wait plus a second TMEM pass replace the g pass at about equal cost [INFERRED]"
  - "Merge advice: take 32,768 from this kernel (-5.2 us rented, about -6 us portal, ~+0.001 of mean). 16,768 (-1.2 us) only if an interleaved merge timing confirms. Keep r11-merge-b for 2,768, 8,768, 8,384 and 16,384 (tc2). Next levers for D: overlap one tile's image wait with the next tile's pass 1 on 2-tile CTAs (32,768, 16,384; worth at most one wait, ~1-2 us), and a lighter pass 2 [INFERRED]"
  - "Pitfall/tooling: /tmp persists between probes in one session here, and nvcc -shared of the tc-only file built in < 2 s, so source-string variants (str.replace on the saved .cu) give a seconds-long build-and-time loop through ctypes + harness_time [probe_b200]"
paths:
  - {max_tokens: 1568, lang: cuda, width: 128, threads: 512, rows: 128, grid: persistent, mem: tma, launches: 1, x: none, w: none, st: none, cluster: 1, note: "tc_kernel<768,49> / <384,196> with the H49 epilogue (per-image-group counter wait, h' stored once) and one-accumulator GEMM2"}
  - {max_tokens: null, lang: cuda, width: 128, threads: 256, rows: 128, grid: persistent, mem: ldg, launches: 1, x: none, w: none, st: none, cluster: 1, note: "r0 fp32 anchor on every non-wide-C row (merge-only candidate)"}
runs_on:
  H200:
    runs: false
    representative: false
    note: "tcgen05/TMEM, tcgen05.st and cp.reduce.async.bulk paths are sm_100a-only"
  A100:
    runs: false
    representative: false
  L40S:
    runs: false
    representative: false
risks:
  - "Only 32,768 clearly beats r11-merge-b. The other wide-C rows are within noise or slower and should not be merged"
  - "The image wait is a cross-CTA spin inside a cooperative launch. It is deadlock-free only because tiles are image-major, publish before waiting, and all CTAs are co-resident. Any later change to tile order or grid size must keep those three properties"
  - "The stamped build has STACK 32/120 B, so stamp numbers are relative; the shipped build is STACK 0"
  - "The 16,768 KS6 MP2 change doubles that row's reduction bytes; its -1.2..-1.5 us is a single-container measurement"
measure_first:
  - B200
compliance:
  single_stream: true
  no_cross_call_state: true
  fp32_math: true
  no_threads_or_fork: true
  no_precompiled_binaries: true
```