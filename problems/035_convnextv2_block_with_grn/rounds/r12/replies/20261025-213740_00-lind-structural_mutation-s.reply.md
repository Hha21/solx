### Rationale
The STEP 0 gate failed, so I'm shipping the parent unchanged, as the task requires. Timing-only stub (a), which skips the weight pass and leaves the fp16 scratch as garbage, saved only about +0.05 µs on average over the five gate rows. The gate needed −1.2 µs. Stub (b) also skips the post-sync TMA of A and still saved only about −0.4 µs. The weight chain is already hidden behind the depthwise conv, so H48 is refuted.

The stamps show where phase A actually goes on the sc rows: the conv runs 2.8–10.5 µs after the planes land. It is bound by latency at low occupancy, not by the weights. In two separate runs, rewriting the conv's d16 and out writes as coalesced stores gained about 0.4–1.4 µs on 4,96, 1,160, 2,96 and 8,192, but was mixed on 1,192 and 4,192. That is recorded as a lead for the next round, not shipped. `run_tests` was not run, because the code is byte-identical to the parent (r11, 16/16 pass). The sc_kernel probe build matches the parent's registers with STACK 0. I did no new precision run, since the arithmetic is unchanged.

```json solution-spec
{"name": "r12-d-e1-h48-gate-parent", "definition": "035_convnextv2_block_with_grn", "author": "solx-loop",
 "description": "H48 refuted at its STEP 0 gate (the sc weight chain is already overlapped by the conv); parent r11-d-e1-g2-mp2-bulk shipped unchanged.",
 "spec": {"languages": ["cuda_cpp"], "target_hardware": ["B200"], "entry_point": "binding.cpp::run",
          "dependencies": ["torch"], "destination_passing_style": true,
          "compile_options": {"cuda_cflags": ["-O3", "--use_fast_math", "-std=c++17"]}}}
```

```cuda file=kernel.cu
// ConvNeXtV2 block with GRN.
// Wide-C small-image rows (2/8/16/32,768,7,7 and 8/16,384,14,14): cooperative kernel tc_kernel with a tcgen05
//   kind::f16 swap-AB engine (A = fp16 weight tile, 128 rows, K-major; B = fp16 activations [channel][token],
//   MN-major; TMA 128B-swizzled ring; whole-warp TMA producer and MMA warp; double-buffered TMEM accumulators).
//   r8: 512 threads; the GEMM1 epilogue runs on all 16 warps (3-4 per TMEM quadrant) with the A&S erfc GELU.
//   r9: phase A issues its W1 loads first (volatile, so they stay ahead of the plane-copy wait), runs the conv in their
//   shadow and converts W1 last; LN vectors and per-unit dwb/b2 arrive by bulk copy with the plane copies; the W2 -> fp16
//   conversion moves into the GEMM1 phase (idle CTAs after sync 1, busy CTAs while their first tile's mainloop runs),
//   always complete before sync 2; both grid barriers of tc_kernel are a flat self-resetting atomic-counter barrier.
//   r10: at C=768 the deferred W2 conversion is issued before the GEMM1 epilogue tile loop instead of inside it
//   (r9 build: 128 regs + 120 B stack, 76 STL/LDL in the GELU/g passes; now 105 regs, no stack); tc2_kernel (balanced
//   image-straddling token ranges, no g, GEMM2 B scaled in place) for the rows in kTC2Rows.
//   r11 (H45): GEMM2 issues one MMA over the adjacent [h|g] B chunks (N = 2 x 64*NAT2) instead of two; at 7x7 the
//   GEMM2 units cover whole images (NI2 images per unit, per-row table) and the epilogue drains TMEM into a dense
//   [128 ch][49] smem tile per image and adds it to the prefilled output with one cp.reduce.async.bulk .add.f32
//   (staged in the free ring on the CTA's last unit); at 32,768 a unit covers 256 output channels (MP = 2: two W2
//   tiles share each landed [h|g] B stage, so the L2-bound B ingest halves).
// Small-C rows (1/4/8,192,28,28; 1/2/4,96,56,56; 1,160,449,7): cooperative kernel sc_kernel, same engine with
//   N-chunked token tiles of one image, red.add GRN sums, and the GRN scale applied to the GEMM2 B tile in smem.
//   (cg grid.sync() barriers: the flat barrier measured 1-2 us slower on these rows.)
// Other rows: the r0 fp32 SIMT cooperative anchor (convnext_fused).
#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cooperative_groups.h>
#include <stdint.h>
#include <string.h>
namespace cg = cooperative_groups;

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

// =====================================================================================================
// tcgen05 swap-AB cooperative path for the wide-C small-image rows (512 threads, 16 warps)
// Phase A: plane loads (+ LN vectors, dwb/b2 slices) by bulk copy at entry, W1 loads in flight during the depthwise
//   conv, LN partials, fp16 d, out = x + b2, then the W1 conversion. Sync 1.
// Phase B: GEMM1 (warp 0 TMA, warp 1 MMA); epilogue on all 16 warps (warps 4-15 on every tile, 3 per TMEM quadrant;
//   warps 0-3 join the CTA's last tile, 4 per quadrant): 16-column chunks, LN fold, A&S erfc GELU, fp16 h kept in
//   registers, per-(image,k) sumsq combined through smem, then g = fp16(gw*gf*h) from the same registers. The W2 -> fp16
//   conversion (and W2.gb) runs here: idle CTAs right after sync 1, busy CTAs on warps 4-15 during the first tile's
//   mainloop (r10: at C=768 issued before the tile loop, which removes the epilogue's 120 B stack frame). Sync 2.
// Phase C: GEMM2 split-K, one MMA over the [h|g] B stage into two adjacent accumulators per 128-channel tile (r11);
//   warps 4-11 drain into the prefilled output: at 7x7 with whole-image units by one bulk reduce per image (r11), else
//   by red.add.
// TC_STAMPS: globaltimer diagnostics (off in the shipped build; the timed path writes no stamp buffer).
// TC_ONEMMA / TC_BULK: r11 ablation switches (1 in the shipped build; 0/0 reproduces the r10 GEMM2 phase).
// =====================================================================================================
#ifndef TC_STAMPS
#define TC_STAMPS 0
#endif
#ifndef TC_ONEMMA
#define TC_ONEMMA 1
#endif
#ifndef TC_BULK
#define TC_BULK 1
#endif
#ifndef TC2_ENABLE
#define TC2_ENABLE 1
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
#define TC_SMEM (TC_OFF_BAR + 27 * 8 + 1024)
#define TC_STG2 49152
// phase-C bulk-reduce staging tile [128 ch][49] fp32 (7x7 rows, units that are not the CTA's last): after the ring,
// over buffers used only in phase B; the last unit stages its tiles in the (then free) ring
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
  CUtensorMap mW1, mW2, mD, mH, mG;
  const float *x, *dww, *dwb, *lnw, *lnb, *w1, *b1, *gw, *gb, *w2, *b2;
  float* out;
  __half *d16, *w1h, *w2h, *hid, *gq;
  float *part, *uk, *vk, *wgb, *gsum;
  unsigned long long* st;
  int B, Mcols, units, NI, T1, STG1, NST1, KS, KB2, U2, NI2, NAT2, MP;
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
// shared -> global bulk add of fp32 (async proxy, bulk-group completion; issuing thread commits and waits)
__device__ __forceinline__ void bulk_red_add(float* dst, const void* src, uint32_t bytes) {
  asm volatile("cp.reduce.async.bulk.global.shared::cta.bulk_group.add.f32 [%0], [%1], %2;" ::"l"(dst), "r"(sa(src)), "r"(bytes) : "memory");
  asm volatile("cp.async.bulk.commit_group;" ::: "memory"); }
// non-coherent 16-B load that stays where it is written relative to other volatile asm (mbarrier waits)
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
__device__ __forceinline__ void ebar() { asm volatile("bar.sync 1, 256;" ::: "memory"); }
__device__ __forceinline__ void nbar(int id, int n) { asm volatile("bar.sync %0, %1;" ::"r"(id), "r"(n) : "memory"); }

// Flat self-resetting grid barrier (r4 B microbench form), used by tc_kernel and tc2_kernel: one arrival counter and one
// generation word, each on its own 128-B line, in module-level device memory (zero at module load, never in the torch
// workspace). Every CTA reads the generation, arrives with an acq_rel atomic; the last arriver resets the counter and
// publishes generation+1 with a release store; the others poll the generation with plain acquire loads (no sleep). All
// CTAs are co-resident (cooperative launch). Callers run fence.proxy.async before it and the TMA warps run
// fence.proxy.async after it, so TMA reads of other CTAs' generic stores are ordered (r10 STEP 0 check).
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
__device__ __forceinline__ uint32_t h2u(__half2 h) { return *reinterpret_cast<uint32_t*>(&h); }
__device__ __forceinline__ uint32_t scale_h2(uint32_t u, float s) {
  const float2 f = __half22float2(*reinterpret_cast<const __half2*>(&u));
  return h2u(__floats2half2_rn(f.x * s, f.y * s));
}
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

// W2 -> fp16 conversion with the per-quarter W2.gb partial (deferred out of phase A). Worker wid of nwk converts units
// wid, wid+nwk, ... (unit = one quarter row of C floats); nwk is a multiple of 4, so the quarter is the same for all.
// All loads of the worker's units are issued before any use; the summation order equals the old phase-A code.
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
  uint32_t* stg_all = reinterpret_cast<uint32_t*>(sm + TC_OFF_STG);
  float* inv_s = reinterpret_cast<float*>(sm + TC_OFF_INV);
  int* cb_s = reinterpret_cast<int*>(sm + TC_OFF_CB);
  uint64_t* bars = reinterpret_cast<uint64_t*>(sm + TC_OFF_BAR);
  uint64_t *full1 = bars, *empty1 = bars + 4, *tf1 = bars + 8, *te1 = bars + 10;
  uint64_t *full2 = bars + 12, *empty2 = bars + 16, *tf2 = bars + 20, *te2 = bars + 22, *cbar = bars + 24, *sbar = bars + 25;
  uint32_t* tslot = reinterpret_cast<uint32_t*>(bars + 26);
  const int tid = threadIdx.x, warp = __shfl_sync(0xffffffffu, tid >> 5, 0), lane = tid & 31;
  const int cta = blockIdx.x, P = gridDim.x;
  if (tid == 0) {
    for (int i = 0; i < 26; i++) mb_init(bars + i, (i == 22 || i == 23) ? 8u : 1u);
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
    // every copy of the CTA is issued here, before any weight is touched: LN vectors first (own barrier, needed by the
    // W1 conversion), then per conv unit the plane group, its 32 depthwise filters, its dwb and b2 slices
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
  // W1 units of C floats each (4C rows): the warp's (at most MW1) units are loaded now and converted after the conv
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

  // ---------------- phase B: GEMM1 (hidden = W1h x d16), LN fold, GELU, fp16 hidden h and g = gw*gf*h
  // deferred W2 conversion workers: CTAs without a GEMM1 tile contribute 16 warps (right now), CTAs with one contribute
  // warps 4-15 (during the first tile's mainloop); worker ids are dense over the whole grid
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
    // chunks j, j+nwq, ...: LN fold, A&S erfc GELU, fp16 h kept in registers, per-image sumsq per 8-column group;
    // partials combined through smem; then g = fp16(gw*gf*h) from the same registers (no h re-read).
    const int q = warp & 3, r = q * 32 + lane;
    const int ntc = (cta < p.T1) ? (p.T1 - cta + P - 1) / P : 0;
    uint32_t* stg = reinterpret_cast<uint32_t*>(ring + TC_B_STG) + warp * (TC_B_STGW / 4);
    float* sqb = reinterpret_cast<float*>(ring + TC_B_SQ);
#if TC_STAMPS
    unsigned long long a_pre = 0, a_wait = 0, a_post = 0;
#endif
    // r10: at C=768 the deferred W2 conversion is issued here, ahead of the tile loop (same point for the first tile);
    // inside the loop it pushed tc_kernel<768,49> to 128 regs + a 120 B stack frame (76 STL/LDL in the GELU and g passes)
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
      // per-token LN statistics (mu, rstd, -, valid): warps 2-3 for the second of two tiles (idle during the first)
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
      // deferred W2 conversion, hidden under this tile's mainloop (the accumulator wait follows); C=384 only (see above)
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
      __half* gbase = p.gq + (size_t)(m0 + q * 32) * p.Mcols + tok0;
      uint32_t hreg[5][8];
      float sq0 = 0.f, sq1 = 0.f, sq2 = 0.f;
#pragma unroll
      for (int c = 0; c < 5; c++) {
        const int ch = j + nwq * c;
        if (ch < nch) {
          uint32_t v[16];
          tld16(taddr + ch * 16, v);
          float s8a = 0.f, s8b = 0.f;
#pragma unroll
          for (int e = 0; e < 16; e += 2) {
            const float4 ma = ms[ch * 16 + e], mb = ms[ch * 16 + e + 1];
            const float ha = gelu_as(fmaf(ma.y, fmaf(-ma.x, uk, __uint_as_float(v[e])), vk)) * ma.w;
            const float hb = gelu_as(fmaf(mb.y, fmaf(-mb.x, uk, __uint_as_float(v[e + 1])), vk)) * mb.w;
            const __half2 hh = __floats2half2_rn(ha, hb);
            hreg[c][e >> 1] = h2u(hh);
            const float2 hr = __half22float2(hh);
            if (e < 8) s8a = fmaf(hr.x, hr.x, fmaf(hr.y, hr.y, s8a));
            else s8b = fmaf(hr.x, hr.x, fmaf(hr.y, hr.y, s8b));
          }
          const int ia = (ch * 16) / HWS, ib = (ch * 16 + 8) / HWS;
          sq0 += (ia == 0 ? s8a : 0.f) + (ib == 0 ? s8b : 0.f);
          sq1 += (ia == 1 ? s8a : 0.f) + (ib == 1 ? s8b : 0.f);
          sq2 += (ia == 2 ? s8a : 0.f) + (ib == 2 ? s8b : 0.f);
          tc_stage16(stg, hreg[c], hbase, p.Mcols, ch, vcols, lane);
        }
      }
      { float* w_ = sqb + (j * 128 + r) * 4; w_[0] = sq0; w_[1] = sq1; w_[2] = sq2; }
      tfb();
      nbar(bid, bn);
      if (warp == 4 && lane == 0) mb_arrive1(te1 + buf);
      float f0, f1, f2;
      {
        float s0 = 0.f, s1 = 0.f, s2 = 0.f;
#pragma unroll
        for (int jj = 0; jj < 4; jj++)
          if (jj < nwq) { const float* w_ = sqb + (jj * 128 + r) * 4; s0 += w_[0]; s1 += w_[1]; s2 += w_[2]; }
        const float g0 = sqrtf(s0), g1 = sqrtf(s1), g2 = sqrtf(s2);
        if (j == 0) {
          const float w0 = warp_sum(g0); if (lane == 0) atomicAdd(p.gsum + img0, w0);
          if (nimg > 1) { const float w1_ = warp_sum(g1); if (lane == 0) atomicAdd(p.gsum + img0 + 1, w1_); }
          if (nimg > 2) { const float w2_ = warp_sum(g2); if (lane == 0) atomicAdd(p.gsum + img0 + 2, w2_); }
        }
        f0 = gwk * g0; f1 = gwk * g1; f2 = gwk * g2;
      }
#pragma unroll
      for (int c = 0; c < 5; c++) {
        const int ch = j + nwq * c;
        if (ch < nch) {
          const int ia = (ch * 16) / HWS, ib = (ch * 16 + 8) / HWS;
          const float fa = (ia == 0) ? f0 : (ia == 1) ? f1 : f2;
          const float fb = (ib == 0) ? f0 : (ib == 1) ? f1 : f2;
          uint32_t w[8];
#pragma unroll
          for (int x = 0; x < 4; x++) { w[x] = scale_h2(hreg[c][x], fa); w[4 + x] = scale_h2(hreg[c][4 + x], fb); }
          tc_stage16(stg, w, gbase, p.Mcols, ch, vcols, lane);
        }
      }
#if TC_STAMPS
      if (warp == 4) { const unsigned long long q3 = gtime(); a_pre += q1 - q0; a_wait += q2 - q1; a_post += q3 - q2; }
#endif
    }
#if TC_STAMPS
    if (warp == 4 && lane == 0) { STAMP(4); p.st[cta * 24 + 11] = a_pre; p.st[cta * 24 + 12] = a_wait; p.st[cta * 24 + 13] = a_post; }
#endif
  }
  asm volatile("fence.proxy.async.global;" ::: "memory");
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  tfb(); gbar(P); tfa();
  if (tid == 0) STAMP(7);

  // ---------------- phase C: GEMM2 split-K, out += W2h x h + inv_b * (W2h x g) (+ W2.gb on the first k-split)
  // Unit = (MP x 128 output channels, column group, K slice). Column group = NI2 whole images (7x7, r11) or 128 padded
  // columns (NI2 = 0). B stage = NAT2 64-token h chunks followed by NAT2 g chunks, after the MP W2 boxes; one MMA per
  // W2 box with N = 2*64*NAT2 writes the h product to accumulator columns [0, 64*NAT2) and the g product to
  // [64*NAT2, 128*NAT2). MP = 2 (64 KB stages, 3-deep ring) fills all 512 TMEM columns, so units do not double-buffer.
  const int NI2 = p.NI2, NAT2 = p.NAT2, MP = p.MP;
  const uint32_t NB2 = (uint32_t)NAT2 * 64u;
  const int SS2 = (MP == 2) ? 65536 : TC_STG2, NS2 = (MP == 2) ? 3 : 4, nbuf = (MP == 2) ? 1 : 2, NCP = CT / MP;
  if (warp == 0) {
    asm volatile("fence.proxy.async.global;" ::: "memory");
    const uint32_t stb = (uint32_t)MP * 16384u + 2u * (uint32_t)NAT2 * 8192u, bo = (uint32_t)MP * 16384u;
    uint32_t it = 0, s = 0, n = 0;
    for (int u = cta; u < p.U2; u += P) {
      const int pair = u / p.KS, ks = u - pair * p.KS, cp = pair % NCP, tt2 = pair / NCP;
      const int c0 = cp * MP * 128, col0 = NI2 ? tt2 * NI2 * HWS : tt2 * 128, kb0 = ks * p.KB2;
      for (int kk = 0; kk < p.KB2; kk++) {
        const int kb = kb0 + kk;
        if (it >= (uint32_t)NS2) mb_wait(empty2 + s, (n & 1) ^ 1);
        if (elect1()) {
          unsigned char* st = ring + s * SS2;
          mb_expect(full2 + s, stb);
          for (int m = 0; m < MP; m++) tma2d(st + m * 16384, &p.mW2, kb * 64, c0 + m * 128, full2 + s);
          for (int a = 0; a < NAT2; a++) {
            tma2d(st + bo + a * 8192, &p.mH, col0 + a * 64, kb * 64, full2 + s);
            tma2d(st + bo + (NAT2 + a) * 8192, &p.mG, col0 + a * 64, kb * 64, full2 + s);
          }
        }
        __syncwarp();
        it++; if (++s == (uint32_t)NS2) { s = 0; n++; }
      }
    }
  } else if (warp == 1) {
    uint32_t it = 0, s = 0, n = 0, ui = 0;
#if TC_ONEMMA
    const uint32_t idesc = (1u << 4) | (1u << 16) | (((2u * NB2) >> 3) << 17) | (8u << 24);
#else
    const uint32_t idesc = (1u << 4) | (1u << 16) | ((NB2 >> 3) << 17) | (8u << 24);
#endif
    for (int u = cta; u < p.U2; u += P, ui++) {
      const uint32_t buf = ui % (uint32_t)nbuf, nb = ui / (uint32_t)nbuf;
      if (nb >= 1) { mb_wait(te2 + buf, (nb - 1) & 1); tfa(); }
      const uint32_t d1 = tbase + buf * 256;
      for (int kk = 0; kk < p.KB2; kk++) {
        mb_wait(full2 + s, n & 1);
        tfa();
#if TC_STAMPS
        if (lane == 0 && it == 0) STAMP(14);
#endif
        if (elect1()) {
          const uint32_t a0 = sa(ring + s * SS2), bh = a0 + (uint32_t)MP * 16384u;
#pragma unroll
          for (int k = 0; k < 4; k++) {
            const uint64_t bd = sdesc(bh + k * 2048, 8192, 1024);
            umma(d1, sdesc(a0 + k * 32, 16, 1024), bd, idesc, (uint32_t)(kk | k));
            if (MP == 2) umma(tbase + 256, sdesc(a0 + 16384 + k * 32, 16, 1024), bd, idesc, (uint32_t)(kk | k));
#if !TC_ONEMMA
            umma(d1 + NB2, sdesc(a0 + k * 32, 16, 1024), sdesc(bh + NAT2 * 8192 + k * 2048, 8192, 1024), idesc, (uint32_t)(kk | k));
#endif
          }
          umma_commit(empty2 + s);
          if (kk == p.KB2 - 1) umma_commit(tf2 + buf);
        }
        __syncwarp();
        it++; if (++s == (uint32_t)NS2) { s = 0; n++; }
      }
    }
#if TC_STAMPS
    if (lane == 0) STAMP(15);
#endif
  } else if (warp >= 4 && warp < 12) {
    const int ew = warp - 4, grp = ew >> 2, q = warp & 3, r = q * 32 + lane, et = ew * 32 + lane;
    bool done = false;
    if constexpr (HW_ == 49) {
      if (NI2 > 0) {
        // whole-image units: per (channel tile, image), TMEM -> dense [128 ch][49] fp32 smem tile (lanes = channels,
        // stride 49 words: conflict-free) -> one bulk add into out[b][c..c+127][0..48] (contiguous 25 KB). Thread et 0
        // owns the bulk group. On the CTA's last unit the ring is free (all MMAs committed, no loads pending), so every
        // tile gets its own ring slot; otherwise one staging tile is reused after cp.async.bulk.wait_group.read.
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
          const float inv0 = 1.f / (__ldcg(p.gsum + bi0) / (float)C4 + p.eps);
          const float inv1 = (nimg > 1) ? 1.f / (__ldcg(p.gsum + bi0 + 1) / (float)C4 + p.eps) : 0.f;
          const uint32_t buf = ui % (uint32_t)nbuf, nb = ui / (uint32_t)nbuf;
          mb_wait(tf2 + buf, nb & 1);
          tfa();
#if TC_STAMPS
          if (warp == 4 && lane == 0) { if (ui == 0) STAMP(17); STAMP(16); }
#endif
          int jt = 0;
          for (int m = 0; m < MP; m++) {
            const uint32_t tq = tbase + ((uint32_t)(q * 32) << 16) + (buf + m) * 256;
            const float wg = m ? wg1 : wg0;
            for (int i = 0; i < nimg; i++, jt++) {
              const float inv = i ? inv1 : inv0;
              float* dsm = lastu ? reinterpret_cast<float*>(ring + jt * (128 * HW_ * 4)) : sst;
              if (!lastu) {
                if (et == 0) asm volatile("cp.async.bulk.wait_group.read 0;" ::: "memory");
                ebar();
              }
              const int ch0 = (i * HWS) >> 4, ch1 = (i * HWS + HW_ - 1) >> 4;
              for (int ch = ch0 + grp; ch <= ch1; ch += 2) {
                uint32_t v1[16], v2[16];
                tld16(tq + ch * 16, v1);
                tld16(tq + NB2 + ch * 16, v2);
#pragma unroll
                for (int jx = 0; jx < 16; jx++) {
                  const int hw = ch * 16 + jx - i * HWS;
                  if (hw >= 0 && hw < HW_) dsm[r * HW_ + hw] = fmaf(inv, __uint_as_float(v2[jx]), __uint_as_float(v1[jx]) + wg);
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
        // smem must stay valid until the bulk engine has read it (CUTLASS tma_store_wait form)
        if (et == 0) asm volatile("cp.async.bulk.wait_group.read 0;" ::: "memory");
        done = true;
      }
    }
    if (!done) {
    float* fst = reinterpret_cast<float*>(stg_all + ew * (TC_STGB / 4));
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
        inv_s[et] = valid ? 1.f / (__ldcg(p.gsum + b) / (float)C4 + p.eps) : 0.f;
        cb_s[et] = valid ? b * C_ * HW_ + hw : -1;
      }
      ebar();
      const uint32_t buf = ui & 1;
      mb_wait(tf2 + buf, (ui >> 1) & 1);
      tfa();
#if TC_STAMPS
      if (warp == 4 && lane == 0) { if (ui == 0) STAMP(17); STAMP(16); }
#endif
      if constexpr (HW_ % 4 == 0) {
        // 14x14: 4 consecutive tokens of one image are 16-B aligned in NCHW, so each lane (output channel) reduces
        // straight from registers with a float4 red.add
        float* orow = p.out + (size_t)(c0 + r) * HW_;
#pragma unroll
        for (int cs = 0; cs < 2; cs++) {
          const int ch = grp + 2 * cs;
          uint32_t v1[32], v2[32];
          tld32(tbase + ((uint32_t)(q * 32) << 16) + buf * 256 + ch * 32, v1);
          tld32(tbase + ((uint32_t)(q * 32) << 16) + buf * 256 + NB2 + ch * 32, v2);
#pragma unroll
          for (int g4 = 0; g4 < 8; g4++) {
            const int jc = g4 * 4;
            const int cb = cb_s[ch * 32 + jc];
            if (cb >= 0) {
              float4 o;
              o.x = fmaf(inv_s[ch * 32 + jc + 0], __uint_as_float(v2[jc + 0]), __uint_as_float(v1[jc + 0]) + wg);
              o.y = fmaf(inv_s[ch * 32 + jc + 1], __uint_as_float(v2[jc + 1]), __uint_as_float(v1[jc + 1]) + wg);
              o.z = fmaf(inv_s[ch * 32 + jc + 2], __uint_as_float(v2[jc + 2]), __uint_as_float(v1[jc + 2]) + wg);
              o.w = fmaf(inv_s[ch * 32 + jc + 3], __uint_as_float(v2[jc + 3]), __uint_as_float(v1[jc + 3]) + wg);
              atomicAdd(reinterpret_cast<float4*>(orow + cb), o);
            }
          }
        }
      } else {
        // 7x7 (legacy, NI2 = 0): transpose 16 columns through shared memory so 16 lanes reduce 16 consecutive tokens
#pragma unroll
        for (int cs = 0; cs < 2; cs++) {
          const int ch = grp + 2 * cs;
          uint32_t v1[32], v2[32];
          tld32(tbase + ((uint32_t)(q * 32) << 16) + buf * 256 + ch * 32, v1);
          tld32(tbase + ((uint32_t)(q * 32) << 16) + buf * 256 + NB2 + ch * 32, v2);
#pragma unroll
          for (int hh = 0; hh < 2; hh++) {
#pragma unroll
            for (int j = 0; j < 16; j++) { const int jc = hh * 16 + j; fst[lane * 17 + j] = fmaf(inv_s[ch * 32 + jc], __uint_as_float(v2[jc]), __uint_as_float(v1[jc]) + wg); }
            __syncwarp();
            const int cj = lane & 15;
            const int cb = cb_s[ch * 32 + hh * 16 + cj];
            if (cb >= 0) {
              float* ob = p.out + cb + (size_t)(c0 + q * 32) * HW_;
#pragma unroll 4
              for (int rr = 0; rr < 16; rr++) { const int row = 2 * rr + (lane >> 4); atomicAdd(ob + (size_t)row * HW_, fst[row * 17 + cj]); }
            }
            __syncwarp();
          }
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

// =====================================================================================================
// tcgen05 swap-AB cooperative path for the small-C rows (C = 96 / 160 / 192 at 28x28, 56x56, 449x7)
// Phase A: the weight unit loads (fp16 W1' with the LN-weight fold, row sums u, v = W1.lnb + b1, fp16 W2, W2.gb
//   partials; at most SC_MAXWU units per warp) are issued first and stay in flight during the conv units: (image,
//   PG-plane group, row strip) from 16-B-aligned superset bulk copies; each conv thread loops over a plane subgroup,
//   writes fp16 d and out = x + b2 from registers and keeps its tokens' LN partial sums in registers; the weight
//   conversion runs last. Sync 1 (cg grid.sync).
//   r9: the dwb and b2 vectors arrive by bulk copy with the plane copies (no cold LDG per plane in the conv loop).
//   r12 (H48 gate, refuted): skipping this weight pass entirely moves the row times by +0.05 us on average; the chain
//   is hidden under the conv, which (latency-bound at low occupancy) is the real phase-A term here.
// Phase B: GEMM1 tiles = 128 hidden rows x one NC1-token chunk of one image; LN fold, exact GELU (A&S erfc form, r8),
//   fp16 h, per-(image,k) sum of squares with one red.add per row per tile. Sync 2.
// Phase C: GEMM2 split-K units (128 output channels x NC2 tokens x K slice); the epilogue warps scale the hidden B
//   tile in smem by s_b[k] = 1 + gw*gf*inv_b before the MMA; coalesced red.add into the prefilled output.
// SC_STAMPS: globaltimer diagnostics (off in the shipped build).
// =====================================================================================================
#ifndef SC_STAMPS
#define SC_STAMPS 0
#endif
#define SC_NT 384
#define SC_RING (192 * 1024)
#define SC_OFF_STG SC_RING
#define SC_OFF_MS (SC_OFF_STG + 8 * TC_STGB)
#define SC_OFF_GF (SC_OFF_MS + 2048)
#define SC_OFF_SS (SC_OFF_GF + 3072)
#define SC_OFF_GW (SC_OFF_SS + 3072)
#define SC_OFF_SQ (SC_OFF_GW + 3072)
#define SC_OFF_BAR (SC_OFF_SQ + 1024)
#define SC_SMEM (SC_OFF_BAR + 40 * 8 + 1024)
#define SC_NST2 6
#define SC_STG2 32768
#define SC_MAXWU 1
struct SCP {
  CUtensorMap mW1, mW2, mD, mH;
  const float *x, *dww, *dwb, *lnw, *lnb, *w1, *b1, *gw, *gb, *w2, *b2;
  float* out;
  __half *d16, *w1h, *w2h, *hid;
  float *part, *uk, *vk, *wgb, *gsq;
  unsigned long long* st;
  int B, H, HW, HWS, Mcols, NS, SR, SLOT, units, PSUB;
  int NC1, N1, NAT1, CPI1, T1, STG1;
  int NC2, N2, NAT2, CPI2, KS, KB2, U2;
  float eps, ln_eps;
};
#if SC_STAMPS
#define SSTAMP(i) do { p.st[cta * 24 + (i)] = gtime(); } while (0)
#else
#define SSTAMP(i) do {} while (0)
#endif

template <int C_, int W_, int PG_>
__global__ void __launch_bounds__(SC_NT, 1) sc_kernel(const __grid_constant__ SCP p) {
  constexpr int C4 = 4 * C_, G = C_ / PG_, HT = C4 / 128, CT = (C_ + 127) / 128, KB1 = (C_ + 63) / 64;
  constexpr int SEG = W_ / 7, NQ = (C_ / 4 + 31) / 32;
  static_assert(W_ % 7 == 0 && C4 % 128 == 0 && C_ % PG_ == 0 && C4 <= 768 && C_ % 4 == 0 && (PG_ * 49) % 4 == 0, "sc shape");
  extern __shared__ __align__(1024) unsigned char sc_raw[];
  unsigned char* sm = sc_raw + ((1024u - (sa(sc_raw) & 1023u)) & 1023u);
  unsigned char* ring = sm;
  uint32_t* stg_all = reinterpret_cast<uint32_t*>(sm + SC_OFF_STG);
  float4* ms_s = reinterpret_cast<float4*>(sm + SC_OFF_MS);
  float* gf_s = reinterpret_cast<float*>(sm + SC_OFF_GF);
  float* ss_s = reinterpret_cast<float*>(sm + SC_OFF_SS);
  float* gw_s = reinterpret_cast<float*>(sm + SC_OFF_GW);
  float* sq_s = reinterpret_cast<float*>(sm + SC_OFF_SQ);
  uint64_t* bars = reinterpret_cast<uint64_t*>(sm + SC_OFF_BAR);
  uint64_t *full1 = bars, *empty1 = bars + 4, *tf1 = bars + 8, *te1 = bars + 10;
  uint64_t *full2 = bars + 12, *ready2 = bars + 18, *empty2 = bars + 24, *tf2 = bars + 30, *te2 = bars + 32, *cbar = bars + 34;
  uint32_t* tslot = reinterpret_cast<uint32_t*>(bars + 36);
  const int tid = threadIdx.x, warp = __shfl_sync(0xffffffffu, tid >> 5, 0), lane = tid & 31;
  const int cta = blockIdx.x, P = gridDim.x;
  cg::grid_group grid = cg::this_grid();
  if (tid == 0) {
    for (int i = 0; i < 35; i++) {
      const bool e8 = (i == 10 || i == 11 || (i >= 18 && i < 24) || i == 32 || i == 33);
      mb_init(bars + i, e8 ? 8u : 1u);
    }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  if (warp == 2) {
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], 512;" ::"r"(sa(tslot)));
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
  }
  tfb(); __syncthreads(); tfa();
  const uint32_t tbase = *tslot;
  if (tid == 0) SSTAMP(0);

  // ---------------- phase A
  const int H = p.H, HW = p.HW, HWS = p.HWS, SR = p.SR, NS = p.NS, SLOT = p.SLOT, PSUB = p.PSUB;
  const int tps = SR * W_;
  const int nu = (cta < p.units) ? (p.units - cta + P - 1) / P : 0;
  float* xs = reinterpret_cast<float*>(ring);
  float* wsm = xs + nu * PG_ * SLOT;
  float* stp = wsm + nu * PG_ * 49;
  float* par = reinterpret_cast<float*>(ring + (((size_t)nu * (PG_ * SLOT + PG_ * 49 + PSUB * tps * 2) * 4 + 15) & ~(size_t)15));
  float* dwb_s = par;
  float* b2_s = par + C_;
  if (tid == 0) {
    // one transaction: gw, dwb, b2 (all CTAs), and per conv unit the PG depthwise filters and PG plane strips (with halo rows)
    uint32_t tot = C4 * 4 + 2 * C_ * 4;
    for (int pass = 0; pass < 2; pass++) {
      if (pass == 1) {
        mb_expect(cbar, tot);
        bulk_g2s(gw_s, p.gw, C4 * 4, cbar);
        bulk_g2s(dwb_s, p.dwb, C_ * 4, cbar);
        bulk_g2s(b2_s, p.b2, C_ * 4, cbar);
      }
      for (int i = 0; i < nu; i++) {
        const int u = cta + i * P, b = u / (G * NS), rr = u - b * G * NS, g = rr / NS, s = rr - g * NS;
        const int h0 = s * SR, h1 = min(H, h0 + SR), r0 = max(0, h0 - 3), r1 = min(H, h1 + 3);
        if (pass == 0) tot += PG_ * 49 * 4;
        else bulk_g2s(wsm + i * PG_ * 49, p.dww + (size_t)g * PG_ * 49, PG_ * 49 * 4, cbar);
        for (int pp = 0; pp < PG_; pp++) {
          const size_t pb = ((size_t)b * C_ + g * PG_ + pp) * HW;
          const size_t a0 = ((pb + (size_t)r0 * W_) * 4) & ~(size_t)15;
          const size_t a1 = ((pb + (size_t)r1 * W_) * 4 + 15) & ~(size_t)15;
          if (pass == 0) tot += (uint32_t)(a1 - a0);
          else bulk_g2s(xs + (i * PG_ + pp) * SLOT, reinterpret_cast<const char*>(p.x) + a0, (uint32_t)(a1 - a0), cbar);
        }
      }
    }
  }
  for (int i = cta * SC_NT + tid; i < p.B * C4; i += P * SC_NT) p.gsq[i] = 0.f;
  // weight units of C floats each, highest CTAs first; at most SC_MAXWU units per warp, loaded now, converted after the conv
  const int nw = P * (SC_NT / 32), wbase = (P - 1 - cta) * (SC_NT / 32) + warp;
  float4 wv[SC_MAXWU][NQ];
#pragma unroll
  for (int k = 0; k < SC_MAXWU; k++) {
    const int wu = wbase + k * nw;
    if (wu < 8 * C_) {
      const bool is1 = wu < C4;
      const int row = is1 ? wu : ((wu - C4) >> 2), qtr = is1 ? 0 : ((wu - C4) & 3);
      const float4* src = reinterpret_cast<const float4*>(is1 ? p.w1 + (size_t)row * C_ : p.w2 + (size_t)row * C4 + qtr * C_);
#pragma unroll
      for (int j = 0; j < NQ; j++) { const int c4 = lane + 32 * j; wv[k][j] = (c4 < C_ / 4) ? ldnc4(src + c4) : make_float4(0.f, 0.f, 0.f, 0.f); }
    }
  }
  if (tid == 0) SSTAMP(6);
  mb_wait(cbar, 0);
  if (nu > 0) {
    if (tid == 0) SSTAMP(7);
    // conv thread = (unit, plane subgroup, row, 7-wide segment): loops over PG/PSUB planes, writes fp16 d and
    // out = x + b2 from registers, keeps its 7 tokens' LN partial sums in registers; rows are clamped, not branched
    const int pos = SR * SEG, per_unit = PSUB * pos;
    for (int itm = tid; itm < nu * per_unit; itm += SC_NT) {
      const int ui = itm / per_unit, r1_ = itm - ui * per_unit, ps = r1_ / pos, r2_ = r1_ - ps * pos;
      const int hl = r2_ / SEG, w0 = (r2_ - hl * SEG) * 7;
      const int u = cta + ui * P, b = u / (G * NS), rr = u - b * G * NS, g = rr / NS, s = rr - g * NS;
      const int h0 = s * SR, h = h0 + hl;
      if (h >= min(H, h0 + SR)) continue;
      const int r0 = max(0, h0 - 3);
      float s1[7], s2[7];
#pragma unroll
      for (int o = 0; o < 7; o++) { s1[o] = 0.f; s2[o] = 0.f; }
#pragma unroll 1
      for (int pp = ps; pp < PG_; pp += PSUB) {
        const int c = g * PG_ + pp;
        const size_t pbase = ((size_t)b * C_ + c) * HW;
        const float* xp = xs + (ui * PG_ + pp) * SLOT + (int)((pbase + (size_t)r0 * W_) & 3) - r0 * W_;
        const float* wp = wsm + (ui * PG_ + pp) * 49;
        float acc[7], xc[7];
#pragma unroll
        for (int o = 0; o < 7; o++) acc[o] = 0.f;
#pragma unroll
        for (int i = 0; i < 7; i++) {
          const int hh = h + i - 3;
          const bool rok = (hh >= 0) && (hh < H);
          const int hs = rok ? hh : h;
          float v[13];
#pragma unroll
          for (int jj = 0; jj < 13; jj++) {
            const int col = w0 - 3 + jj;
            const bool cok = (col >= 0) && (col < W_);
            const float xv = xp[hs * W_ + min(max(col, 0), W_ - 1)];
            v[jj] = (rok && cok) ? xv : 0.f;
          }
          if (i == 3) {
#pragma unroll
            for (int o = 0; o < 7; o++) xc[o] = v[o + 3];
          }
#pragma unroll
          for (int jj = 0; jj < 7; jj++) {
            const float wvv = wp[i * 7 + jj];
#pragma unroll
            for (int o = 0; o < 7; o++) acc[o] = fmaf(v[o + jj], wvv, acc[o]);
          }
        }
        const float bias = dwb_s[c], b2c = b2_s[c];
        __half* dd = p.d16 + (size_t)c * p.Mcols + b * HWS + h * W_ + w0;
        float* oo = p.out + pbase + h * W_ + w0;
#pragma unroll
        for (int o = 0; o < 7; o++) {
          const float dv = acc[o] + bias;
          dd[o] = __float2half_rn(dv);
          oo[o] = xc[o] + b2c;
          s1[o] += dv; s2[o] = fmaf(dv, dv, s2[o]);
        }
      }
      float* sp = stp + ((size_t)(ui * PSUB + ps) * tps + hl * W_ + w0) * 2;
#pragma unroll
      for (int o = 0; o < 7; o++) { sp[2 * o] = s1[o]; sp[2 * o + 1] = s2[o]; }
    }
    __syncthreads();
    if (tid == 0) SSTAMP(8);
    for (int e = tid; e < nu * tps; e += SC_NT) {
      const int ui = e / tps, lt = e - ui * tps;
      const int u = cta + ui * P, b = u / (G * NS), rr = u - b * G * NS, g = rr / NS, s = rr - g * NS;
      const int h0 = s * SR;
      if (h0 + lt / W_ >= min(H, h0 + SR)) continue;
      float a1 = 0.f, a2 = 0.f;
      for (int q2 = 0; q2 < PSUB; q2++) { const float2 v = reinterpret_cast<const float2*>(stp)[(size_t)(ui * PSUB + q2) * tps + lt]; a1 += v.x; a2 += v.y; }
      reinterpret_cast<float2*>(p.part)[((size_t)b * G + g) * HW + h0 * W_ + lt] = make_float2(a1, a2);
    }
    if (HWS > HW) {
      for (int e = tid; e < nu * PG_ * 8; e += SC_NT) {
        const int pl = e >> 3, jj = e & 7, ui = pl / PG_, pp = pl - ui * PG_;
        const int u = cta + ui * P, b = u / (G * NS), rr = u - b * G * NS, g = rr / NS, s = rr - g * NS;
        if (min(H, s * SR + SR) == H && HW + jj < HWS)
          p.d16[(size_t)(g * PG_ + pp) * p.Mcols + b * HWS + HW + jj] = __float2half_rn(0.f);
      }
    }
  }
  {
    // weight conversion (arithmetic and order unchanged)
#pragma unroll
    for (int k = 0; k < SC_MAXWU; k++) {
      const int wu = wbase + k * nw;
      if (wu < 8 * C_) {
        const bool is1 = wu < C4;
        const int row = is1 ? wu : ((wu - C4) >> 2), qtr = is1 ? 0 : ((wu - C4) & 3);
        uint2* dst = reinterpret_cast<uint2*>(is1 ? p.w1h + (size_t)row * C_ : p.w2h + (size_t)row * C4 + qtr * C_);
        if (is1) {
          float su = 0.f, sv = 0.f;
#pragma unroll
          for (int j = 0; j < NQ; j++) {
            const int c4 = lane + 32 * j;
            if (c4 < C_ / 4) {
              const float4 lw = __ldg(reinterpret_cast<const float4*>(p.lnw) + c4);
              const float4 lb = __ldg(reinterpret_cast<const float4*>(p.lnb) + c4);
              const __half2 h0 = __floats2half2_rn(wv[k][j].x * lw.x, wv[k][j].y * lw.y);
              const __half2 h1 = __floats2half2_rn(wv[k][j].z * lw.z, wv[k][j].w * lw.w);
              const float2 f0 = __half22float2(h0), f1 = __half22float2(h1);
              su += (f0.x + f0.y) + (f1.x + f1.y);
              sv += wv[k][j].x * lb.x + wv[k][j].y * lb.y + wv[k][j].z * lb.z + wv[k][j].w * lb.w;
              uint2 o; o.x = h2u(h0); o.y = h2u(h1);
              dst[c4] = o;
            }
          }
          su = warp_sum(su); sv = warp_sum(sv);
          if (lane == 0) { p.uk[row] = su; p.vk[row] = sv + __ldg(p.b1 + row); }
        } else {
          const float4* gb4 = reinterpret_cast<const float4*>(p.gb + qtr * C_);
          float sg = 0.f;
#pragma unroll
          for (int j = 0; j < NQ; j++) {
            const int c4 = lane + 32 * j;
            if (c4 < C_ / 4) {
              const float4 g4 = __ldg(gb4 + c4);
              sg += wv[k][j].x * g4.x + wv[k][j].y * g4.y + wv[k][j].z * g4.z + wv[k][j].w * g4.w;
              uint2 o; o.x = h2u(__floats2half2_rn(wv[k][j].x, wv[k][j].y)); o.y = h2u(__floats2half2_rn(wv[k][j].z, wv[k][j].w));
              dst[c4] = o;
            }
          }
          sg = warp_sum(sg);
          if (lane == 0) p.wgb[row * 4 + qtr] = sg;
        }
      }
    }
  }
#if SC_STAMPS
  __syncthreads(); if (tid == 0) SSTAMP(1);
#endif
  asm volatile("fence.proxy.async.global;" ::: "memory");
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  tfb(); grid.sync(); tfa();
  if (tid == 0) SSTAMP(2);

  // ---------------- phase B: GEMM1 per (hidden tile, token chunk)
  if (warp == 0) {
    asm volatile("fence.proxy.async.global;" ::: "memory");
    uint32_t it = 0;
    for (int t = cta; t < p.T1; t += P) {
      const int ht = t % HT, cj = t / HT, b = cj / p.CPI1, j = cj - b * p.CPI1;
      const int m0 = ht * 128, tok0 = b * HWS + j * p.NC1;
      for (int kb = 0; kb < KB1; kb++, it++) {
        const int s = it & 3; const uint32_t n = it >> 2;
        if (it >= 4) mb_wait(empty1 + s, (n & 1) ^ 1);
        if (elect1()) {
          unsigned char* st = ring + s * p.STG1;
          mb_expect(full1 + s, 16384 + p.NAT1 * 8192);
          tma2d(st, &p.mW1, kb * 64, m0, full1 + s);
          for (int a = 0; a < p.NAT1; a++) tma2d(st + 16384 + a * 8192, &p.mD, tok0 + a * 64, kb * 64, full1 + s);
        }
        __syncwarp();
      }
    }
  } else if (warp == 1) {
    uint32_t it = 0, ti = 0;
    const uint32_t idesc = (1u << 4) | (1u << 16) | ((uint32_t)(p.N1 >> 3) << 17) | (8u << 24);
    for (int t = cta; t < p.T1; t += P, ti++) {
      const uint32_t buf = ti & 1;
      if (ti >= 2) { mb_wait(te1 + buf, ((ti >> 1) & 1) ^ 1); tfa(); }
      const uint32_t d = tbase + buf * 256;
      for (int kb = 0; kb < KB1; kb++, it++) {
        const int s = it & 3; const uint32_t n = it >> 2;
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
  } else if (warp >= 4) {
    const int ew = warp - 4, grp = ew >> 2, q = warp & 3, r = q * 32 + lane, et = ew * 32 + lane;
    uint32_t* stg = stg_all + ew * (TC_STGB / 4);
    const int nch = (p.N1 + 31) >> 5;
    uint32_t ti = 0;
    for (int t = cta; t < p.T1; t += P, ti++) {
      const int ht = t % HT, cj = t / HT, b = cj / p.CPI1, j = cj - b * p.CPI1;
      const int m0 = ht * 128, tok0 = b * HWS + j * p.NC1, vcols = min(p.NC1, HW - j * p.NC1);
      const int k = m0 + r;
      const float uk = __ldcg(p.uk + k), vk = __ldcg(p.vk + k);
      ebar();
      if (et < 128) {
        // per-token LayerNorm statistics from the per-plane-group partials; one token per thread
        float4 m = make_float4(0.f, 0.f, 0.f, 0.f);
        if (et < vcols) {
          const float2* pp = reinterpret_cast<const float2*>(p.part) + (size_t)b * G * HW + j * p.NC1 + et;
          float s1 = 0.f, s2 = 0.f;
#pragma unroll
          for (int g = 0; g < G; g++) { const float2 v = __ldcg(pp + (size_t)g * HW); s1 += v.x; s2 += v.y; }
          const float mu = s1 / (float)C_;
          const float var = fmaxf(s2 / (float)C_ - mu * mu, 0.f);
          m = make_float4(mu, rsqrtf(var + p.ln_eps), 1.f, 0.f);
        }
        ms_s[et] = m;
      }
      ebar();
      const uint32_t buf = ti & 1;
      mb_wait(tf1 + buf, (ti >> 1) & 1);
      tfa();
      const uint32_t taddr = tbase + ((uint32_t)(q * 32) << 16) + buf * 256;
      __half* hbase = p.hid + (size_t)(m0 + q * 32) * p.Mcols + tok0;
      float sq = 0.f;
#pragma unroll 1
      for (int ch = grp; ch < nch; ch += 2) {
        uint32_t v[32];
        tld32(taddr + ch * 32, v);
        uint32_t hw2[16];
#pragma unroll
        for (int jj = 0; jj < 32; jj += 2) {
          float hf[2];
#pragma unroll
          for (int e = 0; e < 2; e++) {
            const float4 m = ms_s[ch * 32 + jj + e];
            const float z = fmaf(m.y, __uint_as_float(v[jj + e]) - m.x * uk, vk);
            const float h = __half2float(__float2half_rn(gelu_as(z)));
            hf[e] = h;
            sq += (m.z != 0.f) ? h * h : 0.f;
          }
          hw2[jj >> 1] = h2u(__floats2half2_rn(hf[0], hf[1]));
        }
        uint4* srow = reinterpret_cast<uint4*>(stg + lane * 20);
#pragma unroll
        for (int qd = 0; qd < 4; qd++) srow[qd] = make_uint4(hw2[4 * qd], hw2[4 * qd + 1], hw2[4 * qd + 2], hw2[4 * qd + 3]);
        __syncwarp();
#pragma unroll
        for (int i2 = 0; i2 < 4; i2++) {
          const int row = i2 * 8 + (lane >> 2), colb = ch * 32 + (lane & 3) * 8;
          const uint4 val = *reinterpret_cast<const uint4*>(stg + row * 20 + (lane & 3) * 4);
          if (colb < vcols) *reinterpret_cast<uint4*>(hbase + (size_t)row * p.Mcols + colb) = val;
        }
        __syncwarp();
      }
      tfb(); __syncwarp();
      if (lane == 0) mb_arrive1(te1 + buf);
      sq_s[grp * 128 + r] = sq;
      ebar();
      if (grp == 0) atomicAdd(p.gsq + (size_t)b * C4 + k, sq_s[r] + sq_s[128 + r]);
    }
    if (warp == 4 && lane == 0) SSTAMP(3);
  }
  asm volatile("fence.proxy.async.global;" ::: "memory");
  tfb(); grid.sync(); tfa();

  // ---------------- phase C: GEMM2 split-K with the GRN scale applied to the hidden tile in smem
  if (warp == 0) {
    asm volatile("fence.proxy.async.global;" ::: "memory");
    uint32_t it = 0;
    for (int u = cta; u < p.U2; u += P) {
      const int ks = u % p.KS, rest = u / p.KS, ct = rest % CT, cj = rest / CT, b = cj / p.CPI2, j = cj - b * p.CPI2;
      const int c0 = ct * 128, tok0 = b * HWS + j * p.NC2, kb0 = ks * p.KB2;
      for (int kk = 0; kk < p.KB2; kk++, it++) {
        const int s = it % SC_NST2; const uint32_t n = it / SC_NST2;
        if (it >= SC_NST2) mb_wait(empty2 + s, (n & 1) ^ 1);
        if (elect1()) {
          unsigned char* st = ring + s * SC_STG2;
          mb_expect(full2 + s, 16384 + p.NAT2 * 8192);
          tma2d(st, &p.mW2, (kb0 + kk) * 64, c0, full2 + s);
          for (int a = 0; a < p.NAT2; a++) tma2d(st + 16384 + a * 8192, &p.mH, tok0 + a * 64, (kb0 + kk) * 64, full2 + s);
        }
        __syncwarp();
      }
    }
  } else if (warp == 1) {
    uint32_t it = 0, ui = 0;
    const uint32_t idesc = (1u << 4) | (1u << 16) | ((uint32_t)(p.N2 >> 3) << 17) | (8u << 24);
    for (int u = cta; u < p.U2; u += P, ui++) {
      const uint32_t buf = ui & 1;
      if (ui >= 2) { mb_wait(te2 + buf, ((ui >> 1) & 1) ^ 1); tfa(); }
      const uint32_t d = tbase + buf * 256;
      for (int kk = 0; kk < p.KB2; kk++, it++) {
        const int s = it % SC_NST2; const uint32_t n = it / SC_NST2;
        mb_wait(ready2 + s, n & 1);
        tfa();
        if (elect1()) {
          const uint32_t a0 = sa(ring + s * SC_STG2), b0 = a0 + 16384;
#pragma unroll
          for (int k = 0; k < 4; k++) umma(d, sdesc(a0 + k * 32, 16, 1024), sdesc(b0 + k * 2048, 8192, 1024), idesc, (uint32_t)(kk | k));
          umma_commit(empty2 + s);
          if (kk == p.KB2 - 1) umma_commit(tf2 + buf);
        }
        __syncwarp();
      }
    }
  } else if (warp >= 4) {
    if (warp == 4 && lane == 0) SSTAMP(5);
    const int ew = warp - 4, grp = ew >> 2, q = warp & 3, r = q * 32 + lane, et = ew * 32 + lane;
    float* fst = reinterpret_cast<float*>(stg_all + ew * (TC_STGB / 4));
    const bool hw4 = (HW & 3) == 0;
    const int nch2 = (p.N2 + 31) >> 5;
    uint32_t ui = 0, itc = 0;
    for (int u = cta; u < p.U2; u += P, ui++) {
      const int ks = u % p.KS, rest = u / p.KS, ct = rest % CT, cj = rest / CT, b = cj / p.CPI2, j = cj - b * p.CPI2;
      const int c0 = ct * 128, kb0 = ks * p.KB2, vcols = min(p.NC2, HW - j * p.NC2);
      float wg = 0.f;
      if (ks == 0 && c0 + r < C_) { const float4 w4 = __ldcg(reinterpret_cast<const float4*>(p.wgb) + (c0 + r)); wg = (w4.x + w4.y) + (w4.z + w4.w); }
      ebar();
      // GRN: gf = sqrt(sum of squares), inv_b = 1 / (mean_k gf + eps), s[k] = 1 + gw[k] * gf[k] * inv_b for this K slice
      {
        float a = 0.f;
        for (int kk2 = et; kk2 < C4; kk2 += 256) { const float g = sqrtf(__ldcg(p.gsq + (size_t)b * C4 + kk2)); gf_s[kk2] = g; a += g; }
        a = warp_sum(a);
        if (lane == 0) sq_s[ew] = a;
      }
      ebar();
      {
        float tot = 0.f;
#pragma unroll
        for (int w8 = 0; w8 < 8; w8++) tot += sq_s[w8];
        const float inv = 1.f / (tot / (float)C4 + p.eps);
        for (int i = et; i < p.KB2 * 64; i += 256) { const int kx = kb0 * 64 + i; ss_s[i] = fmaf(gw_s[kx] * gf_s[kx], inv, 1.f); }
      }
      ebar();
      for (int kk = 0; kk < p.KB2; kk++, itc++) {
        const int s = itc % SC_NST2; const uint32_t n = itc / SC_NST2;
        mb_wait(full2 + s, n & 1);
        uint4* bb = reinterpret_cast<uint4*>(ring + s * SC_STG2 + 16384);
        const float* sk = ss_s + kk * 64;
        for (int ci = et; ci < p.NAT2 * 512; ci += 256) {
          const float sc = sk[(ci >> 3) & 63];
          uint4 w = bb[ci];
          w.x = scale_h2(w.x, sc); w.y = scale_h2(w.y, sc); w.z = scale_h2(w.z, sc); w.w = scale_h2(w.w, sc);
          bb[ci] = w;
        }
        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
        __syncwarp();
        if (lane == 0) mb_arrive1(ready2 + s);
      }
      const uint32_t buf = ui & 1;
      mb_wait(tf2 + buf, (ui >> 1) & 1);
      tfa();
      const int crow0 = c0 + q * 32;
      float* obase = p.out + ((size_t)b * C_ + crow0) * HW + j * p.NC2;
#pragma unroll 1
      for (int ch = grp; ch < nch2; ch += 2) {
        uint32_t v[32];
        tld32(tbase + ((uint32_t)(q * 32) << 16) + buf * 256 + ch * 32, v);
#pragma unroll
        for (int hh = 0; hh < 2; hh++) {
#pragma unroll
          for (int jj = 0; jj < 16; jj++) fst[lane * 17 + jj] = __uint_as_float(v[hh * 16 + jj]) + wg;
          __syncwarp();
          const int colb = ch * 32 + hh * 16;
          if (hw4) {
#pragma unroll
            for (int i4 = 0; i4 < 4; i4++) {
              const int row = i4 * 8 + (lane >> 2), cq = (lane & 3) * 4;
              if (crow0 + row < C_ && colb + cq < vcols) {
                const float* f = fst + row * 17 + cq;
                atomicAdd(reinterpret_cast<float4*>(obase + (size_t)row * HW + colb + cq), make_float4(f[0], f[1], f[2], f[3]));
              }
            }
          } else {
#pragma unroll 4
            for (int rr = 0; rr < 16; rr++) {
              const int row = 2 * rr + (lane >> 4), cx = lane & 15;
              if (crow0 + row < C_ && colb + cx < vcols) atomicAdd(obase + (size_t)row * HW + colb + cx, fst[row * 17 + cx]);
            }
          }
          __syncwarp();
        }
      }
      tfb(); __syncwarp();
      if (lane == 0) mb_arrive1(te2 + buf);
    }
    if (warp == 4 && lane == 0) SSTAMP(4);
  }
  tfb(); __syncthreads();
  if (warp == 2) asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, 512;" ::"r"(tbase));
}

// =====================================================================================================
// r10 tc2_kernel (H41): wide-C rows with balanced image-straddling token ranges, 512 threads, 16 warps.
// Flat layouts d16 [C][B*HW] and hid [4C][B*HW] fp16 (token = b*HW + hw, row pitch B*HW; range starts are multiples
// of 8 tokens, so every TMA box start is 16-B aligned). One GEMM1 tile (128 hidden rows x R1 tokens) and one GEMM2 unit
// (128 output channels x R2 tokens x one K slice) per CTA; the host plan guarantees tiles, units <= grid.
// Phase A: as tc_kernel (d16 stored in the flat layout as half2 pairs; gsum[B][4C] zeroed). Sync 1.
// Phase B: GEMM1 with N = round16(R1) (two MMAs 128 + (N-128) into adjacent TMEM columns when N > 256); LN statistics
//   and the deferred W2 conversion on warps 4-15 during the mainloop; epilogue on all 16 warps: LN fold, A&S erfc GELU,
//   fp16 h stores only (no g); the per-(image,k) sumsq is split at the image boundary inside each 16-column chunk
//   (warp-uniform), summed per warp in smem and red.add-ed into gsum. Sync 2.
// Phase C: warps 4-15 compute gf = sqrt(gsum), mean_k gf per image and s[img][k] = 1 + gw*gf/(mean+eps) for the unit's
//   <= 7 images and K slice into smem while the first stages land (gw slice loaded first); then all 12 warps scale
//   every landed h tile in place by s[img(token)][k] (per element, 8-token vectors can straddle an image),
//   fence.proxy.async and arrive on the stage's ready barrier (NGRP stage-owner groups are supported, NGRP = 1 is
//   used: r10 stamps showed 4-warp groups took ~1.1 us per 56 KB stage); single accumulator; all-warp drain with
//   red.add into the x + b2 prefill (+ W2.gb on the first K slice).
// TC2_STAMPS: globaltimer diagnostics (off in the shipped build).
// =====================================================================================================
#ifndef TC2_STAMPS
#define TC2_STAMPS 0
#endif
#define T2_RING (192 * 1024)
#define T2_OFF_MS T2_RING
#define T2_OFF_SS (T2_OFF_MS + 4608)
#define T2_SS_BYTES 24576
#define T2_OFF_CB (T2_OFF_SS + T2_SS_BYTES)
#define T2_OFF_RED (T2_OFF_CB + 1152)
#define T2_OFF_BAR (T2_OFF_RED + 512)
#define T2_SMEM (T2_OFF_BAR + 32 * 8 + 1024)
#define T2_NSLOT 8
#define T2_NMAX 288
#define T2_NKL 4
struct TC2P {
  CUtensorMap mW1, mW2, mD, mH;
  const float *x, *dww, *dwb, *lnw, *lnb, *w1, *b1, *gw, *gb, *w2, *b2;
  float* out;
  __half *d16, *w1h, *w2h, *hid;
  float *part, *uk, *vk, *wgb, *gsum;
  unsigned long long* st;
  int B, Mtok, units;
  int R1, N1, NAT1, STG1, NST1, T1;
  int R2, N2, NAT2, STG2, NST2, NGRP, KS, KB2, U2;
  float eps, ln_eps;
};
#if TC2_STAMPS
#define ST2(i) do { p.st[cta * 16 + (i)] = gtime(); } while (0)
#else
#define ST2(i) do {} while (0)
#endif
__device__ __forceinline__ uint32_t scale_h2x(uint32_t u, float sa_, float sb_) {
  const float2 f = __half22float2(*reinterpret_cast<const __half2*>(&u));
  return h2u(__floats2half2_rn(f.x * sa_, f.y * sb_));
}

template <int C_, int HW_>
__global__ void __launch_bounds__(TC_NT, 1) tc2_kernel(const __grid_constant__ TC2P p) {
  constexpr int W_ = (HW_ == 49) ? 7 : 14, H_ = W_;
  constexpr int C4 = 4 * C_, G = C_ / 32, HT = C4 / 128, CT = C_ / 128, KB1 = C_ / 64;
  extern __shared__ __align__(1024) unsigned char tc2_raw[];
  unsigned char* sm = tc2_raw + ((1024u - (sa(tc2_raw) & 1023u)) & 1023u);
  unsigned char* ring = sm;
  float4* ms = reinterpret_cast<float4*>(sm + T2_OFF_MS);
  float* ss = reinterpret_cast<float*>(sm + T2_OFF_SS);
  int* cb_s = reinterpret_cast<int*>(sm + T2_OFF_CB);
  float* red_s = reinterpret_cast<float*>(sm + T2_OFF_RED);
  uint64_t* bars = reinterpret_cast<uint64_t*>(sm + T2_OFF_BAR);
  uint64_t *full1 = bars, *empty1 = bars + 4, *tf1 = bars + 8, *full2 = bars + 9, *ready2 = bars + 15;
  uint64_t *empty2 = bars + 21, *tf2 = bars + 27, *cbar = bars + 28, *sbar = bars + 29;
  uint32_t* tslot = reinterpret_cast<uint32_t*>(bars + 30);
  const int tid = threadIdx.x, warp = __shfl_sync(0xffffffffu, tid >> 5, 0), lane = tid & 31;
  const int cta = blockIdx.x, P = gridDim.x;
  if (tid == 0) {
    const uint32_t gwn = (uint32_t)(12 / p.NGRP);
    for (int i = 0; i < 30; i++) mb_init(bars + i, (i >= 15 && i < 21) ? gwn : 1u);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  if (warp == 2) {
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], 512;" ::"r"(sa(tslot)));
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
  }
  tfb(); __syncthreads(); tfa();
  const uint32_t tbase = *tslot;
  if (tid == 0) ST2(0);

  // ---------------- phase A (as tc_kernel): plane loads, W1 loads in flight, conv, LN partials, flat fp16 d, out = x + b2
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
  for (int i = cta * TC_NT + tid; i < p.B * C4; i += P * TC_NT) p.gsum[i] = 0.f;
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
    // flat token layout: half2 stores of globally even token pairs (single halves at an odd plane start or end)
    constexpr int NPI = HW_ / 2 + 1;
    for (int e = tid; e < nu * 32 * NPI; e += TC_NT) {
      const int pl = e / NPI, m = e - pl * NPI;
      const int ui = pl >> 5, pp = pl & 31;
      const int u = cta + ui * P, b = u / G, g = u - b * G;
      const int hw = 2 * m - ((b * HW_) & 1);
      __half* drow = p.d16 + (size_t)(g * 32 + pp) * p.Mtok + b * HW_;
      const float* src = dbuf + pl * HW_;
      if (hw >= 0 && hw + 1 < HW_) *reinterpret_cast<__half2*>(drow + hw) = __floats2half2_rn(src[hw], src[hw + 1]);
      else if (hw == -1) drow[0] = __float2half_rn(src[0]);
      else if (hw == HW_ - 1) drow[hw] = __float2half_rn(src[hw]);
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
#if TC2_STAMPS
  __syncthreads(); if (tid == 0) ST2(1);
#endif
  asm volatile("fence.proxy.async.global;" ::: "memory");
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  tfb(); gbar(P); tfa();
  if (tid == 0) ST2(2);

  // ---------------- phase B: one GEMM1 tile per CTA
  const int Bz = min(P, p.T1), Ez = P - Bz, Wt = Ez * 16 + Bz * 12;
  if (cta >= p.T1) {
    tc_w2_work<C_>(p, (cta - Bz) * 16 + warp, Wt, lane);
  } else {
    const int ht = cta % HT, rg = cta / HT, m0 = ht * 128, t0 = rg * p.R1, R = p.R1;
    float* sqw = ss;  // [4 warps per quadrant][T2_NSLOT image slots][128 rows] sumsq parts (GEMM2 scale area, free here)
    if (warp == 0) {
      asm volatile("fence.proxy.async.global;" ::: "memory");
      const int nst = p.NST1;
      for (int kb = 0; kb < KB1; kb++) {
        const int s = kb % nst; const uint32_t n = (uint32_t)(kb / nst);
        if (kb >= nst) mb_wait(empty1 + s, (n & 1) ^ 1);
        if (elect1()) {
          unsigned char* st = ring + s * p.STG1;
          mb_expect(full1 + s, 16384 + p.NAT1 * 8192);
          tma2d(st, &p.mW1, kb * 64, m0, full1 + s);
          for (int a = 0; a < p.NAT1; a++) tma2d(st + 16384 + a * 8192, &p.mD, t0 + a * 64, kb * 64, full1 + s);
        }
        __syncwarp();
      }
    } else if (warp == 1) {
      const int nst = p.NST1, Na = p.N1 > 256 ? 128 : p.N1, Nb = p.N1 - Na;
      const uint32_t ida = (1u << 4) | (1u << 16) | ((uint32_t)(Na >> 3) << 17) | (8u << 24);
      const uint32_t idb = (1u << 4) | (1u << 16) | ((uint32_t)(Nb >> 3) << 17) | (8u << 24);
      for (int kb = 0; kb < KB1; kb++) {
        const int s = kb % nst; const uint32_t n = (uint32_t)(kb / nst);
        mb_wait(full1 + s, n & 1);
        tfa();
        if (elect1()) {
          const uint32_t a0 = sa(ring + s * p.STG1), b0 = a0 + 16384;
#pragma unroll
          for (int k = 0; k < 4; k++) {
            const uint64_t ad = sdesc(a0 + k * 32, 16, 1024);
            umma(tbase, ad, sdesc(b0 + k * 2048, 8192, 1024), ida, (uint32_t)(kb | k));
            if (Nb > 0) umma(tbase + 128, ad, sdesc(b0 + 16384 + k * 2048, 8192, 1024), idb, (uint32_t)(kb | k));
          }
          umma_commit(empty1 + s);
          if (kb == KB1 - 1) umma_commit(tf1);
        }
        __syncwarp();
      }
    } else if (warp >= 4) {
      // per-token LN statistics (mu, rstd, -, valid) for the tile's tokens, zeroed sumsq parts, deferred W2 conversion
      const int et = tid - 128;
      if (et < T2_NMAX) {
        const int t = t0 + et;
        float4 m = make_float4(0.f, 0.f, 0.f, 0.f);
        if (et < R && t < p.Mtok) {
          const int b = t / HW_, hw = t - b * HW_;
          const float2* pp = reinterpret_cast<const float2*>(p.part) + (size_t)b * G * HW_ + hw;
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
      for (int i = et; i < 4 * T2_NSLOT * 128; i += 384) sqw[i] = 0.f;
      tc_w2_work<C_>(p, Ez * 16 + cta * 12 + (warp - 4), Wt, lane);
    }
    nbar(1, TC_NT);
    mb_wait(tf1, 0);
    tfa();
    if (tid == 0) ST2(3);
    {
      // warp j of quadrant q takes the 16-column chunks j, j+4, ...; the ring is free (all stages consumed)
      const int q = warp & 3, j = warp >> 2, r = q * 32 + lane, k = m0 + r;
      const float uk = __ldcg(p.uk + k), vk = __ldcg(p.vk + k);
      const int vcols = min(R, p.Mtok - t0), img0 = t0 / HW_, nch = (R + 15) >> 4;
      uint32_t* stg = reinterpret_cast<uint32_t*>(ring + warp * TC_B_STGW);
      __half* hbase = p.hid + (size_t)(m0 + q * 32) * p.Mtok + t0;
      const uint32_t taddr = tbase + ((uint32_t)(q * 32) << 16);
#pragma unroll 1
      for (int ch = j; ch < nch; ch += 4) {
        uint32_t v[16];
        tld16(taddr + ch * 16, v);
        const int cs = t0 + ch * 16, ia = cs / HW_, bnd = (ia + 1) * HW_ - cs;
        float slo = 0.f, shi = 0.f;
        uint32_t w[8];
#pragma unroll
        for (int e = 0; e < 16; e += 2) {
          const float4 ma = ms[ch * 16 + e], mb = ms[ch * 16 + e + 1];
          const float ha = gelu_as(fmaf(ma.y, fmaf(-ma.x, uk, __uint_as_float(v[e])), vk)) * ma.w;
          const float hb = gelu_as(fmaf(mb.y, fmaf(-mb.x, uk, __uint_as_float(v[e + 1])), vk)) * mb.w;
          const __half2 hh = __floats2half2_rn(ha, hb);
          w[e >> 1] = h2u(hh);
          const float2 hr = __half22float2(hh);
          const float qa = hr.x * hr.x, qb = hr.y * hr.y;
          if (e < bnd) slo += qa; else shi += qa;
          if (e + 1 < bnd) slo += qb; else shi += qb;
        }
        float* sw = sqw + (j * T2_NSLOT + (ia - img0)) * 128 + r;
        sw[0] += slo;
        if (bnd < 16) sw[128] += shi;
        tc_stage16(stg, w, hbase, p.Mtok, ch, vcols, lane);
      }
    }
    nbar(1, TC_NT);
    {
      const int img0 = t0 / HW_, nimg = (min(t0 + R, p.Mtok) - 1) / HW_ - img0 + 1;
      for (int idx = tid; idx < nimg * 128; idx += TC_NT) {
        const int sl = idx >> 7, rr = idx & 127;
        float s = 0.f;
#pragma unroll
        for (int jj = 0; jj < 4; jj++) s += sqw[(jj * T2_NSLOT + sl) * 128 + rr];
        atomicAdd(p.gsum + (size_t)(img0 + sl) * C4 + m0 + rr, s);
      }
    }
#if TC2_STAMPS
    nbar(1, TC_NT); if (tid == 0) ST2(4);
#endif
  }
  asm volatile("fence.proxy.async.global;" ::: "memory");
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  tfb(); gbar(P); tfa();
  if (tid == 0) ST2(5);

  // ---------------- phase C: one GEMM2 unit per CTA
  if (cta < p.U2) {
    const int ks = cta % p.KS, rest = cta / p.KS, ct = rest % CT, rg = rest / CT;
    const int c0 = ct * 128, t0 = rg * p.R2, R = p.R2, kb0 = ks * p.KB2, KSL = p.KB2 * 64, k0 = kb0 * 64;
    const int img0 = t0 / HW_, nimg = (min(t0 + R, p.Mtok) - 1) / HW_ - img0 + 1;
    if (warp == 0) {
      asm volatile("fence.proxy.async.global;" ::: "memory");
      const int nst = p.NST2;
      for (int kk = 0; kk < p.KB2; kk++) {
        const int s = kk % nst; const uint32_t n = (uint32_t)(kk / nst);
        if (kk >= nst) mb_wait(empty2 + s, (n & 1) ^ 1);
        if (elect1()) {
          unsigned char* st = ring + s * p.STG2;
          mb_expect(full2 + s, 16384 + p.NAT2 * 8192);
          tma2d(st, &p.mW2, (kb0 + kk) * 64, c0, full2 + s);
          for (int a = 0; a < p.NAT2; a++) tma2d(st + 16384 + a * 8192, &p.mH, t0 + a * 64, (kb0 + kk) * 64, full2 + s);
        }
        __syncwarp();
      }
    } else if (warp == 1) {
      const int nst = p.NST2, Na = p.N2 > 256 ? 128 : p.N2, Nb = p.N2 - Na;
      const uint32_t ida = (1u << 4) | (1u << 16) | ((uint32_t)(Na >> 3) << 17) | (8u << 24);
      const uint32_t idb = (1u << 4) | (1u << 16) | ((uint32_t)(Nb >> 3) << 17) | (8u << 24);
      for (int kk = 0; kk < p.KB2; kk++) {
        const int s = kk % nst; const uint32_t n = (uint32_t)(kk / nst);
        mb_wait(ready2 + s, n & 1);
        tfa();
#if TC2_STAMPS
        if (lane == 0 && kk == 0) ST2(7);
        if (lane == 0 && kk == p.KB2 - 1) ST2(8);
#endif
        if (elect1()) {
          const uint32_t a0 = sa(ring + s * p.STG2), b0 = a0 + 16384;
#pragma unroll
          for (int k = 0; k < 4; k++) {
            const uint64_t ad = sdesc(a0 + k * 32, 16, 1024);
            umma(tbase, ad, sdesc(b0 + k * 2048, 8192, 1024), ida, (uint32_t)(kk | k));
            if (Nb > 0) umma(tbase + 128, ad, sdesc(b0 + 16384 + k * 2048, 8192, 1024), idb, (uint32_t)(kk | k));
          }
          umma_commit(empty2 + s);
          if (kk == p.KB2 - 1) umma_commit(tf2);
        }
        __syncwarp();
      }
    } else if (warp >= 4) {
      const int et = tid - 128;
      if (et < T2_NMAX) {
        const int t = t0 + et;
        int cb = -1;
        if (et < R && t < p.Mtok) { const int b = t / HW_; cb = b * C_ * HW_ + (t - b * HW_); }
        cb_s[et] = cb;
      }
      // gw of this thread's K-slice entries first, in the same load round as gsum
      float gwv[T2_NKL];
#pragma unroll
      for (int jx = 0; jx < T2_NKL; jx++) { const int kl = et + 384 * jx; gwv[jx] = (kl < KSL) ? __ldg(p.gw + k0 + kl) : 0.f; }
      {
        // gf = sqrt(gsum) for the unit's images (one predicated load round per 4 images), per-image sums of gf,
        // and the K-slice values of gf kept in smem (row stride KSL)
        constexpr int NV = C4 / 4, NJ = (NV + 383) / 384;
        float psm[T2_NSLOT];
#pragma unroll
        for (int i = 0; i < T2_NSLOT; i++) psm[i] = 0.f;
#pragma unroll
        for (int h4 = 0; h4 < 2; h4++) {
          float4 v[4][NJ];
#pragma unroll
          for (int i = 0; i < 4; i++)
#pragma unroll
            for (int jj = 0; jj < NJ; jj++) {
              const int sl = h4 * 4 + i, idx = et + 384 * jj;
              v[i][jj] = (sl < nimg && idx < NV) ? __ldcg(reinterpret_cast<const float4*>(p.gsum + (size_t)(img0 + sl) * C4) + idx)
                                                 : make_float4(0.f, 0.f, 0.f, 0.f);
            }
#pragma unroll
          for (int i = 0; i < 4; i++)
#pragma unroll
            for (int jj = 0; jj < NJ; jj++) {
              const int sl = h4 * 4 + i, idx = et + 384 * jj, kx = idx * 4 - k0;
              const float4 g = make_float4(sqrtf(v[i][jj].x), sqrtf(v[i][jj].y), sqrtf(v[i][jj].z), sqrtf(v[i][jj].w));
              psm[sl] += (g.x + g.y) + (g.z + g.w);
              if (sl < nimg && idx < NV && kx >= 0 && kx < KSL) *reinterpret_cast<float4*>(ss + sl * KSL + kx) = g;
            }
        }
#pragma unroll
        for (int i = 0; i < T2_NSLOT; i++) { const float a = warp_sum(psm[i]); if (lane == 0) red_s[(warp - 4) * T2_NSLOT + i] = a; }
      }
      nbar(2, 384);
      {
        // s = 1 + gw * gf / (mean_k gf + eps)
        float inv[T2_NSLOT];
#pragma unroll
        for (int sl = 0; sl < T2_NSLOT; sl++) {
          float tot = 0.f;
#pragma unroll
          for (int w = 0; w < 12; w++) tot += red_s[w * T2_NSLOT + sl];
          inv[sl] = 1.f / (tot / (float)C4 + p.eps);
        }
#pragma unroll
        for (int jx = 0; jx < T2_NKL; jx++) {
          const int kl = et + 384 * jx;
          if (kl < KSL) {
#pragma unroll
            for (int sl = 0; sl < T2_NSLOT; sl++)
              if (sl < nimg) { float* sp = ss + sl * KSL + kl; *sp = fmaf(gwv[jx] * *sp, inv[sl], 1.f); }
          }
        }
      }
      nbar(2, 384);
#if TC2_STAMPS
      if (tid == 128) ST2(6);
#endif
      // stage-owner groups: group grp scales k-blocks grp, grp + NGRP, ... (always the same stages, in order)
      const int gwn = 12 / p.NGRP, grp = (warp - 4) / gwn, gt = (warp - 4 - grp * gwn) * 32 + lane, gth = gwn * 32;
      const int nv = p.NAT2 * 512;
      for (int kk = grp; kk < p.KB2; kk += p.NGRP) {
        const int s = kk % p.NST2; const uint32_t n = (uint32_t)(kk / p.NST2);
        mb_wait(full2 + s, n & 1);
        uint4* bt = reinterpret_cast<uint4*>(ring + s * p.STG2 + 16384);
        const float* sk = ss + kk * 64;
        for (int vi = gt; vi < nv; vi += gth) {
          // 16-B vector vi = (64-token chunk, k row, swizzled slot): 8 consecutive tokens of one k row
          const int kr = (vi >> 3) & 63, col = (vi >> 9) * 64 + (((vi & 7) ^ (kr & 7)) << 3);
          if (col < R) {
            const int t = t0 + col, ia = t / HW_, bnd = (ia + 1) * HW_ - t;
            const int sl = min(ia - img0, nimg - 1), sl2 = min(sl + 1, nimg - 1);
            const float fa = sk[sl * KSL + kr], fb = sk[sl2 * KSL + kr];
            uint4 w = bt[vi];
            w.x = scale_h2x(w.x, 0 < bnd ? fa : fb, 1 < bnd ? fa : fb);
            w.y = scale_h2x(w.y, 2 < bnd ? fa : fb, 3 < bnd ? fa : fb);
            w.z = scale_h2x(w.z, 4 < bnd ? fa : fb, 5 < bnd ? fa : fb);
            w.w = scale_h2x(w.w, 6 < bnd ? fa : fb, 7 < bnd ? fa : fb);
            bt[vi] = w;
          }
        }
        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
        __syncwarp();
        if (lane == 0) mb_arrive1(ready2 + s);
      }
    }
    nbar(1, TC_NT);
    {
      // drain on all 16 warps: warp j of quadrant q takes the 32-column chunks j, j+4, ...
      const int q = warp & 3, j = warp >> 2, r = q * 32 + lane;
      float wg = 0.f;
      if (ks == 0) { const float4 w4 = __ldcg(reinterpret_cast<const float4*>(p.wgb) + (c0 + r)); wg = (w4.x + w4.y) + (w4.z + w4.w); }
      mb_wait(tf2, 0);
      tfa();
      if (tid == 0) ST2(9);
      const int nch = (R + 31) >> 5;
      const uint32_t taddr = tbase + ((uint32_t)(q * 32) << 16);
      if constexpr (HW_ % 4 == 0) {
        float* orow = p.out + (size_t)(c0 + r) * HW_;
#pragma unroll 1
        for (int ch = j; ch < nch; ch += 4) {
          uint32_t v[32];
          tld32(taddr + ch * 32, v);
#pragma unroll
          for (int g4 = 0; g4 < 8; g4++) {
            const int jc = g4 * 4, cb = cb_s[ch * 32 + jc];
            if (cb >= 0)
              atomicAdd(reinterpret_cast<float4*>(orow + cb),
                        make_float4(__uint_as_float(v[jc]) + wg, __uint_as_float(v[jc + 1]) + wg,
                                    __uint_as_float(v[jc + 2]) + wg, __uint_as_float(v[jc + 3]) + wg));
          }
        }
      } else {
        float* fst = reinterpret_cast<float*>(ring) + warp * (32 * 17);
#pragma unroll 1
        for (int ch = j; ch < nch; ch += 4) {
          uint32_t v[32];
          tld32(taddr + ch * 32, v);
#pragma unroll
          for (int hh = 0; hh < 2; hh++) {
#pragma unroll
            for (int jj = 0; jj < 16; jj++) fst[lane * 17 + jj] = __uint_as_float(v[hh * 16 + jj]) + wg;
            __syncwarp();
            const int cj = lane & 15, cb = cb_s[ch * 32 + hh * 16 + cj];
            if (cb >= 0) {
              float* ob = p.out + cb + (size_t)(c0 + q * 32) * HW_;
#pragma unroll 4
              for (int rr = 0; rr < 16; rr++) { const int row = 2 * rr + (lane >> 4); atomicAdd(ob + (size_t)row * HW_, fst[row * 17 + cj]); }
            }
            __syncwarp();
          }
        }
      }
    }
  }
  tfb(); __syncthreads();
  if (tid == 0) ST2(10);
  if (warp == 2) asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, 512;" ::"r"(tbase));
}

// ---------------------------------------------------------------- host side of the tcgen05 paths
// NI = images per GEMM1 tile, KS = GEMM2 K split, NI2 = images per GEMM2 unit (r11: whole-image units with the
// bulk-reduce drain, 7x7 only; 0 = 128-column units with red.add), MP = 128-channel tiles per GEMM2 unit (r11: 2 at
// 32,768 only; 16,768 KS 6 / 8,768 KS 12 with MP 2 measured -0.4 / +0.6 us, because the reduction bytes double).
// U2 = (C/128/MP) x column groups x KS = 144 units on 148 SMs at 2/8/16/32,768.
struct TCRow { int B, C, H, W, NI, KS, NI2, MP; };
static const TCRow kTCRows[6] = {
    {2, 768, 7, 7, 1, 12, 1, 1}, {8, 768, 7, 7, 2, 6, 2, 1}, {16, 768, 7, 7, 3, 3, 2, 1},
    {32, 768, 7, 7, 3, 3, 2, 2}, {8, 384, 14, 14, 1, 3, 0, 1}, {16, 384, 14, 14, 1, 3, 0, 1},
};

struct WSArgs {
  const float *x, *dww, *dwb, *lnw, *lnb, *w1, *b1, *gw, *gb, *w2, *b2;
  float* out; void* ws; unsigned long long* stamps;
  int B, C, H, W; float eps, ln_eps;
};

static inline size_t tc_al(size_t v) { return (v + 255) / 256 * 256; }

struct TCPlan { TCRow r; int HW, HWS, Mcols, C4, G, T1, NAT1, STG1, U2, KB2, NI2, NAT2, MP;
                size_t o_d16, o_w1h, o_w2h, o_hid, o_gq, o_part, o_uk, o_vk, o_wgb, o_gsum, bytes; };

static bool tc_plan(int B, int C, int H, int W, TCPlan& pl) {
  int idx = -1;
  for (int i = 0; i < 6; i++)
    if (kTCRows[i].B == B && kTCRows[i].C == C && kTCRows[i].H == H && kTCRows[i].W == W) idx = i;
  if (idx < 0) return false;
  const TCRow r = kTCRows[idx];
  pl.r = r;
  pl.HW = H * W; pl.HWS = (pl.HW + 7) / 8 * 8; pl.Mcols = B * pl.HWS; pl.C4 = 4 * C; pl.G = C / 32;
  pl.T1 = (pl.C4 / 128) * ((B + r.NI - 1) / r.NI);
  pl.NAT1 = (((r.NI * pl.HWS + 15) & ~15) + 63) / 64;
  if (pl.NAT1 > 4 || r.NI > 3) return false;
  pl.STG1 = 16384 + pl.NAT1 * 8192;
  pl.KB2 = (pl.C4 / 64) / r.KS;
  if (pl.KB2 * r.KS * 64 != pl.C4) return false;
  pl.NI2 = (TC_BULK && pl.HW == 49) ? r.NI2 : 0;
  if (pl.NI2 < 0 || pl.NI2 > 2) return false;
  pl.NAT2 = pl.NI2 ? (pl.NI2 * pl.HWS + 63) / 64 : 2;
  if (pl.NAT2 > 2) return false;
  pl.MP = (TC_ONEMMA && pl.NI2 > 0 && (C / 128) % 2 == 0 && r.MP == 2) ? 2 : 1;
  const int ng2 = pl.NI2 ? (B + pl.NI2 - 1) / pl.NI2 : (pl.Mcols + 127) / 128;
  pl.U2 = (C / 128 / pl.MP) * ng2 * r.KS;
  size_t o = 0;
  pl.o_d16 = o; o = tc_al(o + (size_t)C * pl.Mcols * 2);
  pl.o_w1h = o; o = tc_al(o + (size_t)pl.C4 * C * 2);
  pl.o_w2h = o; o = tc_al(o + (size_t)pl.C4 * C * 2);
  pl.o_hid = o; o = tc_al(o + (size_t)pl.C4 * pl.Mcols * 2);
  pl.o_gq = o; o = tc_al(o + (size_t)pl.C4 * pl.Mcols * 2);
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

static const char* tc_launch(const WSArgs* a, int nsm, void* stream_v) {
  TCPlan pl;
  if (!tc_plan(a->B, a->C, a->H, a->W, pl)) return "ws_launch: unsupported shape";
  {
    // phase-A smem budget (slots per conv unit x units per CTA must stay below the LN/param area) and worker counts of
    // the W1 preload (<= MW1 units per warp) and the deferred W2 pass (<= MW units per worker)
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
  p.hid = (__half*)(wsb + pl.o_hid); p.gq = (__half*)(wsb + pl.o_gq);
  p.part = (float*)(wsb + pl.o_part); p.uk = (float*)(wsb + pl.o_uk); p.vk = (float*)(wsb + pl.o_vk);
  p.wgb = (float*)(wsb + pl.o_wgb); p.gsum = (float*)(wsb + pl.o_gsum);
  p.st = a->stamps;  // diagnostics only: read by the kernel only when built with -DTC_STAMPS=1
  const int C = a->C, C4 = pl.C4;
  if (!enc16(&p.mW1, p.w1h, C, C4, 64, 128)) return "ws_launch: map W1";
  if (!enc16(&p.mW2, p.w2h, C4, C, 64, 128)) return "ws_launch: map W2";
  if (!enc16(&p.mD, p.d16, pl.Mcols, C, 64, 64)) return "ws_launch: map D";
  if (!enc16(&p.mH, p.hid, pl.Mcols, C4, 64, 64)) return "ws_launch: map H";
  if (!enc16(&p.mG, p.gq, pl.Mcols, C4, 64, 64)) return "ws_launch: map G";
  p.x = a->x; p.dww = a->dww; p.dwb = a->dwb; p.lnw = a->lnw; p.lnb = a->lnb; p.w1 = a->w1; p.b1 = a->b1;
  p.gw = a->gw; p.gb = a->gb; p.w2 = a->w2; p.b2 = a->b2; p.out = a->out;
  p.B = a->B; p.Mcols = pl.Mcols; p.units = a->B * pl.G; p.NI = pl.r.NI; p.T1 = pl.T1; p.STG1 = pl.STG1;
  p.NST1 = (TC_B_STG / pl.STG1 < 4) ? TC_B_STG / pl.STG1 : 4;
  p.KS = pl.r.KS; p.KB2 = pl.KB2; p.U2 = pl.U2; p.NI2 = pl.NI2; p.NAT2 = pl.NAT2; p.MP = pl.MP;
  p.eps = a->eps; p.ln_eps = a->ln_eps;
  void* args[] = {&p};
  const void* fn = (C == 768) ? (const void*)tc_kernel<768, 49> : (const void*)tc_kernel<384, 196>;
  cudaError_t e = cudaLaunchCooperativeKernel(fn, dim3(nsm), dim3(TC_NT), args, (size_t)TC_SMEM, (cudaStream_t)stream_v);
  if (e != cudaSuccess) return cudaGetErrorString(e);
  return nullptr;
}

// small-C rows: plane group PG (template: 16 at C=192, 4 at C=96/160), row strips NS, GEMM1 token chunk NC1,
// GEMM2 token chunk NC2, GEMM2 K split KS (r12: NS re-swept against 2-3 alternatives per row; the table values win)
struct SCRow { int B, C, H, W, PG, NS, NC1, NC2, KS; };
static const SCRow kSCRows[7] = {
    {1, 192, 28, 28, 16, 10, 56, 56, 4}, {4, 192, 28, 28, 16, 3, 56, 112, 2}, {8, 192, 28, 28, 16, 3, 112, 112, 1},
    {2, 96, 56, 56, 4, 3, 64, 112, 2},   {1, 96, 56, 56, 4, 6, 64, 64, 3},    {1, 160, 449, 7, 4, 3, 112, 112, 2},
    {4, 96, 56, 56, 4, 3, 64, 112, 1},
};
struct SCPlan { SCRow r; int HW, HWS, Mcols, C4, G, SR, SLOT, units, PSUB, unitf, N1, NAT1, STG1, CPI1, T1, N2, NAT2, CPI2, KB2, U2;
                size_t o_d16, o_w1h, o_w2h, o_hid, o_part, o_uk, o_vk, o_wgb, o_gsq, bytes; };

static bool sc_plan(int B, int C, int H, int W, int nsm, SCPlan& pl) {
  int idx = -1;
  for (int i = 0; i < 7; i++)
    if (kSCRows[i].B == B && kSCRows[i].C == C && kSCRows[i].H == H && kSCRows[i].W == W) idx = i;
  if (idx < 0 || nsm <= 0) return false;
  const SCRow r = kSCRows[idx];
  pl.r = r;
  if (!((C == 192 && W == 28 && r.PG == 16) || (C == 96 && W == 56 && r.PG == 4) || (C == 160 && W == 7 && r.PG == 4))) return false;
  if ((8 * C + (SC_NT / 32) * nsm - 1) / ((SC_NT / 32) * nsm) > SC_MAXWU) return false;
  pl.HW = H * W; pl.HWS = (pl.HW + 7) / 8 * 8; pl.Mcols = B * pl.HWS; pl.C4 = 4 * C; pl.G = C / r.PG;
  pl.SR = (H + r.NS - 1) / r.NS;
  if ((r.NS - 1) * pl.SR >= H) return false;
  pl.SLOT = ((pl.SR + 6) * W + 8 + 3) & ~3;
  pl.units = B * pl.G * r.NS;
  {
    // plane subgroups per conv unit so that the CTA's threads are used
    const int numax = (pl.units + nsm - 1) / nsm, pos = pl.SR * (W / 7);
    int target = SC_NT / (numax * pos); if (target < 1) target = 1;
    pl.PSUB = 1;
    for (int d = 1; d <= r.PG; d++) if (r.PG % d == 0 && d <= target) pl.PSUB = d;
  }
  pl.unitf = r.PG * pl.SLOT + r.PG * 49 + pl.PSUB * pl.SR * W * 2;
  pl.N1 = (r.NC1 + 15) / 16 * 16; pl.NAT1 = (pl.N1 + 63) / 64; pl.STG1 = 16384 + pl.NAT1 * 8192;
  pl.CPI1 = (pl.HW + r.NC1 - 1) / r.NC1; pl.T1 = (pl.C4 / 128) * B * pl.CPI1;
  pl.N2 = (r.NC2 + 15) / 16 * 16; pl.NAT2 = (pl.N2 + 63) / 64;
  pl.CPI2 = (pl.HW + r.NC2 - 1) / r.NC2;
  pl.KB2 = (pl.C4 / 64) / r.KS;
  if (pl.KB2 * r.KS * 64 != pl.C4 || pl.KB2 * 64 > 768 || pl.NAT2 > 2 || pl.N1 > 128 || pl.N2 > 256) return false;
  if ((r.NC1 % 8) || (r.NC2 % 8) || 4 * pl.STG1 > SC_RING) return false;
  if ((size_t)((pl.units + nsm - 1) / nsm) * pl.unitf * 4 + (size_t)2 * C * 4 + 16 > (size_t)SC_RING) return false;
  pl.U2 = ((C + 127) / 128) * B * pl.CPI2 * r.KS;
  size_t o = 0;
  pl.o_d16 = o; o = tc_al(o + (size_t)C * pl.Mcols * 2);
  pl.o_w1h = o; o = tc_al(o + (size_t)pl.C4 * C * 2);
  pl.o_w2h = o; o = tc_al(o + (size_t)pl.C4 * C * 2);
  pl.o_hid = o; o = tc_al(o + (size_t)pl.C4 * pl.Mcols * 2);
  pl.o_part = o; o = tc_al(o + (size_t)B * pl.G * pl.HW * 8);
  pl.o_uk = o; o = tc_al(o + (size_t)pl.C4 * 4);
  pl.o_vk = o; o = tc_al(o + (size_t)pl.C4 * 4);
  pl.o_wgb = o; o = tc_al(o + (size_t)C * 16);
  pl.o_gsq = o; o = tc_al(o + (size_t)B * pl.C4 * 4);
  pl.bytes = o;
  return true;
}

static int sm_count() {
  static int nsm = 0;
  if (nsm == 0) { int dev = 0; cudaGetDevice(&dev); cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, dev); }
  return nsm;
}

// r10 tc2 rows: GEMM1 token ranges NR1 (R1 = round8(ceil(B*HW / NR1))), GEMM2 token ranges NR2, GEMM2 K split KS.
// Shape-only table; a row whose plan does not fit (grid < tiles or units, B*HW % 8 != 0) stays on tc_kernel.
// Only rows where tc2 beat tc_kernel in-container are listed (r10: 16,384 only; 8,768 / 16,768 / 32,768 / 8,384 lost).
struct TC2Row { int B, C, H, W, NR1, NR2, KS; };
static const TC2Row kTC2Rows[] = {
    {16, 384, 14, 14, 12, 24, 2},
};
static const int kTC2N = (int)(sizeof(kTC2Rows) / sizeof(kTC2Rows[0]));
struct TC2Plan { int HW, Mtok, C4, G, R1, N1, NAT1, STG1, NST1, T1, R2, N2, NAT2, STG2, NST2, NGRP, KS, KB2, U2;
                 size_t o_d16, o_w1h, o_w2h, o_hid, o_part, o_uk, o_vk, o_wgb, o_gsum, bytes; };

static bool tc2_plan(int B, int C, int H, int W, int nsm, TC2Plan& pl) {
  int idx = -1;
  for (int i = 0; i < kTC2N; i++)
    if (kTC2Rows[i].B == B && kTC2Rows[i].C == C && kTC2Rows[i].H == H && kTC2Rows[i].W == W) idx = i;
  if (idx < 0 || nsm <= 0 || !TC2_ENABLE) return false;
  const TC2Row r = kTC2Rows[idx];
  pl.HW = H * W; pl.Mtok = B * pl.HW; pl.C4 = 4 * C; pl.G = C / 32;
  if (pl.Mtok % 8) return false;
  pl.R1 = ((pl.Mtok + r.NR1 - 1) / r.NR1 + 7) / 8 * 8;
  pl.N1 = (pl.R1 + 15) / 16 * 16; pl.NAT1 = (pl.N1 + 63) / 64; pl.STG1 = 16384 + pl.NAT1 * 8192;
  pl.NST1 = T2_RING / pl.STG1; if (pl.NST1 > 4) pl.NST1 = 4;
  pl.T1 = (pl.C4 / 128) * ((pl.Mtok + pl.R1 - 1) / pl.R1);
  pl.R2 = ((pl.Mtok + r.NR2 - 1) / r.NR2 + 7) / 8 * 8;
  pl.N2 = (pl.R2 + 15) / 16 * 16; pl.NAT2 = (pl.N2 + 63) / 64; pl.STG2 = 16384 + pl.NAT2 * 8192;
  pl.NST2 = T2_RING / pl.STG2; if (pl.NST2 > 6) pl.NST2 = 6;
  pl.NGRP = 1;  // every scaling warp works on every stage (NST2 % NGRP == 0 holds trivially)
  pl.KS = r.KS; pl.KB2 = (pl.C4 / 64) / r.KS;
  if (pl.KB2 * r.KS * 64 != pl.C4 || pl.KB2 * 64 > T2_NKL * 384) return false;
  pl.U2 = (C / 128) * ((pl.Mtok + pl.R2 - 1) / pl.R2) * r.KS;
  if (pl.T1 > nsm || pl.U2 > nsm || pl.N1 > 272 || pl.N2 > 272 || pl.NST1 < 2 || pl.NST2 < 2) return false;
  // image slots per token range (sumsq parts and scale rows), and the scale area [slots][K slice] fp32
  if ((pl.R1 - 1) / pl.HW + 2 > 7 || (pl.R2 - 1) / pl.HW + 2 > 7) return false;
  if ((size_t)((pl.R2 - 1) / pl.HW + 2) * pl.KB2 * 64 * 4 > (size_t)T2_SS_BYTES) return false;
  size_t o = 0;
  pl.o_d16 = o; o = tc_al(o + (size_t)C * pl.Mtok * 2);
  pl.o_w1h = o; o = tc_al(o + (size_t)pl.C4 * C * 2);
  pl.o_w2h = o; o = tc_al(o + (size_t)pl.C4 * C * 2);
  pl.o_hid = o; o = tc_al(o + (size_t)pl.C4 * pl.Mtok * 2);
  pl.o_part = o; o = tc_al(o + (size_t)B * pl.G * pl.HW * 8);
  pl.o_uk = o; o = tc_al(o + (size_t)pl.C4 * 4);
  pl.o_vk = o; o = tc_al(o + (size_t)pl.C4 * 4);
  pl.o_wgb = o; o = tc_al(o + (size_t)C * 16);
  pl.o_gsum = o; o = tc_al(o + (size_t)B * pl.C4 * 4);
  pl.bytes = o;
  return true;
}

static const char* tc2_launch(const WSArgs* a, int nsm, const TC2Plan& pl, void* stream_v) {
  {
    const int units = a->B * pl.G, numax = (units + nsm - 1) / nsm;
    if ((size_t)numax * (2 * 32 * pl.HW + 32 * 49) * 4 > (size_t)TC_PA_MAX) return "tc2: phase A smem budget";
    const int MW1 = (a->C >= 768) ? 2 : 1;
    if (MW1 * nsm * (TC_NT / 32) < pl.C4) return "tc2: W1 worker count";
    const int Bz = nsm < pl.T1 ? nsm : pl.T1, Wt = (nsm - Bz) * 16 + Bz * 12;
    if (MW1 * Wt < pl.C4) return "tc2: W2 worker count";
  }
  TC2P p;
  memset(&p, 0, sizeof(p));
  char* wsb = (char*)a->ws;
  p.d16 = (__half*)(wsb + pl.o_d16); p.w1h = (__half*)(wsb + pl.o_w1h); p.w2h = (__half*)(wsb + pl.o_w2h);
  p.hid = (__half*)(wsb + pl.o_hid);
  p.part = (float*)(wsb + pl.o_part); p.uk = (float*)(wsb + pl.o_uk); p.vk = (float*)(wsb + pl.o_vk);
  p.wgb = (float*)(wsb + pl.o_wgb); p.gsum = (float*)(wsb + pl.o_gsum);
  p.st = a->stamps;  // diagnostics only: read by the kernel only when built with -DTC2_STAMPS=1
  const int C = a->C, C4 = pl.C4;
  if (!enc16(&p.mW1, p.w1h, C, C4, 64, 128)) return "tc2: map W1";
  if (!enc16(&p.mW2, p.w2h, C4, C, 64, 128)) return "tc2: map W2";
  if (!enc16(&p.mD, p.d16, pl.Mtok, C, 64, 64)) return "tc2: map D";
  if (!enc16(&p.mH, p.hid, pl.Mtok, C4, 64, 64)) return "tc2: map H";
  p.x = a->x; p.dww = a->dww; p.dwb = a->dwb; p.lnw = a->lnw; p.lnb = a->lnb; p.w1 = a->w1; p.b1 = a->b1;
  p.gw = a->gw; p.gb = a->gb; p.w2 = a->w2; p.b2 = a->b2; p.out = a->out;
  p.B = a->B; p.Mtok = pl.Mtok; p.units = a->B * pl.G;
  p.R1 = pl.R1; p.N1 = pl.N1; p.NAT1 = pl.NAT1; p.STG1 = pl.STG1; p.NST1 = pl.NST1; p.T1 = pl.T1;
  p.R2 = pl.R2; p.N2 = pl.N2; p.NAT2 = pl.NAT2; p.STG2 = pl.STG2; p.NST2 = pl.NST2; p.NGRP = pl.NGRP;
  p.KS = pl.KS; p.KB2 = pl.KB2; p.U2 = pl.U2; p.eps = a->eps; p.ln_eps = a->ln_eps;
  void* args[] = {&p};
  const void* fn = (C == 768) ? (const void*)tc2_kernel<768, 49> : (const void*)tc2_kernel<384, 196>;
  cudaError_t e = cudaLaunchCooperativeKernel(fn, dim3(nsm), dim3(TC_NT), args, (size_t)T2_SMEM, (cudaStream_t)stream_v);
  if (e != cudaSuccess) return cudaGetErrorString(e);
  return nullptr;
}

extern "C" long long ws_workspace_bytes(int B, int C, int H, int W) {
  long long best = 0;
  TC2Plan p2;
  if (tc2_plan(B, C, H, W, sm_count(), p2)) best = (long long)p2.bytes;
  TCPlan pl;
  if (tc_plan(B, C, H, W, pl)) return best > (long long)pl.bytes ? best : (long long)pl.bytes;
  SCPlan sp;
  if (sc_plan(B, C, H, W, sm_count(), sp)) return (long long)sp.bytes;
  return best;
}

static const char* sc_launch(const WSArgs* a, int nsm, void* stream_v) {
  SCPlan pl;
  if (!sc_plan(a->B, a->C, a->H, a->W, nsm, pl)) return "ws_launch: unsupported shape";
  SCP p;
  memset(&p, 0, sizeof(p));
  char* wsb = (char*)a->ws;
  p.d16 = (__half*)(wsb + pl.o_d16); p.w1h = (__half*)(wsb + pl.o_w1h); p.w2h = (__half*)(wsb + pl.o_w2h);
  p.hid = (__half*)(wsb + pl.o_hid);
  p.part = (float*)(wsb + pl.o_part); p.uk = (float*)(wsb + pl.o_uk); p.vk = (float*)(wsb + pl.o_vk);
  p.wgb = (float*)(wsb + pl.o_wgb); p.gsq = (float*)(wsb + pl.o_gsq);
  p.st = a->stamps;  // diagnostics only: read by the kernel only when built with -DSC_STAMPS=1
  const int C = a->C, C4 = pl.C4;
  if (!enc16(&p.mW1, p.w1h, C, C4, 64, 128)) return "ws_launch: map W1";
  if (!enc16(&p.mW2, p.w2h, C4, C, 64, 128)) return "ws_launch: map W2";
  if (!enc16(&p.mD, p.d16, pl.Mcols, C, 64, 64)) return "ws_launch: map D";
  if (!enc16(&p.mH, p.hid, pl.Mcols, C4, 64, 64)) return "ws_launch: map H";
  p.x = a->x; p.dww = a->dww; p.dwb = a->dwb; p.lnw = a->lnw; p.lnb = a->lnb; p.w1 = a->w1; p.b1 = a->b1;
  p.gw = a->gw; p.gb = a->gb; p.w2 = a->w2; p.b2 = a->b2; p.out = a->out;
  p.B = a->B; p.H = a->H; p.HW = pl.HW; p.HWS = pl.HWS; p.Mcols = pl.Mcols; p.NS = pl.r.NS; p.SR = pl.SR;
  p.SLOT = pl.SLOT; p.units = pl.units; p.PSUB = pl.PSUB;
  p.NC1 = pl.r.NC1; p.N1 = pl.N1; p.NAT1 = pl.NAT1; p.CPI1 = pl.CPI1; p.T1 = pl.T1; p.STG1 = pl.STG1;
  p.NC2 = pl.r.NC2; p.N2 = pl.N2; p.NAT2 = pl.NAT2; p.CPI2 = pl.CPI2; p.KS = pl.r.KS; p.KB2 = pl.KB2; p.U2 = pl.U2;
  p.eps = a->eps; p.ln_eps = a->ln_eps;
  void* args[] = {&p};
  const void* fn = (C == 192) ? (const void*)sc_kernel<192, 28, 16> : (C == 96) ? (const void*)sc_kernel<96, 56, 4>
                                                                                 : (const void*)sc_kernel<160, 7, 4>;
  cudaError_t e = cudaLaunchCooperativeKernel(fn, dim3(nsm), dim3(SC_NT), args, (size_t)SC_SMEM, (cudaStream_t)stream_v);
  if (e != cudaSuccess) return cudaGetErrorString(e);
  return nullptr;
}

// mode: -1 = shape dispatch (tc2 where its plan fits, else tc, else sc); 0 = never tc2; 1 = tc2 only (probe builds)
extern "C" const char* ws_launch_mode(const WSArgs* a, void* stream_v, int mode) {
  static bool attr_done = false;
  const int nsm = sm_count();
  if (!attr_done) {
    // one maximal value for every instantiation (never shrunk later; the portal plans all shapes first)
    cudaError_t e;
    e = cudaFuncSetAttribute(tc_kernel<768, 49>, cudaFuncAttributeMaxDynamicSharedMemorySize, 227 * 1024);
    if (e != cudaSuccess) return cudaGetErrorString(e);
    e = cudaFuncSetAttribute(tc_kernel<384, 196>, cudaFuncAttributeMaxDynamicSharedMemorySize, 227 * 1024);
    if (e != cudaSuccess) return cudaGetErrorString(e);
    e = cudaFuncSetAttribute(tc2_kernel<768, 49>, cudaFuncAttributeMaxDynamicSharedMemorySize, 227 * 1024);
    if (e != cudaSuccess) return cudaGetErrorString(e);
    e = cudaFuncSetAttribute(tc2_kernel<384, 196>, cudaFuncAttributeMaxDynamicSharedMemorySize, 227 * 1024);
    if (e != cudaSuccess) return cudaGetErrorString(e);
    e = cudaFuncSetAttribute(sc_kernel<192, 28, 16>, cudaFuncAttributeMaxDynamicSharedMemorySize, 227 * 1024);
    if (e != cudaSuccess) return cudaGetErrorString(e);
    e = cudaFuncSetAttribute(sc_kernel<96, 56, 4>, cudaFuncAttributeMaxDynamicSharedMemorySize, 227 * 1024);
    if (e != cudaSuccess) return cudaGetErrorString(e);
    e = cudaFuncSetAttribute(sc_kernel<160, 7, 4>, cudaFuncAttributeMaxDynamicSharedMemorySize, 227 * 1024);
    if (e != cudaSuccess) return cudaGetErrorString(e);
    attr_done = true;
  }
  if (mode != 0) {
    TC2Plan p2;
    if (tc2_plan(a->B, a->C, a->H, a->W, nsm, p2)) return tc2_launch(a, nsm, p2, stream_v);
    if (mode == 1) return "ws_launch: tc2 plan does not fit";
  }
  TCPlan pl;
  if (tc_plan(a->B, a->C, a->H, a->W, pl)) return tc_launch(a, nsm, stream_v);
  return sc_launch(a, nsm, stream_v);
}

extern "C" const char* ws_launch(const WSArgs* a, void* stream_v) { return ws_launch_mode(a, stream_v, -1); }
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

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("run", &run, "ConvNeXtV2 block with GRN (tcgen05 swap-AB paths + fp32 anchor)"); }
```

```yaml design-card
id: r12-d-e1-h48-gate-parent
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
  H48 is refuted at its STEP 0 gate. Removing the sc_kernel weight chain entirely changes nothing: no fp32 W1/W2 loads,
  no conversion, no fp16 scratch writes. Stub (a) moved the five gate rows by -0.47 / -0.06 / +0.42 / +0.02 / +0.32 us,
  about +0.05 us on average, where the gate needed -1.2. The weight loads are already in flight under the depthwise
  conv, which takes 2.8-10.5 us after the planes land and is the real phase-A term on these rows. Owner-converted
  resident weights cannot save time that is already hidden. Per the E1 directive, the parent ships unchanged: the
  code is byte-identical apart from two comments.
expected_effect:
  S:
    pct: 0
    confidence: high
  M:
    pct: 0
    confidence: high
  L:
    pct: 0
    confidence: high
resources_sm100a:
  regs_per_thread: 168
  smem_per_cta_bytes: 232448
  threads_per_cta: 384
  ctas_per_sm: 1
  bytes_in_flight_per_sm: 196608
  launches_per_call: 1
knobs:
  sc_rows: "unchanged r10/r11 kSCRows table (NS re-swept in r12: the table values win)"
  TC_ONEMMA: 1
  TC_BULK: 1
  TC_STAMPS: 0
  SC_STAMPS: 0
dispatch:
  - shapes: "2,768,7,7 8,768,7,7 16,768,7,7 32,768,7,7 8,384,14,14"
    kernel: "tc_kernel (r11, unchanged)"
    meta: {grid: "num SMs, cooperative"}
  - shapes: "16,384,14,14"
    kernel: "tc2_kernel<384,196> (r10, unchanged)"
    meta: {}
  - shapes: "C <= 192 rows, 1,160,449,7"
    kernel: "sc_kernel (r10, unchanged)"
    meta: {}
  - shapes: "8,320 / 32,128 / 64,96"
    kernel: "convnext_fused (r0 anchor, merge-only)"
    meta: {}
tests:
  - H48
findings:
  - "Fast loop: an sc_kernel-only .cu (helpers + sc_kernel + sc_plan/launch, about 500 lines) builds 7 nvcc variants in parallel in 5.6 s and is called through ctypes (a WSArgs Structure, workspace from torch.empty, the current stream). harness_time over 5 rows x 4 variants x 3 rounds takes about 4 s. The parent-equivalent build matches the torch fp32 reference within 1.8-2.3e-3 on all 7 sc rows, with the same regs as the parent (168/168/122) and STACK 0 [probe_b200]"
  - "STEP 0 gate, interleaved x3 (rented us, parent / stub a = no weight pass, garbage scratch / stub b = a + no GEMM1 A TMA / stub c = b + no GEMM2 A TMA): 1,192 18.17 / 17.70 / 17.52 / 17.52; 2,96 25.04 / 24.98 / 24.58 / 24.82; 4,192 28.46 / 28.89 / 28.31 / 28.60; 1,160 25.06 / 25.08 / 24.78 / 24.59; 8,192 42.30 / 42.62 / 41.89 / 41.99. Stub (a) vs parent: -0.47 / -0.06 / +0.42 / +0.02 / +0.32 us, mean +0.05 against the -1.2 us gate. GATE FAILED; H48 refuted. Even stub (b), which also skips the post-sync TMA of A, saves only 0.16-0.64 us (mean -0.39) [probe_b200]"
  - "Stamps (globaltimer, after a 252 MB flush, CTA medians from the first CTA start, median of 5 calls; parent -> stub a), in us: weight loads issued / planes landed / conv end / phase-A end / sync-1 release. 1,192 1.79 / 2.30 / 5.12 / 5.89 -> 5.63 / 7.94 -> 7.17. 2,96 1.02 / 2.05 / 7.94 / 9.73 -> 9.47 / 11.52 -> 11.26. 4,192 1.79 / 2.30 / 8.45 / 9.47 -> 9.22 / 11.26 -> 11.01. 1,160 1.28 / 2.05 / 6.14 / 7.94 -> 7.68 / 9.73 -> 9.47. 8,192 2.82 / 3.33 / 13.82 / 15.36 -> 14.85 / 17.41 -> 17.66. The W1/W2 fp32 loads do not delay the plane copies (landing at 2.0-3.3 us is the same with or without them). The conversion after the conv is the only exposed weight term: 0.25-0.5 us of phase-A end, and 0.25-0.8 us at sync 1, which the cg barrier then absorbs. The 2.6-3.1 us chain in the r7 ledger predates r8's loads-first order [probe_b200]"
  - "KEY: the sc rows' phase A is the depthwise conv, 2.8 / 5.9 / 6.2 / 4.1 / 10.5 us after the planes land at 1,192 / 2,96 / 4,192 / 1,160 / 8,192. That is roughly 5-10x its FMA bound (1,192 about 0.2 us, 8,192 about 1.6 us at 1.95 GHz) and about 3x a per-SM LDS estimate (8,192 about 2.6 us). The cause is latency at low occupancy: 1,192 runs 192 thread-items (6 warps per SM, 1 plane each) on 120 of 148 CTAs, and 8,192 runs 320 items of 4 serial planes [probe_b200, INFERRED]"
  - "Lead (not shipped): moving the conv's scattered stores out of the loop (fp16 d staged in smem, then one warp per plane writing d16 and out = x + b2 with lane-contiguous stores from the plane copy; arithmetic and roundings unchanged, max diff vs the parent <= 3e-4 from red.add order). Interleaved x3, two independent runs, rented us: 4,96 -1.38 / -1.36; 1,160 -0.62 / -1.00; 2,96 -0.63 / -0.41; 8,192 -0.47 / -1.03; 1,96 -0.06 / -0.04; 1,192 +0.57 / -0.13; 4,192 +0.30 / -0.26. Regs fall 168 -> 107 at C=96/192. A timing-only stub without any conv stores bounds the store share at 0.1-2.5 us (8,192 -2.5, 4,96 -2.25, 1,160 -1.34). A shape-dispatched coalesced path on 4,96 / 1,160 / 2,96 / 8,192 is worth about -0.5..-1.4 us there (about +0.0005-0.001 of mean). It needs run_tests and a single-seed check before the merge [probe_b200]"
  - "Dead end: NS (row strips) re-swept on all 7 sc rows, with and without coalescing (1,192 NS 10/14/28; 1,96 6/8/14; 2,96 3/4/7; 1,160 3/4/6; 4,96 3/4/7; 4,192 3/4/7; 8,192 3/4/7). The shipped NS is fastest on every row; more strips (more CTAs, fewer rows per item) cost +0.5..+3.4 us because of the extra halo copies and the shorter conv segments per thread [probe_b200]"
  - "Next lever for the sc rows (r13): the conv itself, not the weights. Use all 12 warps per CTA with smaller items (split the 7-wide segment or the tap rows, or give the 28 idle CTAs at 1,192 a share), register-blocked patches reading taps once per lane, and the coalesced store path above. The ceiling is about -1.5 us at 1,192 and -5..-6 us at 8,192 / 4,96 if the conv reached ~2x its LDS bound [INFERRED]"
  - "Budget used: 0 compile_b200, 0 run_tests (the shipped code is byte-identical to r11-d-e1-g2-mp2-bulk, which passed 16/16 including the portal-order re-runs), 7 probes. Precision: arithmetic unchanged, so the r8 10-round log stands (sc 79-89 % of atol, tc 59-65 %). Stamps and stubs were built only in probes; the shipped build has SC_STAMPS / TC_STAMPS = 0 [INFERRED]"
  - "Pitfall: /tmp persisted between some probes in this session and not others (one probe found /tmp/s0 missing). Check for the file and re-embed the source when it is absent [probe_b200]"
paths:
  - {max_tokens: 1568, lang: cuda, width: 128, threads: 512, rows: 128, grid: persistent, mem: tma, launches: 1, x: none, w: none, st: none, cluster: 1, note: "tc_kernel / tc2 (r11, unchanged)"}
  - {max_tokens: 12544, lang: cuda, width: 128, threads: 384, rows: 128, grid: persistent, mem: tma, launches: 1, x: none, w: none, st: none, cluster: 1, note: "sc_kernel unchanged on the C <= 192 rows and 1,160"}
  - {max_tokens: null, lang: cuda, width: 128, threads: 256, rows: 128, grid: persistent, mem: ldg, launches: 1, x: none, w: none, st: none, cluster: 1, note: "r0 fp32 anchor on 8,320 / 32,128 / 64,96 (merge-only)"}
runs_on:
  H200:
    runs: false
    representative: false
    note: "tcgen05/TMEM paths are sm_100a-only"
  A100:
    runs: false
    representative: false
  L40S:
    runs: false
    representative: false
risks:
  - "None new: the code is identical to the parent. The merge should keep r11-merge-b's picks"
measure_first:
  - B200
compliance:
  single_stream: true
  no_cross_call_state: true
  fp32_math: true
  no_threads_or_fork: true
  no_precompiled_binaries: true
```