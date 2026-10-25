### Rationale
I built STEP 2 and it works: the tcgen05 GEMM1 engine was correct on the first build, and h and gsum match torch. But the gate fails: pass 1 takes 547 µs rented at 64,96 against the ≤ 400 µs line, and that also crosses the > 520 µs refutation line. I did not build STEP 3, so the package ships the r0 anchor byte for byte (16/16 pass in run_tests). The validated pass-1 kernel is archived, disabled, in `i_pass1_tc05.cu`.

The ablations show the terms add almost serially: skeleton 255 µs (x-stream latency), conv +100, epilogue maths +120–150, h stores +40–90. Pass 1 is still 18 % faster than G's 668 µs. With G's existing tail the 64,96 row would come to about 744 µs, against 858 µs for r11-merge-b measured in the same container (−13 %). That projection is worth a STEP-3 build next round.

```json solution-spec
{"name": "r12-i-e3-tc05-pass1-547", "definition": "035_convnextv2_block_with_grn", "author": "solx-loop", "spec": {"languages": ["cuda_cpp"], "target_hardware": ["B200"], "entry_point": "binding.cpp::run", "dependencies": ["torch"], "destination_passing_style": true, "compile_options": {"cuda_cflags": ["-O3", "--use_fast_math", "-std=c++17"]}}}
```

```cuda file=kernel.cu
// ConvNeXtV2 block with GRN: one cooperative launch, three phases, fp32 SIMT maths.
//  P1: depthwise 7x7 conv (+bias) -> xn [C][Mp] (K-major), per-(token, 32-channel chunk) LN partial sums;
//      zero GRN sums; transpose weights to [K][N]
//  P2: hidden[4C][Mp] = gelu(LN(xn) @ W1^T + b1); sumsq[b][k] += hidden^2 (per image)
//  P3: out = x + (hidden * (1 + gw*nf) + gb) @ W2^T + b2, written NCHW (split-K with atomics at small sizes)
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

// ---------------------------------------------------------------- phase 1
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

// 4 rows x TW cols output patch of one channel; wt = 49 weights (registers)
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

// MODE 1: patch tiles (TWR = TW in {4,1}); MODE 0: flattened token tiles (TWR = tokens per lane in {1,2,4})
template <int MODE, int TWR>
__device__ void phase1_conv(const Params& p, float* smem) {
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int C = p.C, H = p.H, W = p.W, HW = p.HW, M = p.M, Mp = p.Mp, nch = p.nch;
  float* st_s = smem;                     // [8][P1_SLOTS][2]
  float* wsm = smem + 8 * P1_SLOTS * 2;   // [32][49]
  const int lpr = p.p1_lpr, rg = p.p1_rg, hb = p.p1_hb, wb = p.p1_wb;
  const int rows_per_tile = 4 * rg;
  const int rpt = lpr * TWR;  // slots per row (mode 1)

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

// ---------------------------------------------------------------- GEMM phases
// A = act [K][Mp] (K-major), B = Wt [K][N]. 8 warps = WM x WN spatial warps x KS K-groups. Lane grid 8 x 4.
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

    // per-thread token quad constants
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

    // ---- epilogue: partial tiles -> smem [KS][BN][CS_LD], then one token per thread
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
  if (p.split3 > 1) {  // initialise the output with residual + bias for the split-K atomics
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

// ---------------------------------------------------------------- host
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
  // phase-1 geometry
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
  const int HW = H * W, M = B * HW, C4 = 4 * C, nch = C / 32;
  const int Mp = ((M + 127) / 128) * 128;
  TORCH_CHECK(pwconv1_weight.is_contiguous() && pwconv2_weight.is_contiguous() && dwconv_weight.is_contiguous(), "weights must be contiguous");
  TORCH_CHECK(layernorm_weight.is_contiguous() && layernorm_bias.is_contiguous() && grn_weight.is_contiguous() && grn_bias.is_contiguous(), "params contiguous");
  TORCH_CHECK(dwconv_bias.is_contiguous() && pwconv1_bias.is_contiguous() && pwconv2_bias.is_contiguous(), "biases contiguous");

  // workspace (no state kept across calls)
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

  cudaStream_t stream = at::cuda::getCurrentCUDAStream(x.device().index()).stream();
  const char* err = convnext_launch(p, (void*)stream);
  if (err != nullptr) throw std::runtime_error(std::string("convnext_launch failed: ") + err);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("run", &run, "ConvNeXtV2 block with GRN (fused, fp32)"); }
```

```cuda file=i_pass1_tc05.cu
// Lineage I, r12 E3 STEP 2 (H43): fused pass 1 for 64,96,112,112 with tcgen05 GEMM1. DISABLED (#if 0) in this package:
// every row runs the r0 anchor. Kept as the validated engine for later sessions. Gate (<= 400 us rented) FAILED: 547 us.
// Probe source (NVRTC, extern "C", launch grid 148 x 512 thr, dyn smem 228864 B); host side per call:
//   x map: 3-D FLOAT32 {W, H, B*C}, strides {W*4, H*W*4}, box {120, 8, 2}, no swizzle, OOB zero; start {-4, h0-3, b*C+8g+2sb}
//   w1a  : fp16(lnw*W1) [384][96] in the no-swizzle K-major core-matrix layout W1p.view(48,8,12,8).permute(0,2,1,3)
//          (LBO = 128 B along K, SBO = 1536 B along M: validated, the swapped assignment is wrong)
//   uv   : u[k] = sum_c fp16(lnw*W1)[k][c], v[k] = sum_c W1[k][c]*lnb[c] + b1[k];  dwp = dw padded [C][7][8]
//   hout : fp16 [B][392][HW] (rows 384,385 = 1, 386..391 = 0 for the GEMM2 b2' hi/lo augmentation); gsum [B][384] zeroed.
// Validated vs torch (b = 0 and 63, all rows): h max 1.996e-3 vs a same-rounding fp32 reference (fp16 output rounding),
// 2.6e-3 vs the fp32-weight reference; gsum rel 5.8e-6; stable over repeated launches and harness_time.
// Rented B200 harness_time (us): this build 546.6; direct stores 587; no h stores 509; no epilogue maths/stores 357;
// conv FMAs skipped 521; conv skipped + no epilogue 255 (skeleton: x-stream latency + chain); 4x56 tiles + 4-stage ring 677.
#if 0
typedef unsigned int u32; typedef unsigned long long u64; typedef unsigned short u16;
#ifndef EPI_MODE
#define EPI_MODE 0
#endif
#ifndef STAGE
#define STAGE 1
#endif
#define C 96
#define HH 112
#define WW 112
#define HWv (HH*WW)
#define R 2
#define PW 7
#define CG 8
#define NS 3
#define NROW 8
#define RS 120
#define CS (NROW*RS)
#define SF (CG*CS)
#define NG (C/CG)
#define T (R*WW)
#define SPI (HH/R)
#define C4 384
#define KH 392
#define OFF_D 92160
#define OFF_A 141312
#define OFF_SUM 215040
#define OFF_STAT 216832
#define OFF_BAR 220416
#define OFF_STG 220672
#define NTH 512
__device__ __forceinline__ u32 sa(const void* p){ return (u32)__cvta_generic_to_shared(p); }
__device__ __forceinline__ void mb_init(u64* b, u32 c){ asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(sa(b)), "r"(c)); }
__device__ __forceinline__ void mb_wait(u64* b, u32 par){
  u32 ok;
  do { asm volatile("{\n .reg .pred P;\n mbarrier.try_wait.parity.shared::cta.b64 P, [%1], %2;\n selp.u32 %0,1,0,P;\n}" : "=r"(ok) : "r"(sa(b)), "r"(par) : "memory"); } while (!ok);
}
__device__ __forceinline__ void mb_arrive(u64* b){ asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"(sa(b)) : "memory"); }
__device__ __forceinline__ void mb_expect(u64* b, u32 n){ asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(sa(b)), "r"(n) : "memory"); }
__device__ __forceinline__ void tma3(void* d, const void* map, int c0, int c1, int c2, u64* b){
  asm volatile("cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3, %4}], [%5];"
    :: "r"(sa(d)), "l"(map), "r"(c0), "r"(c1), "r"(c2), "r"(sa(b)) : "memory"); }
__device__ __forceinline__ void bulk1d(void* d, const void* s, u32 n, u64* b){
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];" :: "r"(sa(d)), "l"(s), "r"(n), "r"(sa(b)) : "memory"); }
__device__ __forceinline__ u16 f2h(float f){ u16 h; asm("cvt.rn.f16.f32 %0, %1;" : "=h"(h) : "f"(f)); return h; }
__device__ __forceinline__ float h2f(u16 h){ float f; asm("cvt.f32.f16 %0, %1;" : "=f"(f) : "h"(h)); return f; }
__device__ __forceinline__ u32 pack2(float lo, float hi){ u32 d; asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(d) : "f"(hi), "f"(lo)); return d; }
__device__ __forceinline__ float rcpa(float x){ float y; asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x)); return y; }
__device__ __forceinline__ float ex2a(float x){ float y; asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x)); return y; }
// exact-class GELU: A&S 7.1.26 erfc with the 0.5 folded into the coefficients; relu(x) - |w| covers both signs
__device__ __forceinline__ float gelu_as(float x){
  const float t = rcpa(fmaf(fabsf(x), 0.2316419f, 1.f));
  const float p = fmaf(fmaf(fmaf(fmaf(0.5307027145f, t, -0.7265760135f), t, 0.7107068705f), t, -0.142248368f), t, 0.127414796f);
  const float e = ex2a(x * x * -0.72134752f);
  const float w = (t * p) * e * x;
  return fmaxf(x, 0.f) - fabsf(w);
}
__device__ __forceinline__ bool elect_one(){ u32 p; asm volatile("{\n .reg .pred P;\n elect.sync _|P, 0xffffffff;\n selp.u32 %0,1,0,P;\n}" : "=r"(p)); return p != 0; }
__device__ __forceinline__ u64 sdesc(u32 addr, u32 lbo, u32 sbo, u32 lay){
  return (u64)((addr & 0x3FFFF) >> 4) | ((u64)(lbo >> 4) << 16) | ((u64)(sbo >> 4) << 32) | (1ull << 46) | ((u64)lay << 61); }
__device__ __forceinline__ void tld16(u32 addr, u32 (&r)[16]){
  asm volatile("tcgen05.ld.sync.aligned.32x32b.x16.b32 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, [%16];\n tcgen05.wait::ld.sync.aligned;"
    : "=r"(r[0]),"=r"(r[1]),"=r"(r[2]),"=r"(r[3]),"=r"(r[4]),"=r"(r[5]),"=r"(r[6]),"=r"(r[7]),"=r"(r[8]),"=r"(r[9]),"=r"(r[10]),"=r"(r[11]),"=r"(r[12]),"=r"(r[13]),"=r"(r[14]),"=r"(r[15]) : "r"(addr)); }

// warps 0-7: conv (two groups of 4 alternate 8-channel items over a 3-stage x ring; each warp refills its own 2-channel
// sub-box for item it+3; barrier index it%6 so the two groups never share a barrier: a group running one item ahead
// otherwise sees the previous phase as complete, the r12 intermittent-fault bug). warps 8-15: epilogue (2 per TMEM
// lane quadrant, column halves); warp 8 also issues the MMAs (elect.sync) behind named barriers 2/3 per TMEM slot.
extern "C" __global__ void __launch_bounds__(NTH,1) ipass2(const void* xmap, const float* __restrict__ dwp, const float* __restrict__ dwb,
   const u16* __restrict__ w1a, const float* __restrict__ uv, u16* __restrict__ hout, float* __restrict__ gsum, int B, float ln_eps)
{
  extern __shared__ __align__(1024) unsigned char smem[];
  float* ring = (float*)smem;
  unsigned char* dt = smem + OFF_D;
  float* sumf = (float*)(smem + OFF_SUM);
  u64* bars = (u64*)(smem + OFF_BAR);
  u64* full = bars; u64* dfull = bars + 6; u64* dfree = bars + 7;
  u64* accfull = bars + 8; u64* abar = bars + 12;
  u32* tslot = (u32*)(smem + OFF_BAR + 128);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int ntiles = B * SPI;
  const int tb = (int)((long long)ntiles * blockIdx.x / gridDim.x), te = (int)((long long)ntiles * (blockIdx.x + 1) / gridDim.x);
  const int ntl = te - tb, nitems = ntl * NG;
  if (tid == 0) {
    for (int s = 0; s < 6; s++) mb_init(full + s, 4);
    mb_init(dfull, 1); mb_init(dfree, 1); mb_init(accfull, 1); mb_init(accfull + 1, 1); mb_init(abar, 1);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  for (int i = tid; i < 2 * T; i += NTH) sumf[i] = 0.f;
  if (warp == 8) {
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], 512;" :: "r"(sa(tslot)) : "memory");
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;" ::: "memory");
  }
  asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
  __syncthreads();
  asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
  const u32 tbase = *tslot;
  if (warp < 8) {
    const int grp = warp >> 2, sb = warp & 3, lt = tid & 127, cl = lt >> 4, j = lt & 15, w0 = j * PW;
    auto issue = [&](int it) {
      const int l = it / NG, g = it - l * NG, tile = tb + l;
      const int b = tile / SPI, h0 = (tile - b * SPI) * R, s = it % NS;
      mb_expect(full + it % 6, (u32)(2 * CS * 4));
      tma3(ring + s * SF + 2 * sb * CS, xmap, -4, h0 - 3, b * C + g * CG + 2 * sb, full + it % 6);
    };
    if (grp == 0 && lane == 0) { for (int it = 0; it < 3 && it < nitems; it++) issue(it); }
    int li = 0;
    for (int tile = tb; tile < te; tile++, li++) {
      const int b = tile / SPI, h0 = (tile - b * SPI) * R; const int par = li & 1;
      float ssum[R * PW], ssq[R * PW];
#pragma unroll
      for (int k = 0; k < R * PW; k++) { ssum[k] = 0.f; ssq[k] = 0.f; }
#pragma unroll 1
      for (int qi = 0; qi < NG / 2; qi++) {
        const int g = 2 * qi + grp, it = li * NG + g, s = it % NS;
        const int c = g * CG + cl;
        float wt[49];
        const float4* wp4 = reinterpret_cast<const float4*>(dwp + c * 56);
#pragma unroll
        for (int r = 0; r < 7; r++) { float4 a = __ldg(wp4 + 2 * r), bq = __ldg(wp4 + 2 * r + 1);
          wt[r*7+0]=a.x; wt[r*7+1]=a.y; wt[r*7+2]=a.z; wt[r*7+3]=a.w; wt[r*7+4]=bq.x; wt[r*7+5]=bq.y; wt[r*7+6]=bq.z; }
        const float bias = __ldg(dwb + c);
        mb_wait(full + it % 6, (it / 6) & 1);
        float acc[R][PW];
#pragma unroll
        for (int r = 0; r < R; r++)
#pragma unroll
          for (int cc = 0; cc < PW; cc++) acc[r][cc] = 0.f;
        const float* xb = ring + s * SF + cl * CS + w0 + 1;
#pragma unroll
        for (int i = 0; i < R + 6; i++) {
          float v[PW + 6];
          const float* rp = xb + i * RS;
#pragma unroll
          for (int m = 0; m < PW + 6; m++) v[m] = rp[m];
#pragma unroll
          for (int r = 0; r < R; r++) {
            const int ki = i - r;
            if (ki >= 0 && ki < 7) {
#pragma unroll
              for (int cc = 0; cc < PW; cc++)
#pragma unroll
                for (int jj = 0; jj < 7; jj++) acc[r][cc] = fmaf(v[cc + jj], wt[ki * 7 + jj], acc[r][cc]);
            }
          }
        }
        __syncwarp();
        if (lane == 0 && it + 3 < nitems) issue(it + 3);
        if (qi == 0 && li > 0) mb_wait(dfree, (li - 1) & 1);
#pragma unroll
        for (int r = 0; r < R; r++)
#pragma unroll
          for (int cc = 0; cc < PW; cc++) {
            const int t = r * WW + w0 + cc;
            const u16 hv = f2h(acc[r][cc] + bias);
            const float df = h2f(hv);
            ssum[r * PW + cc] += df; ssq[r * PW + cc] = fmaf(df, df, ssq[r * PW + cc]);
            const int byte = (t >> 6) * (C * 128) + c * 128 + ((((t & 63) >> 3) ^ (c & 7)) << 4) + (t & 7) * 2;
            *reinterpret_cast<u16*>(dt + byte) = hv;
          }
      }
      asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
#pragma unroll
      for (int k = 0; k < R * PW; k++) { ssum[k] += __shfl_xor_sync(0xffffffffu, ssum[k], 16); ssq[k] += __shfl_xor_sync(0xffffffffu, ssq[k], 16); }
      if (lane < 16) {
#pragma unroll
        for (int r = 0; r < R; r++)
#pragma unroll
          for (int cc = 0; cc < PW; cc++) { const int t = r * WW + w0 + cc; atomicAdd(sumf + 2 * t, ssum[r * PW + cc]); atomicAdd(sumf + 2 * t + 1, ssq[r * PW + cc]); }
      }
      asm volatile("bar.sync 1, 256;" ::: "memory");
      if (tid < T) {
        const int t = tid;
        const float s1 = sumf[2 * t], s2 = sumf[2 * t + 1];
        sumf[2 * t] = 0.f; sumf[2 * t + 1] = 0.f;
        const float mu = s1 * (1.f / C);
        const float rs = rsqrtf(fmaxf(s2 * (1.f / C) - mu * mu, 0.f) + ln_eps);
        reinterpret_cast<float2*>(smem + OFF_STAT)[par * T + t] = make_float2(rs, -mu * rs);
        u16* hc = hout + (size_t)(b * KH + C4) * HWv + h0 * WW + t;
#pragma unroll
        for (int rr = 0; rr < 8; rr++) hc[(size_t)rr * HWv] = (rr < 2) ? (u16)0x3C00 : (u16)0;
      }
      asm volatile("bar.sync 1, 256;" ::: "memory");
      if (tid == 0) mb_arrive(dfull);
    }
  } else {
    const int q = warp & 3, half = (warp - 8) >> 2;
    const bool issuer = (warp == 8);
    uint4* stg = reinterpret_cast<uint4*>(smem + OFF_STG + (warp - 8) * 1024);
    if (issuer && lane == 0) {
      mb_expect(abar, 73728u);
      for (int qq = 0; qq < 4; qq++) bulk1d(smem + OFF_A + qq * 18432, (const unsigned char*)w1a + qq * 18432, 18432u, abar);
    }
    const u32 idesc = (1u << 4) | (1u << 16) | ((u32)(T >> 3) << 17) | ((u32)(128 >> 4) << 24);
    const u32 dsa = sa(dt), asa = sa(smem + OFF_A);
    auto mma_block = [&](int n, int jb, bool last) {
      if (elect_one()) {
#pragma unroll
        for (int s = 0; s < 6; s++) {
          const u64 ad = sdesc(asa + jb * 24576 + s * 256, 128u, 1536u, 0);
          const u64 bd = sdesc(dsa + s * 2048, C * 128, 1024, 2);
          asm volatile("{\n .reg .pred p;\n setp.ne.b32 p, %4, 0;\n tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n}"
            :: "r"(tbase + (n & 1) * 256), "l"(ad), "l"(bd), "r"(idesc), "r"(s) : "memory");
        }
        asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];" :: "r"(sa(accfull + (n & 1))) : "memory");
        if (last) asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];" :: "r"(sa(dfree)) : "memory");
      }
      __syncwarp();
    };
    float uu[3], vv[3];
#pragma unroll
    for (int jb = 0; jb < 3; jb++) { const int k = jb * 128 + q * 32 + lane; uu[jb] = uv[k]; vv[jb] = uv[C4 + k]; }
    const float4* stat4 = reinterpret_cast<const float4*>(smem + OFF_STAT);
    const int ntot = 3 * ntl;
    if (issuer) mb_wait(abar, 0);
    int li = 0;
    for (int tile = tb; tile < te; tile++, li++) {
      const int b = tile / SPI, h0 = (tile - b * SPI) * R; const int par = li & 1;
      mb_wait(dfull, li & 1);
      if (issuer) {
        asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
#pragma unroll
        for (int jb = 0; jb < 2; jb++) {
          const int n = 3 * li + jb;
          if (n >= 2) { asm volatile("bar.sync %0, 256;" :: "r"(2 + (n & 1)) : "memory"); asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory"); }
          mma_block(n, jb, false);
        }
      }
#pragma unroll
      for (int jb = 0; jb < 3; jb++) {
        const int n = 3 * li + jb, slot = n & 1, m = n >> 1;
        mb_wait(accfull + slot, m & 1);
        asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
        const int k = jb * 128 + q * 32 + lane;
        u16* hrow = hout + (size_t)(b * KH + k) * HWv + h0 * WW + half * 112;
        u16* hst = hout + (size_t)(b * KH + jb * 128 + q * 32 + (lane >> 1)) * HWv + h0 * WW + half * 112 + (lane & 1) * 8;
        const u32 taddr = tbase + ((u32)(q * 32) << 16) + slot * 256 + half * 112;
        const float cu = uu[jb], cv = vv[jb];
        float ss = 0.f;
#pragma unroll 1
        for (int ch = 0; ch < 7; ch++) {
          u32 r[16];
          tld16(taddr + ch * 16, r);
          u32 pk[8];
#pragma unroll
          for (int cp = 0; cp < 8; cp++) {
            const float4 st = stat4[par * (T / 2) + half * 56 + ch * 8 + cp];
            const float x0 = fmaf(__uint_as_float(r[2 * cp]), st.x, fmaf(st.y, cu, cv));
            const float x1 = fmaf(__uint_as_float(r[2 * cp + 1]), st.z, fmaf(st.w, cu, cv));
            const float g0 = gelu_as(x0), g1 = gelu_as(x1);
            ss = fmaf(g0, g0, ss); ss = fmaf(g1, g1, ss);
            pk[cp] = pack2(g0, g1);
          }
#if STAGE
          stg[lane * 2 + (lane >> 2 & 1)] = make_uint4(pk[0], pk[1], pk[2], pk[3]);
          stg[lane * 2 + 1 - (lane >> 2 & 1)] = make_uint4(pk[4], pk[5], pk[6], pk[7]);
          __syncwarp();
#pragma unroll
          for (int pp = 0; pp < 2; pp++) {
            const int row = pp * 16 + (lane >> 1), part = lane & 1;
            const uint4 vq = stg[row * 2 + (part ^ (row >> 2 & 1))];
            *reinterpret_cast<uint4*>(hst + (size_t)pp * 16 * HWv + ch * 16) = vq;
          }
          __syncwarp();
#else
          uint4* dst = reinterpret_cast<uint4*>(hrow + ch * 16);
          dst[0] = make_uint4(pk[0], pk[1], pk[2], pk[3]);
          dst[1] = make_uint4(pk[4], pk[5], pk[6], pk[7]);
#endif
        }
        asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
        if (issuer) {
          if (jb == 0) {
            asm volatile("bar.sync %0, 256;" :: "r"(2 + slot) : "memory");
            asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
            mma_block(n + 2, 2, true);
          }
        } else if (n + 2 < ntot) {
          asm volatile("bar.arrive %0, 256;" :: "r"(2 + slot) : "memory");
        }
        atomicAdd(gsum + b * C4 + k, ss);
      }
    }
  }
  asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
  __syncthreads();
  if (warp == 8) {
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, 512;" :: "r"(tbase) : "memory");
  }
}
#endif
```

```yaml design-card
id: r12-i-e3-tc05-pass1-547
parents: [r10-i-e3-tmaconv-gate]
operation: structural_mutation
language: cuda_cpp
niche:
  mem: ldg128
  st: direct
  grid: persistent
  launch: fused
  tile: rows128
  red: cta
  cache: default
  spec: all
hypothesis: >-
  H43 STEP 2: fixed-role pass 1 for 64,96 (8 conv warps feeding fp16 d into an MN-major SW128 B tile,
  tcgen05 kind::f16 GEMM1 against a resident W1', epilogue warps doing the LN fold, A&S erfc GELU,
  per-(image,k) sumsq and fp16 h stores) runs within 1.5x of the SM pipe bound, i.e. pass 1 <= 400 us rented.
  Measured: correct on the first engine build, but 547 us (best of 4 structures), so the gate fails and the
  > 520 us refutation line is crossed. Pass 1 is still 18 % faster than G's 668 us. The shipped code is the
  r0 anchor; the pass-1 kernel is archived disabled in i_pass1_tc05.cu. Nothing here is B200-only beyond
  what the archive needs (tcgen05).
expected_effect:
  S: {pct: 0, confidence: high}
  M: {pct: 0, confidence: high}
  L: {pct: 0, confidence: high}
resources_sm100a:
  regs_per_thread: 128
  smem_per_cta_bytes: 70208
  threads_per_cta: 256
  ctas_per_sm: 2
  bytes_in_flight_per_sm: 65536
  launches_per_call: 1
knobs:
  shipped: "r0 anchor unchanged"
  archived_pass1_64_96: "512 thr (8 conv + 8 epilogue warps), 1 CTA/SM, grid 148, tiles 2x112 tokens (N=224), x ring 3 x 30 KB (8 ch, 2-ch sub-boxes 120x8x2), W1' resident 72 KB no-swizzle, d 48 KB single buffer, TMEM 2 x 256-col slots, 1 KB/warp store staging, dyn smem 228864 B"
tests: [H43]
findings:
  - "ENGINE VALIDATED first build: A = fp16(lnw*W1) resident in no-swizzle K-major core-matrix layout (LBO = 128 B along K, SBO = 1536 B along M, +256 B per k16, +24576 B per 128-row block; the swapped LBO/SBO is wrong); B = gate d writer (MN-major SW128, LBO = C*128 = 12288, SBO 1024, +2048 per k16); idesc (1<<4)|(1<<16)|(224>>3)<<17|(128>>4)<<24, N=224 accumulators in 2 slots of 256 TMEM columns. h vs same-rounding fp32 torch: max 1.996e-3 (fp16 output rounding of |h| up to ~9), rms 1.9e-4; vs fp32-weight reference max 2.6e-3; gsum rel 5.8e-6; const rows 384..391 correct [probe_b200]"
  - "BUG FOUND (would also affect any ring shared by alternating consumer groups): two conv warp groups alternating items over a 3-stage ring with one barrier per stage deadlock/fault intermittently ('unspecified launch failure', mostly under harness_time cold L2). A group running one item ahead waits on parity (p+1) while phase p is pending and sees phase p-1 as complete, then corrupts arrival counts. Fix: barriers indexed it % (2*NS) so each barrier is only waited on by one group, in program order. Stable over 5 direct repeats and 4 harness_time runs after the fix [probe_b200]"
  - "Register cap: 448 threads (14 warps) still compiles to 128 regs, not 144; registers appear to be allocated per SMSP (4 warps on one scheduler -> 16384/128). So 512 threads costs nothing more than 448 [probe_b200]"
  - "v1 (dedicated producer + MMA warps, 4 epilogue warps, 448 thr), harness_time rented us: full 608, no epilogue maths/stores 357, conv FMAs skipped 534, both skipped 286 [probe_b200]"
  - "v2 (8 epilogue warps; conv warps refill their own 2-channel TMA sub-box; epilogue warp 8 issues the MMAs behind per-slot named barriers): full 585-588; no dfree wait (racy, timing only) 578 -> the single-d-buffer stall costs ~8 us; GELU replaced by identity 567 -> the A&S GELU costs only ~18 us; conv skipped 521; conv skipped + no epilogue 255; no dfree + no epilogue 349 [probe_b200]"
  - "Stores: per-lane STG.128 to 32 different h rows is expensive: no h stores 509, store-only (no maths) 560, store-only with a 1 KB/warp smem transpose (16 rows x 32 B per STG) 462; full with staging 546.6 (best, -40 us vs direct) [probe_b200]"
  - "Dead end: 4x56-token tiles (box 64x10x2, 1.5x fewer x bytes per tile) with a 4-stage same-group ring (refill it+4) ran 677 us, slower than 547: the refill slack (one item per group) matters more than bytes per tile; the skeleton is bound by x-ring latency, not bandwidth [probe_b200]"
  - "Diagnosis: the terms add almost serially: skeleton 255 us (x-stream: 3 x 30 KB in flight, refilled on consumption, ~2 us per round; the r10 gate copy-only was 118 us with 123 KB in flight), conv +100, epilogue maths +120-150, h stores +40-90 (staged). Issue-count estimate for conv + epilogue is ~300 us at perfect IPC, so 400 needed ~75 % issue efficiency; we reach ~55 % [probe_b200, INFERRED]"
  - "H43 verdict: gate (pass 1 <= 400 us) FAILED; refutation clause (pass 1 > 520 us after tuning, i.e. < 22 % better than G's 668) FIRES at 547 us (-18 %). The role layout fits (128 regs, STACK 0 in the measured ipass2 builds, 224 KB smem with 8 conv warps), so the second refutation half does not fire [probe_b200]"
  - "Value left: pass 1 547 + G's tail (K0 3 + K3 8 + cuBLASLt G2 186, r7 numbers) ~ 744 us projected for the 64,96 row vs r11-merge-b 857.7 us measured in the same container (-13 %); the success line (<= 650) is missed, but at ~+0.0015 of mean STEP 3 is still worth building as a plain improvement. Not the +0.006 H43 promised [INFERRED, run_tests]"
  - "Next levers, in order: (1) a deeper x ring: smem is spent on resident W1' (72 KB); stream W1' per hidden block or shrink d, and use NS >= 5 with cross-group refill (it+NS consumed by the other group); (2) bigger store staging (2 KB/warp -> 8 rows x 64 B per STG); (3) packed f32x2 (FFMA2/FMUL2) in the LN fold and GELU, since x0/x1 column pairs are natural pairs; (4) the 32,128 instantiation was not built [INFERRED]"
  - "Package: compile_b200 128 regs / 0 spill on the anchor; the archive TU is preprocessed away. run_tests 16/16 PASS with the anchor (same container: 64,96 5614 vs r11-merge-b 857.7 us; anchor rows only). run_tests used a stub for i_pass1_tc05.cu; the full archive file was compile-checked in compile_b200 (code inside #if 0 only) [compile_b200, run_tests]"
  - "Not done: STEP 3 (K0/K3/cuBLASLt tail), the 10-round fp64 precision log (roundings match G r7's logged path: d16, fp16(lnw*W1), h16, A&S erfc; GRN sums now from fp32 g), stamps per tile, the 32,128 instantiation, cuobjdump STACK on an nvcc build of the pass-1 kernel (NVRTC resources showed STACK 0) [INFERRED]"
  - "Pitfall: probe containers reset /tmp between probes unpredictably; embed full sources each time [probe_b200]"
paths:
  - {max_tokens: null, lang: cuda, width: 128, threads: 256, rows: 128, grid: persistent, mem: ldg, launches: 1, x: none, w: none, st: none, cluster: 1}
runs_on:
  H200: {runs: true, representative: false, note: "shipped code is the r0 anchor; target_hardware B200 only"}
  A100: {runs: true, representative: false}
  L40S: {runs: true, representative: false}
risks:
  - "Ships nothing new: every row runs the r0 anchor (portal ~0.30); its value is the validated tcgen05 pass-1 engine, the parity-aliasing fix and the ablation numbers"
  - "The 547 us pass 1 is a rented-clock number; the portal ratio for this SIMT/issue-bound kernel is likely ~1.25-1.3"
  - "The pass-1 h layout [B][392][HW] with constant rows assumes G's cuBLASLt hi/lo b2' recipe for the tail (unbuilt here)"
measure_first: [H200]
compliance:
  single_stream: true
  no_cross_call_state: true
  fp32_math: true
  no_threads_or_fork: true
  no_precompiled_binaries: true
```