"""Per-size targets built from NVIDIA's SOLAR roofline: what T_SOL is, how far our times sit above it, and what time each
size needs for the mean score to pass the live #1.

  .venv/bin/python loop/sol_targets.py L2/035_convnextv2_block_with_grn            # or SOLX_PROBLEM
  .venv/bin/python loop/sol_targets.py L2/035_convnextv2_block_with_grn --live-first 0.7509 --margin 0.002

SOLAR (github.com/NVlabs/SOLAR, read 2026-10-20, not run): T_SOL = max(MACs / MAC_per_cycle, fused bytes / DRAM_byte_per_cycle)
at 1.5 GHz with the B200 numbers of configs/arch/B200.yaml. MACs count only einsum ops (conv, matmul, linear); layer norm,
GELU and the GRN elementwise/reduction ops are CUDA-core work that perf_model.py leaves out of T_SOL. Fused bytes are the
unique external reads (inputs and weights once) plus the model outputs; intermediates are free. The MAC rate and the
bytes per element both follow SOLAR's --precision flag (default fp16), and the portal's recovered T_SOL for #129 matches
the fp16/bf16 tensor rate and 2-byte elements, not fp32 SIMT, TF32 or 4-byte tensors, plus a constant 0.42 us whose
source is in neither repo. Only the rules for problems in RULES are encoded; any other problem falls back to the T_SOL
that planner.anchors() recovers from the portal scores (which is what scores a kernel in every case).

Score per size: S = (Tb - Ts) / ((T - Ts) + (Tb - Ts)), mean over sizes. dS/dT = -S^2 / (Tb - Ts).
Standard library plus the loop's planner/archive/problem modules.
"""

import argparse
import math
import os
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "loop"))

import archive
import planner
import problem

LIVE_FIRST = {"035_convnextv2_block_with_grn": 0.750880, "045_fused_linear_gelu_grn_linear": 0.854141}   # live board #1 by problem (the user's figures, 2026-10-20);
                                                             # other problems use the saved board; --live-first overrides
MARGIN = 0.002               # aim this far above the live #1
# configs/arch/B200.yaml in SOLAR: 1.5 GHz; DRAM 8 TB/s nominal = 5111.6 B/cycle (7.67 TB/s at 1.5 GHz); MACs per cycle.
FREQ_GHZ, DRAM_B_PER_CYC = 1.5, 5111.6
MAC_PER_CYC = {"fp16": 603751, "tf32": 301875, "fp32_sm": 40251}      # fp16 = bf16 = 1.81 PFLOP/s
PRECISION, BYTES_PER_ELEM = "fp16", 2     # what the recovered #129 T_SOL matches (SOLAR's DEFAULT_PRECISION)
OFFSET_US = 0.42             # constant that the recovered T_SOL carries on top of the SOLAR roofline (mean 0.425, sd 0.04 over 16 sizes)
STEP = 0.002                 # greedy cut per move, as a fraction of a size's current time


def _convnext(ax):
    """(MACs, fused bytes) of the 035_convnextv2_block_with_grn workload with these axes."""
    n, c = ax["batch_size"] * ax["height"] * ax["width"], ax["dim"]
    macs = n * (49 * c + 8 * c * c)             # 7x7 depthwise + C->4C + 4C->C pointwise GEMMs
    elems = 2 * n * c + 8 * c * c + 65 * c      # x and y once; two weights 4C^2; 65 C of dw weight and the five biases/scales
    return macs, elems * BYTES_PER_ELEM


RULES = {"035_convnextv2_block_with_grn": _convnext}


def solar(key):
    """SOLAR's own roofline for a workload key, or None when its rules are not encoded for the current problem."""
    p = problem.current()
    rule = RULES.get(p.name)
    if not rule:
        return None
    macs, nbytes = rule(p.axes(key))
    t_cmp = macs / MAC_PER_CYC[PRECISION] / (FREQ_GHZ * 1e3)      # cycles / (cycles per us)
    t_mem = nbytes / DRAM_B_PER_CYC / (FREQ_GHZ * 1e3)
    return dict(gflop=2 * macs / 1e9, mb=nbytes / 1e6, ai=macs / nbytes, bound="compute" if t_cmp >= t_mem else "memory",
                t=max(t_cmp, t_mem) + OFFSET_US)


def best_submission():
    """(submission id, {key: (T us, reported S)}) of our best complete accepted portal submission for the current problem."""
    p = problem.current()
    ok = {r["submission_id"] for r in archive.portal_rows(archive.PORTAL) if r.get("result") in (None, "", "AC")}
    by = {}
    for r in archive.portal_rows(archive.PORTAL_WL):
        by.setdefault(r["submission_id"], {})[p.key_from_label(r["workload"])] = (float(r["latency_ms"]) * 1e3, float(r["sol_score"]))
    full = {s: d for s, d in by.items() if len(d) == len(p.keys())}
    full = {s: d for s, d in full.items() if s in ok} or full
    sid = max(full, key=lambda s: statistics.mean(v[1] for v in full[s].values()))
    return sid, full[sid]


def score(t, tb, ts):
    return (tb - ts) / ((t - ts) + (tb - ts))


def mean_score(times, anc):
    return statistics.mean(score(times[k], *anc[k]) for k in anc)


def equal_s(anc, target):
    """T per size at which every size scores exactly `target`."""
    return {k: ts + (tb - ts) * (1 - target) / target for k, (tb, ts) in anc.items()}


def greedy(times, anc, target):
    """Cheapest allocation: repeatedly cut STEP of the time of the size whose score rises most, until the mean reaches target.

    Each size's score is concave in log T (diminishing returns), so equal small moves to the best marginal gain minimise the
    total relative cut."""
    t = dict(times)
    for _ in range(20000):
        if mean_score(t, anc) >= target:
            break
        k = max(t, key=lambda k: score(t[k] * (1 - STEP), *anc[k]) - score(t[k], *anc[k]))
        t[k] *= 1 - STEP
    return t


def practical_floor(key):
    """A floor for fp32-in/out kernels: fp16 tensor MMA at the portal clock, tensors moved once at the read ceiling, + fixed."""
    try:
        sys.path.insert(0, str(ROOT / "poc"))
        import leader_gap as g
        pf, tbs, fixed = dict(g.CEIL)["FP16/BF16 tensor"], g.DRAM_READ_TBS, g.FIXED_US
    except Exception:
        pf, tbs, fixed = 1.82e15, 7.5, 2.0
    p = problem.current()
    return fixed + max(p.gflop(key) * 1e9 / pf, p.mbytes(key) * 1e6 / (tbs * 1e12)) * 1e6


def uniform_cut(times, anc, target):
    """One relative cut r applied to every size that reaches the target mean (bisection)."""
    lo, hi = 0.0, 0.99
    for _ in range(40):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if mean_score({k: t * (1 - mid) for k, t in times.items()}, anc) < target else (lo, mid)
    return hi


def pct(x, digits=0):
    """Signed percent, '-' for no change."""
    return "-" if abs(x) < 5e-4 else f"{x:+.{digits}%}"


def gm(xs):
    return math.exp(sum(map(math.log, xs)) / len(xs))


def targets_text(name=None, live_first=None, margin=MARGIN):
    """Markdown tables and a paragraph for the current problem (or `name`, which sets SOLX_PROBLEM)."""
    if name:
        os.environ["SOLX_PROBLEM"] = name
    p = problem.current()
    anc = planner.anchors()
    if not anc:
        return f"(sol targets: no portal results for {p.name} yet)"
    sid, best = best_submission()
    keys = sorted(anc, key=p.mbytes)
    times = {k: best[k][0] for k in keys}
    n = len(keys)
    live, src = live_first, "given"
    if live is None:
        live, src = LIVE_FIRST.get(p.name), "LIVE_FIRST"
        if live is None:                         # no live figure for this problem: the saved public board
            try:
                sys.path.insert(0, str(ROOT / "poc"))
                import leader_gap
                live, src = leader_gap.leaderboard_scores(p.name)[0]["s1"], "saved board"
            except Exception:
                pass
    target = live + margin
    now = mean_score(times, anc)
    t_eq, t_gr = equal_s(anc, target), greedy(times, anc, target)
    rules = {k: solar(k) for k in keys}
    lines = [f"SOL targets for {p.level}/{p.name}: our best is #{sid}, mean {now:.4f} by the recovered anchors (portal {statistics.mean(v[1] for v in best.values()):.4f}); "
             f"live #1 {live:.6f} ({src}), target mean {target:.6f} (live + {margin}), a gain of {target - now:+.4f}.",
             "T_SOL is recovered from the portal scores (planner.anchors(); it is what scores a kernel)."]
    if all(rules.values()):
        dev = max(abs(rules[k]["t"] - anc[k][1]) for k in keys)
        lines.append(f"SOLAR rules (B200 1.5 GHz, {PRECISION} tensor rate {2 * MAC_PER_CYC[PRECISION] * FREQ_GHZ / 1e6:.2f} PFLOP/s, "
                     f"{BYTES_PER_ELEM}-byte tensors, {DRAM_B_PER_CYC * FREQ_GHZ / 1e3:.2f} TB/s) + {OFFSET_US} us reproduce it "
                     f"within {dev:.2f} us at all {n} sizes. AI is SOLAR's MAC per byte; its ridge point is "
                     f"{MAC_PER_CYC[PRECISION] / DRAM_B_PER_CYC:.0f}.")
    else:
        lines.append("(SOLAR rules are not encoded for this problem: GFLOP and MB are the compulsory I/O of the definition.)")
    lines += [f"| workload ({','.join(p.var_axes)}) | GFLOP | fused MB | AI | binds | T_SOL us | T_b us | ours us | S | T/T_SOL | dMean/dT x1e-4 per us |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
    for k in keys:
        tb, ts = anc[k]
        r = rules[k]
        gf, mb, ai, bd = (r["gflop"], r["mb"], f"{r['ai']:.0f}", r["bound"]) if r else (p.gflop(k), p.mbytes(k), "-", "-")
        s = score(times[k], tb, ts)
        lines.append(f"| {k} | {gf:.2f} | {mb:.1f} | {ai} | {bd} | {ts:.2f} | {tb:.1f} | {times[k]:.1f} | {s:.3f} | "
                     f"{times[k] / ts:.0f}x | {s * s / (tb - ts) / n * 1e4:.2f} |")
    lines += ["", f"Time each size needs for mean {target:.4f}: (a) every size at the same S = {target:.4f}; (b) cheapest allocation = "
              f"greedy over {STEP:.1%} cuts of the size with the largest score gain until the mean reaches the target "
              "(minimises the sum of relative cuts). Cut is relative to our time; + means we are already faster than (a) needs.",
              "| workload | ours us | (a) equal-S us | (a) cut | (b) cheapest us | (b) cut | S now -> (b) |",
              "|---|---|---|---|---|---|---|"]
    for k in keys:
        lines.append(f"| {k} | {times[k]:.1f} | {t_eq[k]:.1f} | {pct(t_eq[k] / times[k] - 1)} | {t_gr[k]:.1f} | "
                     f"{pct(t_gr[k] / times[k] - 1, 1)} | {score(times[k], *anc[k]):.3f} -> {score(t_gr[k], *anc[k]):.3f} |")
    cut_a = sum(max(0.0, 1 - t_eq[k] / times[k]) for k in keys)
    cut_b = sum(1 - t_gr[k] / times[k] for k in keys)
    lines.append(f"Sum of relative cuts: (a) {cut_a:.2f} (cuts only), (b) {cut_b:.2f}; (b) reaches mean {mean_score(t_gr, anc):.4f}, "
                 f"saves {sum(times[k] - t_gr[k] for k in keys):.0f} us in total. For scale, one uniform cut of "
                 f"{uniform_cut(times, anc, target):.1%} on all {n} sizes also reaches it (sum {n * uniform_cut(times, anc, target):.2f}).")
    # Summary paragraph: distance above T_SOL and above a practical floor, by size band.
    band = {b: [k for k in keys if p.band(k) == b] for b in "SML"}
    ratio = {b: gm([times[k] / anc[k][1] for k in ks]) for b, ks in band.items() if ks}
    fl = {k: practical_floor(k) for k in keys}
    allr = [times[k] / anc[k][1] for k in keys]
    rng = lambda xs: f"{min(xs):.0f}-{max(xs):.0f}"
    s_ks, l_ks = band["S"], band["L"]
    lat = bool(s_ks) and gm([times[k] / fl[k] for k in s_ks]) > 2
    thr = bool(l_ks) and gm([times[k] / fl[k] for k in l_ks]) > 2
    omitted = " (for #129: the 4C intermediate, the GRN pass, the fp32 depthwise conv and the GELU)" if p.name in RULES else ""
    lines += ["", f"Summary: our times are {min(allr):.0f}-{max(allr):.0f}x T_SOL (geometric mean S {ratio.get('S', 0):.0f}x, M {ratio.get('M', 0):.0f}x, "
              f"L {ratio.get('L', 0):.0f}x). "
              + ("T_SOL is not reachable for an fp32 problem (it assumes the fp16/bf16 tensor rate and 2-byte traffic and leaves "
                 "CUDA-core work out), so " if all(rules.values()) else "")
              + f"a practical floor is the fairer yardstick (fp16 tensor MMA at 1.82 PF/s, "
              f"fp32 tensors moved once at 7.5 TB/s, + 2 us); we are "
              f"{min(times[k] / fl[k] for k in keys):.1f}-{max(times[k] / fl[k] for k in keys):.1f}x above it at every size. "
              + (f"In the S band T is {rng([times[k] for k in s_ks])} us against floors of {rng([fl[k] for k in s_ks])} us, so "
                 f"latency, launch and synchronisation bind there, not the roofline. " if lat else
                 "The S band is close to the roofline. ")
              + (f"In the L band (the largest sizes by bytes) we are still "
                 f"{min(times[k] / fl[k] for k in l_ks):.1f}-{max(times[k] / fl[k] for k in l_ks):.1f}x above the floor, so a throughput "
                 f"constraint binds there as well as fixed latency: traffic and CUDA-core work that the roofline omits{omitted}, not "
                 f"T_SOL. " if thr else "The L band is close to the roofline. ")
              + "A microsecond saved pays most at the small sizes (dMean/dT column), because T_b - T_SOL is small there."]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("problem", nargs="?", help="e.g. L2/035_convnextv2_block_with_grn (default: $SOLX_PROBLEM)")
    ap.add_argument("--live-first", type=float, help="live #1 mean score (default: LIVE_FIRST, else the saved board)")
    ap.add_argument("--margin", type=float, default=MARGIN, help="aim this far above the live #1")
    a = ap.parse_args()
    print(targets_text(a.problem, a.live_first, a.margin))


if __name__ == "__main__":
    main()
