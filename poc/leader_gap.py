"""What must the leaders be doing? Back out the per-workload times that the public top scores imply and compare them
with what the B200 can do at the portal's locked clock, per precision.

  python3 poc/leader_gap.py L1/035          # or a problem folder name; SOLX_PROBLEM works too
  python3 poc/leader_gap.py L1/035 --scores 0.7945,0.7442,0.7378,0.691   # newer leaderboard numbers

Method: score S = (Tb - Ts) / ((T - Ts) + (Tb - Ts)) per workload, averaged. Tb (hidden baseline) comes from our own
portal pages; Ts is back-solved from our pages (median over submissions, ill-conditioned near S = 0.5, so those
are skipped). Assuming a leader scores the same S on every workload (the simplest reading of a mean), its time is
T = Ts + (Tb - Ts) (1 - S) / S. Dividing the workload's compulsory bytes and FLOPs by T gives the bandwidth and the
compute rate the leader must sustain; the cheapest precision whose ceiling covers that rate is printed.
If the implied rate exceeds every ceiling we allow ourselves, the gap is not tuning: a rule, a precision, an algorithm
(less traffic or fewer FLOPs) or the uniform-S assumption differs. That is the first thing to look for.

Ceilings are portal-lock numbers measured on our own submissions where possible; edit CEIL when better data exists.
Standard library plus the loop's problem module.
"""

import argparse
import csv
import os
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "loop"))
LB = ROOT / "poc" / "results" / "leaderboards" / "leaderboards_b200_v1.1.csv"
WL = ROOT / "poc" / "results" / "b200_portal_workloads.csv"

# Portal-lock ceilings. DRAM: 8.18 TB/s peak; public read+write streams reach 7.05-7.1 TB/s marginal (b200_sota.md),
# read-heavy streams ~7.5 (STREAM triad 7.48); our #88 portal fit was 6.82 + ~2-5 us fixed. Tensor cores: datasheet
# per-SM rates at the portal's 1500 MHz lock (harness device_config 1500/3996) x 148 SMs: kind::tf32 4,096 FLOP/clk/SM
# = 0.91 PF/s, kind::f16 8,192 = 1.82 PF/s, kind::f8f6f4 16,384 = 3.64 PF/s. Our #35 r6 ledger's MMA ideal (928 clk per
# 64-k block at N = 464 per SM) is this rate. Achieved, for scale: cuBLAS bf16 1.47 PF/s at 1500 MHz (#30 probe), #35
# r5 TF32 tc_ws about 0.62 PF/s on the portal at B=919. (Until 2026-10-11 this table used 0.65/1.3/2.6, our *achieved*
# TF32 rate x 1/2/4: that understated every tensor ceiling by 1.4x.) FP32 SIMT: 128 lanes x 2 FLOP x 148 x 1.5 GHz.
CEIL = [("fp32 SIMT", 0.057e15), ("TF32 tensor", 0.91e15), ("FP16/BF16 tensor", 1.82e15), ("FP8 tensor", 3.64e15)]
DRAM_TBS = 7.05      # read+write streams; read-dominated streams can reach about DRAM_READ_TBS
DRAM_READ_TBS = 7.5
FIXED_US = 2.0


def leaderboard_scores(name):
    for r in csv.DictReader(open(LB)):
        if r["name"] == name:
            return {k: float(r[k]) for k in ("s1", "s2", "s3", "s5", "s10") if r.get(k)}, r.get("rank1_date", "")
    return {}, ""


def our_rows(name):
    rows = [r for r in csv.DictReader(open(WL))]
    import problem
    p = problem.current()
    return [r for r in rows if r.get("submission_id") and _kernel_of(r["submission_id"]) == name]


_KERNEL = {}


def _kernel_of(sid):
    if not _KERNEL:
        for r in csv.DictReader(open(ROOT / "poc" / "results" / "b200_portal.csv")):
            _KERNEL[r.get("submission_id")] = (r.get("kernel"), r.get("result"))
    return (_KERNEL.get(sid) or (None, None))[0]


def _result_of(sid):
    _kernel_of(sid)
    return (_KERNEL.get(sid) or (None, None))[1]


def report(scores=None):
    import problem
    p = problem.current()
    lb, date = leaderboard_scores(p.name)
    if scores:
        lb, date = dict(zip(("s1", "s2", "s3", "s5"), scores)), "live figures given"
    rows = our_rows(p.name)
    if not rows:
        return f"(leader gap: no portal results for {p.name} yet)"
    by_key = {}
    for r in rows:
        by_key.setdefault(p.key_from_label(r["workload"]), []).append(r)
    full = {r["submission_id"] for r in rows if _result_of(r["submission_id"]) in ("AC", "", None)} or \
        {r["submission_id"] for r in rows}                    # a WA page's passing rows are not a full submission
    best_sid = max(full,
                   key=lambda s: statistics.mean(float(r["sol_score"]) for r in rows if r["submission_id"] == s))
    best = {p.key_from_label(r["workload"]): r for r in rows if r["submission_id"] == best_sid}
    our_mean = statistics.mean(float(r["sol_score"]) for r in best.values())
    lines = [f"Leader gap for {p.level}/{p.name} (public board {date or 'as given'}): "
             + ", ".join(f"{k} {v:.4f}" for k, v in lb.items()) + f"; our best {our_mean:.4f} (#{best_sid})",
             "Assumes the leader scores the same S on every workload. Ceilings at the portal lock: DRAM "
             f"{DRAM_TBS} TB/s read+write, {DRAM_READ_TBS} read-only, + {FIXED_US} us; " + ", ".join(f"{n} {c / 1e15:.2f} PF/s" for n, c in CEIL) + ".",
             "| workload | MB | GFLOP | Tb us | Ts us | ours us (S) | leader us | leader TB/s | leader TF/s | needs |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    s1 = lb.get("s1")
    beyond = 0
    ceil_s = {n: [] for n, _ in CEIL}           # best score per workload if a kernel ran at the ceilings (optimistic)
    for k in sorted(by_key, key=p.mbytes):
        rs = by_key[k]
        tb = statistics.median(float(r["baseline_ms"]) * 1e3 for r in rs)
        ts_list = [float(r["t_sol_ms_derived"]) * 1e3 for r in rs if r.get("t_sol_ms_derived")]
        if not ts_list or not s1 or k not in best:
            continue
        ts = statistics.median(ts_list)
        t1 = ts + (tb - ts) * (1 - s1) / s1
        mb, gf = p.mbytes(k), p.gflop(k)
        bw = mb * 1e6 / (max(t1 - FIXED_US, 0.1) * 1e-6) / 1e12
        rate = gf * 1e9 / (t1 * 1e-6)
        needs = []
        t_mem = FIXED_US + mb * 1e6 / (DRAM_READ_TBS * 1e12) * 1e6
        ours_s = float(best[k]["sol_score"])
        for n, c in CEIL:
            t_floor = max(t_mem, gf * 1e9 / c * 1e6) if gf else t_mem
            ceil_s[n].append(max(ours_s, (tb - ts) / ((max(t_floor, ts) - ts) + (tb - ts))))
        if t1 < 5:
            needs.append("launch-bound size: uniform S is not meaningful here")
        elif bw > DRAM_READ_TBS:
            needs.append(f"> DRAM ({bw / DRAM_READ_TBS:.2f}x the read ceiling): fewer bytes")
        elif bw > DRAM_TBS:
            needs.append("DRAM near peak (read-only stream, no extra traffic)")
        if gf:
            fit = next((n for n, c in CEIL if rate <= c), None)
            needs.append(fit or f"> {CEIL[-1][0]}")
        beyond += any(x.startswith(">") for x in needs) and t1 >= 5
        ours = best[k]
        lines.append(f"| {ours['workload']} | {mb:.1f} | {gf:.1f} | {tb:.1f} | {ts:.1f} | "
                     f"{float(ours['latency_ms']) * 1e3:.1f} ({float(ours['sol_score']):.3f}) | {t1:.1f} | {bw:.2f} | "
                     f"{rate / 1e12:.0f} | {'; '.join(needs) or 'DRAM ok'} |")
    if any(ceil_s.values()):
        lines.append("Best mean score if every workload ran at the ceiling (optimistic; DRAM read ceiling, no harness "
                     "flush penalty), by the precision the maths uses: "
                     + ", ".join(f"{n} {statistics.mean(v):.3f}" for n, v in ceil_s.items() if v)
                     + (". Memory-bound problem: precision does not matter." if not p.gflop(next(iter(by_key))) else "."))
    lines.append(f"{beyond} workload(s) imply more than every ceiling: there the leader must move fewer bytes, do fewer "
                 "FLOPs, use a cheaper precision, or not score uniformly." if beyond else
                 "Every implied time fits under a ceiling listed in 'needs': the leader's score is reachable with that "
                 "precision and bandwidth.")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("problem", nargs="?", help="e.g. L1/035 (default: $SOLX_PROBLEM)")
    ap.add_argument("--scores", help="comma-separated s1,s2,s3,s5 to use instead of the saved leaderboard CSV")
    a = ap.parse_args()
    if a.problem:
        os.environ["SOLX_PROBLEM"] = a.problem
    print(report([float(x) for x in a.scores.split(",")] if a.scores else None))


if __name__ == "__main__":
    main()
