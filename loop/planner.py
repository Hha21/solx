"""Decide what a round should attempt, and which tested candidates deserve the B200 portal.

Both decisions use the B200 per-workload portal data: the hidden baseline Tb per workload, and Tsol recovered from
the per-workload scores, so a predicted set of B200 times can be turned into a predicted portal score exactly.

  plan(arc)        -> tasks for the next round: exploit the band with the most score to gain on the best kernel,
                      the next band, one exploration of an untried niche, and repairs of recent failures.
  shortlist(arc,r) -> candidates of round r ranked by predicted B200 score, plus B200-only designs that cheap GPUs
                      cannot predict (exploration slots).
"""

import math
import statistics
from collections import defaultdict

import archive
import problem
from archive import ROOT


def P():
    return problem.current()


FIXED_TARGET_US = 2.0               # public B200 floor for a tiny streaming kernel is ~2.2-2.5 µs in total (b200_sota.md)
BW_TARGET_TBS = 7.1                 # best public B200 read+write streams fit 7.05-7.10 TB/s marginal (b200_sota.md)
EXPLORE_NICHES = [                  # (axis, tag, instruction) tried in order until one is missing from the archive
    # Dropped: launch:pdl (one-kernel designs gain nothing) and red:halfwarp (the r3 CuTe winner already does it).
    ("lang", "cutedsl", "Write the design in CuTe DSL (Python), which compiles for sm_100a; use it to reach a memory "
                        "path Triton cannot express."),
    ("tile", "token", "Process one token's 48 heads per program (tile:token) so the weight tile is constant."),
    ("lang", "gluon", "Write the design in Gluon (triton.experimental.gluon) for explicit layout and memory control."),
    ("grid", "capped4", "Small/medium specialist after FlashInfer's CAKE B200 kernels (b200_sota.md section 4): a persistent "
                        "grid capped at 4 CTAs per SM, each thread keeping 2-4 rows' 256-bit loads in flight and issuing "
                        "the next row's loads before the current row's reduction; parent's kernel for large sizes."),
    ("tile", "fat", "Cut the CTA count 4-16x: an empty grid of 49,152 CTAs already spans 26 us on B200 "
                    "(the problem card). Give each thread several rows (loads for all of them issued first, then "
                    "the reductions), or larger CTAs, keeping registers low enough for full occupancy. Measure the "
                    "CTA-count effect with probe_b200 before writing the full kernel."),
    ("st", "l2order", "Order the work so that the outputs written last (the ones still in L2 when the kernel ends) are "
                      "as large a share as possible, e.g. reverse or interleave the tile order; keep the parent's "
                      "evict_last stores."),
]


def anchors():
    """Per workload key: (Tb, Tsol) in µs for the current problem, recovered from every ingested portal page.

    Tb is shown on the page; Tsol follows from the per-workload score S = (Tb - Tsol) / ((T - Tsol) + (Tb - Tsol)).
    That inversion is ill-conditioned near S = 0.5, so those workloads get the roofline fitted by sol_model()."""
    tb, ts, rows = {}, defaultdict(list), []
    for r in archive.portal_rows(archive.PORTAL_WL):
        k = P().key_from_label(r["workload"])
        t, b, s = float(r["latency_ms"]) * 1e3, float(r["baseline_ms"]) * 1e3, float(r["sol_score"])
        tb[k] = b
        rows.append((k, t, b, s))
        x = (b - s * t - s * b) / (1 - 2 * s) if abs(1 - 2 * s) > 0.1 else None
        if x is not None and 0 < x < min(t, b):
            ts[k].append(x)
    model = sol_model(rows)
    return {k: (tb[k], statistics.median(ts[k]) if ts.get(k) else model(k)) for k in tb}


def sol_model(rows):
    """Roofline Tsol(key) = max(MB / TB/s, GFLOP / PFLOP/s), fitted to the reported per-workload scores.

    rows are (key, T, Tb, S). #218's portal scores fit this with 6.75 TB/s and 1.81 PFLOP/s to within rounding; a
    problem without a sol.yaml `flops` expression gets the bandwidth term only."""
    if not rows:
        return lambda k: 0.0
    flops = any(P().gflop(k) for k, *_ in rows)
    tsol = lambda k, bw, pf: max(mb(k) / bw, P().gflop(k) / pf if flops else 0.0)

    def err(bw, pf):
        e = 0.0
        for k, t, b, s in rows:
            x = tsol(k, bw, pf)
            d = (t - x) + (b - x)
            e += ((b - x) / d - s) ** 2 if d > 0 and x < b else 1.0
        return e

    bw, pf, step = 7.0, 2.0, 0.5                      # coarse-to-fine search in log space
    for _ in range(12):
        grid = [(bw * math.exp(i * step), pf * math.exp(j * step)) for i in range(-4, 5) for j in (range(-4, 5) if flops else [0])]
        bw, pf = min(grid, key=lambda p: err(*p))
        step /= 2
    return lambda k: tsol(k, bw, pf)


def score(times, anc):
    """Mean per-workload portal score for times {workload key: µs} (every anchored workload must be present)."""
    vals = [(tb - ts) / ((times[k] - ts) + (tb - ts)) for k, (tb, ts) in anc.items()]
    return sum(vals) / len(vals)


def b200_times(rec):
    """Portal µs per workload key for an archive record."""
    return dict((rec.get("b200") or {}).get("timings") or {})


def mb(key):
    return P().mbytes(key)


def band_of(key):
    return P().band(key)


def fit_fixed_bw(times):
    """Least-squares t = a + bytes/bw over S and M sizes; returns (fixed µs, TB/s)."""
    pts = [(mb(k), t) for k, t in times.items() if band_of(k) in "SM"]
    n = len(pts)
    mx, my = sum(x for x, _ in pts) / n, sum(y for _, y in pts) / n
    slope = sum((x - mx) * (y - my) for x, y in pts) / sum((x - mx) ** 2 for x, _ in pts)
    return my - slope * mx, 1 / slope            # µs per MB -> MB/µs = TB/s


def best_kernel(arc):
    """The kernel with the best portal score, or None before the problem's first portal result."""
    measured = [r for r in arc.values() if r.get("b200")]
    return max(measured, key=lambda r: r["b200"]["score"]) if measured else None


def band_tbs(times, band):
    xs = [mb(k) / t for k, t in times.items() if band_of(k) == band]
    return math.exp(sum(map(math.log, xs)) / len(xs)) if xs else None


def gains(best, anc):
    t = b200_times(best)
    base = score(t, anc)
    fixed, bw = fit_fixed_bw(t)
    s_t = {k: (min(v, FIXED_TARGET_US + mb(k) / bw) if band_of(k) == "S" else v) for k, v in t.items()}
    # M/L: the best kernel already beats the physical read+write rate (outputs left in L2 at kernel end are written
    # back after the timed window), so a bandwidth target no longer fits; value a further 5% instead.
    l_t = {k: (min(v, fixed + mb(k) / BW_TARGET_TBS, 0.95 * v) if bw < BW_TARGET_TBS else 0.95 * v)
           if band_of(k) != "S" else v for k, v in t.items()}
    return dict(base=base, fixed=fixed, bw=bw, S=score(s_t, anc) - base, ML=score(l_t, anc) - base,
                tbs_M=band_tbs(t, "M"), tbs_L=band_tbs(t, "L"))


def worse_in_band(arc, best, bands):
    """Kernels already measured on B200 that are slower than the best in these bands (what not to repeat)."""
    bt = b200_times(best)
    out = []
    for r in arc.values():
        t = b200_times(r)
        if not t or r["id"] == best["id"]:
            continue
        rel = [t[k] / bt[k] for k in t if band_of(k) in bands and k in bt]
        if rel and sum(rel) / len(rel) > 1.02:
            n = r.get("niche") or {}
            out.append(f"{r['id']} ({n.get('mem', '?')}/{n.get('grid', '?')}): {100 * (sum(rel) / len(rel) - 1):+.0f}%")
    return out


def in_flight(arc):
    """Proposed or tested ideas with no B200 result yet, so the LLM does not propose them again."""
    out = []
    for r in arc.values():
        if not r.get("b200") and str(r.get("round", "")).startswith("r") and r.get("status") in ("proposed", "passed"):
            n = r.get("niche") or {}
            out.append(f"{r['id']} ({n.get('mem', '?')}/{n.get('grid', '?')}, {r.get('language', '?')})")
    return out


EXPLORE_COMMON = (
    " This is an EXPLORATION task: the goal is to learn whether this idea can move some size band by 3% or more on the "
    "portal, not to edge out the current best by 0.1%. You may discard the parent's design entirely; it is given for "
    "reference. Test the idea's core assumption with probe_b200 first (seconds per probe). A missed gate or success "
    "line is not a reason to stop: switch to the best configuration your own probes found (or the experiment's next "
    "variant), build it into a full kernel and measure it with run_tests, and use the rest of the budget on it. Hand "
    "in a measured kernel that implements this experiment's design on at least one workload (others may fall back); "
    "the r0 baseline or the unchanged parent is not a valid candidate unless a measured hardware fact rules the design "
    "out (name it, with the probe's number). Record what you learned in the card's findings: negative results are "
    "valuable and go into the lab notebook for later rounds. CUDA C++ uses the two-file layout.")
MEMORY_NOTE = (" Keep the kernel copy-like (memory-bound): the portal locks the SM clock lower than the test bench and "
               "penalises extra SM work per byte (the problem card).")


def explore_common():
    """EXPLORE_COMMON plus, for memory-bound problems only, the copy-like advice."""
    return EXPLORE_COMMON + (MEMORY_NOTE if P().kind == "memory" else "")

EXPLORE_IDEAS = [
    ("explore:one-wave-fat", "One wave of fat CTAs. Launching and retiring CTAs is not free on B200: an empty grid of "
     "49,152 CTAs spans 26 us and 768 CTAs 1.6 us (the problem card). Design a kernel whose grid is exactly one "
     "resident wave (148 SMs x the CTAs that fit) at every size, each CTA streaming a contiguous chunk of rows with "
     "several rows' 256-bit loads in flight per thread (issue all loads, then reduce), keeping evict_last stores. "
     "First probe a plain copy: one wave of fat CTAs vs many small CTAs, at 12.6, 101 and 805 MB."),
    ("explore:bulk-pipeline", "A hand-written bulk-copy pipeline in CUDA C++: cp.async.bulk (1-D TMA) global to shared "
     "memory with mbarrier completion, normalise from shared memory, then store (cp.async.bulk shared to global, or "
     "st.global with evict_last), double or triple buffered in a persistent single wave. The aim is fewer SM "
     "instructions per byte, which the portal's 1500 MHz SM clock rewards. Earlier Triton TMA designs (g2-tma, "
     "g2-tmaws) were slow for other reasons; this is a different bet. First probe a bulk-copy pipeline copy against a "
     "plain LDG/STG copy at 12.6, 101 and 805 MB."),
    ("explore:work-steal", "A persistent single wave with dynamic work stealing: CTAs take tiles from an atomic counter "
     "in a deliberate order (for example interleaving Q and K, or spreading consecutive tiles across the chip), so "
     "there is no tail and no per-CTA launch cost. The counter must be correct across back-to-back calls with no host "
     "synchronisation and no extra launch (for example reset by the last CTA to finish); allocate it once outside "
     "run(). First probe whether a work-stealing copy beats a one-shot copy at 12.6, 101 and 805 MB."),
    ("explore:l2-endstate", "Engineer the L2 end state. r5's evict_last stores gained 7-9% at M because output still "
     "in L2 when the kernel ends is written back after the timer stops. Find out which outputs end up resident (the "
     "L2 is 126 MB; the harness fills it with dirty lines before every call), then change the order of work and the "
     "store and load policies so that more output stays resident at M and L sizes. Probe the L2 behaviour first, "
     "for example a copy whose last X MB of stores are evict_last versus all of them."),
    ("explore:wildcard", "Wildcard: propose the idea you believe has the best chance of moving some size band by 3% "
     "or more on the portal, reasoning from first principles, the briefing and the lab notebook. It must differ "
     "structurally from the best kernel, and its core assumption must be tested with probe_b200 before you build."),
]


def plan_explore(arc, max_tasks=5):
    best = best_kernel(arc)
    tried = {str((r.get("task") or {}).get("niche", "")) for r in arc.values()}
    ideas = [i for i in EXPLORE_IDEAS if i[0] not in tried or i[0] == "explore:wildcard"]
    return [dict(operation="new_design", parents=[best["id"]], band="all", niche=tag, gain=None,
                 instructions=text + explore_common()) for tag, text in ideas[:max_tasks]]


def plan(arc, max_tasks=3, last_round=None, mode="exploit"):
    if mode == "explore":
        best = best_kernel(arc)
        return dict(best=best["id"], best_score=best["b200"]["score"], gains=gains(best, anchors()),
                    tasks=plan_explore(arc, max_tasks))
    anc = anchors()
    best = best_kernel(arc)
    g = gains(best, anc)
    tasks = []
    pending = "; ".join(in_flight(arc)) or "none"
    note = f" Ideas already proposed but not yet measured on B200 (do not repeat them): {pending}."
    # Repairs first: candidates that failed correctness or build in the last round.
    for r in arc.values():
        if last_round and r.get("round") == last_round and str(r.get("status", "")).startswith("failed") \
                and r.get("failure_log") and len(tasks) < max_tasks:
            tasks.append(dict(operation="repair", parents=[r["id"]], band="all",
                              instructions="Fix this candidate so it passes the harness; keep its design.",
                              feedback=r["failure_log"]))
    s_task = dict(operation="specialisation", parents=[best["id"]], band="S", gain=g["S"], instructions=(
        f"Goal: cut the fixed cost on small inputs (the S band, <= {P().s_max:.0f} MB). On B200 the parent fits {g['fixed']:.1f} us fixed "
        f"+ bytes / {g['bw']:.1f} TB/s on small and medium sizes; reaching {FIXED_TARGET_US} us fixed would add about "
        f"{g['S']:+.3f} to the score. Write a small-input kernel behind a dispatch:size wrapper that routes every larger "
        f"size to the parent's kernel with the parent's exact configuration, so the B200 result isolates the change. "
        f"The best public B200 streaming kernels take about 2.2-2.5 us in total at the smallest sizes, so about 1 us is "
        f"recoverable; see b200_sota.md section 4 for what they do. "
        f"Keep the answer focused: one candidate, and say in the card what you think the fixed cost consists of. "
        f"Slower on B200 than the parent in this band already: {'; '.join(worse_in_band(arc, best, "S")) or 'none'}." + note))
    l_task = dict(operation="structural_mutation", parents=[best["id"]], band="M,L", gain=g["ML"], instructions=(
        f"Goal: cut medium and large input latency (the M and L bands) on B200. The parent reaches {g['tbs_M']:.2f} TB/s (M) "
        f"and {g['tbs_L']:.2f} TB/s (L) effective; a further 5% would add about {g['ML']:+.3f} to the score. Its fit is "
        f"{g['bw']:.1f} TB/s, above the physical read+write rate: outputs still in L2 when the kernel ends are written "
        f"back after the timed window (see the problem card), so think about what occupies L2 at the end, not "
        f"only bandwidth. Propose the single change most likely to help, correct for every shape, one launch. "
        f"Slower on B200 than the parent in these bands already: "
        f"{'; '.join(worse_in_band(arc, best, "ML")) or 'none'}." + note))
    for t in sorted([s_task, l_task], key=lambda t: -t["gain"]):
        if len(tasks) < max_tasks:
            tasks.append(t)
    tried = {(k, str(v)) for r in arc.values() for k, v in (r.get("niche") or {}).items()} | \
            {("lang", str(r.get("language", "")).replace("cuda_cpp", "cuda").replace("cute_dsl", "cutedsl"))
             for r in arc.values()} | \
            {tuple(str(r["task"]["niche"]).split(":", 1)) for r in arc.values()     # the niche a task asked for
             if ":" in str((r.get("task") or {}).get("niche", ""))}
    for axis, tag, text in EXPLORE_NICHES:
        if (axis, tag) not in tried and len(tasks) < max_tasks:
            tasks.append(dict(operation="new_design", parents=[best["id"]], band="all", niche=f"{axis}:{tag}",
                              instructions=f"Explore the empty niche {axis}:{tag}. {text} It must be correct for every "
                                           f"shape; aim to beat the parent somewhere and say where in the card."))
            break
    return dict(best=best["id"], best_score=best["b200"]["score"], gains=g, tasks=tasks)


def rented_exponents(arc, best=None):
    """Per size band, k such that (portal time ratio) ~ (rented-B200 time ratio) ** k, both relative to the best kernel.

    Least squares through the origin in log space over every kernel timed on both. The rented B200 runs unlocked
    clocks, which exaggerated S/M differences about 1.5-2x in the 7 October calibration (k < 1); L matched (k ~ 1).
    """
    best = best or best_kernel(arc)
    bp, br = b200_times(best), dict((best.get("timings") or {}).get("B200r", {}))
    num, den = defaultdict(float), defaultdict(float)
    for r in arc.values():
        if r["id"] == best["id"]:
            continue
        p, q = b200_times(r), dict((r.get("timings") or {}).get("B200r", {}))
        for k in set(p) & set(q) & set(bp) & set(br):
            if not (p[k] and q[k] and bp[k] and br[k]):
                continue
            x, y = math.log(q[k] / br[k]), math.log(p[k] / bp[k])
            num[band_of(k)] += x * y
            den[band_of(k)] += x * x
    return {b: min(1.5, max(0.3, num[b] / den[b])) if den[b] > 1e-6 else 1.0 for b in "SML"}


def representative_gpu(rec):
    """(gpu with timings, gpu the card names but without timings yet), H200 preferred."""
    card = archive.parse_card(rec.get("card", ""))
    runs = card.get("runs_on") or {}
    def says_representative(v):
        if isinstance(v, dict):
            return str(v.get("representative")).lower() == "true"
        return "representative:true" in str(v).replace(" ", "").replace("'", "").replace('"', "").lower()

    named = [g for g in ("H200", "A100") if says_representative(runs.get(g, ""))]
    have = [g for g in named if rec.get("timings", {}).get(g)]
    return (have[0] if have else None), (named[0] if named and not have else None)


def shortlist(arc, round_id, k_predicted=3, k_explore=2):
    anc = anchors()
    best = best_kernel(arc)
    bt = b200_times(best)
    ranked, explore, awaiting = [], [], []
    try:
        import emulator
        emu = emulator.Emulator(arc)
    except Exception:
        emu = None
    for r in arc.values():
        if r.get("round") != round_id or r.get("status") != "passed" or r.get("b200"):
            continue
        gpu, waiting = representative_gpu(r)
        if (r.get("timings") or {}).get("B200r") and (best.get("timings") or {}).get("B200r"):
            gpu, waiting = "B200r", None          # a rented B200 beats any cheap GPU (calibration 7 October)
        if waiting:
            awaiting.append(dict(id=r["id"], gpu=waiting))
            continue
        ref = (best.get("timings") or {}).get(gpu or "", {})
        if gpu == "B200r" and emu is not None:
            card = archive.parse_card(r.get("card", ""))
            paths = card.get("paths") if isinstance(card.get("paths"), list) else None
            parent = next((pp for pp in (r.get("parents") or []) if pp in emu.feats), best["id"])
            pred = emu.predict(paths or emu.feats.get(parent) or [{}], emulator.rented_times(r),
                               best_score=best["b200"]["score"])
            rel = {k: v / ref[k] for k, v in r["timings"]["B200r"].items() if v and ref.get(k)}
            ranked.append(dict(id=r["id"], gpu=f"B200r (emulator, {pred['n_train']} kernels)"
                               + ("" if paths else ", parent's features"),
                               predicted_score=pred["score"], sd=pred["sd"], p_better=pred["p_better"],
                               rel="/".join(fmt_rel(rel, b) for b in "SML")))
            continue
        if gpu and ref:
            rel = {k: v / ref[k] for k, v in r["timings"][gpu].items() if v and ref.get(k)}
            kx = rented_exponents(arc, best) if gpu == "B200r" else {"S": 1.0, "M": 1.0, "L": 1.0}
            pred = {k: t * (rel[k] ** kx[band_of(k)] if k in rel else 1.0) for k, t in bt.items()}
            ranked.append(dict(id=r["id"], gpu=gpu, predicted_score=score(pred, anc),
                               rel="/".join(fmt_rel(rel, b) for b in "SML")))
        else:
            explore.append(dict(id=r["id"], gpu=None, predicted_score=None,
                                rel="no representative cheap GPU (B200-only features)"))
    ranked.sort(key=lambda x: (-x.get("p_better", -1), -x["predicted_score"]))
    return dict(best=best["id"], best_score=best["b200"]["score"], awaiting=awaiting,
                picks=ranked[:k_predicted] + explore[:k_explore], ranked=ranked, explore=explore)


def fmt_rel(rel, band):
    xs = [v for k, v in rel.items() if band_of(k) == band]
    return f"{100 * (math.exp(sum(map(math.log, xs)) / len(xs)) - 1):+.0f}%" if xs else "-"
