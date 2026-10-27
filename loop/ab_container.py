"""Interleaved A/B on the rented B200: time a portal-measured anchor kernel and a candidate alternately in ONE container,
then project the candidate's portal score from the anchor's portal times (per row: portal_anchor x cand/anchor).

Cross-session drift on Modal is 5-12 us per row, larger than many real gains; within one container it mostly cancels.
Each test lists the workloads twice (portal-order check); both passes count as samples.

  .venv/bin/python loop/ab_container.py --problem L2/035_convnextv2_block_with_grn --anchor-sid 65879 \
      anchor.json candidate.json [--reps 2]
"""
import argparse
import csv
import json
import math
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
ROOT = Path(__file__).resolve().parent.parent
FLAKE = ("Expected kernel activity sequence not found", "No timing results for methodology: cupti")


def portal_times(sid):
    out = {}
    for r in csv.DictReader(open(ROOT / "poc" / "results" / "b200_portal_workloads.csv")):
        if r["submission_id"] == str(sid):
            out[",".join(re.findall(r"=(\d+)", r["workload"]))] = float(r["latency_ms"]) * 1e3
    return out


def interleave(sols, reps, tag="ab"):
    """Test each solution ({name: solution}) `reps` times, alternating, inside the caller's b200_modal.app.run().
    Returns ({name: {workload key: [µs]}} from PASSED rows of first and later passes, [(name, rep, workload, status,
    log tail)] for failures that are not timing flakes). Even reps go in order, odd reps in reverse."""
    import b200_modal
    samples, fails, names = {n: {} for n in sols}, [], list(sols)
    for rep in range(reps):
        for who in names if rep % 2 == 0 else names[::-1]:
            res = b200_modal.call(dict(id=f"{tag}-{who}-{rep}", kind="test", repeat=2, solution=sols[who],
                                                problem=b200_modal.problem_ref()))
            if res.get("error"):
                print(f"{who} rep {rep}: error {res['error'][-800:]}")
                fails.append((who, rep, "*", "ERROR", res["error"][-200:]))
            for r in (res.get("workloads") or []) + (res.get("later_passes") or []):
                if r["status"] == "PASSED" and r["latency_us"]:
                    samples[who].setdefault(r["workload"], []).append(r["latency_us"])
                elif not any(f in r["log"] for f in FLAKE):
                    fails.append((who, rep, r["workload"], r["status"], r["log"][-200:]))
            print(f"{who} rep {rep}: {res.get('seconds')} s, {res.get('gpu')}", flush=True)
    return samples, fails


def med(xs):
    return sorted(xs)[len(xs) // 2]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--problem", required=True)
    ap.add_argument("--anchor-sid", required=True, help="portal submission id of the anchor kernel")
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("anchor", type=Path)
    ap.add_argument("candidate", type=Path)
    a = ap.parse_args()
    os.environ["SOLX_PROBLEM"] = a.problem
    import b200_modal
    import planner
    b200_modal.load_token()
    sols = {"anchor": json.loads(a.anchor.read_text()), "cand": json.loads(a.candidate.read_text())}
    with b200_modal.app.run():
        samples, order_fail = interleave(sols, a.reps)
    portal = portal_times(a.anchor_sid)
    proj, rows = {}, []
    for k in sorted(portal, key=lambda k: portal[k]):
        an, ca = samples["anchor"].get(k), samples["cand"].get(k)
        if not an or not ca:
            rows.append(f"{k:18s} missing samples (anchor {len(an or [])}, cand {len(ca or [])})")
            continue
        ratio = med(ca) / med(an)
        proj[k] = portal[k] * ratio
        rows.append(f"{k:18s} anchor {med(an):8.1f} cand {med(ca):8.1f} ({100 * (ratio - 1):+5.1f}%)  n={len(an)}/{len(ca)}"
                    f"  portal anchor {portal[k]:8.1f} -> cand {proj[k]:8.1f}")
    print("\n".join(rows))
    anc = planner.anchors()
    if len(proj) == len(portal):
        print(f"anchor portal score {planner.score(portal, anc):.4f}; candidate projected {planner.score(proj, anc):.4f}")
    for f in order_fail:
        print("FAILED (not a timing flake):", f)
    sys.exit(1 if order_fail else 0)


if __name__ == "__main__":
    main()
