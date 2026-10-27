"""Calibrate the rented B200 (Modal) against the portal: time kernels that already have portal results and compare.

For each kernel: per-size latency on the rented B200 vs the portal, their ratio, and the kernel's rank by geomean.
If the ranking matches the portal's, the rented B200 is a trustworthy test bench for the design sessions.

  .venv/bin/python loop/calibrate_b200.py [ids ...]
"""

import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import archive  # noqa: E402
import b200_modal  # noqa: E402
import problem  # noqa: E402

DEFAULT = ["r5-ldg256-os-r16-stel", "r3-cute-ldg256-os-r16", "d1-os-r8w4-nohint", "g2-os-r8w4", "r2-ldg256-os-r16"]


def gmean(xs):
    return math.exp(sum(map(math.log, xs)) / len(xs))


def main():
    ids = sys.argv[1:] or DEFAULT
    arc = archive.load()
    b200_modal.load_token()
    out = dict(time=time.strftime("%Y-%m-%dT%H:%M:%S"), kernels={})
    import modal
    with modal.enable_output(), b200_modal.app.run():
        out["info"] = b200_modal.info.remote()
        print(json.dumps(out["info"], indent=1), flush=True)
        for cid in ids:
            sol = json.loads((archive.ROOT / arc[cid]["solution"]).read_text())
            t0 = time.time()
            res = b200_modal.call(dict(id=cid, kind="test", solution=sol, problem=b200_modal.problem_ref()))
            wl = res.get("workloads") or []
            ok = wl and all(w["status"] == "PASSED" for w in wl)
            print(f"{cid}: {sum(w['status'] == 'PASSED' for w in wl)}/{len(wl)} passed in {time.time() - t0:.0f} s"
                  + ("" if ok else f"\n{str(res.get('error') or res.get('console_tail'))[-2000:]}"), flush=True)
            b200_modal.write_trace(problem.current().b200 / "timing" / "B200r" / f"{cid}.jsonl", res)
            out["kernels"][cid] = dict(rented={w["workload"]: w["latency_us"] for w in wl if w.get("latency_us")},
                                       portal=arc[cid]["b200"]["timings"], gpu=res.get("gpu"))
    d = problem.current().b200
    d.mkdir(exist_ok=True)
    path = d / f"calibration_{time.strftime('%Y%m%d-%H%M%S')}.json"
    path.write_text(json.dumps(out, indent=1))
    rows = []
    for cid, k in out["kernels"].items():
        keys = [x for x in k["portal"] if x in k["rented"]]
        if not keys:
            continue
        ratio = {x: k["rented"][x] / k["portal"][x] for x in keys}
        bands = {b: [ratio[x] for x in keys if problem.current().band(x) == b] for b in "SML"}
        rows.append((cid, gmean([k["portal"][x] for x in keys]), gmean([k["rented"][x] for x in keys]),
                     {b: gmean(v) for b, v in bands.items() if v}))
    print(f"\n{'kernel':26} {'portal µs':>9} {'rented µs':>9}   rented/portal by band S/M/L")
    for cid, p, r, b in rows:
        print(f"{cid:26} {p:9.2f} {r:9.2f}   " + " / ".join(f"{b.get(x, float('nan')):.3f}" for x in "SML"))
    po = [c for c, *_ in sorted(rows, key=lambda r: r[1])]
    ro = [c for c, *_ in sorted(rows, key=lambda r: r[2])]
    print(f"\nportal order: {po}\nrented order: {ro}\nsame order: {po == ro}\nsaved {path.relative_to(archive.ROOT)}")


if __name__ == "__main__":
    main()
