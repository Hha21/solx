"""Time every portal-measured kernel on the rented B200 (twice) and record its sm_100a compile statistics.

The pairs (portal time, rented time) per workload are the training data for the multi-fidelity emulator
(loop/emulator.py). Rep 1 -> loop/b200/timing/B200r, rep 2 -> loop/b200/timing/B200r2, compile -> loop/b200/static_B200r.jsonl.

  .venv/bin/python loop/pair_timings.py [ids ...]      (default: every archive kernel with a portal result)
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import archive  # noqa: E402
import b200_modal  # noqa: E402

import problem  # noqa: E402

D = problem.current().b200


def main():
    arc = archive.load()
    ids = sys.argv[1:] or sorted(c for c, r in arc.items()
                                 if (r.get("b200") or {}).get("timings") and r.get("solution")
                                 and (archive.ROOT / r["solution"]).exists())
    b200_modal.load_token()
    statics = {}
    sf = D / "static_B200r.jsonl"
    if sf.exists():
        statics = {json.loads(l)["id"]: json.loads(l) for l in open(sf) if l.strip()}
    t0 = time.time()
    with b200_modal.app.run():
        for cid in ids:
            sol = json.loads((archive.ROOT / arc[cid]["solution"]).read_text())
            for rep, gdir in ((1, "B200r"), (2, "B200r2")):
                res = b200_modal.call(dict(id=f"{cid}-{rep}", kind="test", solution=sol, problem=b200_modal.problem_ref()))
                b200_modal.write_trace(D / "timing" / gdir / f"{cid}.jsonl", res)
                wl = res.get("workloads") or []
                print(f"{cid:28} rep {rep}: {sum(w['status'] == 'PASSED' for w in wl)}/{len(wl)} passed, "
                      f"{res.get('seconds')} s", flush=True)
            if cid not in statics:
                comp = b200_modal.call(dict(id=f"{cid}-c", kind="compile", solution=sol, problem=b200_modal.problem_ref()))
                statics[cid] = dict(id=cid, kernels=comp.get("kernels") or [], error=comp.get("error") or "")
                sf.write_text("".join(json.dumps(x) + "\n" for x in statics.values()))
    print(f"{len(ids)} kernels in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
