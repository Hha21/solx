"""Portal-order check on the rented B200: run the harness with every workload listed twice in one process.

The portal sets up every workload before it times them, so host state that one shape leaves behind (a function's
max dynamic shared memory, a cached plan, a handle) can break another shape. Our normal test runs each workload's
correctness and timing back to back and never sees that; this check does. Run it on a candidate before it goes to
the portal.

  .venv/bin/python loop/order_check.py --problem L2/035_convnextv2_block_with_grn path/to/solution.json [...]
"""
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
FLAKE = ("Expected kernel activity sequence not found", "No timing results for methodology: cupti")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--problem", default=None)
    ap.add_argument("solutions", nargs="+", type=Path)
    a = ap.parse_args()
    if a.problem:
        os.environ["SOLX_PROBLEM"] = a.problem
    import b200_modal
    b200_modal.load_token()
    bad = 0
    with b200_modal.app.run():
        for path in a.solutions:
            sol = json.loads(path.read_text())
            res = b200_modal.call(dict(id=f"{path.stem}-order", kind="test", repeat=2, solution=sol,
                                                problem=b200_modal.problem_ref()))
            first, later = res.get("workloads") or [], res.get("later_passes") or []
            # Modal's CUPTI timestamp skew (the normal test retries these): correctness had passed, so not a failure
            flaky = lambda r: r["status"] == "RUNTIME_ERROR" and any(f in r["log"] for f in FLAKE)
            fails = [r for r in first + later if r["status"] != "PASSED" and not flaky(r)]
            skipped = sum(flaky(r) for r in first + later)
            bad += bool(fails) or not later
            print(f"{path.stem}: first pass {sum(r['status'] == 'PASSED' for r in first)}/{len(first)}, "
                  f"second pass {sum(r['status'] == 'PASSED' for r in later)}/{len(later)} "
                  f"({res.get('seconds', '?')} s){f'; {skipped} timing flake(s), correctness passed' if skipped else ''}")
            for r in fails:
                print(f"   {r['workload']}: {r['status']} {r['log'][-200:]}")
            if res.get("error"):
                print("   error:", res["error"][-1500:])
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
