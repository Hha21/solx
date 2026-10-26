"""The design-session tools as an MCP server, for design sessions and the research lead run as headless Claude Code
(`claude -p`) on the user's Claude plan instead of OpenRouter.

Each `claude -p` session starts its own copy of this server (stdio). It reuses designer.Session's tool logic, budgets
and emulator, so a headless session gets the same tools and limits as an OpenRouter one. The B200 (Modal) connection
is opened on the first GPU call, so the research lead (--no-gpu) never starts it.

  .venv/bin/python loop/mcp_tools.py --name <session> --round r11 --dir loop/rounds/r11/sessions [--no-gpu]
"""

import argparse
import atexit
import json
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import archive  # noqa: E402
import designer  # noqa: E402
import planner  # noqa: E402
import problem  # noqa: E402

from mcp.server.mcpserver import MCPServer  # noqa: E402


def eprint(*args, **kw):
    kw.pop("file", None)
    print(*args, file=sys.stderr, **kw)


designer.print = eprint                           # stdout carries the MCP protocol; status lines go to stderr

ap = argparse.ArgumentParser()
ap.add_argument("--name", required=True)
ap.add_argument("--round", required=True)
ap.add_argument("--dir", type=Path, required=True)
ap.add_argument("--gpu-calls", type=int, default=6)
ap.add_argument("--probes", type=int, default=30)
ap.add_argument("--b200-minutes", type=float, default=10)
ap.add_argument("--no-gpu", action="store_true", help="research lead: archive and score-model tools only")
ap.add_argument("--check", action="store_true", help="load everything the server needs, then exit (start-up test)")
ap.add_argument("--fresh", action="store_true", help="fresh-eyes session: no get_kernel (earlier designs stay hidden)")
ap.add_argument("--lineage", help="portfolio session: get_kernel reads only this lineage's kernels and r0 baselines")
A = ap.parse_args()
A.dir.mkdir(parents=True, exist_ok=True)

ARC = archive.load()
BEST = planner.best_kernel(ARC)
ANC = planner.anchors()
try:
    import emulator
    EMU = emulator.Emulator(ARC)
except Exception:
    EMU = None
SESSION = designer.Session(A.name, {"round": A.round}, "", "", None, BEST, ANC, ARC,
                           argparse.Namespace(gpu_calls=A.gpu_calls, probes=A.probes), A.dir, EMU)
LOCK = threading.Lock()
LOG = A.dir / f"{A.name}.tools.jsonl"


def gpu_server():
    """Open the Modal B200 on first use and time the reference kernel there (once per session)."""
    if SESSION.server is None:
        srv = designer.ModalB200(A.round, BEST["solution"] if BEST else None, budget_minutes=A.b200_minutes,
                                 workspace=f"{problem.current().name}/{A.round}/{A.name}")
        atexit.register(srv.stop)
        srv.wait_alive(0)
        SESSION.server = srv
    return SESSION.server


def call(fn, args):
    if fn == "get_kernel" and A.lineage:          # lineages share conclusions, not code (loop/portfolio.py)
        import portfolio
        rec = ARC.get(args.get("id", ""))
        if rec is None or not portfolio.visible(rec, A.lineage):
            ids = sorted(k for k, r in ARC.items() if r.get("solution") and portfolio.visible(r, A.lineage))
            return (f"{args.get('id')!r} is not readable from lineage {A.lineage} (other lineages share conclusions, "
                    f"not code). Readable ids: {', '.join(ids) or 'none'}")
    with LOCK:                                    # one tool call at a time per session, like the OpenRouter loop
        if fn in ("compile_b200", "run_tests", "probe_b200"):
            if A.no_gpu:
                return "GPU tools are not available to the research lead."
            gpu_server()
        out = SESSION.tool({"id": "mcp", "function": {"name": fn, "arguments": json.dumps(args)}})
        if A.fresh and BEST:                      # the best kernel's id names its design: keep it out of fresh sessions
            out = out.replace(BEST["id"], "current-best")
        with open(LOG, "a") as f:
            f.write(json.dumps({"tool": fn, "args": args, "result": out[:20000]}) + "\n")
        return out


def describe(name):
    return next(t["function"]["description"] for t in designer.TOOLS if t["function"]["name"] == name)


server = MCPServer("solx", instructions="Tools for designing SOL-ExecBench kernels: archive, score model, and (in design "
                                        "sessions) a rented NVIDIA B200 for compiling, testing and probing.")


if not A.fresh:
    @server.tool(description=describe("get_kernel"))
    def get_kernel(id: str, part: str = "") -> str:
        return call("get_kernel", {"id": id, "part": part})


@server.tool(description=describe("read_example"))
def read_example(name: str = "") -> str:
    return call("read_example", {"name": name})


@server.tool(description=describe("predict_score"))
def predict_score(pct_change: dict | None = None, fixed_us: float | None = None, tbs: float | None = None) -> str:
    return call("predict_score", {k: v for k, v in dict(pct_change=pct_change, fixed_us=fixed_us, tbs=tbs).items()
                                  if v is not None})


if not A.no_gpu:
    @server.tool(description=describe("compile_b200"))
    def compile_b200(solution_spec: dict, files: list[dict], note: str = "", paths: list[dict] | None = None) -> str:
        return call("compile_b200", dict(solution_spec=solution_spec, files=files, note=note, paths=paths))

    @server.tool(description=describe("run_tests"))
    def run_tests(solution_spec: dict, files: list[dict], note: str = "", paths: list[dict] | None = None) -> str:
        return call("run_tests", dict(solution_spec=solution_spec, files=files, note=note, paths=paths))

    @server.tool(description=describe("probe_b200"))
    def probe_b200(script: str, note: str = "") -> str:
        return call("probe_b200", dict(script=script, note=note))


if __name__ == "__main__":
    if A.check:
        print(f"ok: {len(ARC)} archive records, best {BEST['id'] if BEST else None}, emulator {EMU is not None}",
              file=sys.stderr)
        sys.exit(0)
    server.run("stdio")
