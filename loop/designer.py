"""Interactive design sessions: the inner LLM drafts a kernel, checks it with tools, revises it, then submits.

Tools offered to the model:
  compile_b200  compile a draft for sm_100a (B200): registers, shared memory, spills, CTAs per SM, load/store widths
  run_tests     NVIDIA's harness on a cheap GPU: correctness on every workload, timing against the current best
                kernel on the same GPU, and the B200 score that timing projects to (the emulator)
  predict_score the planner's maths model: per-band latency changes -> B200 score
  get_kernel    any archive kernel's source, design card and measurements
The two GPU tools go to loop/toolserver.py, a Slurm job on CSF3 that serves every session of a round through a
file queue on /scratch. Transcripts are saved to loop/rounds/<round>/sessions/<name>.{jsonl,md}.
"""

import json
import math
import subprocess
import threading
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import archive
import llm
import planner
import problem
import prompts
import reply

REMOTE, RSOLX = "csf3", "/scratch/t95317ha/solx"
CACHE = {"type": "ephemeral", "ttl": "1h"}
EXAMPLES = archive.ROOT / "loop" / "context" / "examples"

PROTOCOL = """=== DESIGN SESSION PROTOCOL (interactive) ===
This is an interactive design session. Use the tools to check and improve your design before you commit to it:
1. Draft. 2. compile_b200: fix compile errors; check registers, CTAs per SM, spills, and that the B200 SASS uses the
memory instructions you intended (e.g. LDG.256 / STG.256). 3. run_tests: correctness on every workload, timing on a
cheap GPU against the current best kernel, and the projected B200 score. 4. Revise while it pays. 5. Final answer.
- Local timing runs on {gpu}. Unless that is a B200, it cannot run B200-only paths (256-bit loads, sm_100-only
  instructions) or show B200's 126 MB L2; those run a fallback there, so judge them from compile_b200 output and
  reasoning, not from local speed. A rented B200 runs unlocked clocks (the portal locks SM 1500 / DRAM 3996 MHz), so
  trust its relative timings more than its absolute ones. Timing noise is
  about 1% on medium/large sizes and up to 3% on the smallest; re-test before trusting a change that small.
- predict_score turns per-band latency changes into a B200 score, so you can see which band is worth the effort.
- probe_b200 (rented B200 only) runs a short Python experiment in seconds: use it to check a property (what a
  construct compiles to, how a cache hint or launch shape behaves, a latency) before spending a full test on it. A
  fast loop for CUDA ideas: iterate on the extern "C" kernel with b200probe.cuda_kernel + harness_time on the real
  shapes (seconds each), then run_tests the finished solution once.
- CUDA C++ solutions use two files (generation protocol section 3): kernel.cu without PyTorch headers plus
  binding.cpp. A full B200 test then takes about 10-20 s instead of about a minute. Triton and CuTe DSL tests take
  about 10 s.
- get_kernel returns any archive kernel's source, design card and measurements; read_example shows annotated
  excerpts of the best public B200 norm kernels (see b200_sota.md).
- Budget: {gpu_calls} GPU calls (compile_b200 + run_tests) and {turns} turns. GPU calls queue on one shared GPU; a test
  takes 1-4 minutes, a compile under 1. compile_b200 and run_tests need complete files every time. probe_b200 runs in
  a working directory that persists for this session, across probes and containers: write your kernel sources there
  once and change them with short Python edits in later probes instead of resending them.
- Finish with a reply that makes no tool call and contains exactly one candidate in the OUTPUT CONTRACT format. Put
  what you measured into the design card (hypothesis, expected_effect), and list what you learned under findings,
  dead ends included: later sessions read them in the lab notebook. Submit the best version you tested, with its
  code unchanged apart from comments; if you clean it up, run_tests it again first. If its benefit is B200-only, say
  so in the card."""

DRAFT = {"type": "object", "properties": {
    "solution_spec": {"type": "object", "description": "the solution JSON without sources, e.g. {\"languages\": "
                      "[\"cute_dsl\"], \"entry_point\": \"kernel.py::run\", \"dependencies\": [\"torch\"], "
                      "\"destination_passing_style\": true}"},
    "files": {"type": "array", "items": {"type": "object", "properties": {
        "path": {"type": "string"}, "content": {"type": "string", "description": "complete file"}},
        "required": ["path", "content"]}},
    "note": {"type": "string", "description": "one line: what this draft changes"},
    "paths": {"type": "array", "description": "the design card's `paths` block for this draft (what each size "
              "range's code path does: lang, width, threads, rows, grid, mem, x, w, st, prefetch...). run_tests uses "
              "it to predict the portal score with the emulator; without it the prediction has wider error bars.",
              "items": {"type": "object"}}},
    "required": ["solution_spec", "files"]}

TOOLS = [
    {"type": "function", "function": {"name": "compile_b200", "parameters": DRAFT, "description":
        "Compile a draft for sm_100a (B200) and report each kernel's registers, shared memory, local (spill) bytes, "
        "CTAs per SM when known, and the count of global load/store instructions by width (LDG.256, LDG.128, ...). "
        "Returns compiler errors if it does not build. Under a minute."}},
    {"type": "function", "function": {"name": "run_tests", "parameters": DRAFT, "description":
        "Run NVIDIA's harness on the session GPU: correctness on every workload (with errors and logs on failure) "
        "and latency per workload, compared with the current best kernel timed on the same GPU, plus the projected "
        "B200 score. 1-4 minutes."}},
    {"type": "function", "function": {"name": "probe_b200", "description":
        "Run a short Python experiment on the B200 in seconds, to check a property before writing a full kernel: "
        "a micro-benchmark, which instructions a construct compiles to, how a cache hint behaves, a latency, an "
        "occupancy. Module b200probe is importable: cuda_kernel(src, name) compiles extern \"C\" CUDA with NVRTC in "
        "about a second and returns a launcher k(grid, block, *args); harness_time(fn, inputs, outputs) times like "
        "NVIDIA's harness (CUPTI, cold L2, shifted pointers) and returns µs; sass(src) / resources(src) show SASS and "
        "registers; flush_l2(). torch, triton and cutlass (CuTe DSL) are also available. Print what you need; output "
        "is cut to 6000 characters; 120 s limit. Probes do not count as GPU test calls but use the B200 budget.",
        "parameters": {"type": "object", "properties": {
            "script": {"type": "string", "description": "complete Python script"},
            "note": {"type": "string", "description": "one line: what this probe checks"}},
            "required": ["script"]}}},
    {"type": "function", "function": {"name": "predict_score", "description":
        "Predict the B200 score from latency changes relative to the current best kernel's B200 times, using the "
        "real scoring formula and the hidden baseline per workload. Give pct_change per band (negative = faster), "
        "or fixed_us and tbs to model every size as fixed_us + bytes / tbs. Also reports how much a 10% gain in "
        "each band is worth.", "parameters": {"type": "object", "properties": {
            "pct_change": {"type": "object", "properties": {"S": {"type": "number"}, "M": {"type": "number"},
                                                            "L": {"type": "number"}}},
            "fixed_us": {"type": "number"}, "tbs": {"type": "number"}}}}},
    {"type": "function", "function": {"name": "read_example", "description":
        "Annotated excerpts of state-of-the-art public kernels (FlashInfer, quack, SGLang, CAKE), with source and "
        "licence. Call with no name for the list. Learn techniques from them; do not copy them wholesale.",
        "parameters": {"type": "object", "properties": {"name": {"type": "string"}}}}},
    {"type": "function", "function": {"name": "get_kernel", "description":
        "Source, design card, measurements and sm_100a statistics of an archive kernel, by id. A large kernel returns "
        "its card and a file index first; read a file with part=\"<path>\" or lines with part=\"<path>:120-480\".",
        "parameters": {"type": "object", "properties": {"id": {"type": "string"}, "part": {"type": "string"}},
                       "required": ["id"]}}},
]


def ssh(cmd, stdin=None, check=True):
    return subprocess.run(["ssh", "-o", "BatchMode=yes", REMOTE, cmd], input=stdin, capture_output=True, text=True,
                          check=check).stdout


band_of = planner.band_of


def gmean(xs):
    return math.exp(sum(map(math.log, xs)) / len(xs)) if xs else None


class ToolServer:
    """Client side of loop/toolserver.py: start the Slurm job, send requests, wait for results."""

    def __init__(self, round_id, gpus, slurm, ref_solution, wait_s=2700):
        """gpus: one GPU type or several sharing a Slurm account ("A100,L40S"); the job runs on whichever frees first."""
        self.name = f"{round_id}-{time.strftime('%m%d-%H%M%S')}"
        self.queue = f"{RSOLX}/toolq/{self.name}"
        self.wait_s, self.ref_solution = wait_s, ref_solution
        accts = {slurm[g][0] for g in gpus.split(",")}
        if len(accts) != 1:
            raise SystemExit(f"--tool-gpu {gpus}: these GPUs use different Slurm accounts; pick one account's GPUs")
        parts = ",".join(slurm[g][1] for g in gpus.split(","))
        ssh(f"mkdir -p {self.queue}/req")
        self.job = ssh(f"cd {RSOLX} && sbatch --parsable -A {accts.pop()} -p {parts} --export=ALL,QUEUE={self.queue},"
                       f"REF={ref_solution} loop/jobs/toolserver.sbatch").strip()
        self.gpu, self.ref, self.lock, self.n = gpus, None, threading.Lock(), 0
        print(f"tool server: {gpus} job {self.job}, queue {self.queue}", flush=True)

    def wait_alive(self, timeout_s):
        """Block until the server job runs (returns the GPU name) or the timeout passes (returns None)."""
        t0, last = time.time(), None
        while time.time() - t0 < timeout_s:
            name = ssh(f"cat {self.queue}/alive 2>/dev/null", check=False).strip()
            if name:
                self.gpu = name
                return name
            state = ssh(f"squeue -h -j {self.job} -o '%T %R'", check=False).strip()
            if state != last:
                print(f"tool server job {self.job}: {state or 'not in queue'}", flush=True)
                last = state
            if not state:
                return None                               # job ended without starting the server
            time.sleep(30)
        return None

    def stop(self):
        ssh(f"touch {self.queue}/stop; scancel {self.job}", check=False)

    def call(self, kind, sol, **extra):
        with self.lock:
            self.n += 1
            rid = f"{kind}-{self.n:03d}"
        ssh(f"cat > {self.queue}/req/.{rid}.part && mv {self.queue}/req/.{rid}.part {self.queue}/req/{rid}.json",
            stdin=json.dumps(dict(id=rid, kind=kind, solution=sol, **extra)))
        t0 = time.time()
        while time.time() - t0 < self.wait_s:
            out = ssh(f"cat {self.queue}/res/{rid}.json 2>/dev/null", check=False)
            if out.strip():
                return json.loads(out)
            time.sleep(5)
        ssh(f"rm -f {self.queue}/req/{rid}.json", check=False)
        return dict(error=f"no result after {self.wait_s // 60} minutes (the GPU job may still be queued); "
                          "continue without this measurement")

    def reference(self):
        if self.ref is None:
            out = ssh(f"cat {self.queue}/ref.json 2>/dev/null", check=False)
            if out.strip():
                self.ref = json.loads(out)
        return self.ref


class ModalB200:
    """Same interface as ToolServer, backed by a real B200 on Modal (loop/b200_modal.py), billed per second.

    Calls are serialised on one container (max_containers=1), so timings never overlap. A cap on B200 minutes per
    round stops the GPU tools (sessions are told to finish) rather than letting spend run on.
    """

    def __init__(self, round_id, ref_solution, budget_minutes=60, workspace=None):
        import modal
        import b200_modal
        b200_modal.load_token()
        self.name, self.ref_solution, self.budget_s = f"{round_id}-modal", ref_solution, budget_minutes * 60
        self.workspace = workspace                    # probe_b200's persistent working directory on the B200
        self.problem = b200_modal.problem_ref()
        self._ctx = b200_modal.app.run()
        self._ctx.__enter__()
        self.fn, self.gpu, self.ref = b200_modal.handle, "B200 (Modal)", None
        self.lock, self.n, self.used_s = threading.Lock(), 0, 0.0
        print(f"tool server: Modal B200, budget {budget_minutes} min", flush=True)

    def call(self, kind, sol, **extra):
        with self.lock:
            if self.used_s >= self.budget_s:
                return dict(error=f"the round's B200 budget ({self.budget_s // 60:.0f} min) is used up; finish "
                                  "with what you have")
            self.n += 1
            rid = f"{kind}-{self.n:03d}"
        t0 = time.time()
        req = dict(id=rid, kind=kind, solution=sol, problem=self.problem, **extra)
        if kind == "probe" and self.workspace:
            req["workspace"] = self.workspace
        try:
            res = self.fn.remote(req)
        except Exception as e:
            res = dict(id=rid, kind=kind, error=f"Modal call failed: {type(e).__name__}: {e}")
            # a network drop stops the ephemeral app ("function ... is stopped"); start a fresh one and retry once
            if any(m in str(e) for m in ("is stopped", "not running", "hydrated")) and self._restart():
                try:
                    res = self.fn.remote(req)
                except Exception as e2:
                    res = dict(id=rid, kind=kind, error=f"Modal call failed after restart: {type(e2).__name__}: {e2}")
        with self.lock:      # container time for this call (+5 s overhead); queueing behind other sessions is free
            self.used_s += float(res.get("seconds") or (time.time() - t0)) + 5
        if res.get("gpu") and "B200" not in res["gpu"]:
            res = dict(id=rid, kind=kind, error=f"wrong GPU {res['gpu']!r}; result discarded")
        return res

    def _restart(self):
        import b200_modal
        with self.lock:
            try:
                self._ctx.__exit__(None, None, None)
            except Exception:
                pass
            try:
                self._ctx = b200_modal.app.run()
                self._ctx.__enter__()
                self.fn = b200_modal.handle
            except Exception as e:
                print(f"tool server: Modal app restart failed: {type(e).__name__}: {e}", flush=True)
                return False
        print("tool server: the Modal app was stopped; started a new one", flush=True)
        return True

    def wait_alive(self, timeout_s):
        """Time the reference kernel (this also starts and warms the container); returns the GPU name."""
        if not self.ref_solution:                     # new problem: nothing on the portal yet, no reference kernel
            return self.gpu
        sol = json.loads((archive.ROOT / self.ref_solution).read_text())
        res = self.call("test", sol)
        if not res.get("workloads"):
            print(f"reference run failed: {str(res.get('error') or res.get('console_tail'))[-1500:]}", flush=True)
            return None
        self.ref = dict(ref=Path(self.ref_solution).stem, gpu=res.get("gpu"), workloads=res["workloads"])
        self.gpu = res.get("gpu") or self.gpu
        return self.gpu

    def reference(self):
        return self.ref

    def stop(self):
        print(f"Modal B200 used for about {self.used_s / 60:.1f} min (~${self.used_s / 3600 * 6.25:.2f})", flush=True)
        self._ctx.__exit__(None, None, None)


def code_key(sol):
    """Hash of a solution's code with comment-only and blank lines dropped, so a cleaned-up copy of a tested draft
    matches it. Anything else that changed means a new measurement."""
    import hashlib
    h = hashlib.sha256(json.dumps({k: v for k, v in sol["spec"].items() if k != "description"}, sort_keys=True).encode())
    for src in sorted(sol["sources"], key=lambda x: x["path"]):
        h.update(src["path"].encode())
        for line in src["content"].splitlines():
            t = line.strip()
            if t and not t.startswith(("//", "#", "/*", "*")):
                h.update(line.rstrip().encode())
    return h.hexdigest()[:16]


def make_solution(args, name):
    """Tool arguments -> (solution JSON, problems)."""
    spec = dict(args.get("solution_spec") or {})
    spec = dict(spec.get("spec", spec))
    files = [f for f in args.get("files") or [] if isinstance(f, dict) and f.get("path")]
    problems = [] if files else ["no files given"]
    entry = spec.get("entry_point", "")
    if files and entry.split("::")[0] not in [f["path"] for f in files]:
        problems.append(f"entry_point {entry!r} is not one of the files")
    spec.setdefault("target_hardware", ["B200", "LOCAL"])
    spec.setdefault("destination_passing_style", True)
    sol = {"name": name, "definition": problem.current().name, "author": "solx-loop",
           "description": str(args.get("note") or "")[:500], "spec": spec,
           "sources": [{"path": f["path"], "content": f.get("content", "")} for f in files]}
    return sol, problems + reply.lint(sol)


def fmt_compile(res):
    lines = [f"compile_b200 for sm_100a ({res.get('seconds', '?')} s): {len(res.get('kernels') or [])} kernel(s)"]
    for k in res.get("kernels") or []:
        ops = ", ".join(f"{op} x{n}" for op, n in sorted((k.get("ops") or {}).items()))
        fit = k.get("resident_per_sm")
        lines.append(f"- {str(k.get('kernel'))[:60]}: {k.get('regs')} regs/thread, {k.get('smem')} B shared, "
                     f"{k.get('local')} B local (spills), {'%s CTAs/SM' % fit if fit else 'CTAs/SM unknown'}; "
                     f"memory ops in SASS: {ops or 'none found'}")
    if res.get("error"):
        lines.append(f"error:\n{res['error'][-3000:]}")
    return "\n".join(lines)


def emulator_line(emu, paths, wl, ref, best):
    """Predicted portal score from this B200 run, via the multi-fidelity emulator (loop/emulator.py)."""
    rented = {w["workload"]: w["latency_us"] for w in wl if w.get("latency_us")}
    ref_now = {w["workload"]: w["latency_us"] for w in ref["workloads"] if w.get("latency_us")}
    import emulator
    ref_train = emulator.rented_times(emu.arc.get(best["id"]) or {})
    if ref_train:      # this container may run a little faster or slower than the training runs: rescale by the reference
        rented = {k: v * ref_train[k] / ref_now[k] for k, v in rented.items() if k in ref_train and k in ref_now}
    guessed = not paths
    p = emu.predict(paths or emu.feats.get(best["id"]) or [{}], rented, best_score=best["b200"]["score"])
    sd = math.sqrt(p["sd"] ** 2 + (0.004 ** 2 if guessed else 0))
    return (f"predicted portal score {p['score']:.4f} ± {sd:.4f} (emulator trained on {p['n_train']} portal kernel(s) of "
            f"this problem{': treat as a rough guide' if p['n_train'] < 5 else ''}; current best "
            f"{best['b200']['score']:.4f} on the portal; "
            f"P(beats it) = {p['p_better']:.0%}). From the multi-fidelity emulator: portal = this B200's time x a "
            f"correction learned from {len(emu.kernels)} kernels measured on both, by size and by design features; "
            f"the ± grows for designs unlike those." + (" No `paths` given, so the parent's features were assumed; "
                                                         "pass `paths` for a sharper prediction." if guessed else ""))


def copy_lag_line(wl):
    """Each workload's time against a plain read+write copy of the same bytes on the rented B200 (if measured)."""
    import emulator
    copy = emulator.copy_reference()
    cells = [f"{w['workload']} {w['latency_us']:.1f} us ({w['latency_us'] / copy[w['workload']]:.2f}x copy)"
             if copy.get(w["workload"]) else f"{w['workload']} {w['latency_us']:.1f} us"
             for w in sorted(wl, key=lambda w: problem.current().mbytes(w["workload"])) if w.get("latency_us")]
    return "times (µs; x copy = against a plain copy of the same bytes): " + "; ".join(cells)


def fmt_test(res, ref, best, anc, k=None, emu=None, paths=None):
    wl = res.get("workloads") or []
    passed = sum(w["status"] == "PASSED" for w in wl)
    lines = [f"run_tests on {res.get('gpu', '?')} ({res.get('seconds', '?')} s): {passed}/{len(wl) or '?'} workloads "
             "PASSED"]
    for w in wl:
        if w["status"] != "PASSED":
            lines.append(f"- {w['workload']}: {w['status']}, max abs err {w['max_abs_err']}, max rel err "
                         f"{w['max_rel_err']}\n  {w['log'][-800:]}")
    if res.get("console_tail"):
        lines.append(f"harness output (tail):\n{res['console_tail'][-3000:]}")
    if res.get("error"):
        lines.append(f"error: {res['error']}")
    if not wl or passed < len(wl):
        return "\n".join(lines)
    if not ref or not ref.get("workloads") or best is None:
        return "\n".join(lines + [copy_lag_line(wl), "No portal result for this problem yet, so there is no reference "
                                   "kernel or score model: compare designs by these times and their lag behind a "
                                   "plain copy. The first portal submission unlocks predicted portal scores."])
    rt = {w["workload"]: w["latency_us"] for w in ref["workloads"] if w.get("latency_us")}
    rel = {}
    rows = []
    for w in wl:
        if w["latency_us"] and rt.get(w["workload"]):
            r = w["latency_us"] / rt[w["workload"]]
            rel[w["workload"]] = r
            rows.append(f"{w['workload']} {w['latency_us']:.1f} vs {rt[w['workload']]:.1f} ({100 * (r - 1):+.1f}%)")
    lines.append(f"latency vs reference {ref['ref']} on the same GPU (µs, this vs reference):\n  " + "; ".join(rows))
    by_band = {b: gmean([v for k, v in rel.items() if band_of(k) == b]) for b in "SML"}
    lines.append("change by band (geomean): " + ", ".join(f"{b} {100 * (v - 1):+.1f}%" for b, v in by_band.items() if v))
    on_b200 = "B200" in str(res.get("gpu"))
    if on_b200 and emu is not None:
        lines.append(emulator_line(emu, paths, wl, ref, best))
        return "\n".join(lines)
    bt = planner.b200_times(best)
    k = (k or {}) if on_b200 else {}
    pred = {key: t * (rel[key] ** k.get(band_of(key), 1.0) if key in rel else 1.0) for key, t in bt.items()}
    if on_b200:
        lines.append(f"projected portal score: {planner.score(pred, anc):.4f} (current best {best['b200']['score']:.4f}). "
                     f"This is a real B200 with unlocked clocks; differences at S/M come out larger than on the portal, "
                     f"so the projection shrinks them (portal change ~ measured change ** k, k = "
                     f"{', '.join(f'{b} {v:.2f}' for b, v in k.items())}, fitted on kernels measured on both).")
    else:
        lines.append(f"projected B200 score (emulator: the reference's B200 times x this GPU's relative speed): "
                     f"{planner.score(pred, anc):.4f} (current best {best['b200']['score']:.4f}). Not valid for "
                     "B200-only paths, which run a fallback on this GPU.")
    return "\n".join(lines)


def predict(args, best, anc):
    if best is None or not anc:
        return ("No portal result for this problem yet, so the score model has no anchors (hidden baseline and SOL "
                "time per workload). The first portal submission provides them.")
    bt = planner.b200_times(best)
    if args.get("fixed_us") is not None and args.get("tbs"):
        pred = {bs: args["fixed_us"] + planner.mb(bs) / args["tbs"] for bs in bt}
        how = f"every size = {args['fixed_us']} µs + bytes / {args['tbs']} TB/s"
    else:
        pct = {k: float(v) for k, v in (args.get("pct_change") or {}).items()}
        pred = {bs: t * (1 + pct.get(band_of(bs), 0) / 100) for bs, t in bt.items()}
        how = f"per-band change {pct} vs {best['id']}"
    base = planner.score(bt, anc)
    worth = {b: planner.score({bs: t * (0.9 if band_of(bs) == b else 1) for bs, t in bt.items()}, anc) - base
             for b in "SML"}
    gm = {b: gmean([t for bs, t in pred.items() if band_of(bs) == b]) for b in "SML"}
    return (f"predicted B200 score {planner.score(pred, anc):.4f} for {how} (this model gives the current best "
            f"{base:.4f}; its portal score is {best['b200']['score']:.4f}). "
            f"Predicted B200 geomean µs: S {gm['S']:.1f}, M {gm['M']:.1f}, L {gm['L']:.1f}. "
            f"Worth of a 10% latency cut alone: S {worth['S']:+.4f}, M {worth['M']:+.4f}, L {worth['L']:+.4f}.")


def with_tail_cache(messages):
    """Copy of messages with a cache breakpoint on the last one, so the next turn re-reads the conversation cheaply."""
    out = list(messages)
    last = dict(out[-1])
    if isinstance(last.get("content"), str) and last["role"] in ("tool", "user"):
        last["content"] = [{"type": "text", "text": last["content"], "cache_control": CACHE}]
        out[-1] = last
    return out


class Session:
    def __init__(self, name, task, static, dyn, server, best, anc, arc, a, d, emu=None):
        self.name, self.task, self.server, self.best, self.anc, self.arc, self.a = name, task, server, best, anc, arc, a
        self.messages = [llm.system_message(static),
                         {"role": "user", "content": [{"type": "text", "text": dyn, "cache_control": CACHE}]}]
        self.gpu_calls, self.drafts, self.probes = 0, 0, 0
        self.k = planner.rented_exponents(arc, best) if best else {}
        self.emu = emu
        self.totals = dict(prompt_tokens=0, completion_tokens=0, cached_tokens=0, reasoning_tokens=0, cost_usd=0.0,
                           seconds=0.0)
        self.log_json, self.log_md = d / f"{name}.jsonl", d / f"{name}.md"
        self.measured, self.measured_path = {}, d / f"{name}.b200.json"
        self.md = [f"# Design session {name}\n\nTask: {json.dumps(task)}\n"]

    def note(self, text):
        self.md.append(text)
        self.log_md.write_text("\n".join(self.md))

    def tool(self, call):
        fn = call["function"]["name"]
        try:
            args = json.loads(call["function"].get("arguments") or "{}")
        except json.JSONDecodeError as e:
            return f"invalid JSON arguments: {e}"
        if fn == "get_kernel":
            rec = self.arc.get(args.get("id", ""))
            return prompts.kernel_view(rec, args.get("part") or "") if rec and rec.get("solution") else \
                f"unknown id {args.get('id')!r}; ids: {', '.join(sorted(self.arc))}"
        if fn == "predict_score":
            return predict(args, self.best, self.anc)
        if fn == "read_example":
            files = {f.name: f for f in sorted(EXAMPLES.glob("*")) if f.is_file()}
            name = args.get("name") or ""
            if name in files:
                return files[name].read_text()
            return "examples: " + ", ".join(f"{n} ({len(f.read_text().splitlines())} lines)" for n, f in files.items())
        if fn == "probe_b200":
            if not isinstance(self.server, ModalB200):
                return "probe_b200 needs the rented B200 (--tool-gpu B200); not available in this session"
            if self.probes >= self.a.probes:
                return "probe budget used up for this session"
            self.probes += 1
            self.note(f"\n### probe_b200 ({self.probes}): {args.get('note', '')}\n")
            res = self.server.call("probe", None, script=str(args.get("script") or ""))
            if res.get("error"):
                return f"probe failed to run: {res['error']}"
            return (f"probe on {res.get('gpu')} ({res.get('seconds')} s), exit code {res.get('returncode')}\n"
                    f"stdout:\n{res.get('stdout') or ''}\n" + (f"stderr:\n{res['stderr']}" if res.get("stderr") else ""))
        if fn not in ("compile_b200", "run_tests"):
            return f"unknown tool {fn}"
        if self.gpu_calls >= self.a.gpu_calls:
            return "GPU budget used up: give your final answer now."
        self.drafts += 1
        sol, problems = make_solution(args, f"{self.name}-d{self.drafts}")
        if problems:
            return "not sent to the GPU, fix these first:\n- " + "\n- ".join(problems)
        self.gpu_calls += 1
        self.note(f"\n### {fn} (draft {self.drafts}): {args.get('note', '')}\n")
        kind = "compile" if fn == "compile_b200" else "test"
        res = self.server.call(kind, sol, **({"repeat": 2} if kind == "test" else {}))
        if "B200" in str(res.get("gpu")):           # kept so the round's final B200 test can reuse it
            self.measured.setdefault(code_key(sol), {})[kind] = res
            self.measured_path.write_text(json.dumps(self.measured))
        if kind == "compile":
            return fmt_compile(res)
        return (fmt_test(res, self.server.reference(), self.best, self.anc, self.k, self.emu, args.get("paths"))
                + order_check_text(res))

    def turn(self, n, tool_choice=None):
        effort = self.a.effort if n == 0 else self.a.turn_effort
        for attempt in range(3):
            try:
                return llm.complete(with_tail_cache(self.messages), tools=TOOLS, tool_choice=tool_choice,
                                    model=self.a.model, max_tokens=self.a.max_tokens if n == 0 else 32000,
                                    effort=effort, tag=f"{self.name}/t{n}")
            except Exception as e:                       # network or provider error: back off and retry
                print(f"  {self.name} turn {n}: {e}; retrying", flush=True)
                time.sleep(30 * (attempt + 1))
        raise RuntimeError(f"{self.name}: three failed attempts at turn {n}")

    def run(self, first_turn_done=None):
        text, nudged, info = "", False, {}
        for n in range(self.a.turns):
            last_chance = n == self.a.turns - 1 or self.totals["cost_usd"] >= self.a.session_budget
            try:
                m, info = self.turn(n, tool_choice="none" if last_chance else None)
            finally:
                if first_turn_done:
                    first_turn_done.set()                 # release the other sessions even if this turn failed
            for k in self.totals:
                self.totals[k] += info.get(k) or 0
            self.messages.append(m)
            text = m.get("content") or ""
            calls = m.get("tool_calls") or []
            self.note(f"\n## Turn {n} ({info.get('completion_tokens')} tokens out, ${info.get('cost_usd')})\n\n{text}\n")
            if not calls:
                if reply.candidates(text) or nudged:
                    break
                nudged = True
                self.messages.append({"role": "user", "content": "No candidate in the OUTPUT CONTRACT format was found "
                                      "in your reply. Reply now with exactly one candidate in that format."})
                continue
            for c in calls:
                result = self.tool(c)
                self.note(f"**{c['function']['name']}** →\n```\n{result}\n```\n")
                self.messages.append({"role": "tool", "tool_call_id": c["id"], "content": result})
            left = self.a.turns - 1 - n
            if left <= 2 or self.gpu_calls >= self.a.gpu_calls:
                self.messages[-1]["content"] += (f"\n\n[session: {left} turn(s) and {self.a.gpu_calls - self.gpu_calls}"
                                                 " GPU call(s) left; give your final answer soon]")
            self.log_json.write_text("\n".join(json.dumps(x) for x in self.messages[1:]))
        self.log_json.write_text("\n".join(json.dumps(x) for x in self.messages[1:]))
        info = dict(info, **{k: round(v, 4) if isinstance(v, float) else v for k, v in self.totals.items()},
                    turns=n + 1, gpu_calls=self.gpu_calls)
        self.note(f"\n---\nsession total: {info['turns']} turns, {self.gpu_calls} GPU calls, ${info['cost_usd']:.2f}\n")
        return text, info


def order_check_failures(res):
    """Later-pass failures of a repeat=2 test (every workload set up before any re-run, as on the portal); timing
    flakes of the rented GPU excluded."""
    flake = ("Expected kernel activity sequence not found", "No timing results for methodology: cupti")
    return [r for r in res.get("later_passes") or [] if r["status"] != "PASSED"
            and not (r["status"] == "RUNTIME_ERROR" and any(f in r["log"] for f in flake))]


def order_check_text(res):
    bad = order_check_failures(res)
    if not bad:
        return ""
    return ("\n\nPORTAL-ORDER CHECK FAILED: the portal sets up every workload before it times them. Re-running each "
            "workload after all of them had run once in this process failed on:\n"
            + "\n".join(f"- {r['workload']}: {r['status']} {r['log'][-300:]}" for r in bad)
            + "\nHost state left by another shape breaks these (a function attribute such as max dynamic shared "
            "memory set per shape, a cached plan or handle). Make that state correct for every shape in any order; "
            "this candidate would fail on the portal.")


def run_sessions(round_id, jobs, static, a, slurm, sessions_dir):
    """jobs = [(i, task, dynamic prompt, name)] -> {i: (final text, info)}; one shared GPU tool server."""
    arc = archive.load()
    best, anc = planner.best_kernel(arc), planner.anchors()
    sessions_dir.mkdir(parents=True, exist_ok=True)
    if a.tool_gpu == "B200":
        server = ModalB200(round_id, best["solution"] if best else None, budget_minutes=a.b200_minutes)
    else:
        server = ToolServer(round_id, a.tool_gpu, slurm, best["solution"])
    try:
        # No LLM turns until a GPU is ours: a session that cannot test would just be the one-shot mode, at more cost.
        gpu = server.wait_alive(a.gpu_wait * 60)
        if not gpu:
            raise SystemExit(f"no GPU for the tool server within {a.gpu_wait} minutes; nothing was spent on the LLM. "
                             "Try again later, or another --tool-gpu.")
        print(f"tool server up on {gpu}", flush=True)
        static = static + "\n\n" + PROTOCOL.format(gpu=gpu, gpu_calls=a.gpu_calls, turns=a.turns)
        emu = None
        if isinstance(server, ModalB200):
            try:
                import emulator
                emu = emulator.Emulator(arc)
                print(f"emulator: {len(emu.kernels)} paired kernels, hyperparameters {emu.hp}", flush=True)
            except Exception as e:                    # fall back to the per-band correction
                print(f"emulator unavailable ({type(e).__name__}: {e}); using the per-band correction", flush=True)
        sess = {i: Session(name, t, static, dyn, server, best, anc, arc, a, sessions_dir, emu)
                for i, t, dyn, name in jobs}
        warm = threading.Event()
        with ThreadPoolExecutor(max_workers=len(sess)) as pool:
            first = min(sess)
            futs = {first: pool.submit(sess[first].run, warm)}
            warm.wait()                                   # the first turn writes the prompt cache for the rest
            futs.update({i: pool.submit(s.run) for i, s in sess.items() if i != first})
            results = {}
            for i, f in futs.items():
                try:
                    results[i] = f.result()
                except Exception as e:
                    print(f"  session {sess[i].name} failed: {e}", flush=True)
                    results[i] = ("", dict(sess[i].totals, finish="error", model=a.model))
        return results
    finally:
        server.stop()
