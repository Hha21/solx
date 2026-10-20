"""Run the kernel loop: plan a round, propose candidates with the inner LLM, test them on CSF3, collect results,
and shortlist the best for the B200 portal.

  python3 loop/round.py auto      --round r3 [--max-tasks 3] [--gpus A100,H200]   plan + propose + test
                                  [--interactive [--tool-gpu A100,L40S] [--turns 12] [--gpu-calls 8]]
  python3 loop/round.py plan      --round r3                                    show what the planner would do
  python3 loop/round.py propose   --round r3 --operation structural_mutation --parents g2-os-r16w8 \
                                  --band L --instructions "..." [--dry-run]      one hand-written task
  python3 loop/round.py test      --round r3 --gpus A100,H200
  python3 loop/round.py status    --round r3
  python3 loop/round.py collect   --round r3
  python3 loop/round.py shortlist --round r3                                    -> rounds/r3/portal/
  .venv/bin/python loop/round.py merge --round r5    per-workload merge of r5's kernels into the best (auto --portfolio
                                                     runs it after testing)
  python3 loop/round.py table
  .venv/bin/python loop/round.py ideas --round r1 --backend claude          portfolio round: idea sessions + critic
  .venv/bin/python loop/round.py lead  --round r1 --backend claude --portfolio   lead manages 2-3 lineages

With --interactive, each task is a design session (loop/designer.py): the LLM can compile its drafts for sm_100a,
test and time them on a CSF3 GPU against the current best, and query the score model before it answers.

Candidates live in loop/rounds/<round>/candidates/ (solution JSON + design card); lint failures go to rejected/.
Portal results arrive through poc/ingest_portal.py and are picked up by collect, shortlist and table.
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import archive  # noqa: E402
import planner  # noqa: E402
import problem  # noqa: E402
import prompts  # noqa: E402
import reply  # noqa: E402

ROOT = archive.ROOT
REMOTE = "csf3"
RSOLX = "/scratch/t95317ha/solx"
SLURM = {"A100": ("gpu-sk01", "gpuA"), "L40S": ("gpu-sk01", "gpuL"), "H200": ("gpu-cdt-dmcs", "gpuH_short")}
MODEL = "anthropic/claude-opus-5.5"


def rdir(r):
    return problem.current().rounds / r


def ssh(cmd):
    return subprocess.run(["ssh", "-o", "BatchMode=yes", REMOTE, cmd], capture_output=True, text=True, check=True).stdout


def get_archive():
    arc = archive.load()
    if not arc and problem.current().name == "038_flux_multi_head_rmsnorm_qk":
        arc = archive.seed()                       # #38's history (proof of concept, generation 2); no other problem has one
    archive.ingest_portal(arc)
    return arc


def slug(task, i):
    lin = f"lin{re.sub(r'[^a-z0-9]+', '', str(task['lineage']).lower())}-" if task.get("lineage") else ""
    return f"{i:02d}-{lin}{task['operation']}-{re.sub(r'[^a-z0-9]+', '', task.get('band', 'all').lower())}"


def call_llm(static, dyn, task, round_id, i, a):
    """One task -> (text, info); retries once at medium effort if the answer ran out of tokens while thinking."""
    import llm
    tag = f"{round_id}/{slug(task, i)}"
    text, info = llm.chat(static, dyn, model=a.model, max_tokens=a.max_tokens, effort=a.effort, tag=tag)
    if info["finish"] == "length" and not reply.candidates(text):
        print(f"  {tag}: ran out of tokens without an answer; retrying at medium effort", flush=True)
        text2, info2 = llm.chat(static, dyn, model=a.model, max_tokens=a.max_tokens, effort="medium", tag=tag + "/retry")
        info2["cost_usd"] = (info2.get("cost_usd") or 0) + (info.get("cost_usd") or 0)
        return text2, info2
    return text, info


CLAUDE_NOTE = """=== RUNNING AS A HEADLESS CLAUDE CODE SESSION ===
The design tools are MCP tools named mcp__solx__<tool> (compile_b200, run_tests, probe_b200, predict_score,
get_kernel, read_example). No other tools are available. When you are done, your final message must contain the
candidate in the OUTPUT CONTRACT format (fenced blocks), with its design card including paths and findings."""


RESUME_NOTE = """Your session was cut off (by a usage limit, or by the rented B200 dropping out: tool calls failing with
"function ... is stopped"); the tools work again, with a fresh GPU budget for this resumed part, so spend it on
finishing, not on starting over. Everything you did before is above; if you handed in a placeholder candidate because
the tools were down, this replaces it (give the new kernel a new name). Continue the
experiment from where you stopped, keep the remaining steps short, then reply with exactly one candidate in the OUTPUT
CONTRACT format, with the card's findings covering what you measured before and after the interruption."""


def run_claude(prompt, system_file, model, effort, mcp_args, sdir, name, timeout_s, resume=None, max_turns=None):
    """One headless `claude -p` run with our MCP tools only. Returns the parsed JSON result (or an error dict)."""
    cfg = sdir / f"{name}.mcp.json"
    cfg.write_text(json.dumps({"mcpServers": {"solx": {"command": str(ROOT / ".venv" / "bin" / "python"),
                                                       "args": [str(ROOT / "loop" / "mcp_tools.py"), *mcp_args]}}}))
    # A tool server that fails to start leaves Claude running with no tools (it then writes tool calls as text):
    # check it starts before spending anything on the session.
    chk = subprocess.run([str(ROOT / ".venv" / "bin" / "python"), str(ROOT / "loop" / "mcp_tools.py"), *mcp_args,
                          "--check"], capture_output=True, text=True, timeout=300)
    if chk.returncode:
        raise RuntimeError(f"the MCP tool server does not start:\n{chk.stderr[-2500:]}")
    cmd = ["claude", "-p", "--model", model, "--effort", effort, "--mcp-config", str(cfg), "--strict-mcp-config",
           "--tools", "", "--allowedTools", "mcp__solx", "--output-format", "stream-json", "--verbose"]
    if system_file:
        cmd += ["--append-system-prompt-file", str(system_file)]
    if max_turns:
        cmd += ["--max-turns", str(max_turns)]
    if resume:
        cmd += ["--resume", resume]
    # stream-json: every event (model text, tool calls, tool results) is appended to <name>.stream.jsonl as it
    # arrives, so a session can be watched live (loop/dashboard); the final "result" event carries the same fields
    # as --output-format json.
    t0 = time.time()
    out, timed_out, sid = None, threading.Event(), None
    with open(sdir / f"{name}.stream.jsonl", "a") as log, open(sdir / f"{name}.stderr.txt", "a") as err:
        p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=err, text=True, cwd=sdir)
        timer = threading.Timer(timeout_s, lambda: (timed_out.set(), p.kill()))
        timer.start()
        try:
            p.stdin.write(prompt)
            p.stdin.close()
            for line in p.stdout:
                log.write(line)
                log.flush()
                if line.startswith("{"):
                    try:
                        ev = json.loads(line)
                    except ValueError:
                        continue
                    sid = ev.get("session_id") or sid       # lets a session killed at its time limit be resumed
                    if ev.get("type") == "result":
                        out = ev
            p.wait()
        finally:
            timer.cancel()
    if out is None:
        tail = (sdir / f"{name}.stderr.txt").read_text()[-3000:]
        out = dict(is_error=True, result="", subtype="timeout" if timed_out.is_set() else "error", stderr=tail,
                   session_id=sid)
    out["seconds"] = round(time.time() - t0, 1)
    return out


def claude_sessions(round_id, jobs, static, a, sdir):
    """Design sessions as headless Claude Code runs on the user's plan, in parallel; each has its own MCP tool server
    (and its own Modal B200 container, opened on its first GPU call)."""
    import designer
    sdir.mkdir(parents=True, exist_ok=True)
    system_file = sdir / "briefing.txt"
    protocol = designer.PROTOCOL.format(gpu="a rented NVIDIA B200 (Modal; unlocked clocks)", gpu_calls=a.gpu_calls,
                                        turns="about 15")
    system_file.write_text(static + "\n\n" + protocol + "\n\n" + CLAUDE_NOTE)
    if any(t.get("fresh") for _, t, _, _ in jobs):        # fresh-eyes sessions: no history, no get_kernel
        fresh_file = sdir / "briefing_fresh.txt"
        fresh_protocol = re.sub(r"- get_kernel returns.*?\n(?=- )", "- read_example shows annotated excerpts of public "
                                "B200 norm kernels. This session has no get_kernel: earlier designs stay hidden.\n",
                                protocol, flags=re.S)
        fresh_file.write_text(prompts.static_prompt(fresh=True) + "\n\n" + fresh_protocol + "\n\n"
                              + CLAUDE_NOTE.replace("get_kernel, ", ""))
    minutes = max(3.0, a.b200_minutes / max(1, len(jobs)))

    def one(job):
        i, t, dyn, name = job
        mcp_args = ["--name", name, "--round", round_id, "--dir", str(sdir), "--gpu-calls", str(a.gpu_calls),
                    "--probes", str(a.probes), "--b200-minutes", str(minutes)] + (["--fresh"] if t.get("fresh") else [])
        if t.get("lineage"):                              # get_kernel then reads only this lineage's kernels
            mcp_args += ["--lineage", str(t["lineage"])]
        model = t.get("model") or a.session_model
        brief = sdir / "briefing_fresh.txt" if t.get("fresh") else system_file
        prev = sdir / f"{name}.claude.json"
        if getattr(a, "resume", False) and prev.exists() and json.loads(prev.read_text()).get("session_id"):
            out = run_claude(RESUME_NOTE, brief, model, a.effort, mcp_args, sdir, name, a.session_timeout * 60,
                             resume=json.loads(prev.read_text())["session_id"], max_turns=a.max_turns)
        else:
            out = run_claude(dyn, brief, model, a.effort, mcp_args, sdir, name, a.session_timeout * 60,
                             max_turns=a.max_turns)
        text = out.get("result") or ""
        if not reply.candidates(text) and out.get("session_id"):          # one nudge for the contract
            more = run_claude("Reply now with exactly one candidate in the OUTPUT CONTRACT format.", None,
                              model, a.effort, mcp_args, sdir, name, 900, resume=out["session_id"])
            text = more.get("result") or text
            out["total_cost_usd"] = (out.get("total_cost_usd") or 0) + (more.get("total_cost_usd") or 0)
        (sdir / f"{name}.claude.json").write_text(json.dumps({k: v for k, v in out.items() if k != "result"}, indent=1))
        return i, (text, log_usage(out, model, f"{round_id}/{name}"))

    with ThreadPoolExecutor(max_workers=max(1, min(a.parallel, len(jobs)))) as pool:    # each session = claude + MCP
        return dict(pool.map(one, jobs))


def log_usage(out, model, tag):
    """A claude run's tokens and API-equivalent cost -> loop/runs/usage.jsonl; returns the info dict."""
    u = out.get("usage") or {}
    info = dict(model=f"claude:{model}", finish=out.get("subtype") or ("error" if out.get("is_error") else "stop"),
                seconds=out.get("seconds"), turns=out.get("num_turns"), gpu_calls="-",
                prompt_tokens=(u.get("input_tokens") or 0) + (u.get("cache_read_input_tokens") or 0)
                + (u.get("cache_creation_input_tokens") or 0), cached_tokens=u.get("cache_read_input_tokens"),
                completion_tokens=u.get("output_tokens"), reasoning_tokens=None,
                cost_usd=round(out.get("total_cost_usd") or 0, 4))
    with open(ROOT / "loop" / "runs" / "usage.jsonl", "a") as f:
        f.write(json.dumps(dict(info, time=time.strftime("%Y-%m-%dT%H:%M:%S"), tag=tag,
                                billing="claude-plan (cost is the API-equivalent estimate)")) + "\n")
    return info



LEAD_PROTOCOL = """=== RESEARCH LEAD PROTOCOL ===
You are the research lead of this project. You do not write kernels. Each round you:
1. Read the evidence below: the lab notebook (portal vs bench outcomes and findings recorded by design sessions),
   the archive, the hypothesis ledger, and the emulator's learned portal-vs-bench effects. Use mcp__solx__get_kernel
   to read any kernel's code and results, mcp__solx__read_example for public SOTA kernels, and
   mcp__solx__predict_score to see what a change per size band is worth.
   Start from the LEADER GAP section: it backs out what the public top scores imply per workload and the best score
   reachable at the hardware ceilings for each precision. If the leader is above what our current approach can reach
   (for example a TF32 ceiling below the leader's score), the gap is a different precision, algorithm, traffic
   pattern or rule reading, not tuning; name what must differ before planning refinements of the current best.
   tricks.md lists verified cross-problem lessons; check that the current best already uses the ones that apply.
2. Update the ledger: change a status only on evidence; add hypotheses that findings suggest; cite evidence (kernel
   ids, portal numbers, probe findings); keep each entry short. Keep the constraints.
3. Design the next round: {n} experiments for design sessions. Each session is a Claude agent with a rented B200
   (probes take seconds, full tests 10-20 s; run_tests predicts the portal score with the emulator, whose error bars
   are only realistic once a problem has ~5 portal results, and it misranks effects under ~3%;
   bench noise is about 1% per size, so small effects need repeated, interleaved probes). Each experiment tests one
   hypothesis (or a new idea you add to the ledger) and states what to build or measure, the success criterion, and
   what result would refute it. Prefer experiments that could move the portal score by 0.005 or more, or that
   settle an open hypothesis blocking a large gain. Avoid tweaks worth about 1%. Exploration is encouraged.
   Diversify across strategy lenses: library, memory, compute, fusion, precision, reduction, scheduling,
   custom-sm100. Plateau rule: when the last two portal results each moved the best score by less than 0.005 (or
   the leader gap shows the current approach's ceiling is below the leader), at least one experiment must be a
   fundamentally different option, not a refinement of the current best: another lens or precision, a library
   path versus a custom kernel, another data layout or algorithm. Say which lens each experiment uses.
4. Give each experiment a model: opus for open-ended design and debugging, sonnet for well-specified work.
Output, in this order:
### Assessment
(at most 250 words: where the remaining score is, what we now know, what this round tests and why)
```yaml ledger
(the complete updated ledger, same schema)
```
```json tasks
[{{"id": "E1", "hypothesis": "H8", "title": "...", "operation": "new_design", "parents": ["<archive id>"],
  "band": "S|M|L|all", "instructions": "what to build or measure, step by step", "success": "...",
  "refuted_if": "...", "model": "opus|sonnet"}}]
```"""


def leader_gap_text():
    """The leader-gap analysis for the current problem: implied leader times vs the hardware ceilings per precision."""
    try:
        sys.path.insert(0, str(ROOT / "poc"))
        import leader_gap
        import sol_targets
        live = sol_targets.LIVE_FIRST.get(problem.current().name)      # the live board's #1 when we have it
        return leader_gap.report([live] if live else None)
    except Exception as e:
        return f"(unavailable: {e})"


def evidence_sections(arc, effects=True):
    """What the lead and the idea sessions read: problem, ledger, notebook, portal results, leader gap, archive."""
    import emulator
    measured = [r for r in arc.values() if r.get("b200")]
    calib = "\n".join(f"- {r['id']}: portal {r['b200']['score']:.4f}, S/M/L {archive.fmt(archive.band_geomeans(r['b200'].get('timings', {})))} µs"
                      for r in sorted(measured, key=lambda r: -r["b200"]["score"]))
    out = [f"=== PROBLEM ===\n{problem.current().summary()}",
           f"=== HYPOTHESIS LEDGER ({problem.current().ledger.relative_to(ROOT)}) ===\n"
           f"{problem.current().ledger.read_text() if problem.current().ledger.exists() else '(none yet)'}",
           f"=== LAB NOTEBOOK ===\n{prompts.lab_notebook(arc, limit=260)}",
           f"=== PORTAL RESULTS (best first) ===\n{calib or '(none yet)'}",
           f"=== LEADER GAP (what the public top scores imply; poc/leader_gap.py) ===\n{leader_gap_text()}"]
    try:
        import sol_targets
        out.append(f"=== SOL TARGETS (loop/sol_targets.py) ===\n{sol_targets.targets_text()}")
    except Exception as e:                      # a failure here must never break a round
        out.append(f"=== SOL TARGETS (loop/sol_targets.py) ===\n(unavailable: {e})")
    if effects:
        try:
            eff = emulator.effects_markdown(emulator.Emulator(arc))
        except Exception as e:
            eff = f"(unavailable: {e})"
        out.append(f"=== EMULATOR: portal minus bench, by design feature ===\n{eff}")
    out.append(f"=== ARCHIVE ===\n{archive.table(arc, limit=60)}")
    return out


def cmd_ideas(a):
    """Ideas tier of a portfolio round: independent idea sessions and a premise audit (no GPU), run in parallel; the
    merged, shuffled pool goes to rounds/<round>/ideas/pool.json for `round.py lead --portfolio`."""
    import portfolio
    arc = get_archive()
    d = rdir(a.round) / "ideas"
    d.mkdir(parents=True, exist_ok=True)
    evidence = evidence_sections(arc, effects=False) + [f"=== LINEAGES ===\n{portfolio.lineages_text()}"]
    static = prompts.static_prompt()
    jobs = [(f"ideas{i}-{m}", m, portfolio.IDEAS_PROTOCOL,
             f"Round {a.round}: write 10 design ideas and 3 leader stories for this problem.")
            for i, m in enumerate(x for x in a.idea_models.split(",") if x)]
    if a.critic_model:
        jobs.append(("critic", a.critic_model, portfolio.CRITIC_PROTOCOL,
                     f"Round {a.round}: audit the premises this project relies on."))

    def one(job):
        name, model, protocol, task = job
        (d / f"{name}.briefing.txt").write_text(static + "\n\n" + protocol)
        prompt = "\n\n".join([f"=== TASK ===\n{task}"] + evidence)
        (d / f"{name}.prompt.txt").write_text(prompt)
        out = run_claude(prompt, d / f"{name}.briefing.txt", model, a.effort,
                         ["--name", name, "--round", a.round, "--dir", str(d), "--no-gpu"], d, name,
                         a.session_timeout * 60, max_turns=20)
        text = out.get("result") or ""
        (d / f"{name}.reply.md").write_text(text)
        (d / f"{name}.claude.json").write_text(json.dumps({k: v for k, v in out.items() if k != "result"}, indent=1))
        info = log_usage(out, model, f"{a.round}/ideas/{name}")
        print(f"{name}: {info['turns']} turns, {info['seconds']} s, API-equivalent ${info['cost_usd']:.2f}", flush=True)
        return name, text

    with ThreadPoolExecutor(max_workers=max(1, min(a.parallel, len(jobs)))) as pool:
        replies = dict(pool.map(one, jobs))
    p = portfolio.pool(replies, seed=hash(a.round) & 0xffff)
    (d / "pool.json").write_text(json.dumps(p, indent=1))
    fams = sorted({str(i.get("family", "-")) for i in p["ideas"]})
    print(f"pool: {len(p['ideas'])} ideas in {len(fams)} families, {len(p['leader_stories'])} leader stories, "
          f"{len(p['premises'])} doubtful premises -> {(d / 'pool.json').relative_to(ROOT)}")
    print(f"next: .venv/bin/python loop/round.py lead --problem {problem.current().level}/{problem.current().name} "
          f"--round {a.round} --backend claude --portfolio")


def cmd_lead(a):
    """Run the research lead: update the hypothesis ledger and write the round's experiments to plan.json. With
    --portfolio the lead also reads the round's ideas pool and premise audit and manages 2-3 lineages."""
    import portfolio
    arc = get_archive()
    d = rdir(a.round)
    sdir = d / "lead"
    sdir.mkdir(parents=True, exist_ok=True)
    system_file = sdir / "briefing.txt"
    protocol = portfolio.LEAD_PROTOCOL if a.portfolio else LEAD_PROTOCOL
    system_file.write_text(prompts.static_prompt() + "\n\n" + protocol.format(n=a.max_tasks))
    extra = []
    if a.portfolio:
        pool_file = d / "ideas" / "pool.json"
        if not pool_file.exists():
            raise SystemExit(f"no ideas pool for {a.round}: run round.py ideas --round {a.round} first")
        extra = [*portfolio.pool_text(json.loads(pool_file.read_text())),
                 f"=== LINEAGES ({portfolio.lineages_file().relative_to(ROOT)}) ===\n{portfolio.lineages_text()}"]
    prompt = "\n\n".join([
        f"=== TASK ===\nRound {a.round}: update the hypothesis ledger"
        + (", manage the lineages" if a.portfolio else "") + f" and design {a.max_tasks} experiments."
        + (f"\n\nBRIEF FROM THE OPERATOR FOR THIS ROUND:\n{a.brief}" if getattr(a, "brief", None) else ""),
        *extra, *evidence_sections(arc)])
    (sdir / "prompt.txt").write_text(prompt)
    if getattr(a, "from_reply", False):             # re-parse a saved reply (e.g. after a ledger YAML error)
        text, out = (sdir / "reply.md").read_text(), json.loads((sdir / "lead.claude.json").read_text())
    else:
        out = run_claude(prompt, system_file, a.lead_model, a.effort,
                         ["--name", "lead", "--round", a.round, "--dir", str(sdir), "--no-gpu"], sdir, "lead",
                         a.session_timeout * 60, max_turns=25)
        text = out.get("result") or ""
        (sdir / "reply.md").write_text(text)
        (sdir / "lead.claude.json").write_text(json.dumps({k: v for k, v in out.items() if k != "result"}, indent=1))
    blocks = reply.named_blocks(text) or {m.group(1).strip(): m.group(2) for m in reply.FENCE.finditer(text)}
    ledger = next((v for k, v in blocks.items() if k.startswith("yaml") and "ledger" in k), None)
    tasks_js = next((v for k, v in blocks.items() if k.startswith("json") and "tasks" in k), None)
    if not (ledger and tasks_js):
        raise SystemExit(f"lead reply had no ledger or tasks block; see {sdir / 'reply.md'}")
    try:                                                    # must parse before it replaces the ledger
        problem.current().ledger.write_text(load_ledger_text(ledger))
    except Exception as e:                                  # keep the old ledger and the plan; save the raw text
        (sdir / "ledger.raw.yaml").write_text(ledger)
        print(f"lead ledger is not valid YAML ({str(e)[:200]}); kept the old ledger, raw text in {sdir / 'ledger.raw.yaml'}")
    lins = None
    if a.portfolio:
        lin_text = next((v for k, v in blocks.items() if k.startswith("yaml") and "lineages" in k), None)
        try:
            lins = portfolio.save_lineages(load_ledger_text(lin_text)) if lin_text else None
        except Exception as e:
            (sdir / "lineages.raw.yaml").write_text(lin_text or "")
            print(f"lead lineages block is not valid ({str(e)[:200]}); kept the old lineages, raw text in "
                  f"{sdir / 'lineages.raw.yaml'}")
        lins = lins if lins is not None else portfolio.load_lineages()
    tasks = []
    for e in json.loads(tasks_js)[:a.max_tasks]:
        best0 = planner.best_kernel(arc)
        lin = e.get("lineage") if a.portfolio else None
        parents = [p for p in (e.get("parents") or []) if p in arc]
        if lin:                                             # a lineage builds on its own kernels, or starts fresh
            parents = [p for p in parents if portfolio.visible(arc[p], lin)]
        elif not parents:
            parents = [best0["id"]] if best0 else []
        tasks.append(dict(operation=e.get("operation") or "new_design", parents=parents, band=e.get("band") or "all",
                          niche=f"{'lin:' + str(lin) + ':' if lin else 'exp:'}{e.get('id', '?')}:"
                                f"{e.get('hypothesis', '-')}", model=e.get("model"), lineage=lin,
                          instructions=(f"Experiment {e.get('id')} (tests {e.get('hypothesis')}): {e.get('title', '')}. "
                                        f"{e.get('instructions', '')} Success: {e.get('success', '')} Refuted if: "
                                        f"{e.get('refuted_if', '')}" + planner.explore_common() +
                                        " Set the card's tests: field to the hypothesis id(s).")))
    best = planner.best_kernel(arc) or {"id": None, "b200": {"score": None}}
    (d / "plan.json").write_text(json.dumps(dict(best=best["id"], best_score=best["b200"]["score"], tasks=tasks,
                                                 lead=dict(model=a.lead_model, cost_usd=out.get("total_cost_usd"))),
                                            indent=1))
    print(text.split("```")[0].strip()[:3000])
    for i, t in enumerate(tasks):
        print(f"  task {i}: {t['niche']} parents={t['parents']} band={t['band']} model={t.get('model')}"
              + (f" lineage={t['lineage']}" if t.get("lineage") else ""))
    u = out.get("usage") or {}
    print(f"lead: {out.get('num_turns')} turns, {out.get('seconds')} s, API-equivalent ${out.get('total_cost_usd') or 0:.2f}")
    return dict(tasks=tasks)


def register(arc, round_id, text, t):
    """Parse a reply's candidates, lint them, and file them under the round (candidates/ or rejected/)."""
    d = rdir(round_id)
    for sub in ("candidates", "rejected"):
        (d / sub).mkdir(parents=True, exist_ok=True)
    cands = reply.candidates(text)
    if not cands:
        print("  no parseable candidate in the reply")
    for c in cands:
        cid, sol, card, problems = reply.build(c, round_id, set(arc))
        if sol is None:
            print(f"  rejected: {problems}")
            continue
        hits = reply.lint(sol)
        ok = not (problems or hits)
        where = d / ("candidates" if ok else "rejected")
        (where / f"{cid}.json").write_text(json.dumps(sol, indent=1))
        (where / f"{cid}.card.yaml").write_text(card)
        pc = archive.parse_card(card)
        arc[cid] = dict(id=cid, round=round_id, parents=pc.get("parents") or t.get("parents") or [],
                        operation=pc.get("operation") or t.get("operation"), language=pc.get("language", "triton"),
                        niche=pc.get("niche") or {}, card=card, task=t,
                        solution=str((where / f"{cid}.json").relative_to(ROOT)),
                        status="proposed" if ok else "rejected: " + "; ".join(problems + hits), timings={})
        print(f"  {cid}: {arc[cid]['status']}")


def load_ledger_text(text):
    """Parse an agent's ledger, repairing the usual mistake (an unquoted value containing ': ') by quoting it."""
    import yaml
    try:
        yaml.safe_load(text)
        return text
    except yaml.YAMLError:
        pass
    # Fold continuation lines of plain multi-line values into their parent line first ("statement: ... with a\n
    #   Triton kernel: int32 ..." is invalid YAML), leaving block scalars (| and >) alone.
    folded = []
    key_or_item = re.compile(r"^\s*(- |[\w-]+:(\s|$)|#)")
    for line in text.splitlines():
        prev = folded[-1] if folded else ""
        indent = lambda l: len(l) - len(l.lstrip())
        if (folded and line.strip() and not key_or_item.match(line) and indent(line) > indent(prev)
                and not re.search(r":\s*[|>][-+]?\s*$", prev) and not prev.rstrip().endswith(":")):
            folded[-1] = prev.rstrip() + " " + line.strip()
        else:
            folded.append(line)
    fixed = []
    for line in folded:
        m = re.match(r"^(\s*(?:- )?(?:[\w-]+: |- ))(.*)$", line)
        if m and ": " in m.group(2) and not m.group(2).lstrip().startswith(("'", '"', "[", "{", "|", ">")):
            line = m.group(1) + "'" + m.group(2).replace("'", "''") + "'"
        fixed.append(line)
    fixed = "\n".join(fixed)
    yaml.safe_load(fixed)                          # still invalid: let the caller see the error
    return fixed


def cmd_research(a):
    """Research phase for a problem: B200 measurements, then a research agent writes the problem card, the first
    hypothesis ledger and a simple, correct first kernel (round r0)."""
    import research
    p = problem.current()
    p.dir.mkdir(parents=True, exist_ok=True)
    rows_file = p.b200 / "reference_times.json"
    d = rdir("r0") / "research"
    if a.from_reply:                                # finish from a saved reply without re-running the agent
        text, out = (d / "reply.md").read_text(), json.loads((d / "research.claude.json").read_text())
        return finish_research(p, d, text, out)
    if a.remeasure or not rows_file.exists():
        print(f"measuring {p.level}/{p.name} on the rented B200 (copy floor and PyTorch reference per workload)...",
              flush=True)
        rows = research.measure()
    else:
        rows = json.loads(rows_file.read_text())
    table = research.measurements_table(rows)
    print(table, flush=True)
    d = rdir("r0") / "research"
    d.mkdir(parents=True, exist_ok=True)
    example = problem.get("L1/030_attention_output_projection_with_residual" if p.kind == "compute"
                          else "L1/038_flux_multi_head_rmsnorm_qk")      # a finished card of the same kind
    example_card = example.card.read_text() if example.card.exists() else ""
    example_ledger = example.ledger.read_text() if example.ledger.exists() else ""
    system_file = d / "briefing.txt"
    system_file.write_text(prompts.static_prompt() + "\n\n" + research.RESEARCH_PROTOCOL + "\n\n" + CLAUDE_NOTE)
    prompt = "\n\n".join([
        f"=== TASK ===\nResearch phase for {p.level}/{p.name}.",
        f"=== PROBLEM ===\n{p.summary()}",
        f"=== definition.json ===\n{(p.data / 'definition.json').read_text()}",
        f"=== reference.py ===\n{(p.data / 'reference.py').read_text()}",
        f"=== workload.jsonl (first 3 lines) ===\n" + "\n".join((p.data / "workload.jsonl").read_text().splitlines()[:3]),
        f"=== MEASUREMENTS ON THE RENTED B200 (unlocked clocks; the portal locks SM 1500 MHz) ===\n{table}",
        f"=== EXAMPLE: the finished problem card of another problem (format and depth to aim for) ===\n"
        f"{example_card[:12000]}",
        f"=== EXAMPLE: that problem's ledger (copy its constraints; write your own hypotheses) ===\n{example_ledger[:6000]}",
    ])
    (d / "prompt.txt").write_text(prompt)
    out = run_claude(prompt, system_file, a.lead_model, a.effort,
                     ["--name", "research", "--round", "r0", "--dir", str(d), "--gpu-calls", str(a.gpu_calls),
                      "--probes", str(a.probes), "--b200-minutes", str(a.b200_minutes)], d, "research",
                     a.session_timeout * 60, max_turns=a.max_turns)
    text = out.get("result") or ""
    if not (reply.candidates(text) and "card" in text) and out.get("session_id"):   # e.g. killed at its time limit
        more = run_claude("Time is up. Do not call any more tools. Write your final answer now in the output format "
                          "of the RESEARCH PHASE PROTOCOL (card, ledger, first kernel). Hand in exactly the source of "
                          "the last version that passed run_tests everywhere; put later probe results in the card.",
                          None, a.lead_model, a.effort, ["--name", "research", "--round", "r0", "--dir", str(d),
                          "--no-gpu"], d, "research", 1800, resume=out["session_id"], max_turns=4)
        text = more.get("result") or text
        out["total_cost_usd"] = (out.get("total_cost_usd") or 0) + (more.get("total_cost_usd") or 0)
    (d / "reply.md").write_text(text)
    (d / "research.claude.json").write_text(json.dumps({k: v for k, v in out.items() if k != "result"}, indent=1))
    finish_research(p, d, text, out)


def finish_research(p, d, text, out):
    blocks = reply.named_blocks(text)          # cards often contain ```python examples: FENCE would cut them short
    card = next((v for k, v in blocks.items() if k.startswith("markdown") and "card" in k), None)
    ledger = next((v for k, v in blocks.items() if k.startswith("yaml") and "ledger" in k), None)
    if card:
        p.card.write_text(card)
    if ledger:
        try:
            p.ledger.write_text(load_ledger_text(ledger))
        except Exception as e:                      # keep the raw text for a human; do not lose the kernel
            (p.dir / "ledger.raw.yaml").write_text(ledger)
            print(f"ledger is not valid YAML ({str(e)[:200]}); saved as {p.dir / 'ledger.raw.yaml'}")
    arc = get_archive()
    if not any((rdir("r0") / "candidates").glob("*.json")):    # --from-reply after a run: keep the registered kernel
        register(arc, "r0", text, dict(operation="new_design", parents=[], band="all", niche="research:first-kernel"))
        archive.save(arc)
    print(f"research: {out.get('num_turns')} turns, {out.get('seconds')} s, API-equivalent "
          f"${out.get('total_cost_usd') or 0:.2f}; card {'written' if card else 'MISSING'}, ledger "
          f"{'written' if ledger else 'MISSING'}; see {d.relative_to(ROOT)}/reply.md")
    if any((rdir("r0") / "candidates").glob("*.json")):
        b200_test("r0")
        first = ", ".join(str(c.relative_to(ROOT)) for c in sorted((rdir("r0") / "candidates").glob("*.json")))
        print(f"next: submit {first} to the portal (its result gives "
              f"the per-workload anchors), ingest the page, then: round.py lead --problem {p.level}/{p.name} --round r1")


def propose_tasks(round_id, tasks, a):
    arc = get_archive()
    d = rdir(round_id)
    for sub in ("candidates", "rejected", "replies"):
        (d / sub).mkdir(parents=True, exist_ok=True)
    static = prompts.static_prompt()
    stamp = time.strftime("%Y%m%d-%H%M%S")
    jobs = []
    for i, t in enumerate(tasks):
        if getattr(a, "resume", False):               # same session name and prompt as the interrupted run
            old = sorted((d / "replies").glob(f"*_{slug(t, i)}.prompt.txt"))
            if old:
                name = old[-1].name[:-len(".prompt.txt")]
                reply_md = d / "replies" / f"{name}.reply.md"
                if reply_md.exists() and reply.candidates(reply_md.read_text()):
                    print(f"{name}: already gave a candidate; not resumed")
                    continue
                jobs.append((i, t, old[-1].read_text(), name))
                continue
        if t.get("fresh"):
            dyn = prompts.fresh_prompt(arc, round_id, extra=t.get("instructions", ""))
        else:
            extra = t.get("instructions", "")
            if t.get("lineage"):
                import portfolio
                extra += portfolio.session_brief(t["lineage"])
            dyn = prompts.dynamic_prompt(arc, round_id, t["operation"], t["parents"], n=1, band=t.get("band", "all"),
                                         niche=t.get("niche", "any empty niche"), extra=extra,
                                         feedback=t.get("feedback", "none"))
        name = f"{stamp}_{slug(t, i)}"
        (d / "replies" / f"{name}.prompt.txt").write_text(dyn)
        jobs.append((i, t, dyn, name))
    if a.dry_run:
        for i, t, dyn, name in jobs:
            print(f"dry run {name}: static ~{len(static) // 4:,} tokens (cached), dynamic ~{len(dyn) // 4:,} tokens")
        return
    if a.backend == "claude":
        results = claude_sessions(round_id, jobs, static, a, d / "sessions")
    elif a.interactive:
        import designer
        if a.tool_gpu != "B200":
            sync_remote()
        results = designer.run_sessions(round_id, jobs, static, a, SLURM, d / "sessions")
    else:
        results = {}
        first, rest = jobs[0], jobs[1:]
        results[first[0]] = call_llm(static, first[2], first[1], round_id, first[0], a)   # warms the prompt cache
        with ThreadPoolExecutor(max_workers=4) as pool:
            futs = {j[0]: pool.submit(call_llm, static, j[2], j[1], round_id, j[0], a) for j in rest}
            for i, f in futs.items():
                results[i] = f.result()
    total = 0.0
    for i, t, dyn, name in jobs:
        text, info = results[i]
        (d / "replies" / f"{name}.reply.md").write_text(text)
        total += info.get("cost_usd") or 0
        extra = f", {info['turns']} turns, {info['gpu_calls']} GPU calls" if "turns" in info else ""
        print(f"{name}: {info['prompt_tokens']} in ({info['cached_tokens']} cached), {info['completion_tokens']} out "
              f"({info['reasoning_tokens']} reasoning), ${info['cost_usd']}, {info['seconds']} s, "
              f"finish={info.get('finish')}{extra}")
        register(arc, round_id, text, t)
    archive.save(arc)
    print(f"round {round_id}: {len(jobs)} calls, ${total:.2f}")


def cmd_propose(a):
    task = dict(operation=a.operation, parents=a.parents, band=a.band, niche=a.niche, instructions=a.instructions)
    propose_tasks(a.round, [task], a)


def make_plan(a):
    arc = get_archive()
    p = planner.plan(arc, max_tasks=a.max_tasks, last_round=a.last_round, mode=a.mode)
    d = rdir(a.round)
    d.mkdir(parents=True, exist_ok=True)
    (d / "plan.json").write_text(json.dumps(p, indent=1, default=str))
    g = p["gains"]
    print(f"best on B200: {p['best']} (score {p['best_score']:.4f}); fit {g['fixed']:.1f} us fixed + {g['bw']:.1f} TB/s; "
          f"M {g['tbs_M']:.2f} TB/s, L {g['tbs_L']:.2f} TB/s")
    print(f"estimated score gain: small-input fixed cost -> {planner.FIXED_TARGET_US} us: {g['S']:+.3f}; "
          f"medium/large a further 5%: {g['ML']:+.3f}")
    for i, t in enumerate(p["tasks"]):
        print(f"  task {i}: {t['operation']} parents={t['parents']} band={t.get('band')} niche={t.get('niche', '-')}")
    return p


def cmd_plan(a):
    make_plan(a)


def cmd_auto(a):
    if a.use_plan:                                 # run a plan already written (e.g. by `round.py lead`)
        p = json.loads((rdir(a.round) / "plan.json").read_text())
        print(f"using the saved plan: {len(p['tasks'])} task(s)")
    else:
        p = cmd_lead(a) if a.lead else make_plan(a)
    propose_tasks(a.round, p["tasks"], a)
    if not a.dry_run and any((rdir(a.round) / "candidates").glob("*.json")):
        a.gpus = a.gpus or ("B200" if a.tool_gpu == "B200" or a.backend == "claude" else "A100,H200")
        cmd_test(a)
        csf = [g for g in a.gpus.split(",") if g != "B200"]
        if a.portfolio and not csf and not a.no_merge:
            cmd_merge(a)
        print(f"next: python3 loop/round.py collect --round {a.round} (after the CSF3 jobs finish); then shortlist"
              if csf else f"next: python3 loop/round.py shortlist --problem {problem.current().level}/"
                         f"{problem.current().name} --round {a.round}")


def cmd_merge(a):
    """Per-workload merge: time the round's tested kernels and the best portal kernel side by side in one container
    (loop/ab_container.py), route each workload to the kernel that beat the best by more than --margin there, build the
    merged kernel (loop/merge.py), test it, and register it as <round>-merge with its projected portal score."""
    import math
    import yaml
    import ab_container
    import b200_modal
    import merge
    import portfolio
    arc = get_archive()
    best = planner.best_kernel(arc)
    if not best:
        raise SystemExit("merge: no portal-measured kernel yet")
    portal = ab_container.portal_times(best["b200"]["submission"])
    cid, d = f"{a.round}-merge", rdir(a.round)
    cands = [r for r in arc.values() if r.get("round") == a.round and r["id"] != cid and not r.get("b200")
             and (r.get("timings") or {}).get("B200r") and (r.get("task") or {}).get("niche") != "merge"]
    if not cands:
        print(f"merge: no kernel of {a.round} passed any workload on the rented B200; nothing to merge")
        return
    sols = {r["id"]: json.loads((ROOT / r["solution"]).read_text()) for r in [best] + cands}
    b200_modal.load_token()
    print(f"merge: timing {best['id']} (portal #{best['b200']['submission']}) against "
          f"{', '.join(r['id'] for r in cands)}, {a.merge_reps} reps in one container", flush=True)
    with b200_modal.app.run():
        samples, fails = ab_container.interleave(sols, a.merge_reps, tag=cid)
        bad = {(f[0], f[2]) for f in fails}
        an, med = samples[best["id"]], ab_container.med
        # The portal/rented ratio is a property of the kernel's launch structure, not only of the workload: on #66310
        # the 5-6 launch pipeline (lineage A) ran 1.40-1.55x its rented time on S/M rows where the single-launch
        # cooperative kernels (B, D) ran 1.25-1.31x. So a row's candidate is projected with the ratio its own lineage
        # shows on the anchor row closest in rented time, and an unseen lineage needs --margin-new.
        flat = {c: merge.flatten(sols[c], c) for c in sols}
        fam = lambda leaf: portfolio.lineage_of(arc.get(leaf)) or leaf
        afam = {k: fam(flat[best["id"]][1](k)) for k in portal}
        rat = {k: portal[k] / med(an[k]) for k in portal if an.get(k)}

        def est(c, k):
            t, f = med(samples[c][k]), fam(flat[c][1](k))
            if afam[k] == f:
                return t * rat[k], True
            same = [j for j in rat if afam[j] == f]
            if same:
                j = min(same, key=lambda j: abs(math.log(med(an[j]) / t)))
                return t * rat[j], True
            return t * rat[k], False

        pick, rows = {}, [f"| workload | {best['id']} portal us (rented) | " + " | ".join(r["id"] for r in cands)
                          + " | routed to |", "|---" * (len(cands) + 3) + "|"]
        for k in sorted(portal, key=portal.get):
            opts, cells = [], []
            for c in samples:
                if c == best["id"]:
                    continue
                if not (an.get(k) and samples[c].get(k)) or (c, k) in bad or (c, "*") in bad:
                    cells.append("fail" if (c, k) in bad or (c, "*") in bad else "-")
                    continue
                e, known = est(c, k)
                cells.append(f"{e:.1f} ({100 * (med(samples[c][k]) / med(an[k]) - 1):+.1f}%)" + ("" if known else " new"))
                if e < portal[k] * (1 - (a.margin if known else a.margin_new)):
                    opts.append((e, c))
            if opts:
                e, c = min(opts)
                pick[k] = (c, e / portal[k])
            rows.append(f"| {k} | {portal[k]:.1f} ({med(an[k]):.1f}) | " if an.get(k) else f"| {k} | {portal[k]:.1f} (-) | ")
            rows[-1] += " | ".join(cells) + f" | {pick[k][0] if k in pick else best['id']} |"
        for f in fails:
            print("failed (not a timing flake):", f)
        report = [f"# Per-workload merge for round {a.round}", "",
                  f"Projected portal us per workload (rented change in the same container in brackets; median of "
                  f"{2 * a.merge_reps} passes each). A candidate's rented time is scaled by the portal/rented ratio its own "
                  f"lineage shows in {best['id']}'s portal result; a workload moves when the projection beats the best's "
                  f"portal time by {100 * a.margin:.0f}% ({100 * a.margin_new:.0f}% for a lineage with no portal result yet, "
                  f"marked new).", "", *rows, ""]
        if not pick:
            report.append(f"No round kernel beat {best['id']} by the margin on any workload: no merge.")
            (d / "merge.md").write_text("\n".join(report) + "\n")
            print("\n".join(report))
            return
        leaves, route = merge.flatten(sols[best["id"]], best["id"])
        routes = {}
        for k in portal:
            if k in pick:
                sub, rt = merge.flatten(sols[pick[k][0]], pick[k][0])
                leaves.update(sub)
                routes[k] = rt(k)
            else:
                routes[k] = route(k)
        default = max(sorted(set(routes.values())), key=list(routes.values()).count)
        used = sorted({c for c, _ in pick.values()})
        desc = (f"Per-workload merge (loop/merge.py, kernels unchanged) of {best['id']} with {', '.join(used)} on "
                f"{', '.join(sorted(pick))}, where they beat it side by side.")
        sol = merge.build(leaves, routes, default, cid, description=desc)
        (d / "candidates").mkdir(parents=True, exist_ok=True)
        path = d / "candidates" / f"{cid}.json"
        path.write_text(json.dumps(sol, indent=1))
        res = b200_modal.handle.remote(dict(id=cid, kind="test", repeat=2, solution=sol,
                                            problem=b200_modal.problem_ref()))
    b200_modal.write_trace(d / "results" / "timing" / "B200r" / f"{cid}.jsonl", res)
    order_warning(cid, res)
    wl = res.get("workloads") or []
    proj = {k: portal[k] * (pick[k][1] if k in pick else 1.0) for k in portal}
    anc = planner.anchors()
    s0, s1 = planner.score(portal, anc), planner.score(proj, anc)
    card = dict(id=cid, parents=[best["id"]] + used, operation="repair", language=best.get("language", "cuda_cpp"),
                niche={**(best.get("niche") or {}), "spec": "dispatch:size"},
                hypothesis=f"Operator per-workload merge: {desc} Projected portal score {s1:.4f} (best {s0:.4f}).",
                dispatch=[dict(kernel=lid, shapes=" ".join(k for k in portal if routes[k] == lid))
                          for lid in sorted(set(routes.values()))])
    card_text = yaml.safe_dump(card, sort_keys=False, width=120)
    (d / "candidates" / f"{cid}.card.yaml").write_text(card_text)
    arc = get_archive()
    arc[cid] = dict(id=cid, round=a.round, parents=card["parents"], operation="repair", language=card["language"],
                    niche=card["niche"], card=card_text, solution=str(path.relative_to(ROOT)), status="proposed",
                    timings={}, task=dict(operation="repair", parents=card["parents"], band="all", niche="merge",
                                          instructions="Operator per-workload merge (round.py merge)."))
    archive.save(archive.refresh(arc))
    passed = sum(w["status"] == "PASSED" for w in wl)
    report += [f"Merged kernel `{path.relative_to(ROOT)}`: {passed}/{len(wl)} workloads passed its test "
               f"(timing flakes count as failures here; see the trace).",
               f"Projected portal score {s1:.4f} against the best's {s0:.4f} (#{best['b200']['submission']})."]
    (d / "merge.md").write_text("\n".join(report) + "\n")
    print("\n".join(report))


def sync_remote():
    subprocess.run(["rsync", "-a", "--exclude", "__pycache__", f"{ROOT}/loop/", f"{REMOTE}:{RSOLX}/loop/"], check=True)
    subprocess.run(["rsync", "-a", f"{ROOT}/poc/run_timing.py", f"{REMOTE}:{RSOLX}/poc/"], check=True)


def order_warning(cid, res):
    """Say loudly when a candidate fails the portal-order check (later passes of a repeat=2 test)."""
    import designer
    if "later_passes" not in res:
        print(f"B200r {cid}: no portal-order check in this test (run loop/order_check.py before submitting)")
    elif designer.order_check_failures(res):
        print(f"B200r {cid}: PORTAL-ORDER CHECK FAILED on "
              f"{', '.join(r['workload'] for r in designer.order_check_failures(res))}: do not submit")


def b200_test(round_id):
    """Test every candidate of a round on the rented B200 (Modal): timings -> results/timing/B200r, compile stats ->
    results/static_B200r.jsonl. Runs in this process (a few minutes); candidates already timed are skipped."""
    import b200_modal
    b200_modal.load_token()
    d = rdir(round_id)
    out = d / "results" / "timing" / "B200r"
    todo = [p for p in sorted((d / "candidates").glob("*.json")) if not (out / f"{p.stem}.jsonl").exists()]
    import designer
    measured = {}                                     # code hash -> {test, compile} from this round's sessions
    for f in (d / "sessions").glob("*.b200.json"):
        for key, v in json.loads(f.read_text()).items():
            measured.setdefault(key, {}).update(v)
    statics, reused, ran = [], 0, []
    t0 = time.time()
    for sol_path in todo:
        sol = json.loads(sol_path.read_text())
        m = measured.get(designer.code_key(sol), {})
        test_ok = m.get("test") and all(w["status"] == "PASSED" for w in m["test"].get("workloads") or [{}])
        if test_ok:
            b200_modal.write_trace(out / f"{sol_path.stem}.jsonl", m["test"])
            order_warning(sol_path.stem, m["test"])
            reused += 1
            print(f"B200r {sol_path.stem}: reused the session's B200 timing (same code)", flush=True)
            if m.get("compile"):
                statics.append(dict(id=sol_path.stem, kernels=m["compile"].get("kernels") or [],
                                    error=m["compile"].get("error") or ""))
                continue
        ran.append((sol_path, sol, m, not test_ok))
    if ran:
        with b200_modal.app.run():
            for sol_path, sol, m, need_test in ran:
                if not need_test:                     # timing reused; only the compile statistics are missing
                    comp = b200_modal.handle.remote(dict(id=f"{sol_path.stem}-c", kind="compile", solution=sol,
                                                                         problem=b200_modal.problem_ref()))
                    statics.append(dict(id=sol_path.stem, kernels=comp.get("kernels") or [],
                                        error=comp.get("error") or ""))
                    continue
                res = b200_modal.handle.remote(dict(id=sol_path.stem, kind="test", solution=sol, repeat=2,
                                                    problem=b200_modal.problem_ref()))
                b200_modal.write_trace(out / f"{sol_path.stem}.jsonl", res)
                order_warning(sol_path.stem, res)
                wl = res.get("workloads") or []
                comp = m.get("compile") or b200_modal.handle.remote(dict(id=f"{sol_path.stem}-c", kind="compile",
                                                                         problem=b200_modal.problem_ref(),
                                                                         solution=sol))
                statics.append(dict(id=sol_path.stem, kernels=comp.get("kernels") or [], error=comp.get("error") or ""))
                print(f"B200r {sol_path.stem}: {sum(w['status'] == 'PASSED' for w in wl)}/{len(wl)} passed", flush=True)
    if statics:
        with open(d / "results" / "static_B200r.jsonl", "a") as f:
            f.write("".join(json.dumps(x) + "\n" for x in statics))
    print(f"rented B200: {sum(1 for r in ran if r[3])} candidate(s) tested in {(time.time() - t0) / 60:.1f} min, "
          f"{reused} timing(s) reused from the sessions", flush=True)


def cmd_test(a):
    gpus = a.gpus.split(",")
    if "B200" in gpus:
        b200_test(a.round)
        gpus = [g for g in gpus if g != "B200"]
    if not gpus:
        return
    sync_remote()
    jobs = {}
    for gpu in gpus:
        acct, part = SLURM[gpu]
        out = ssh(f"cd {RSOLX} && sbatch --parsable -A {acct} -p {part} --export=ALL,ROUND={a.round} "
                  f"loop/jobs/test_round.sbatch")
        jobs[gpu] = out.strip()
        print(f"{gpu}: job {jobs[gpu]}")
    f = rdir(a.round) / "jobs.json"
    old = json.loads(f.read_text()) if f.exists() else {}
    f.write_text(json.dumps({**old, **jobs}, indent=1))


def cmd_status(a):
    jobs = json.loads((rdir(a.round) / "jobs.json").read_text())
    print(ssh(f"sacct -X -n -P -j {','.join(jobs.values())} -o JobID,Partition,State,Start,Elapsed"))


def cmd_collect(a):
    d = rdir(a.round)
    (d / "results").mkdir(parents=True, exist_ok=True)
    r = subprocess.run(["rsync", "-a", f"{REMOTE}:{RSOLX}/loop/rounds/{a.round}/results/", f"{d}/results/"])
    if r.returncode:
        print("(no CSF3 results for this round; using local results only)")
    arc = get_archive()            # archive.load() re-reads every result file, including the ones just pulled
    archive.save(arc)
    print(archive.table({k: v for k, v in arc.items() if v.get("round") == a.round or v.get("b200")}))


def cmd_shortlist(a):
    arc = get_archive()
    s = planner.shortlist(arc, a.round)
    out = rdir(a.round) / "portal"
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    lines = [f"# Portal shortlist for round {a.round}", "",
             f"Reference: {s['best']} (B200 score {s['best_score']:.4f}). Predicted scores project each candidate's "
             f"speed relative to the reference, per workload, on the rented B200 when measured there (else its most "
             f"representative cheap GPU), onto the reference's "
             f"B200 per-workload times.", "",
             "| file | basis | change vs reference S/M/L (rented) | predicted portal score | P(beats best) |",
             "|---|---|---|---|---|"]
    for w in s.get("awaiting", []):
        lines.append(f"| (not copied) `{w['id']}` | waiting for {w['gpu']} timings | - | run collect again when they land | - |")
    for p in s["picks"]:
        shutil.copy(ROOT / arc[p["id"]]["solution"], out / f"{p['id']}.json")
        pred = f"{p['predicted_score']:.4f}" if p["predicted_score"] else "unknown (exploration)"
        if p.get("sd"):
            pred += f" ± {p['sd']:.4f}"
        pb = f"{p['p_better']:.0%}" if p.get("p_better") is not None else "-"
        lines.append(f"| `{p['id']}.json` | {p['gpu'] or '-'} | {p['rel']} | {pred} | {pb} |")
    (out / "README.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def cmd_table(a):
    arc = get_archive()
    archive.save(arc)
    print(archive.table(arc))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--problem", default=None, help="problem name, e.g. L1/038 or 038_flux_multi_head_rmsnorm_qk "
                        "(default: $SOLX_PROBLEM or #38)")
    llm_args = argparse.ArgumentParser(add_help=False, parents=[common])
    llm_args.add_argument("--model", default=MODEL)
    llm_args.add_argument("--effort", default="high")
    llm_args.add_argument("--max-tokens", type=int, default=64000)
    llm_args.add_argument("--dry-run", action="store_true")
    llm_args.add_argument("--backend", choices=("openrouter", "claude"), default="openrouter",
                          help="claude: run design sessions as headless Claude Code on your Claude plan (implies "
                          "interactive tools on the rented B200)")
    llm_args.add_argument("--session-model", default="opus", help="claude backend: model for design sessions")
    llm_args.add_argument("--lead-model", default="fable", help="claude backend: model for the research lead")
    llm_args.add_argument("--session-timeout", type=int, default=60, help="claude backend: minutes per session")
    llm_args.add_argument("--parallel", type=int, default=4, help="claude backend: sessions at once (each runs a "
                          "claude process and an MCP server; keep low on a laptop)")
    llm_args.add_argument("--interactive", action="store_true",
                          help="design sessions with tools (compile, test, score model) on a CSF3 GPU tool server")
    llm_args.add_argument("--tool-gpu", default="A100,L40S", help="GPU type(s) for the session tool server: CSF3 types "
                          "(several with one Slurm account are queued together; the first free one is used), or B200 "
                          "for a rented B200 on Modal (run round.py with .venv/bin/python)")
    llm_args.add_argument("--b200-minutes", type=int, default=20, help="interactive with --tool-gpu B200: cap on "
                          "Modal B200 minutes per round (about $6.25 an hour)")
    llm_args.add_argument("--gpu-wait", type=int, default=180, help="interactive: minutes to wait for the tool server")
    llm_args.add_argument("--turns", type=int, default=12, help="interactive: max LLM turns per session")
    llm_args.add_argument("--max-turns", type=int, default=40, help="claude backend: turn cap per design/research "
                          "session (the lead gets 25)")
    llm_args.add_argument("--probes", type=int, default=30, help="interactive with --tool-gpu B200: max probe_b200 "
                          "experiments per session")
    llm_args.add_argument("--gpu-calls", type=int, default=6, help="interactive: max compile/test calls per session")
    llm_args.add_argument("--turn-effort", default="medium", help="interactive: reasoning effort after the first turn")
    llm_args.add_argument("--session-budget", type=float, default=5.0, help="interactive: USD per session before "
                          "the model is told to finish")
    p = sub.add_parser("propose", parents=[llm_args])
    p.add_argument("--round", required=True)
    p.add_argument("--operation", required=True)
    p.add_argument("--parents", nargs="*", default=[])
    p.add_argument("--band", default="all")
    p.add_argument("--niche", default="any empty niche")
    p.add_argument("--instructions", default="")
    for name in ("plan", "auto", "lead"):
        q = sub.add_parser(name, parents=[llm_args])
        q.add_argument("--round", required=True)
        q.add_argument("--max-tasks", type=int, default=3)
        q.add_argument("--last-round", help="round whose failures get repair tasks")
        q.add_argument("--use-plan", action="store_true", help="auto: run rounds/<round>/plan.json as it is")
        q.add_argument("--brief", help="extra guidance for the research lead this round")
        q.add_argument("--lead", action="store_true", help="auto: let the research lead (claude backend) update "
                       "the hypothesis ledger and choose the experiments instead of the planner")
        q.add_argument("--mode", choices=("exploit", "explore"), default="exploit",
                       help="exploit: improve the best kernel band by band; explore: one big idea per session")
        q.add_argument("--gpus", help="where to test the final candidates: B200 (rented, Modal) and/or CSF3 GPUs; "
                       "default B200 with --tool-gpu B200, else A100,H200")
        q.add_argument("--portfolio", action="store_true", help="lead: read the round's ideas pool and premise audit "
                       "(round.py ideas) and manage 2-3 parallel lineages (loop/portfolio.py)")
        q.add_argument("--no-merge", action="store_true", help="auto --portfolio: skip the per-workload merge step")
        q.add_argument("--merge-reps", type=int, default=2, help="merge: side-by-side reps per kernel")
        q.add_argument("--margin", type=float, default=0.02, help="merge: a workload moves to a round kernel only "
                       "when it is this much faster there than the best")
        q.add_argument("--margin-new", type=float, default=0.10, help="merge: the margin for a kernel whose lineage "
                       "has no portal result yet (its portal/rented ratio is unknown)")
        q.add_argument("--resume", action="store_true", help="auto --use-plan --backend claude: continue this "
                       "round's sessions that ended without a candidate (e.g. cut off by the plan's session limit) "
                       "from their saved Claude session; tasks that already gave a candidate are skipped")
    q = sub.add_parser("ideas", parents=[llm_args], help="portfolio round: idea sessions + premise audit (no GPU)")
    q.add_argument("--round", required=True)
    q.add_argument("--idea-models", default="opus,sonnet,fable", help="one idea session per model (comma-separated)")
    q.add_argument("--critic-model", default="opus", help="model for the premise audit ('' for none)")
    q = sub.add_parser("research", parents=[llm_args], help="research phase for a new problem")
    q.add_argument("--remeasure", action="store_true", help="redo the B200 copy-floor and reference measurements")
    q.add_argument("--from-reply", action="store_true", help="finish from the saved research reply (no agent run)")
    for q in (sub.choices["lead"],):
        q.add_argument("--from-reply", action="store_true", help="re-parse the saved lead reply (no agent run)")
    q = sub.add_parser("merge", parents=[common], help="per-workload merge of a round's kernels into the best one "
                       "(rented B200, one container)")
    q.add_argument("--round", required=True)
    q.add_argument("--merge-reps", type=int, default=2, help="side-by-side reps per kernel")
    q.add_argument("--margin", type=float, default=0.02, help="a workload moves to a round kernel only when it is "
                   "this much faster there than the best")
    q.add_argument("--margin-new", type=float, default=0.10, help="the margin for a kernel whose lineage "
                   "has no portal result yet (its portal/rented ratio is unknown)")
    for name in ("test", "status", "collect", "shortlist"):
        q = sub.add_parser(name, parents=[common])
        q.add_argument("--round", required=True)
        if name == "test":
            q.add_argument("--gpus", default="B200", help="B200 (rented, Modal; default) and/or CSF3 GPU types "
                           "(A100, H200, L40S: needs the CSF3 VPN/SSH)")
    sub.add_parser("table", parents=[common])
    a = ap.parse_args()
    if a.problem:                                   # every module and subprocess reads the problem from here
        os.environ["SOLX_PROBLEM"] = f"{problem.get(a.problem).level}/{problem.get(a.problem).name}"
    if getattr(a, "round", None) and not re.fullmatch(r"[a-z0-9-]+", a.round):
        sys.exit("--round must be lowercase letters, digits and dashes")
    dict(propose=cmd_propose, plan=cmd_plan, auto=cmd_auto, lead=cmd_lead, ideas=cmd_ideas, research=cmd_research,
         merge=cmd_merge, test=cmd_test, status=cmd_status, collect=cmd_collect, shortlist=cmd_shortlist, table=cmd_table)[a.cmd](a)


if __name__ == "__main__":
    main()
