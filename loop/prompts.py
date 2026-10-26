"""Assemble inner-LLM prompts in the generation-protocol layout.

The static part (core brief, problem card, reference docs, protocol) is identical for every call, so it is sent as
a cached system prompt. The dynamic part (task, archive, parents, feedback) changes per call.
"""

import csv
import json
import re
from pathlib import Path

from archive import ROOT, band_geomeans, fmt, table

CTX = ROOT / "loop" / "context"
STATIC_DOCS = ["core_brief.md", "tricks.md", "PROBLEM_CARD", "harness_scoring.md", "b200_arch.md", "b200_sota.md",
               "playbook_membound.md", "playbook_compute.md", "generation_protocol.md"]   # PROBLEM_CARD = problems/<name>/card.md
CONTRACT = """=== OUTPUT CONTRACT ===
Optional "### Rationale" (<=150 words). Then, per candidate:
1) ```json solution-spec``` (the solution JSON without "sources")
2) one ```<lang> file=<path>``` block per source file (complete files)
3) ```yaml design-card``` following the schema in the generation protocol (block style, one key per line).
No other fenced code blocks. Never use a forbidden behaviour; if the task needs one, return no candidate and explain."""


MEMORY_ONLY_DOCS = ("b200_sota.md", "playbook_membound.md")
COMPUTE_ONLY_DOCS = ("playbook_compute.md",)

# Fresh-eyes sessions get the rules, the harness and the hardware, but none of the project's own design history
# (problem card, playbook, SOTA notes with our results, ledger, notebook, archive kernels): every normal session reads
# that history and tends to refine the existing design instead of looking for a different, possibly simpler one.
FRESH_DOCS = ["core_brief.md", "tricks.md", "harness_scoring.md", "b200_arch.md", "OUTPUT_FORMAT"]


def protocol_section(n):
    """Section n of generation_protocol.md (section 3 = the output format and design card schema)."""
    text = (CTX / "generation_protocol.md").read_text()
    m = re.search(rf"^## {n}\..*?(?=^## {n + 1}\.|\Z)", text, re.S | re.M)
    return m.group(0).strip() if m else text


def static_prompt(fresh=False):
    parts = ["You are the kernel-design step of an automated optimisation loop for NVIDIA's SOL-ExecBench B200 "
             "leaderboard. The documents below are your briefing; follow the output contract exactly."]
    import problem
    p = problem.current()
    docs = FRESH_DOCS if fresh else STATIC_DOCS
    if p.kind == "compute":                         # the playbook and SOTA notes are about memory-bound norms (#38)
        docs = [d for d in docs if d not in MEMORY_ONLY_DOCS]
    else:
        docs = [d for d in docs if d not in COMPUTE_ONLY_DOCS]
    docs = [d for d in docs if d in ("OUTPUT_FORMAT", "PROBLEM_CARD") or (CTX / d).exists()]
    for name in docs:
        if name == "OUTPUT_FORMAT":
            parts.append(f"=== output format (generation_protocol.md section 3) ===\n{protocol_section(3)}")
            continue
        if name == "PROBLEM_CARD":
            card = p.card.read_text() if p.card.exists() else "(no problem card yet: run round.py research first)"
            parts.append(f"=== problem card: {p.level}/{p.name} ===\n{card}")
        else:
            parts.append(f"=== {name} ===\n{(CTX / name).read_text()}")
    parts.append(CONTRACT)
    return "\n\n".join(parts)


def b200_workload_table(vid):
    """Per-workload B200 results for a parent, with the hidden baseline, from the ingested portal pages."""
    import archive
    import problem
    rows = [r for r in archive.portal_rows(archive.PORTAL_WL) if r["vid"] == vid]
    if not rows:
        return ""
    out = ["| workload | latency µs | baseline µs | score |", "|---|---|---|---|"]
    for r in sorted(rows, key=lambda r: problem.current().mbytes(problem.current().key_from_label(r["workload"]))):
        out.append(f"| {r['workload']} | {float(r['latency_ms']) * 1e3:.1f} | {float(r['baseline_ms']) * 1e3:.1f} | "
                   f"{float(r['sol_score']):.3f} |")
    return "\n".join(out)


def parent_block(rec, sources=True):
    sol = json.loads((ROOT / rec["solution"]).read_text())
    src = "\n\n".join(f"--- file {s['path']} ---\n{s['content']}" for s in sol["sources"]) if sources else "(see parts)"
    meas = "; ".join(f"{g} {fmt(band_geomeans(t))}" for g, t in rec.get("timings", {}).items() if t)
    b200 = rec.get("b200") or {}
    static = json.dumps(rec.get("static_sm100") or [])
    return (f"--- parent {rec['id']} ---\n"
            f"design card:\n{rec.get('card', '')}\n"
            f"measurements (geomean µs per band S/M/L): {meas or 'none yet'}\n"
            f"B200 portal: score {b200.get('score', 'not submitted')}, geomean {b200.get('geomean_us', '-')} µs\n"
            f"{b200_workload_table(rec['id'])}\n"
            f"sm_100a static features: {static}\n"
            f"source:\n{src}\n")


GET_KERNEL_MAX = 60000      # characters per get_kernel answer; MCP clients refuse much larger tool results


def kernel_view(rec, part=""):
    """get_kernel's answer: the whole kernel if it fits, else the card and a file index; part picks a file or lines."""
    files = {s["path"]: s["content"] for s in json.loads((ROOT / rec["solution"]).read_text())["sources"]}
    if not part:
        full = parent_block(rec)
        if len(full) <= GET_KERNEL_MAX:
            return full
        part = "card"
    if part == "card":
        index = "\n".join(f"  {p}: {len(c.splitlines())} lines, {len(c)} chars" for p, c in files.items())
        return (parent_block(rec, sources=False) + "files (read one with part=\"<path>\" or a line range with "
                f"part=\"<path>:<first>-<last>\"; at most {GET_KERNEL_MAX} characters per call):\n{index}\n")
    path, _, rng = part.partition(":")
    if path not in files:
        return f"unknown part {part!r} of {rec['id']}; parts: card, " + ", ".join(files)
    lines = files[path].splitlines()
    first, last = 1, len(lines)
    if rng:
        try:
            first, last = (int(x) for x in rng.split("-", 1))
        except ValueError:
            return f"line range must look like {path}:120-480"
    out, n = [], 0
    for i in range(max(first, 1), min(last, len(lines)) + 1):
        if n + len(lines[i - 1]) + 1 > GET_KERNEL_MAX:
            last = i - 1
            break
        out.append(lines[i - 1])
        n += len(lines[i - 1]) + 1
    more = f"; continue with part=\"{path}:{last + 1}-{len(lines)}\"" if last < len(lines) else ""
    return f"--- {rec['id']} file {path}, lines {first}-{last} of {len(lines)}{more} ---\n" + "\n".join(out) + "\n"


def lab_notebook(arc, limit=160, per_round=12):
    """What earlier rounds learned: portal outcomes against the bench, and findings recorded in design cards."""
    import math
    import planner
    from archive import parse_card
    lines = []
    best = planner.best_kernel(arc)
    bp, br = (best["b200"]["timings"], (best.get("timings") or {}).get("B200r", {})) if best else ({}, {})

    def rel(t, ref, band):
        ks = [k for k in t if planner.band_of(k) == band and k in ref and t[k] and ref[k]]
        return 100 * (math.exp(sum(math.log(t[k] / ref[k]) for k in ks) / len(ks)) - 1) if ks else None
    out = []
    for r in arc.values():
        p_t, r_t = (r.get("b200") or {}).get("timings"), (r.get("timings") or {}).get("B200r")
        if best and p_t and r_t and r["id"] != best["id"] and str(r.get("round", "")).startswith(("r", "d", "c")):
            cells = []
            for b in "SML":
                a, c = rel(p_t, bp, b), rel(r_t, br, b)
                if a is not None and c is not None:
                    cells.append(f"{b} bench {c:+.1f}% / portal {a:+.1f}%")
            out.append(f"- {r['id']} (portal {r['b200']['score']:.4f}) vs {best['id']}: " + "; ".join(cells))
    if out:
        lines.append("Portal outcomes vs the bench (change vs the current best, same kernel pair):")
        lines.extend(sorted(out))
    # Findings grouped by round, newest round first. Rounds are ordered by when their kernels entered the archive
    # (registration order), not by name: "r9" > "r14" as strings, and c/d/x operator rounds sit between r-rounds.
    # The two newest rounds are shown in full, older ones keep their last `per_round` findings, up to `limit` lines.
    order, groups = {}, {}
    for r in arc.values():
        rnd = str(r.get("round", ""))
        card = parse_card(r.get("card", "")) if r.get("card") else {}
        found = [f"- [{r['id']}] {str(f).strip()}" for f in card.get("findings") or []]
        if found:
            order[rnd] = len(order) if rnd not in order else order[rnd]
            groups.setdefault(rnd, []).extend(found)
    if groups:
        lines.append("Findings recorded by design sessions, newest round first:")
        shown = omitted = 0
        for i, rnd in enumerate(sorted(groups, key=lambda x: order[x], reverse=True)):
            take = groups[rnd] if i < 2 else groups[rnd][-per_round:]
            take = take[:max(0, limit - shown)]
            if take:
                lines.append(f"Round {rnd}:")
                lines.extend(take)
            shown += len(take)
            omitted += len(groups[rnd]) - len(take)
        if omitted:
            lines.append(f"({omitted} older findings not shown; the hypothesis ledger summarises them, and get_kernel "
                         f"returns any kernel's full card)")
    return "\n".join(lines) or "empty"


def fresh_prompt(arc, round_id, extra=""):
    """Task prompt for a fresh-eyes session: the problem itself and the times to beat, nothing about how."""
    import planner
    import problem
    p = problem.current()
    d = p.definition
    anc = planner.anchors()
    best = planner.best_kernel(arc)
    bt = planner.b200_times(best) if best else {}
    S = lambda k, t: (anc[k][0] - anc[k][1]) / ((t - anc[k][1]) + (anc[k][0] - anc[k][1]))
    rows = ["| workload | MB moved | band | Tb (S=0.5) µs | Tsol (S=1) µs | time to beat µs | its score |",
            "|---|---|---|---|---|---|---|"]
    for k in sorted(anc, key=p.mbytes):
        t = bt.get(k)
        rows.append(f"| {', '.join(f'{a}={v}' for a, v in zip(p.var_axes, k.split(',')))} | {p.mbytes(k):.1f} | {p.band(k)} | {anc[k][0]:.1f} | "
                    f"{anc[k][1]:.1f} | {t if t is None else f'{t:.1f}'} | {'' if t is None else f'{S(k, t):.3f}'} |")
    try:
        import yaml
        rules = [r for r in yaml.safe_load(p.ledger.read_text()).get("constraints") or []
                 if not r.startswith("Clusters only")]          # a performance finding, not a rule: keep it hidden
    except Exception:
        rules = []
    io = {sect: {k: (v.get("shape"), v["dtype"]) for k, v in d[sect].items()} for sect in ("inputs", "outputs")}
    return "\n\n".join([
        f"=== TASK ===\nRound: {round_id}   Problem: {p.level}/{p.name}\nFRESH-EYES SESSION. You are deliberately "
        f"given no earlier designs, notes or results beyond the times to beat below. Work from first principles and "
        f"from your own measurements. Return 1 candidate following the OUTPUT CONTRACT, with an id starting with "
        f"'{round_id}-', parents [] and operation new_design.\n{extra}",
        f"=== PROBLEM DEFINITION ===\nname: {d['name']}\ndescription: {d.get('description', '')}\n"
        f"axes: {json.dumps(d['axes'])}\ninputs/outputs (shape, dtype): {json.dumps(io)}\n\nreference:\n{d['reference']}",
        "=== WORKLOADS AND TIMES TO BEAT (portal B200, µs) ===\n" + "\n".join(rows) +
        f"\n\nThe score is the mean over workloads of S = (Tb - Tsol) / ((t - Tsol) + (Tb - Tsol)). The 'time to beat' "
        + (f"is the current best submission (portal score {best['b200']['score']:.4f}). " if best else "") +
        "On the rented B200 the reference and your kernel are timed side by side; run_tests reports both.",
        "=== RULES OF THIS PROJECT (on top of the core brief's forbidden list) ===\n" +
        ("\n".join(f"- {r}" for r in rules) or "- none"),
    ])


def dynamic_prompt(arc, round_id, operation, parents, n=1, band="all", niche="any empty niche", extra="",
                   feedback="none"):
    import problem
    measured = [r for r in arc.values() if r.get("b200")]
    calib = "\n".join(f"- {r['id']}: B200 score {r['b200']['score']:.3f}, geomean {r['b200']['geomean_us']:.1f} µs, "
                      f"B200 S/M/L {fmt(band_geomeans(r['b200'].get('timings', {})))} µs" for r in measured)
    return "\n\n".join([
        f"=== TASK ===\nRound: {round_id}   Operation: {operation}\nProblem: {problem.current().level}/"
        f"{problem.current().name}   "
        f"Target niche: {niche}   Target band: {band}\nReturn {n} candidate(s). Each must follow OUTPUT CONTRACT "
        f"exactly. Give every candidate a new unique id starting with '{round_id}-'.\n{extra}",
        "=== ARCHIVE (best per kernel; times are geomean µs per size band S/M/L; S = B·S ≤ 600, M ≤ 2100, L above) "
        f"===\n{table(arc)}\nB200 results so far:\n{calib or 'none'}",
        f"=== HYPOTHESIS LEDGER ===\n{problem.current().ledger.read_text() if problem.current().ledger.exists() else 'none'}",
        f"=== LAB NOTEBOOK ===\n{lab_notebook(arc)}",
        "=== PARENTS ===\n" + ("\n".join(parent_block(arc[p]) for p in parents) if parents else "none"),
        f"=== FEEDBACK FROM LAST ATTEMPT ===\n{feedback}",
    ])
