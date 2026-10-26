"""Portfolio rounds: independent idea sessions and a premise audit before the lead plans, and 2-3 parallel lineages.

Why (docs/reviews/2026-10-11-sota-agentic-review.md): a single lineage refined under +-0.008 portal noise is a noisy
hill-climb that converges on the first workable design. What has evidence for widening the search: ideas written in
plain language before any code, sampled with a stated probability that an expert would propose them (verbalized
sampling, with required long shots), deduplicated by a judge; several lineages that share conclusions but not code
(islands); incubation so a new idea is not judged on its first crude kernel; and a critic that audits the project's
own premises against its measurements (a wrong tensor-core ceiling once hid real headroom for three rounds).

  round.py ideas --round r1          idea sessions (no GPU) + premise audit -> rounds/r1/ideas/pool.json
  round.py lead  --round r1 --portfolio      the lead reads the pool and the audit, manages lineages, writes plan.json
  round.py auto  --round r1 --use-plan       one design session per experiment, each sees only its lineage's code

State: problems/<p>/lineages.yaml (written by the lead). A kernel's lineage is its task's `lineage` field.
"""

import json
import random

import yaml

import problem

IDEAS_PROTOCOL = """=== IDEAS PROTOCOL ===
You are one of several independent idea generators for this problem. This session writes no kernel and has no GPU.
Your output is a list of design ideas in plain language. The research lead deduplicates the ideas of all generators,
weighs them and starts the best, most different ones as parallel lineages (independent lines of development, each
built and measured by its own design session on a rented B200).
1. Read the problem, the bounds and the evidence below (ledger, notebook, portal results, leader gap, lineages).
   You may read kernels (get_kernel), public SOTA excerpts (read_example) and value a speed-up per size band
   (predict_score).
2. Write 10 ideas. Each is a whole-kernel design or a decisive change to one, specific enough to start building: what
   is computed where, what moves through DRAM / L2 / shared memory / TMEM / registers, how many launches, which
   precision per stage, and why that is fast on B200 for these shapes. Work out the numbers (bytes, FLOPs, launches)
   per size band. Aim at least half of the ideas at the measured terms in the newest design cards' findings (stamps,
   per-phase times, what each card says binds) and at the SOL TARGETS cheapest allocation: name the term an idea removes
   and by how many microseconds. The rest may be whole-kernel long shots. Skip ideas the ledger already records as built
   or refuted unless you say what is different.
3. Spread them out. At most 3 ideas may share a family. At least 3 must be long shots: ideas a strong B200 kernel
   engineer would propose with probability below 0.2, that could still win. Rewriting the maths (reordering, folding
   or fusing operations), another decomposition of the work, a different precision per stage, unusual use of the
   memory hierarchy, and fewer or more launches are all fair game within the rules.
4. For each idea: p_expert (probability, 0-1, that a strong engineer proposes it unprompted), the expected effect per
   band with the arithmetic behind it, the main risk (what would kill it), and the cheapest test (one probe of about
   10 minutes that would most change our belief in it).
5. Leader stories: assume the public leader's implied per-workload times (LEADER GAP) are real. Give 3 different
   explanations of how they reach them that fit our measurements, each with a test.
Rules: never propose a forbidden behaviour (core brief, ledger constraints). Any precision is allowed as long as every
workload meets its tolerance; say how you would check that.
This session returns ideas, not a kernel: ignore the OUTPUT CONTRACT and answer in this format:
### Notes
(at most 150 words)
```json ideas
[{"title": "...", "family": "two or three words", "design": "...", "p_expert": 0.3, "effect": "...",
  "risk": "...", "test": "..."}]
```
```json leader-stories
[{"story": "...", "consistent_with": "...", "test": "..."}]
```"""

CRITIC_PROTOCOL = """=== PREMISE AUDIT PROTOCOL ===
You are the critic. You propose no designs. Find the premises this project relies on that are wrong, stale,
unmeasured or stronger than their evidence, before the next round builds on them. (A ceiling computed from our own
achieved rate instead of the datasheet once made the project dismiss a real ~30% headroom for three rounds: that is
the kind of error you are here to catch.)
1. List the load-bearing claims: ledger statuses (supported / refuted), "settled" numbers, ceilings and floors,
   "X does not help", the problem card's bounds and design priorities, and the newest findings.
2. Check each against first principles (B200 datasheet rates at the portal's 1500 MHz SM lock, bytes and FLOPs
   arithmetic, the tolerance) and against the measurements recorded here (portal per-workload results, bench
   numbers, probe findings). Use get_kernel to read the code and design cards a claim rests on.
3. Report only claims that are wrong, unsupported or overstated: the claim and where it lives, why it is doubtful
   (with numbers), what changes if it is wrong, and the cheapest measurement that settles it. Flag hypotheses refuted
   or supported on one noisy result (portal noise is about +-0.008 score per submission; bench effects under ~3%
   need interleaved repeats). Fewer, solid objections beat many weak ones; "no doubtful premises" is a valid answer.
4. Probe the pool. You see this round's IDEAS POOL and, when the rented B200 is attached, have probe_b200 (seconds per
   probe; no compile_b200 or run_tests). For the ideas with the best expected value whose stated cheapest test fits in a
   probe, and for your own doubtful premises, run the test and report the number, so the lead seeds lineages whose
   premise has survived. Time only compile-time variants: runtime flags inside a timed kernel slow it and mislead (r11).
This session returns an audit, not a kernel: ignore the OUTPUT CONTRACT and answer in this format:
### Audit
(at most 400 words)
```json premises
[{"claim": "...", "where": "ledger H7 / card section 4 / finding in <kernel id>", "verdict": "wrong|unsupported|overstated",
  "why": "...", "if_wrong": "...", "test": "..."}]
```
```json probes
[{"idea": "I7 or a premise", "test": "what the probe measured", "result": "the number, with units",
  "verdict": "holds|fails|unclear", "note": "what it means for the idea"}]
```"""

LEAD_PROTOCOL = """=== RESEARCH LEAD PROTOCOL (PORTFOLIO) ===
You are the research lead. You do not write kernels. You run a portfolio of 2-3 lineages: independent lines of
development, each with its own thesis and its own best kernel, developed in parallel by design sessions on rented
B200s. Lineages share conclusions (the lab notebook and the lineage conclusions), not code: a design session can read
only its own lineage's kernels (and the round-r0 baseline), so lineages stay different. The operator only checks
rules and cost; the scientific choices are yours.
Each round:
1. Read the evidence below: the IDEAS POOL (from independent idea sessions; order and sources hidden), the PREMISE
   AUDIT (a critic's list of doubtful claims), the LINEAGES, the ledger, the portal results, the LEADER GAP (what the
   public top scores imply per workload, and the best score reachable per precision at the hardware ceilings), the
   lab notebook and the archive. Use get_kernel, read_example and predict_score as needed.
2. Answer the audit: for each doubtful premise, correct the ledger now (cite the evidence), schedule the critic's
   test inside an experiment, or say in one line why it stands.
3. Deduplicate the pool into distinct families. Weigh each: the score gain if it works (predict_score, leader gap)
   times your probability that it works, against its cost in rounds. Do not discard an idea for being unusual: a low
   p_expert with a sound mechanism is what lineages are for.
4. Manage the portfolio ({n} experiments this round in total, 2-3 active lineages):
   - Keep a lineage while it improves, or while its ceiling (argued with numbers) is above the best current score.
   - Incubation: a lineage started in the last 2 rounds cannot be retired or paused, and a lineage started last round
     gets an experiment this round. Its second session continues the first: set "continue" to the first session's
     kernel id and the session resumes that conversation; it builds on that kernel and its findings and does not
     re-run the first session's gate. The only exception is a premise refuted by a measured hardware fact
     (name it). First kernels are expected to be crude.
   - Retire a lineage after incubation when it is behind and its argued ceiling is below the best lineage's current
     score. Record why in its conclusions; a retired lineage's ideas stay in the pool for later.
   - When a slot is free, seed a new lineage from the pool: among the families with good expected value, pick the one
     most different from the active lineages.
   - Give the leading lineage at most half of the experiments.
   - At least one experiment each round is a NEW DESIGN: operation new_design, a kernel built from scratch (parents []
     or the r0 baseline) around a pool family that no lineage has built yet, not a change to an existing lineage's
     kernel. It starts a new lineage (take a free slot, or retire or pause one and say why). Prefer families aimed at
     the workloads furthest from the leader.
   - Every third round (round number divisible by 3), make one experiment a fresh-eyes task ("fresh": true): that
     session sees only the definition, the times to beat and the rules, no notebook and no earlier kernels, and designs
     from scratch. It is the loop's check on anchoring; it starts a lineage like any new design.
   - After the round the operator merges per workload automatically: for every workload the merged kernel runs whichever
     kernel was fastest there (this round's kernels against the current best, timed side by side in one container). A
     kernel therefore earns score by winning some workloads, not all of them; a specialist that wins only the workloads
     furthest from the leader is worth building, and it may leave the other workloads to a simple fallback.
5. Design the experiments. Each belongs to one lineage, builds on that lineage's best kernel (or starts the lineage,
   with parents [] or the r0 baseline), tests one hypothesis, and states what to build or measure, the success
   criterion and what would refute it. Gates for a new lineage are relative to the design's own projection (for
   example within 1.5x of its own pipe bound), not to beating the current best this round; a kill line is only for a
   hardware premise a probe measures (co-residency, an exchange cost, a pipe rate). A missed gate means the session
   builds and measures its best fallback configuration, never that it hands in a placeholder. Each session has a rented B200 (probes take seconds, full tests 10-20 s;
   run_tests projects the portal score with the emulator, which misranks effects under ~3%). Prefer experiments that
   could move the portal score by 0.005 or more. Assign a model: opus for open-ended design and debugging, sonnet for
   well-specified work.
6. Update the ledger and the lineages: thesis, status, best kernel, and at most 5 short conclusion lines each (all
   sessions read every lineage's conclusions). The ledger is updated by a diff: list only the hypotheses you add or
   change, each complete (id, statement, status, evidence); a status moves only with new evidence cited in that entry.
   Constraints are the operator's and cannot be changed here.
Output, in this order:
### Assessment
(at most 300 words: where the remaining score is, what each lineage has taught us, what this round tests and why)
```yaml lineages
lineages:
  - id: A
    thesis: one sentence
    status: active
    started: r1
    best: null
    conclusions:
      - one short line
```
```yaml ledger-changes
hypotheses:          # only new or changed entries, each complete; unlisted entries stay as they are
  - id: H51
    statement: ...
    status: open     # open | supported | refuted | parked
    evidence:
      - ...
notes:               # optional: the complete new notes list, only if you change it
  - ...
```
```json tasks
[{{"id": "E1", "lineage": "A", "hypothesis": "H8", "title": "...", "operation": "new_design",
  "parents": ["<an archive id from this lineage, or the r0 baseline, or nothing to start fresh>"], "band": "S|M|L|all",
  "instructions": "what to build or measure, step by step", "success": "...", "refuted_if": "...",
  "model": "opus|sonnet", "fresh": false, "continue": "<last round's kernel id of this lineage, or omit>"}}]
```"""


def lineages_file():
    return problem.current().dir / "lineages.yaml"


def load_lineages():
    f = lineages_file()
    if not f.exists():
        return []
    return (yaml.safe_load(f.read_text()) or {}).get("lineages") or []


def lineage_of(rec):
    """The lineage of an archive record (its task's `lineage`), or None for baselines and pre-portfolio kernels."""
    return ((rec or {}).get("task") or {}).get("lineage")


def visible(rec, lineage):
    """Whether a session of `lineage` may read this kernel: its own lineage, r0 baselines and pre-portfolio kernels."""
    own = lineage_of(rec)
    return lineage is None or own is None or own == lineage or str(rec.get("round")) == "r0"


def lineages_text(lins=None):
    lins = load_lineages() if lins is None else lins
    if not lins:
        return "(no lineages yet: this is the first portfolio round; seed 2-3 from the ideas pool)"
    out = []
    for l in lins:
        out.append(f"- {l.get('id')} [{l.get('status', 'active')}, started {l.get('started', '?')}, best "
                   f"{l.get('best') or 'none'}]: {l.get('thesis', '')}")
        out.extend(f"    - {c}" for c in l.get("conclusions") or [])
    return "\n".join(out)


def session_brief(lineage):
    """The lineage section of a design session's task: its own thesis and conclusions, and the others' conclusions."""
    lins = load_lineages()
    own = next((l for l in lins if l.get("id") == lineage), None)
    others = [l for l in lins if l.get("id") != lineage]
    text = (f"\n\nLINEAGE {lineage}: {own.get('thesis', '') if own else '(new)'}\n"
            f"You develop this lineage. get_kernel reads only this lineage's kernels and the r0 baseline; other "
            f"lineages share their conclusions, not their code. Stay with the thesis: if you find it cannot win, say so "
            f"with numbers in the design card's findings rather than switching to another lineage's design.")
    if own and own.get("conclusions"):
        text += "\nThis lineage so far:\n" + "\n".join(f"- {c}" for c in own["conclusions"])
    if others:
        text += "\nOther lineages (conclusions only):\n" + lineages_text(others)
    return text


def save_lineages(text):
    """Validate and save the lead's ```yaml lineages``` block; returns the parsed list."""
    data = yaml.safe_load(text) or {}
    lins = data.get("lineages") if isinstance(data, dict) else data
    if not isinstance(lins, list) or not all(isinstance(l, dict) and l.get("id") for l in lins):
        raise ValueError("lineages block must be a list of mappings with an id")
    lineages_file().write_text(yaml.safe_dump({"lineages": lins}, sort_keys=False, allow_unicode=True, width=120))
    return lins


def pool(replies, seed=0):
    """Merge idea-session replies into one pool: ideas shuffled and renumbered (sources hidden from the lead)."""
    import reply
    ideas, stories, premises, probes, sources = [], [], [], [], {}
    for name, text in replies.items():
        blocks = reply.named_blocks(text)
        for key, body in blocks.items():
            try:
                items = json.loads(body)
            except ValueError:
                continue
            if not isinstance(items, list):
                continue
            if key.startswith("json") and "ideas" in key:
                ideas.extend((name, i) for i in items if isinstance(i, dict))
            elif key.startswith("json") and "leader" in key:
                stories.extend(s for s in items if isinstance(s, dict))
            elif key.startswith("json") and "premises" in key:
                premises.extend(p for p in items if isinstance(p, dict))
            elif key.startswith("json") and "probes" in key:
                probes.extend(p for p in items if isinstance(p, dict))
    random.Random(seed).shuffle(ideas)
    out = []
    for n, (src, idea) in enumerate(ideas, 1):
        sources[f"I{n}"] = src
        out.append(dict(id=f"I{n}", **{k: v for k, v in idea.items() if k != "id"}))
    return dict(ideas=out, leader_stories=stories, premises=premises, probes=probes, sources=sources)


def pool_text(p):
    """The ideas pool and the audit as prompt sections for the lead."""
    ideas = "\n".join(f"- {i['id']} [{i.get('family', '-')}, p_expert {i.get('p_expert', '?')}] {i.get('title', '')}: "
                      f"{i.get('design', '')} | effect: {i.get('effect', '')} | risk: {i.get('risk', '')} | test: "
                      f"{i.get('test', '')}" for i in p.get("ideas") or [])
    stories = "\n".join(f"- {s.get('story', '')} (fits: {s.get('consistent_with', '')}; test: {s.get('test', '')})"
                        for s in p.get("leader_stories") or [])
    audit = "\n".join(f"- [{q.get('verdict', '?')}] {q.get('claim', '')} ({q.get('where', '')}): {q.get('why', '')} | "
                      f"if wrong: {q.get('if_wrong', '')} | test: {q.get('test', '')}" for q in p.get("premises") or [])
    probes = "\n".join(f"- [{q.get('verdict', '?')}] {q.get('idea', '')}: {q.get('test', '')} -> {q.get('result', '')}"
                       f"{' | ' + q['note'] if q.get('note') else ''}" for q in p.get("probes") or [])
    return (f"=== IDEAS POOL ({len(p.get('ideas') or [])} ideas; order and sources hidden) ===\n{ideas or '(none)'}\n\n"
            f"Leader stories:\n{stories or '(none)'}",
            f"=== PREMISE AUDIT (critic) ===\n{audit or '(no doubtful premises reported)'}\n\n"
            f"Premise probes (critic, measured on the rented B200):\n{probes or '(none run)'}")
