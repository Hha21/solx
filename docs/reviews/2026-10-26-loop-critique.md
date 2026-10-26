# Outside critique of the solx loop after round 12 (2026-10-26)

Read-only review of the #129 (`035_convnextv2_block_with_grn`) rounds r3-r12 at `cd6b5f3`. Evidence is cited as
round / file / card. "Rented" means the Modal B200; "portal" means NVIDIA's scoring. No code was changed and nothing was
run on a GPU. Costs are API-equivalent dollars from `loop/runs/usage.jsonl`; the operator pays in Max-plan quota.

## Summary

1. **The loop is a very good refinement engine and a poor incubator.** Lineage D has had 12 of the 35 planned
   sessions since r3 (A 5, B 4, G 4, C 2, I 2, E/F/H/J/K/L one each; `rounds/r*/plan.json`) and has won almost every
   row since r6. Every design started since r8 (H, I, J, K, L) got one 90-minute session from scratch, a numeric kill
   line set by the lead, and a fallback of "ship the r0 anchor". Five of the eight non-merge candidates of r11-r12 are
   r0 placeholders or the unchanged parent (`r11-j-e3-gate-killed`, `r11-k-e4-overlap-gate`, `r12-i-e3-tc05-pass1-547`,
   `r12-l-e4-toeplitz-gate`, `r12-d-e1-h48-gate-parent`).
2. **New designs die at the setup, not at the idea.** J was killed on a configuration its own probes showed passes
   (KS2 cells 0.54-0.66 us against a 1.5 us line; `r11-j-e3-gate-killed` findings), after 15 minutes and 5 probes of a
   90-minute session. K stopped at 13 minutes. I reached a correct tcgen05 pass 1 that is 18 % faster than the kernel
   that holds the row, projected -13 % on 64,96 (about +0.0015), and shipped nothing because 547 us missed a 400 us
   line (`r12-i-e3-tc05-pass1-547`). The sessions do not run out of time, GPU or turns; they run out of permission.
3. **The incumbent bias is structural, and most of it is the operator's.** The r8-r12 briefs name "the six wide-C
   rows" every round, r11 ordered "all four experiments at the six wide-C sizes" (`rounds/r11/lead/prompt.txt`), so J
   and K were built as specialists for exactly the rows D has tuned for ten rounds. The lead's own half-rule was
   broken in r9 (D 2 of 3). New lineages face a 10 % merge margin against D's 3 % and can never calibrate their portal
   ratio because they never win a row (`round.py cmd_merge`, `--margin-new`). "Incubation" forbids retiring but allows
   pausing: G, I, J and K were all paused without a second session (`lineages.yaml`).
4. **D's own sessions now say its structure is near its floor.** r11 E1 found GEMM2 is L2-bandwidth-bound (16 TB/s
   cross-die), r11 E2 found the fp32 weight read costs >= 3.6 us of DRAM "whatever the access instruction", r12 E2 found
   the GEMM1 epilogue is issue-bound and the counter wait replaces the g pass at equal cost, r12 E1 found the sc rows'
   phase A is a latency-bound conv. The r12 lead: "Rounds on the six wide-C rows now return only +0.002 to +0.005 each."
   Gains r8-r12: +0.011, +0.005, +0.003, +0.001, +0.002. The live first place needs +0.0084 more, and the cheapest route
   is -15 to -23 % on five rows (SOL TARGETS, r12 lead prompt). Refining D will not get there at this rate.
5. **The ideas pool is exhausted and was reused to save about $10 a round.** The r4 pool served r5-r9 and the r10
   pool served r11-r12 (byte-identical `ideas/pool.json`, `SOURCE.txt`). The lead's "Pool families" ledger note at r12
   shows every family built, refuted, parked or "noise-level" except cache hints and the TF32 sc kernel. L was the last
   long shot (p_expert 0.05). The critic's five r10 premises were answered three rounds running because the critic was
   never re-run ("same five items re-raised", `ledger.yaml` notes).
6. **Sessions spend most of their output re-typing.** Nothing persists between tool calls (`designer.PROTOCOL`: "Pass
   complete files every time"), so one D session re-sent 410 KB of source to compile/run_tests plus 194 KB of probe
   scripts, about 151k tokens of tool-call text out of 490k output (`r11 .../00-lind-*.tools.jsonl`). Three of the four
   r12 cards record "/tmp does not persist between probes; embed full sources". The tcgen05 plumbing (idesc, descriptors,
   alloc, mbarrier ring) is re-typed in 5-8 probe scripts per session, in every lineage, every round (probe-script grep,
   section 3). The 2026-10-11 review's items 5 (session workspace) and 6 (context diet) were never built.
7. **The lead plans D experiments without being able to read D.** `get_kernel` on D's kernels returns 100-685 KB,
   above the tool limit; the r11 and r12 leads both open with "I could not read the r10/r11 D kernel cards"
   (`rounds/r11/lead/reply.md`, `rounds/r12/lead/reply.md`). The lead also re-emits the whole 46-56 KB ledger every
   round (`reply.md` block sizes), which both earlier reviews asked to replace with a validated diff.
8. **What to do, in one line each:** stop killing sessions at gates and forbid r0 placeholders; take the row list out
   of the operator brief; give the critic a GPU so kill-line facts are measured before a lineage is seeded; persist
   session files and page `get_kernel`; build a small kit of D's validated primitives that every lineage may include;
   guarantee a new lineage two consecutive sessions, the second resuming the first; refresh the pool every round but
   aim it at measured terms. The measurement side (merge projection within 0.001 for six rounds) needs nothing.

## 1. Why do new designs die?

### The record

| Lineage | Round | Session | Gate the lead set | Result | What the session left on the table |
|---|---|---|---|---|---|
| H dataflow queue | r8 | full kernel built, 15 probes, 16/16 pass | within 15 % of D on 16,384 / 8,384 (incubation line) | +58..+98 % vs D | per-item cost arithmetic was decisive; counter mechanics later reused by D r12 (H49). No second session: paused r9 "no slot", retired r10 |
| I top-row pass 1 | r10 | Modal app stopped for the whole first part; resumed; 9 probes, 7 min of model time | conv-only <= 250 us continue, <= 180 success | 233 us (2.2x FMA bound) | STEP 2/3 unbuilt; paused r11 "by directive" |
| J cluster split-K | r11 | 15 min wall, 5 probes, 1 run_tests, $3.6 | 50 KB exchange <= 1.5 us | 1.63-2.11 us at KS4-6 | its own probe: KS2 cells 0.54-0.66 us and fp16 partials pass; 75 minutes unused; nothing built |
| K overlapped engine | r11 | 13 min wall, 8 probes, $3.6 | epilogue and MMA each keep >= 80 % of solo rate when co-run | 60-74 % | 77 minutes unused; but STEP 0a produced r11's most valuable fact (straight-line issue reaches the MMA ideal at N >= 128), which refuted the "105 clk floor" that had steered G, I and K tile choices for two rounds |
| I (resumed) | r12 | 39 min, 17 probes, 2 builds, $8.8 | pass 1 <= 400 us | 547 us, correct on first engine build | pass 1 is -18 % vs G's 668; projected row -13 % (~+0.0015); STEP 3 unbuilt; budget went to a ring-barrier parity bug |
| L Toeplitz conv | r12 | 50 min, 11 probes, $9.4 | conv <= 100 us at 64,96, <= 25 at 32,128 | 155 / 57.8 us | smem operand traffic (28 B per output) binds; the per-channel tile cannot feed a fused GEMM1, a dead end a desk check could have found before the session |

Sources: `rounds/r8/candidates/r8-h-dataflow-queue.card.yaml`, `rounds/r10/candidates/r10-i-e3-*.card.yaml`,
`rounds/r11/candidates/r11-{j,k}-*.card.yaml`, `rounds/r12/candidates/r12-{i,l}-*.card.yaml`; wall time and cost from
`rounds/r*/sessions/*.claude.json`.

### Diagnosis

- **One session from scratch against a 12-session incumbent.** D's engine needed r3 (weight-stationary coop), r4
  (tcgen05 swap-AB), r6 (epilogue spill fix), r8 (all-warp epilogue), r10 (STACK-120 spill fix, "the round's real
  gain") to reach its current form. A new lineage is asked to beat that in 90 minutes, including re-deriving the engine
  from notebook prose. H r8, G r10 and I r12 each report "engine validated on the first build", which is to their
  credit, but the validation itself is a third of each session's probes.
- **Kill lines are tuned to "beats D this round", not to "worth a second session".** J's 1.5 us, K's 80 %, I's 400 us
  and L's 100 us were all derived in the lead's arithmetic from what the lineage would need to take a row from D in one
  step (`rounds/r11/plan.json`, `rounds/r12/plan.json` instructions). The instructions then say "If the gate fails,
  stop building and report the limiter" (K). A design that is 18 % better than the row's holder on its first build (I
  r12) is a success by any incubation standard and a failure by this one.
- **No reusable validated components.** The only shared assets are prose: the notebook recipe (H r8), D's conclusion
  line "Validated and reusable - MN-major SW128 B (LBO 8192, SBO 1024, +2048 per k16) ..." (`lineages.yaml`), and
  `b200probe.py`, which has `cuda_kernel`, `harness_time`, `sass`, `resources`, `flush_l2` and nothing tcgen05-specific
  (`loop/b200probe.py`). Every session re-types idesc, descriptors, alloc/relinquish, the mbarrier ring, the flat
  barrier and the stamps. Probe-script grep over r6-r12: `idesc` appears in 5-8 probe scripts in each of the D, G, H,
  I, K and L sessions. I r12 found "the swapped LBO/SBO is wrong" and a ring-parity deadlock on its own time.
- **Incubation exists on paper.** `LEAD_PROTOCOL`: "a lineage started in the last 2 rounds cannot be retired". The
  lead complied by pausing: G and I "paused r11 (operator directive)", J and K "paused r12 in incubation (not
  retirable)" (`lineages.yaml`). I's "incubation clock restarts" at r12. A rule that forbids retiring but not starving
  is not an incubation rule.
- **Lineage isolation isolates the wrong thing.** `portfolio.visible` hides other lineages' code so designs stay
  different. The intent is right (the SOTA review's islands), but it also hides the plumbing, so the isolation tax is
  paid in engine re-derivation rather than in design independence. The ideas that mattered to J, K and L (exchange cost,
  co-run efficiency, operand bandwidth) have nothing to do with seeing D's code.

### What would give a promising new design a fair chance

A new lineage should get: (a) its premise probed before it is seeded (section 4, change 3); (b) a kit of validated
primitives so the first session builds the design, not the engine (change 5); (c) a session contract that says "ship
the best measured kernel on at least one row; if the gate misses, spend the rest of the budget on the best fallback
configuration and measure it", never an r0 placeholder (change 1); (d) a second session in the next round that resumes
the first (change 6); (e) target rows where the incumbent is weakest relative to the leader *and* where its structure
is not already at its floor, which is the lead's call, not the brief's (change 2).

## 2. Is the loop biased towards the incumbent?

Yes, through five channels, and the bias has been cheap so far but is now costing score.

1. **The operator brief.** r8: "The six wide-C rows plus 8,320 and 4,96 are now the bottom". r9: "The six wide-C rows
   decide this round". r10: planner arithmetic for the six wide-C rows. r11: "plan FOUR experiments, all aimed at the
   six wide-C sizes". r12: "cheapest allocation now puts the cuts on five wide-C sizes" (`rounds/r*/lead/prompt.txt`,
   BRIEF sections). The brief also lists D's routing and D's "next levers" from D's cards. The r12 Opus lead pushed
   back: "This plan departs on purpose from the operator's preference for the five wide-C rows. The cheapest-allocation
   table is a cost model, and those rows now return the least per round." It was right, and E3/E4 (the top rows) were
   the only experiments that measured something structurally new. The memory note "lead-driven exploration" says the
   operator only checks rules and cost; the briefs do more than that.
2. **The lead's choices and parents.** D got 2 of 3 in r9 against the half-rule, with the justification "the user
   approved spending the round on the lowest rows" (`rounds/r9/lead/reply.md`). D's parents are always D's best; new
   lineages' parents are `[]`. The lead reads a 205 KB prompt in which the ledger (12k tokens) and notebook (19k) are
   D's history; the pool is 10.6k tokens and the lineages 3k (`rounds/r12/lead/prompt.txt` section sizes).
3. **The merge margin.** A known lineage moves a row at 3 % projected gain; a lineage with no portal result needs 10 %
   (`--margin-new`), because its portal/rented ratio is unknown. A new lineage can only learn its ratio by winning a row,
   which it cannot do without a 10 % win. G cleared it in r7 only because A was weak on 64,96 / 32,128 (-10.5 / -15.4 %).
   This is a sound guard against the r5 miss, but it is also a moat.
4. **The evidence the lead and sessions read.** `lab_notebook` shows the two newest rounds in full and 12 findings per
   older round; D ships three or four cards a round with 10-19 findings each, so D's findings dominate the window. The
   premise audit was answered three rounds in a row with the same five items because the critic was not re-run.
5. **The pool.** The r10 pool's 20 ideas (`rounds/r10/ideas/pool.json`) were by r12 all built, refuted, parked or
   folded into D (ledger "Pool families (dedup r12)"). Of the r4 pool's 30 ideas, the Toeplitz conv appeared three
   times (I13, I15, I29) and waited eight rounds for a session; I14 (die-local scratch placement), I19/I21 (hidden never
   materialised, TMEM-resident) and I12 (per-image dataflow with an L2-resident hidden ring) were never built. Most
   "new design" slots since r10 were drawn from the pool's bottom, which lowers the prior on new designs generally.

**Is it costing score?** Through r10, no: D delivered +0.05 of the +0.07 since r6, and the alternatives that were
measured (E, F, H) lost on arithmetic. From r11 on, yes: D's four r11-r12 experiments refuted H44 and H45, refuted H48,
and partly supported H49; the merge took one row in r12 (+0.0021). The five lowest rows (16,384 0.697; 8,384 0.710;
16,768 0.716; 8,768 0.720; 8,192 0.727 on #75393) need -15 to -23 %, and D's own cards attribute the remaining time to
DRAM, L2 bandwidth and issue limits of D's structure. Nothing else was allowed to mature enough to be compared.

## 3. Where the loop spends effort that does not pay

| Item | Evidence | Size |
|---|---|---|
| Source retransmission | `designer.PROTOCOL` "Pass complete files every time; nothing persists between calls"; r11 D E1 sent 410 KB of files plus 194 KB of probe scripts, ~151k tokens of ~490k output; r12 D E2 ~69k of 255k; L ~46k of 274k (`sessions/*.tools.jsonl`) | 25-30 % of session output |
| Parent source in the prompt | r12 D E2 dynamic prompt 74k tokens, of which PARENTS 38k (138 KB source), LEDGER 14k, NOTEBOOK 16k (`rounds/r12/replies/*01-lind*.prompt.txt`); sessions also have `get_kernel` | ~40k cached input tokens per D session |
| Re-deriving the engine | `idesc` / `tcgen05.alloc` / `mbarrier` in 5-8 probe scripts per session, every lineage, r6-r12; I r12's ring-parity bug; "the swapped LBO/SBO is wrong" | a third of a new lineage's probes |
| Re-building the measurement harness | r12 D E1 built a ctypes fast loop ("7 nvcc variants in 5.6 s ... harness_time over 5 rows x 4 variants x 3 rounds in 4 s") inside the session; it is not kept anywhere | one session's STEP 0 per round |
| Placeholder candidates through the merge | 5 of 8 non-merge r11-r12 candidates are r0 anchors or the unchanged parent; the merge times each on 16 rows x 4 passes (`merge.md` r11, r12) | most of the ~$7 Modal per round |
| Lead ledger rewrite | the ledger block is 46-56 KB of the 75-84 KB reply every round (`rounds/r9..r12/lead/reply.md`); no diff or validation (`cmd_lead` writes it whole; both earlier reviews flagged this) | ~15k output tokens per lead; drift risk |
| Stale audit | the r10 critic's five items answered identically in r10, r11 and r12 ("same five items re-raised", `ledger.yaml` notes) | lead text, no new information |
| Lead blind on D | `get_kernel` returns 100-685 KB for D; the r11 and r12 leads could not read the parent they planned for | STEP 0 of every D task becomes "read the parent card in parts" |
| STEP 0 rituals prescribed per task | interleaved parent x3, globaltimer stamps, STACK check, 1500-call barrier stress, 10-round fp64 precision log, all written out in each task (`plan.json` instructions are 3-4 KB each); r12 D E2: "NOT DONE from STEP 0 ... the budget went to H49 itself" | the rails are right; the per-session re-implementation is the waste |
| Pool reuse | saved ~$10-12 per round (r10 ideas cost $12.2, r4 $9.9) but left r11-r12's new designs to the pool's leftovers | false economy |

Where effort *does* pay and should be left alone: the per-size merge (projection within 0.001 for r7-r12), same-container
interleaved timing, the precision log, the STACK check (the r10 win), the findings lines (the notebook is why H's
counter mechanics could be reused by D r12 and why K's STEP 0a changed the ledger).

## 4. What would most raise creativity and gains per round

Weighing the examples the operator named:

- **A shared library of validated building blocks: yes, and it is the only item that changes what a new lineage can
  reach in one session.** The content already exists as prose and as D's code: tcgen05 descriptors and idesc, alloc /
  relinquish, the TMA 3-D box ring with OOB zero fill (I r10), the flat self-resetting barrier and the per-image counters
  (D r10, H r8), the A&S erfc GELU epilogue and staged coalesced stores (D r6/G r9), the LN fold algebra, the stamp
  macros, the STACK check, and the ctypes fast loop (D r12 E1). Shipped as headers and a `b200probe` module that probe
  containers and compile/run_tests can include, it removes the engine re-derivation and the harness rebuild from every
  session. Risk: the kit nudges designs toward D's skeleton. Mitigation: ship primitives (descriptor builders, barrier,
  ring, GELU, stamps), never a kernel skeleton, and keep lineage code isolation for everything above the primitives.
- **Multi-round incubation: yes, but as a session continuation, not as a status.** The resume mechanism exists
  (`run_claude(resume=session_id)`, `RESUME_NOTE`) and worked for the r10 outage. A second session of a new lineage
  should resume the first's transcript (or at minimum receive its archived probe files and card) and must build the
  design, not re-gate it. Two consecutive sessions per new lineage is the minimum that H, I, J and K did not get.
- **Fresh pools every round: yes, but targeted.** The generic "10 ideas, 3 long shots" prompt yielded 30 and 20 ideas
  of which the lead could use 3-4 a round, and by the second reuse the pool was empty. A pool aimed at the measured
  terms ("the fp32 weight read costs 3.6 us of DRAM at C=768 every call", "GEMM2 stages are L2-bound at 16 TB/s", "the
  GEMM1 epilogue is 3.5 us per tile issue-bound", "the sc rows' conv is latency-bound at 6 warps per SM") with the
  cards' stamps attached would produce ideas the lead can schedule. Cost ~$10-12. The saving from reuse was smaller
  than one placeholder session.
- **More, cheaper sessions vs fewer, deeper: neither.** Sessions do not exhaust depth: GPU time 0.3-4.7 minutes of a
  20-30 minute allowance, J and K ended at 13-15 of 90 minutes, turns 5-29 of 40. The binding constraint is one idea
  per session with a kill switch. Keep the 90 minutes, give each session a primary and a fallback configuration, and
  move the premise probes to a cheaper pre-round tier (below).
- **Different models per role: marginal.** Opus led r11-r12 at $3 against Fable's $6-7 and wrote the sharper
  assessments (r12 pushed back on the brief; r11 ordered a premise test that paid). The idea tier already mixes opus /
  sonnet / fable. The one role that would benefit from a model change is none; the role that needs a *tool* change is
  the critic (a GPU).
- **Sessions that continue a previous session's work: yes** (above). The archive already carries the disabled probe
  kernels (`i_tmaconv_gate.cu`, `i_pass1_tc05.cu`), so the cost is the resume plumbing and one protocol sentence.
- **Not asked, but cheap: fresh-eyes sessions were never run on #129.** `prompts.fresh_prompt` and `FRESH_DOCS` exist
  and no plan r3-r12 used them (`plan.json`, zero `fresh` tasks). One fresh session per three rounds, given only the
  definition, the per-row times to beat and the rules, is the cheapest test of whether the loop has anchored on D.
- **Also not asked: give the critic a GPU.** J, K and L were each killed by one fact a single probe measures (cluster
  co-residency at size >= 3 is 105-135 CTAs; a 50 KB DSMEM exchange costs 1.6 us; co-run efficiency is 60-70 %;
  Toeplitz operands cost 28 B per output). The ideas protocol already asks each idea for "the cheapest test (one probe
  of about 10 minutes)". Running those probes in the ideas tier, before the lead plans, turns gate-killed sessions into
  pre-filtered ideas and lets the design session start building.

## 5. What more compute would buy

- **A free B200.** Saves about $7 a round and removes the Modal outages (r10 lost two sessions to "function ... is
  stopped"; `commit be91a1d`). On its own, little: GPU time is not the bottleneck. It becomes valuable with two builds
  that convert quota into GPU time: a persistent per-session workspace (no retransmission, files survive between
  probes) and an unattended sweep harness for the tile tables (KS / MP / NS / N) that sessions now sweep by hand with
  3-7 probes each (`r12-d-e2` "GEMM2 tables after g removal", `r12-d-e1` "NS re-swept on all 7 sc rows"). An overnight
  sweep costs no Claude tokens and is the kind of exhaustive tuning that found the STACK-120 and MP=2 wins.
- **More parallel GPUs.** Not binding. Four sessions already run in parallel, each with its own container.
- **A portal-clocked GPU (1500 MHz lock).** The merge projection has matched the portal within 0.001 for six rounds
  with per-lineage ratios; the remaining uncertainty is +-1.2 us on the S rows, which a lock would not remove (it is
  host launch behaviour). Low value for #129 now. Medium value only if the loop moves to a compute-bound problem.
- **More Claude quota.** This is the binding resource, and the loop spends about $40-65 a round on sessions plus
  $3-7 on the lead. A fresh targeted pool (+$12) and a second session for each new lineage (+$5-10) are the two items
  that need more of it. The larger lever is spending the same quota differently: no placeholders, no retransmission,
  no 55 KB ledger copies, no stale audit answers. Those changes free about a third of a round's session output.

## 6. Ranked changes

Ordered by expected value per unit of quota. "Edit" = prompt or protocol text; "Build" = code.

| # | Change | Kind | Why (evidence) | Expected effect per round | Cost | Risk |
|---|---|---|---|---|---|---|
| 1 | **Gate reform and a no-placeholder rule.** In `portfolio.LEAD_PROTOCOL` step 5 and `designer.PROTOCOL`: a gate miss means "switch to the best fallback configuration from your own probes and measure it", not "stop and report"; every session ships a measured kernel on at least one row, and a card whose candidate is the r0 anchor or the unchanged parent is not a valid hand-in. Gates for new lineages are relative to the design's own projection ("within 1.5x of the pipe bound") and may only be kill lines when the premise is hardware (co-residency, exchange cost). | Edit | J 15 min / K 13 min of 90; I r12 left a -13 % row unbuilt; 5 of 8 r11-r12 candidates were placeholders | 1-2 more measured kernels per round from the same sessions; the merge gets candidates instead of anchors | $0 | sessions ship weak kernels; the merge margin already filters them |
| 2 | **Take the row list and D's levers out of the operator brief.** The brief states the score, the gap, the live #1, the budget and rules. Row choice, lineage choice and levers are the lead's (`cmd_lead --brief`). | Process | r8-r12 briefs name the six wide-C rows every round; r11 ordered all four there; the r12 lead's pushback was the round's best decision | new designs aimed where the incumbent is weak relative to the leader, not at D's rows | $0 | the lead chooses badly; one round shows it |
| 3 | **Premise probes in the ideas tier.** Give the critic (and optionally each idea session) 5-10 `probe_b200` calls (`cmd_ideas` currently passes `--no-gpu`); the critic runs each pool idea's stated "cheapest test" and attaches the number; the lead seeds only lineages whose premise survived. | Build (small) + Edit | J, K, L were each killed by one probe-sized fact; the IDEAS_PROTOCOL already asks for that test | design sessions start building; fewer gate deaths; the pool carries measured priors | +$3-5 Modal, +$3-5 quota | probe results on stubs mislead (K's runtime-flag pitfall); require compile-time variants |
| 4 | **Session workspace and `get_kernel` paging.** Files persist in the session's container or a Modal volume; `compile_b200` / `run_tests` / `probe_b200` take paths and `edit_file(old, new)`; `get_kernel(id, part="card" or "sources" or a file name)`. | Build (half a day to a day) | 25-30 % of session output is retransmitted source; "/tmp does not persist" in 3 of 4 r12 cards; the lead could not read D for two rounds | ~30 % more experiments per session at the same quota; the lead plans with the kernel in view | $0 recurring | a session edits the wrong file; keep the full-file path as a fallback |
| 5 | **A kit of validated primitives.** `loop/kit/` headers plus `b200probe` helpers: tcgen05 descriptor and idesc builders, alloc / relinquish, TMA box ring with mbarrier, flat self-resetting barrier and per-image counters, A&S erfc GELU epilogue with staged coalesced stores, LN fold, stamp macros, STACK check, the ctypes fast loop. Ship it into the probe container and let any lineage `#include` it. Primitives only; no kernel skeleton. | Build (1-2 days; one no-GPU session can extract it from D r11, H r8, I r10/r12) | engine re-typed in 5-8 probes per session; I r12's ring bug; the lineages.yaml "Validated and reusable" line is prose | a new lineage reaches a measured full kernel in one session instead of a gate stub; D sessions skip STEP 0 plumbing | one extraction session (~$10-20) + a validation probe | designs converge on D's structure; mitigated by shipping primitives only |
| 6 | **Two consecutive sessions per new lineage, the second resuming the first.** `LEAD_PROTOCOL` step 4: a lineage started last round gets a session this round, no exceptions; `cmd_auto` resumes the previous session id for that lineage (the `--resume` path exists) with a note "build, do not re-gate". | Edit + Build (small) | G, I, J, K paused without a second session; I lost two rounds and re-derived its engine; the resume path already works | new designs get measured against D after two sessions instead of never | +$5-10 per round | fewer D slots; D's return per slot is now +0.001-0.002 |
| 7 | **Fresh, targeted pool every round.** `cmd_ideas` runs each round; `IDEAS_PROTOCOL` adds a section "the measured terms" built from the newest cards' stamps and the SOL TARGETS cheapest allocation, and asks for ideas against those terms as well as whole-kernel long shots; near-duplicates of ledger entries are dropped by the critic. | Edit | both pools were exhausted by their second reuse; r11-r12 new designs came from the pool's bottom; the audit was stale for three rounds | the lead has 3-4 schedulable new families a round and a current audit | +$10-12 per round | idea quality varies; the critic's probes (change 3) filter |
| 8 | **The lead emits a ledger diff, not the ledger.** `LEAD_PROTOCOL` output: a `yaml ledger-changes` block (new / changed hypotheses by id, status changes with evidence, note edits); `cmd_lead` applies it and validates (ids preserved, constraints unchanged, statuses only move with evidence). | Edit + Build (small) | 46-56 KB of every lead reply is a verbatim ledger copy; both earlier reviews asked for this | ~15k fewer output tokens per lead; no silent drift | $0 | the diff format fails to parse; keep the raw block on failure as now |
| 9 | **One fresh-eyes session every third round.** `fresh: true` task with `prompts.fresh_prompt` (definition, times to beat, rules only), no notebook, no lineages. | Edit | zero fresh-eyes tasks on #129 in r3-r12 though the code exists; the only test of anchoring | an independent design or an independent confirmation that D's structure is right | one session (~$5-10) every third round | it reproduces D; that is information too |
| 10 | **Unattended tile-table sweeps.** A `round.py sweep` step that takes a kernel's compile-time table (KS, MP, NS, N, warp split) and sweeps it interleaved against the parent in one container, overnight, no Claude tokens. | Build (half a day) | sessions sweep tables by hand with 3-7 probes each; the STACK fix and MP=2 came from exactly this kind of tuning | converts GPU time, which is cheap, into the tuning sessions now do; frees sessions for structure | Modal only (or a free B200) | sweeps overfit rented clocks; validate in the merge as now |

Not recommended now: buying a clock-locked GPU (section 5); changing the merge margins (3 % / 10 % are right; the fix
for the 10 % moat is change 6, which earns a lineage its portal ratio); personas, temperature or model shuffling for
creativity (the 2026-10-11 review's evidence table still holds).

A round after changes 1-3 and 6-7 would look like: ideas tier with a GPU ($15, 40 min), lead ($3), four sessions of
which two continue last round's new lineages and two are D or a fresh seed ($35-45), merge ($7). About the same quota
as r11-r12, with every session shipping a measured kernel.

## 7. The earlier reviews: what was adopted and whether it worked

| Recommendation (2026-10-10 / 10-11) | Status at r12 | Did it work? |
|---|---|---|
| Fix lab notebook ordering and truncation | Done (`4cefe39`) | Yes; findings carry across rounds (H's counters reused r12) |
| Constraint lint, probe and turn budgets | Done | Yes; budgets no longer bind at all (section 3) |
| Keep 2-3 lineages with incubation and per-band elites | Done as `portfolio.py` and the merge | The merge is the loop's best mechanism (+0.05 since r6). Incubation did not work: paused is the new retired |
| Idea tier with verbalized sampling and dedup | Done (`cmd_ideas`, `IDEAS_PROTOCOL`) | Produced the families that became G, H, I, J, K, L; then reused until empty |
| Premise audit before each lead | Done (`CRITIC_PROTOCOL`) | Caught the 105 clk premise (took two rounds to be tested); then reused stale for three rounds |
| Session workspace and diff edits | Not done | Retransmission is still 25-30 % of session output |
| Context diet and retrieval (state card, paged tools) | Not done | 74k-token dynamic prompts; the lead cannot read D |
| Validate the lead's ledger rewrites, store a diff | Not done (both reviews) | 55 KB copied per round |
| Clock-locked B200 (Verda) | Open | No longer needed for #129 (projection within 0.001) |
| Ask NVIDIA about reduced-precision operands | Unknown from the repo | Still the largest rules risk for a public #129 entry |

## Open questions for the operator

1. Is the goal to pass 0.7509 on #129, or to build a loop that would do so on the next problem too? Changes 4, 5 and
   10 pay on every future problem; changes 1-3 and 6-7 pay on #129 within two rounds.
2. Would you accept a round in which D gets one session or none? Its last four experiments returned +0.001-0.002
   each and its own cards place the remaining terms at hardware limits of its structure.
3. Has the FP16-operand question been put to NVIDIA? Every row of the current submission rounds d, W1, h and W2 to
   fp16 (`ledger.yaml` precision constraint); the 2026-10-11 review's item 11 is still the one thing that could zero
   the board.
4. Is a Modal volume (or a free B200 with a disk) acceptable for session files? It is the cheapest form of change 4.
