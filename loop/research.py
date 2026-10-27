"""Research phase for a new problem: fixed B200 measurements, then a research agent writes the problem's briefing.

1. measure(): on the rented B200, for every workload, (a) a plain one-shot read+write stream over exactly the
   workload's input and output bytes (the memory floor; saved as problems/<p>/b200/copy_reference.json, which the
   emulator and run_tests use) and (b) NVIDIA's PyTorch reference timed like the harness times it
   (reference_times.json).
2. round.py research runs a research agent (headless Claude, default Fable) with the design tools. It studies the
   definition, the reference code and those numbers, and writes the problem card (card.md), the first hypothesis
   ledger (ledger.yaml) and a simple, correct first kernel (round r0). Submitting that kernel gives the portal anchors
   (hidden baseline and SOL time per workload) that the score model and the emulator need.
"""

import json

import problem

PROBE = r'''
import importlib.util, json, sys, torch
import b200probe as bp
from sol_execbench.core.data import Definition, Workload
from sol_execbench.core.bench.io import gen_inputs, allocate_outputs
prob = "/problems/PROBLEM"
d = Definition.model_validate(json.load(open(f"{prob}/definition.json")))
rs = importlib.util.spec_from_file_location("ref", f"{prob}/reference.py"); ref = importlib.util.module_from_spec(rs)
rs.loader.exec_module(ref)
custom = getattr(ref, d.custom_inputs_entrypoint) if d.custom_inputs_entrypoint else None
wls = [Workload.model_validate(json.loads(l)) for l in open(f"{prob}/workload.jsonl") if l.strip()]
SRC = r"""
extern "C" __global__ void stream_io(const float4* __restrict__ a, long n_in4, float4* __restrict__ b, long n_out4) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
    if (i < n_in4) v = __ldg(a + i);
    if (i < n_out4) b[i] = v;
}"""
k = bp.cuda_kernel(SRC, "stream_io")
nbytes = lambda xs: sum(x.numel() * x.element_size() for x in xs if isinstance(x, torch.Tensor))
out, seen = [], set()
for w in wls:
    sig = tuple(sorted(w.axes.items()))
    if sig in seen:
        continue
    seen.add(sig)
    rec = {"axes": dict(w.axes)}
    try:
        inputs = gen_inputs(d, w, "cuda", custom_inputs_fn=custom)
        outputs = allocate_outputs(d, d.get_resolved_axes_values(w.axes), "cuda")
        bin_, bout = nbytes(inputs), nbytes(outputs)
        rec.update(in_mb=bin_ / 1e6, out_mb=bout / 1e6)
        a = torch.empty(max(16, (bin_ + 15) // 16 * 4), device="cuda"); b = torch.empty(max(16, (bout + 15) // 16 * 4), device="cuda")
        n_in4, n_out4 = bin_ // 16, bout // 16
        grid = (max(n_in4, n_out4, 1) + 255) // 256
        rec["copy_us"] = bp.harness_time(lambda x, y: k((grid,), (256,), x, bp.i64(n_in4), y, bp.i64(n_out4)), [a], [b])
        try:
            rec["ref_us"] = bp.harness_time(lambda *xs: ref.run(*xs), inputs, [])
        except Exception as e:
            rec["ref_error"] = f"{type(e).__name__}: {e}"[:300]
    except Exception as e:
        rec["error"] = f"{type(e).__name__}: {e}"[:300]
    out.append(rec)
print("RESULT " + json.dumps(out))
'''


def measure():
    """Copy floor and PyTorch reference time per workload on the rented B200; saved under problems/<p>/b200/."""
    import b200_modal
    p = problem.current()
    b200_modal.load_token()
    with b200_modal.app.run():
        r = b200_modal.call(dict(id="research", kind="probe", timeout=600,
                                          script=PROBE.replace("PROBLEM", f"{p.level}/{p.name}")))
    line = next((l for l in (r.get("stdout") or "").splitlines() if l.startswith("RESULT ")), None)
    if not line:
        raise SystemExit(f"research measurements failed:\n{(r.get('stdout') or '')[-1500:]}\n{(r.get('stderr') or '')[-2500:]}")
    recs = json.loads(line[len("RESULT "):])
    p.b200.mkdir(parents=True, exist_ok=True)
    copy = {p.key(x["axes"]): round(x["copy_us"], 3) for x in recs if x.get("copy_us")}
    (p.b200 / "copy_reference.json").write_text(json.dumps(copy, indent=1))
    rows = {p.key(x["axes"]): {k: v for k, v in x.items() if k != "axes"} for x in recs}
    (p.b200 / "reference_times.json").write_text(json.dumps(rows, indent=1))
    return rows


def measurements_table(rows):
    p = problem.current()
    lines = ["| workload | band | traffic MB (in + out) | plain stream floor µs (rented B200) | PyTorch reference µs | "
             "reference / floor |", "|---|---|---|---|---|---|"]
    for k in sorted(rows, key=p.mbytes):
        r = rows[k]
        traffic = f"{r['in_mb']:.2f} + {r['out_mb']:.2f}" if "in_mb" in r else "-"
        floor = f"{r['copy_us']:.2f}" if r.get("copy_us") else "-"
        ref = f"{r['ref_us']:.2f}" if r.get("ref_us") else (r.get("ref_error") or r.get("error") or "-")[:60]
        ratio = f"{r['ref_us'] / r['copy_us']:.1f}x" if r.get("ref_us") and r.get("copy_us") else "-"
        lines.append(f"| {k} | {p.band(k)} | {traffic} | {floor} | {ref} | {ratio} |")
    return "\n".join(lines)


RESEARCH_PROTOCOL = """=== RESEARCH PHASE PROTOCOL ===
You are starting work on a new SOL-ExecBench problem. No kernel exists yet. Your job is to understand the problem well
enough that later design rounds start from solid ground, and to hand in a simple, correct first kernel.
1. Study the definition (axes, shapes, dtypes), the PyTorch reference (the exact maths, numerics and edge cases), the
   workloads (sizes, tolerances) and the measurements below: a plain read+write stream over exactly each workload's
   bytes (the memory floor on the rented B200) and the PyTorch reference timed like the harness times it.
2. Work out the bounds per workload: compulsory bytes, FLOPs, arithmetic intensity against B200's ~8 TB/s and its
   tensor/FP32 throughput, so whether each size is memory-, compute- or launch-bound; where the reference loses
   (extra passes, materialised intermediates, launches); what the SOL bound probably is. Use probe_b200 (seconds) to
   check anything you are unsure of, and read_example / the briefing for B200 techniques.
3. Write a first kernel: simple, correct for every workload and tolerance, reasonably fast (one fused launch if the
   maths allows). Test it with run_tests until it passes everywhere. It will be submitted to the portal to obtain the
   hidden baseline and SOL anchors per workload, so correctness matters more than speed.
   Stop at the first version that passes everywhere: do not optimise it (the design rounds do that, with several
   independent approaches). The session is killed after 60 minutes with nothing saved, so write your final answer by
   about 40 minutes even if questions remain open; list them in the card instead.
Output, in this order:
```markdown card
(the problem card, in the style of the example card: 1. semantics; 2. numerics and tolerance; 3. the workloads
table with bytes, FLOPs, band and the measured floor/reference times; 4. bounds and what dominates per band;
5. where the reference loses (facts: extra passes, materialised intermediates, launches); 6. the decisions any design
must make, each with its options and their arithmetic, without choosing (independent idea sessions and the research
lead choose later; a card that picks one design narrows every later session); 7. results so far (your first kernel's
bench numbers); 8. open questions)
```
```yaml ledger
(constraints copied unchanged, then 4-10 hypotheses H1..Hn about what will matter, each with status open and the
evidence you have, using the same schema as the example ledger)
```
Then the first kernel in the OUTPUT CONTRACT format (solution-spec, files, design card with paths and findings).
"""
