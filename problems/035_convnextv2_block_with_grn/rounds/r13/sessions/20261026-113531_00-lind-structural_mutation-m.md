# Design session 20261026-113531_00-lind-structural_mutation-m

Task: {"round": "r13"}


### probe_b200 (1): check /tmp state


### probe_b200 (2): H53 build: balanced token-range tiles; correctness first


### probe_b200 (3): check whether k.cu got written


### probe_b200 (4): H53 build: balanced token-range tiles; correctness first


### probe_b200 (5): H53 timing new vs old tables, alt tables, hints, stamps


### probe_b200 (6): single-round inv reduction loads


### probe_b200 (7): split pass 2: reduction vs stores; no-reduction stub


### probe_b200 (8): float4 predicated reduction loads


### probe_b200 (9): whole-image fast path for 7x7 + 14x14 balanced + GEMM2 KS sweep


### probe_b200 (10): self-contained: whole-image fast path + 14x14 balanced + KS sweep


### probe_b200 (11): precision log + counter stress on the shipped table


### run_tests (draft 1): 


### probe_b200 (12): diagnose harness vs stamp gap


### probe_b200 (13): stage r12 reference build (same container next)


### probe_b200 (14): persistence check


### probe_b200 (15): probe assembly check (phase B text needed)


### probe_b200 (16): A/B: r12 tc_kernel vs new (ring full / 160K) vs new with whole 14x14 tables


### probe_b200 (17): bisect regression: publish fence vs per-thread gsum loads


### probe_b200 (18): stage shared kernel parts (r12 and new share phase A/C)


### probe_b200 (19): stage r12 phase B


### probe_b200 (20): bisect: r12 vs new code variants in one container


### run_tests (draft 2): 


### probe_b200 (21): final build: STACK, 10-round log, 400-call stress, A/B vs r12


### compile_b200 (draft 3): 
