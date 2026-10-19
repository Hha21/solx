# Design session 20261019-183907_00-lind-structural_mutation-all

Task: {"round": "r10"}


### probe_b200 (1): check tmp persistence and tools


### probe_b200 (2): persistence check


### probe_b200 (3): harness_time API


### probe_b200 (4): write part 1 of kernel.cu (anchor path, verbatim)


### probe_b200 (5): part 2: tc_kernel verbatim (tc_w2_work templated on params)


### compile_b200 (draft 1): r10 E1: tc2_kernel balanced token ranges, no g, in-place scaled GEMM2


### run_tests (draft 2): r10 E1 v1: tc2 on 8/16/32,768 and 8/16,384


### probe_b200 (6): test container reuse for staged writes


### probe_b200 (7): persistence check 2


### probe_b200 (8): stage helper part of probe source


### probe_b200 (9): stage parent tc_kernel with extra stamps


### probe_b200 (10): stage tc2 kernel (stamped) for probe


### probe_b200 (11): build stamped probe extension


### probe_b200 (12): stamps tc2 vs parent at 16,384 / 8,768 / 32,768


### probe_b200 (13): patch v2: NGRP=1, gw preload, half2 d16, runtime KSL stride; rebuild


### run_tests (draft 3): r10 E1 v2: NGRP=1 scaling, gw preload, half2 d16, KS=2 at 16,384/32,768


### probe_b200 (14): stage helpers for precision/stress probe


### probe_b200 (15): stage parent tc_kernel (unstamped)


### probe_b200 (16): stage tc2 v2 kernel (shipped arithmetic)


### probe_b200 (17): build precision/stress probe ext + cuobjdump res-usage


### probe_b200 (18): precision log, 1500-call stress, interleaved timing, STL count


### probe_b200 (19): spill fix attempt in tc_kernel<768> 7x7 drain (tld16 halves)


### probe_b200 (20): two spill-fix variants for tc_kernel<768>, check STACK


### probe_b200 (21): locate spill sites in tc_kernel<768> SASS


### probe_b200 (22): spill fix variants C (hoist W2 work) and D (+hreg sized per C)


### probe_b200 (23): time spill-free tc_kernel (variant C) vs shipped parent, interleaved x3
